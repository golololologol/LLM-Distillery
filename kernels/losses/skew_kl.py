import torch
from typing import Protocol, cast
from kernels.compiler import _compile_kernel, _compile_kernel_with_header, _cuda_available


class _SkewKlKernel(Protocol):
    def fused_skew_kl(self, *args: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


class _FusedTrainSkewKlKernel(Protocol):
    def fused_train_skew_kl(self, *args: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


_TRAINING_SKEW_KL_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_skew_kl_kernel(
    const float* __restrict__ student,      // [N, 256]
    const float* __restrict__ teacher,      // [N, 256]
    const int*   __restrict__ actual_bytes, // [N]
    float* __restrict__ grad_student,       // [N, 256]
    float* __restrict__ loss_srkl,          // [N]
    float* __restrict__ loss_ce,            // [N]
    float* __restrict__ loss_tce,           // [N]
    int N,
    float lam,
    float alpha
) {
    const int pos = blockIdx.x;
    const int byte_idx = threadIdx.x;
    if (pos >= N || byte_idx >= 256) return;

    const float eps = 1e-8f;
    const float inv_N = 1.0f / (float)N;
    const int base = pos * 256;

    float S = student[base + byte_idx];
    float T_val = teacher[base + byte_idx];
    float S_eps = S + eps;
    float mix = (1.0f - lam) * T_val + lam * S;
    float mix_eps = mix + eps;

    // Per-byte SRKL contribution
    float skew_kl_contrib = S * (logf(S_eps) - logf(mix_eps));

    // Block-reduce to get per-position SRKL
    float srkl_sum = blockReduceSum(skew_kl_contrib);

    int actual_b = actual_bytes[pos];

    if (byte_idx == 0) {
        loss_srkl[pos] = srkl_sum;
        loss_ce[pos] = -logf(student[base + actual_b] + eps);
        loss_tce[pos] = -logf(teacher[base + actual_b] + eps);
    }

    // Gradient of loss w.r.t. S[b]
    // d(srkl_i)/d(S_b) = log(S_b+eps) + S_b/(S_b+eps) - log(mix_b+eps) - lam*S_b/(mix_b+eps)
    float grad = inv_N * (logf(S_eps) + S / S_eps - logf(mix_eps) - lam * S / mix_eps);

    if (byte_idx == actual_b) {
        grad += alpha * inv_N * (-1.0f / S_eps);
    }

    grad_student[base + byte_idx] = grad;
}

std::vector<torch::Tensor> fused_skew_kl(
    torch::Tensor student,
    torch::Tensor teacher,
    torch::Tensor actual_bytes,
    float lam,
    float alpha
) {
    int N = student.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(student.device());
    auto grad_student = torch::empty({N, 256}, opts);
    auto loss_srkl = torch::empty({N}, opts);
    auto loss_ce = torch::empty({N}, opts);
    auto loss_tce = torch::empty({N}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_skew_kl_kernel<<<N, 256, 0, stream>>>(
        student.data_ptr<float>(),
        teacher.data_ptr<float>(),
        actual_bytes.data_ptr<int>(),
        grad_student.data_ptr<float>(),
        loss_srkl.data_ptr<float>(),
        loss_ce.data_ptr<float>(),
        loss_tce.data_ptr<float>(),
        N, lam, alpha
    );

    return {grad_student, loss_srkl, loss_ce, loss_tce};
}
"""

_TRAINING_SKEW_KL_CPP = """
std::vector<torch::Tensor> fused_skew_kl(
    torch::Tensor student,
    torch::Tensor teacher,
    torch::Tensor actual_bytes,
    float lam,
    float alpha
);
"""


def get_skew_kl_kernel() -> _SkewKlKernel | None:
    if not _cuda_available:
        return None
    kernel = _compile_kernel_with_header(
        "fused_skew_kl",
        _TRAINING_SKEW_KL_CPP,
        _TRAINING_SKEW_KL_CUDA,
        ["fused_skew_kl"],
    )
    return None if kernel is None else cast(_SkewKlKernel, kernel)



def fused_skew_kl_forward_backward(student_dists, teacher_dists, actual_bytes, alpha, lam=0.1, entropy_weights=None):
    kernel = get_skew_kl_kernel()
    if kernel is None:
        return None, None

    grad_student, loss_srkl, loss_ce, loss_tce = kernel.fused_skew_kl(
        student_dists.float().contiguous(),
        teacher_dists.float().contiguous(),
        actual_bytes.int().contiguous(),
        lam, alpha,
    )

    if entropy_weights is not None:
        N = loss_srkl.shape[0]
        w_sum = entropy_weights.sum().clamp(min=1e-8)
        scale = entropy_weights * N / w_sum
        grad_student = grad_student * scale.unsqueeze(1)
        kl_loss = (loss_srkl * entropy_weights).sum() / w_sum
        ce_loss = (loss_ce * entropy_weights).sum() / w_sum
    else:
        kl_loss = loss_srkl.mean()
        ce_loss = loss_ce.mean()
    total_loss = kl_loss + alpha * ce_loss

    return grad_student, {
        "train_loss": total_loss,
        "custom loss": total_loss,
        "CE loss": loss_ce.mean().detach(),
        "kl_div": loss_srkl.mean().detach(),
        "teacher CE loss": loss_tce.mean().detach(),
    }


# ============================================================
# Training kernel: fused abomination loss + gradient (two-pass)
# ============================================================



_FUSED_TRAIN_SKEW_KL_CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Process one depth: normalize bins, compute loss, compute grad_unnorm, write back to bins.
// bins: pointer to 256-float bin array (shared or global), must be normalized in-place first.
// Returns nothing; bins are overwritten with grad_unnorm values.
__device__ void process_depth_skew_kl(
    float* bins, float Z,
    const float* teacher_src, float* teacher_row, float* reduce_buf,
    int dist_idx, const int* actual_bytes,
    const uint8_t* byte_mask,
    float* loss_per_dist, float* ce_per_dist, float* tce_per_dist,
    float* teacher_entropy_out,
    float inv_N, float lam, float alpha, int tid
) {
    const float eps = 1e-8f;

    if (byte_mask[dist_idx] == 0) {
        if (tid == 0) {
            loss_per_dist[dist_idx] = 0.0f;
            ce_per_dist[dist_idx] = 0.0f;
            tce_per_dist[dist_idx] = 0.0f;
            teacher_entropy_out[dist_idx] = 0.0f;
        }
        if (tid < 256) bins[tid] = 0.0f;
        __syncthreads();
        return;
    }

    // Normalize bins
    if (tid < 256) {
        bins[tid] /= Z;
        teacher_row[tid] = teacher_src[tid];
    }
    __syncthreads();

    // Compute per-byte SRKL and reduce for per-dist loss
    float srkl_b = 0.0f;
    if (tid < 256) {
        float S = bins[tid];
        float T_val = teacher_row[tid];
        float mix = (1.0f - lam) * T_val + lam * S;
        srkl_b = S * (logf(S + eps) - logf(mix + eps));
    }
    float srkl_sum = blockReduceSum(srkl_b);
    __syncthreads();

    float ent_b = 0.0f;
    if (tid < 256) {
        float T = teacher_row[tid];
        ent_b = (T > 1e-8f) ? -T * logf(T + 1e-8f) : 0.0f;
    }
    float ent_sum = blockReduceSum(ent_b);
    if (tid == 0) {
        teacher_entropy_out[dist_idx] = ent_sum;
    }
    __syncthreads();

    int actual_b = actual_bytes[dist_idx];
    if (tid == 0) {
        loss_per_dist[dist_idx] = srkl_sum;
        ce_per_dist[dist_idx] = -logf(bins[actual_b] + eps);
        tce_per_dist[dist_idx] = -logf(teacher_row[actual_b] + eps);
    }
    __syncthreads();

    // Compute grad_byte and dot(grad, q)
    float grad_b = 0.0f;
    float S = 0.0f;
    if (tid < 256) {
        S = bins[tid];
        float T_val = teacher_row[tid];
        float S_eps = S + eps;
        float mix = (1.0f - lam) * T_val + lam * S;
        float mix_eps = mix + eps;
        grad_b = inv_N * (logf(S_eps) + S / S_eps - logf(mix_eps) - lam * S / mix_eps);
        if (tid == actual_b) grad_b += alpha * inv_N * (-1.0f / S_eps);
    }
    float dot_prod = blockReduceSum(grad_b * S);
    // Broadcast dot_prod
    if (tid == 0) reduce_buf[0] = dot_prod;
    __syncthreads();
    dot_prod = reduce_buf[0];
    __syncthreads();

    // Write grad_unnorm back to bins
    if (tid < 256) {
        bins[tid] = (grad_b - dot_prod) / Z;
    }
    __syncthreads();
}

// 1 block = 1 token position, 256 threads (8 warps)
__global__ __launch_bounds__(256, 3)
void fused_train_skew_kl_kernel(
    const __half* __restrict__ logits,           // [T1, V]
    const int* __restrict__ first_bytes,          // [V]
    const int* __restrict__ second_bytes,         // [V]
    const int* __restrict__ third_bytes,          // [V]
    const int* __restrict__ byte_lens,            // [V]
    const int* __restrict__ token_byte_seqs_flat, // [V, max_byte_len]
    const int* __restrict__ target_byte_seqs,     // [T1, max_byte_len]
    const int* __restrict__ target_byte_lens,     // [T1]
    const float* __restrict__ teacher_dists,      // [N_total, 256]
    const int* __restrict__ teacher_offsets,       // [T1]
    const int* __restrict__ actual_bytes,          // [N_total]
    const uint8_t* __restrict__ byte_mask,         // [N_total]
    float* __restrict__ grad_logits,              // [T1, V]
    float* __restrict__ loss_per_dist,            // [N_total]
    float* __restrict__ ce_per_dist,              // [N_total]
    float* __restrict__ tce_per_dist,             // [N_total]
    float* __restrict__ teacher_entropy_out,      // [N_total]
    float* __restrict__ grad_unnorm_extra,        // [T1, max_extra, 256] for d3+
    int V, int max_byte_len, int N_total, int max_extra,
    float lam, float alpha, int n_active
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    // Shared memory: d0[256] + d1[256] + d2[256] + reduce[16] + teacher[256] + extra[256]
    extern __shared__ float shared[];
    float* d0_warp_bins = shared;                // [8 * 256] per-warp bins
    float* d1_bins = d0_warp_bins + 8 * 256;     // [256]
    float* d2_bins = d1_bins + 256;              // [256]
    float* reduce_buf = d2_bins + 256;           // [16]
    float* teacher_row = reduce_buf + 16;        // [256]
    float* extra_bins = teacher_row + 256;       // [256] reusable for d3+
    int num_tiles = (V / 2 + 255) / 256;
    float* tile_maxima = extra_bins + 256;        // [num_tiles]

    int my_num_depths = target_byte_lens[t];
    int my_offset = teacher_offsets[t];

    // Preload target byte sequence for this position
    int target_b0 = target_byte_seqs[t * max_byte_len];
    int target_b1 = (my_num_depths >= 2) ? target_byte_seqs[t * max_byte_len + 1] : -1;
    int target_b2 = (my_num_depths >= 3) ? target_byte_seqs[t * max_byte_len + 2] : -1;

    const __half* row = logits + (long long)t * V;
    float* grad_row = grad_logits + (long long)t * V;
    const half2* row_h2 = reinterpret_cast<const half2*>(row);
    int V2 = V / 2;
    int V2_padded = (V2 + 255) & ~255;

    // ============ Zero bins ============
    for (int i = tid; i < 8 * 256; i += 256) d0_warp_bins[i] = 0.0f;
    if (tid < 256) { d1_bins[tid] = 0.0f; d2_bins[tid] = 0.0f; }
    for (int i = tid; i < num_tiles; i += 256) tile_maxima[i] = -3.4e38f;
    __syncthreads();

    // ============ Fused softmax + scatter ============
    float m = -3.4e38f;
    float d = 0.0f;
    for (int i = tid; i < V2_padded; i += 256) {
        float val0, val1;
        int v0, v1;
        bool valid = (i < V2);
        if (valid) {
            half2 h2 = row_h2[i];
            val0 = __half2float(h2.x);
            val1 = __half2float(h2.y);
            v0 = 2 * i;
            v1 = v0 + 1;
        } else {
            val0 = -3.4e38f;
            val1 = -3.4e38f;
            v0 = 0;
            v1 = 0;
        }

        // Online softmax update - invalid threads contribute -inf (no effect)
        float m_new = fmaxf(m, val0);
        d = d * __expf(m - m_new) + __expf(val0 - m_new);
        m = m_new;
        m_new = fmaxf(m, val1);
        d = d * __expf(m - m_new) + __expf(val1 - m_new);
        m = m_new;

        // Track tile max for gradient filtering (warp reduce to minimize atomicMax contention)
        int tile_id = (i - tid) / 256;
        float tile_local = fmaxf(val0, val1);
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_local = fmaxf(tile_local, __shfl_down_sync(0xffffffff, tile_local, offset));
        if (lane == 0) atomicMaxFloat(&tile_maxima[tile_id], tile_local);

        if (!valid) continue;

        // Bin scatter in C=0 frame
        float e0 = __expf(fminf(val0, 80.0f));
        float e1 = __expf(fminf(val1, 80.0f));

        int fb0 = first_bytes[v0];
        int fb1 = first_bytes[v1];
        atomicAdd(&d0_warp_bins[wid * 256 + fb0], e0);
        atomicAdd(&d0_warp_bins[wid * 256 + fb1], e1);

        if (my_num_depths >= 2 && fb0 == target_b0 && byte_lens[v0] >= 2) {
            int sb = second_bytes[v0];
            atomicAdd(&d1_bins[sb], e0);
            if (my_num_depths >= 3 && sb == target_b1 && byte_lens[v0] >= 3) {
                int tb = third_bytes[v0];
                atomicAdd(&d2_bins[tb], e0);
            }
        }
        if (my_num_depths >= 2 && fb1 == target_b0 && byte_lens[v1] >= 2) {
            int sb = second_bytes[v1];
            atomicAdd(&d1_bins[sb], e1);
            if (my_num_depths >= 3 && sb == target_b1 && byte_lens[v1] >= 3) {
                int tb = third_bytes[v1];
                atomicAdd(&d2_bins[tb], e1);
            }
        }
    }
    blockReduceOnlineSoftmax(reduce_buf, m, d);
    float global_max = reduce_buf[0];
    float global_sum = reduce_buf[8];
    float logsumexp = global_max + logf(global_sum);
    __syncthreads();

    // ============ Correct bins: reduce d0 warps + apply exp(-logsumexp) ============
    float correction = __expf(-logsumexp);
    {
        float sum = 0.0f;
        for (int w = 0; w < 8; w++) sum += d0_warp_bins[w * 256 + tid];
        d0_warp_bins[tid] = sum * correction;
    }
    __syncthreads();
    if (tid < 256) {
        d1_bins[tid] *= correction;
        d2_bins[tid] *= correction;
    }
    __syncthreads();
    float* d0_bins = d0_warp_bins;

    // Compute Z for d0, d1, d2
    float z0_local = (tid < 256) ? d0_bins[tid] : 0.0f;
    float Z0 = blockReduceSum(z0_local);
    if (tid == 0) reduce_buf[0] = fmaxf(Z0, 1e-30f);
    __syncthreads(); Z0 = reduce_buf[0]; __syncthreads();

    float z1_local = (tid < 256) ? d1_bins[tid] : 0.0f;
    float Z1 = blockReduceSum(z1_local);
    if (tid == 0) reduce_buf[0] = fmaxf(Z1, 1e-30f);
    __syncthreads(); Z1 = reduce_buf[0]; __syncthreads();

    float z2_local = (tid < 256) ? d2_bins[tid] : 0.0f;
    float Z2 = blockReduceSum(z2_local);
    if (tid == 0) reduce_buf[0] = fmaxf(Z2, 1e-30f);
    __syncthreads(); Z2 = reduce_buf[0]; __syncthreads();

    // ============ PASS 3: Loss + backward per depth ============
    const float inv_N = 1.0f / (float)n_active;

    // d0
    process_depth_skew_kl(d0_bins, Z0, teacher_dists + (long long)my_offset * 256,
        teacher_row, reduce_buf, my_offset, actual_bytes, byte_mask,
        loss_per_dist, ce_per_dist, tce_per_dist, teacher_entropy_out, inv_N, lam, alpha, tid);

    // d1
    if (my_num_depths >= 2) {
        process_depth_skew_kl(d1_bins, Z1, teacher_dists + ((long long)my_offset + 1) * 256,
            teacher_row, reduce_buf, my_offset + 1, actual_bytes, byte_mask,
            loss_per_dist, ce_per_dist, tce_per_dist, teacher_entropy_out, inv_N, lam, alpha, tid);
    }

    // d2
    if (my_num_depths >= 3) {
        process_depth_skew_kl(d2_bins, Z2, teacher_dists + ((long long)my_offset + 2) * 256,
            teacher_row, reduce_buf, my_offset + 2, actual_bytes, byte_mask,
            loss_per_dist, ce_per_dist, tce_per_dist, teacher_entropy_out, inv_N, lam, alpha, tid);
    }

    // d3+ : process each extra depth using extra_bins
    for (int dep = 3; dep < my_num_depths; dep++) {
        // Zero extra_bins
        if (tid < 256) extra_bins[tid] = 0.0f;
        __syncthreads();

        // Scatter: check prefix match for first dep bytes, scatter (dep+1)-th byte
        for (int v = tid; v < V; v += 256) {
            if (byte_lens[v] <= dep) continue;
            if (first_bytes[v] != target_b0) continue;
            if (second_bytes[v] != target_b1) continue;
            if (third_bytes[v] != target_b2) continue;
            // Check bytes 3..dep-1 via token_byte_seqs_flat
            bool match = true;
            long long voff = (long long)v * max_byte_len;
            for (int k = 3; k < dep && match; k++) {
                match = (token_byte_seqs_flat[voff + k] == target_byte_seqs[t * max_byte_len + k]);
            }
            if (!match) continue;
            float val = __half2float(row[v]);
            float prob = __expf(val - logsumexp);
            int scatter_byte = token_byte_seqs_flat[voff + dep];
            atomicAdd(&extra_bins[scatter_byte], prob);
        }
        __syncthreads();

        // Compute Z for this depth
        float zx = (tid < 256) ? extra_bins[tid] : 0.0f;
        float Zx = blockReduceSum(zx);
        if (tid == 0) reduce_buf[0] = fmaxf(Zx, 1e-30f);
        __syncthreads(); Zx = reduce_buf[0]; __syncthreads();

        // Process loss + backward, writes grad_unnorm into extra_bins
        process_depth_skew_kl(extra_bins, Zx, teacher_dists + ((long long)my_offset + dep) * 256,
            teacher_row, reduce_buf, my_offset + dep, actual_bytes, byte_mask,
            loss_per_dist, ce_per_dist, tce_per_dist, teacher_entropy_out, inv_N, lam, alpha, tid);

        // Copy grad_unnorm to global memory for Pass 4
        if (tid < 256) {
            grad_unnorm_extra[((long long)t * max_extra + (dep - 3)) * 256 + tid] = extra_bins[tid];
        }
        __syncthreads();
    }

    // ============ PASS 4: Backward through scatter + softmax ============
    float tile_threshold = logsumexp - 20.0f;
    for (int i = tid; i < V2; i += 256) {
        int tile_id = (i - tid) / 256;
        if (tile_maxima[tile_id] < tile_threshold) continue;

        half2 h2 = row_h2[i];
        float val0 = __half2float(h2.x);
        float val1 = __half2float(h2.y);
        int v0 = 2 * i;
        int v1 = v0 + 1;

        float prob0 = __expf(val0 - logsumexp);
        float prob1 = __expf(val1 - logsumexp);

        if (prob0 < 1e-6f) {
            grad_row[v0] = 0.0f;
        } else {
            int fb0 = first_bytes[v0];
            float gp0 = d0_bins[fb0];
            if (my_num_depths >= 2 && fb0 == target_b0 && byte_lens[v0] >= 2) {
                int sb = second_bytes[v0];
                gp0 += d1_bins[sb];
                if (my_num_depths >= 3 && sb == target_b1 && byte_lens[v0] >= 3) {
                    int tb = third_bytes[v0];
                    gp0 += d2_bins[tb];
                    if (my_num_depths > 3 && tb == target_b2 && byte_lens[v0] > 3) {
                        long long voff = (long long)v0 * max_byte_len;
                        for (int dep = 3; dep < my_num_depths; dep++) {
                            if (byte_lens[v0] <= dep) break;
                            int bv = token_byte_seqs_flat[voff + dep];
                            gp0 += grad_unnorm_extra[((long long)t * max_extra + (dep - 3)) * 256 + bv];
                            if (dep + 1 < my_num_depths && bv != target_byte_seqs[t * max_byte_len + dep])
                                break;
                        }
                    }
                }
            }
            grad_row[v0] = prob0 * gp0;
        }

        if (prob1 < 1e-6f) {
            grad_row[v1] = 0.0f;
        } else {
            int fb1 = first_bytes[v1];
            float gp1 = d0_bins[fb1];
            if (my_num_depths >= 2 && fb1 == target_b0 && byte_lens[v1] >= 2) {
                int sb = second_bytes[v1];
                gp1 += d1_bins[sb];
                if (my_num_depths >= 3 && sb == target_b1 && byte_lens[v1] >= 3) {
                    int tb = third_bytes[v1];
                    gp1 += d2_bins[tb];
                    if (my_num_depths > 3 && tb == target_b2 && byte_lens[v1] > 3) {
                        long long voff = (long long)v1 * max_byte_len;
                        for (int dep = 3; dep < my_num_depths; dep++) {
                            if (byte_lens[v1] <= dep) break;
                            int bv = token_byte_seqs_flat[voff + dep];
                            gp1 += grad_unnorm_extra[((long long)t * max_extra + (dep - 3)) * 256 + bv];
                            if (dep + 1 < my_num_depths && bv != target_byte_seqs[t * max_byte_len + dep])
                                break;
                        }
                    }
                }
            }
            grad_row[v1] = prob1 * gp1;
        }
    }
}

std::vector<torch::Tensor> fused_train_skew_kl(
    torch::Tensor logits,           // [T1, V] fp16
    torch::Tensor first_bytes,      // [V] int
    torch::Tensor second_bytes,     // [V] int
    torch::Tensor third_bytes,      // [V] int
    torch::Tensor byte_lens,        // [V] int
    torch::Tensor token_byte_seqs,  // [V, max_byte_len] int
    torch::Tensor target_byte_seqs, // [T1, max_byte_len] int
    torch::Tensor target_byte_lens, // [T1] int
    torch::Tensor teacher_dists,    // [N_total, 256] float
    torch::Tensor teacher_offsets,  // [T1] int
    torch::Tensor actual_bytes,     // [N_total] int
    torch::Tensor byte_mask,        // [N_total] uint8
    int N_total,
    int n_active,
    float lam,
    float alpha
) {
    const int T1 = logits.size(0);
    const int V = logits.size(1);
    TORCH_CHECK(V % 2 == 0, "Vocab size must be even for half2 vectorization");
    const int max_byte_len = target_byte_seqs.size(1);
    const int max_extra = (max_byte_len > 3) ? (max_byte_len - 3) : 0;

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
    auto grad_logits = torch::zeros({T1, V}, opts_f32);
    auto loss_srkl = torch::empty({N_total}, opts_f32);
    auto loss_ce = torch::empty({N_total}, opts_f32);
    auto loss_tce = torch::empty({N_total}, opts_f32);
    auto teacher_entropy_out = torch::empty({N_total}, opts_f32);

    // d3+ grad_unnorm storage
    auto grad_extra = (max_extra > 0)
        ? torch::empty({T1, max_extra, 256}, opts_f32)
        : torch::empty({0}, opts_f32);

    // Shared: d0_warp[8*256] + d1[256] + d2[256] + reduce[16] + teacher[256] + extra[256] + tile_maxima[num_tiles]
    int num_tiles = (V + 511) / 512;
    const int shared_mem = (8*256 + 256 + 256 + 16 + 256 + 256 + num_tiles) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_train_skew_kl_kernel<<<T1, 256, shared_mem, stream>>>(
        reinterpret_cast<const __half*>(logits.data_ptr<at::Half>()),
        first_bytes.data_ptr<int>(),
        second_bytes.data_ptr<int>(),
        third_bytes.data_ptr<int>(),
        byte_lens.data_ptr<int>(),
        token_byte_seqs.data_ptr<int>(),
        target_byte_seqs.data_ptr<int>(),
        target_byte_lens.data_ptr<int>(),
        teacher_dists.data_ptr<float>(),
        teacher_offsets.data_ptr<int>(),
        actual_bytes.data_ptr<int>(),
        byte_mask.data_ptr<uint8_t>(),
        grad_logits.data_ptr<float>(),
        loss_srkl.data_ptr<float>(),
        loss_ce.data_ptr<float>(),
        loss_tce.data_ptr<float>(),
        teacher_entropy_out.data_ptr<float>(),
        grad_extra.data_ptr<float>(),
        V, max_byte_len, N_total, max_extra,
        lam, alpha, n_active
    );

    return {grad_logits, loss_srkl, loss_ce, loss_tce, teacher_entropy_out};
}
"""

_FUSED_TRAIN_SKEW_KL_CPP = """
std::vector<torch::Tensor> fused_train_skew_kl(
    torch::Tensor logits,
    torch::Tensor first_bytes,
    torch::Tensor second_bytes,
    torch::Tensor third_bytes,
    torch::Tensor byte_lens,
    torch::Tensor token_byte_seqs,
    torch::Tensor target_byte_seqs,
    torch::Tensor target_byte_lens,
    torch::Tensor teacher_dists,
    torch::Tensor teacher_offsets,
    torch::Tensor actual_bytes,
    torch::Tensor byte_mask,
    int N_total,
    int n_active,
    float lam,
    float alpha
);
"""


def get_fused_train_skew_kl_kernel() -> _FusedTrainSkewKlKernel | None:
    if not _cuda_available:
        return None
    kernel = _compile_kernel_with_header(
        "fused_train_skew_kl",
        _FUSED_TRAIN_SKEW_KL_CPP,
        _FUSED_TRAIN_SKEW_KL_CUDA,
        ["fused_train_skew_kl"],
    )
    return None if kernel is None else cast(_FusedTrainSkewKlKernel, kernel)



def fused_train_forward_backward_skew_kl(logits, token_ids, byte_vocab, teacher_dists_flat,
                                          actual_bytes_flat, teacher_offsets,
                                          target_byte_lens, alpha, lam=0.1, entropy_weighting=False,
                                          byte_mask=None):
    kernel = get_fused_train_skew_kl_kernel()
    if kernel is None:
        return None, None

    T = logits.shape[0]
    T1 = T - 1

    next_tokens = token_ids[1:]
    next_byte_lens = byte_vocab.token_byte_lens[next_tokens]
    next_byte_seqs = byte_vocab.token_byte_seqs[next_tokens]  # [T1, max_byte_len]
    N_total = int(teacher_offsets[-1].item()) + int(target_byte_lens[-1].item())

    if byte_mask is None:
        byte_mask = torch.ones(N_total, dtype=torch.uint8, device=logits.device)
        n_active = N_total
    else:
        byte_mask = byte_mask.to(torch.uint8).contiguous()
        n_active = int(byte_mask.sum().item())
    if n_active == 0:
        return None, None

    results = kernel.fused_train_skew_kl(
        logits[:T1].half().contiguous(),
        byte_vocab.first_bytes_i32.contiguous(),
        byte_vocab.second_bytes_i32.contiguous(),
        byte_vocab.third_bytes_i32.contiguous(),
        byte_vocab.byte_lens_i32.contiguous(),
        byte_vocab.token_byte_seqs_i32.contiguous(),
        next_byte_seqs.int().contiguous(),
        next_byte_lens.int().contiguous(),
        teacher_dists_flat.float().contiguous(),
        teacher_offsets.int().contiguous(),
        actual_bytes_flat.int().contiguous(),
        byte_mask,
        N_total, n_active, lam, alpha,
    )

    grad_logits, loss_srkl, loss_ce, loss_tce, teacher_entropy = results

    if entropy_weighting:
        ew = teacher_entropy / 5.545177
        ew_sum = ew.sum().clamp(min=1e-8)

        kl_loss = (loss_srkl * ew).sum() / ew_sum
        ce_loss = (loss_ce * ew).sum() / ew_sum

        token_indices = torch.repeat_interleave(
            torch.arange(T1, device=ew.device),
            target_byte_lens
        )
        ew_per_token = torch.zeros(T1, device=ew.device).scatter_add_(0, token_indices, ew)
        ew_per_token = ew_per_token / target_byte_lens.float().clamp(min=1)
        ew_token_sum = ew_per_token.sum().clamp(min=1e-8)
        grad_scale = ew_per_token * T1 / ew_token_sum
        grad_logits = grad_logits * grad_scale.unsqueeze(1)
    else:
        kl_loss = loss_srkl.sum() / n_active
        ce_loss = loss_ce.sum() / n_active

    total_loss = kl_loss + alpha * ce_loss

    return grad_logits, {
        "train_loss": total_loss,
        "custom loss": total_loss,
        "CE loss": ce_loss.detach(),
        "kl_div": kl_loss.detach(),
        "teacher CE loss": (loss_tce.sum() / n_active).detach(),
    }


