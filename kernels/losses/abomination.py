import torch
from typing import Protocol, cast
from kernels.compiler import _compile_kernel, _compile_kernel_with_header, _cuda_available


class _AbominationKernel(Protocol):
    def abomination_pass1(self, *args: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def abomination_pass2(self, *args: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class _FusedTrainAbominationKernel(Protocol):
    def fused_train_abomination_pass1(self, *args: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def fused_train_abomination_pass2(self, *args: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


_TRAINING_ABOMINATION_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void abomination_pass1_kernel(
    const float* __restrict__ student,
    const float* __restrict__ teacher,
    const int*   __restrict__ actual_bytes,
    float* __restrict__ fwd_kl,
    float* __restrict__ rev_kl,
    float* __restrict__ student_ce,
    float* __restrict__ teacher_ce,
    int N
) {
    const int pos = blockIdx.x;
    const int byte_idx = threadIdx.x;
    if (pos >= N || byte_idx >= 256) return;

    const float eps = 1e-8f;
    const int base = pos * 256;

    float S = student[base + byte_idx];
    float T = teacher[base + byte_idx];
    float log_S = logf(S + eps);
    float log_T = logf(T + eps);

    float fwd_b = T * (log_T - log_S);
    float rev_b = S * (log_S - log_T);

    float fwd_sum = blockReduceSum(fwd_b);
    float rev_sum = blockReduceSum(rev_b);

    if (byte_idx == 0) {
        fwd_kl[pos] = fwd_sum;
        rev_kl[pos] = rev_sum;
        int actual_byte = actual_bytes[pos];
        student_ce[pos] = -logf(student[base + actual_byte] + eps);
        teacher_ce[pos] = -logf(teacher[base + actual_byte] + eps);
    }
}

__global__ void abomination_pass2_kernel(
    const float* __restrict__ student,
    const float* __restrict__ teacher,
    const int*   __restrict__ actual_bytes,
    const float* __restrict__ fwd_kl,
    const float* __restrict__ rev_kl,
    const float* __restrict__ ce_diff,
    const float* __restrict__ ce_diff_grad,
    float fwd_kl_mean,
    float rev_kl_mean,
    float* __restrict__ grad_student,
    float* __restrict__ weighted_fwd,
    float* __restrict__ weighted_rev,
    int N,
    float alpha
) {
    const int pos = blockIdx.x;
    const int byte_idx = threadIdx.x;
    if (pos >= N || byte_idx >= 256) return;

    const float eps = 1e-8f;
    const float inv_2N = 1.0f / (2.0f * (float)N);
    const int base = pos * 256;

    float F_i = fwd_kl[pos];
    float R_i = rev_kl[pos];
    float cd_i = ce_diff[pos];
    float cdg_i = ce_diff_grad[pos];
    int actual_b = actual_bytes[pos];

    float S = student[base + byte_idx];
    float T_val = teacher[base + byte_idx];
    float S_eps = S + eps;
    float log_S = logf(S_eps);
    float log_T = logf(T_val + eps);

    float ratio_F = F_i / fwd_kl_mean + 1.0f;
    float ratio_R = R_i / rev_kl_mean + 1.0f;
    float w_F = powf(ratio_F, alpha) + cd_i;
    float w_R = powf(ratio_R, alpha) + cd_i;

    if (byte_idx == 0) {
        weighted_fwd[pos] = F_i * w_F;
        weighted_rev[pos] = R_i * w_R;
    }

    float dFdS = -T_val / S_eps;
    float dRdS = log_S - log_T + S / S_eps;

    float pw_F = alpha * powf(ratio_F, alpha - 1.0f) / fwd_kl_mean;
    float pw_R = alpha * powf(ratio_R, alpha - 1.0f) / rev_kl_mean;

    float grad = inv_2N * (
        dFdS * (w_F + F_i * pw_F) +
        dRdS * (w_R + R_i * pw_R)
    );

    if (byte_idx == actual_b && cd_i < 5.0f) {
        grad += inv_2N * (F_i + R_i) * cdg_i * (-1.0f / S_eps);
    }

    grad_student[base + byte_idx] = grad;
}

std::vector<torch::Tensor> abomination_pass1(
    torch::Tensor student,
    torch::Tensor teacher,
    torch::Tensor actual_bytes
) {
    int N = student.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(student.device());
    auto fwd_kl = torch::empty({N}, opts);
    auto rev_kl = torch::empty({N}, opts);
    auto student_ce = torch::empty({N}, opts);
    auto teacher_ce = torch::empty({N}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    abomination_pass1_kernel<<<N, 256, 0, stream>>>(
        student.data_ptr<float>(),
        teacher.data_ptr<float>(),
        actual_bytes.data_ptr<int>(),
        fwd_kl.data_ptr<float>(),
        rev_kl.data_ptr<float>(),
        student_ce.data_ptr<float>(),
        teacher_ce.data_ptr<float>(),
        N
    );

    return {fwd_kl, rev_kl, student_ce, teacher_ce};
}

std::vector<torch::Tensor> abomination_pass2(
    torch::Tensor student,
    torch::Tensor teacher,
    torch::Tensor actual_bytes,
    torch::Tensor fwd_kl,
    torch::Tensor rev_kl,
    torch::Tensor ce_diff,
    torch::Tensor ce_diff_grad,
    float fwd_kl_mean,
    float rev_kl_mean,
    float alpha
) {
    int N = student.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(student.device());
    auto grad_student = torch::empty({N, 256}, opts);
    auto weighted_fkl = torch::empty({N}, opts);
    auto weighted_rkl = torch::empty({N}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    abomination_pass2_kernel<<<N, 256, 0, stream>>>(
        student.data_ptr<float>(),
        teacher.data_ptr<float>(),
        actual_bytes.data_ptr<int>(),
        fwd_kl.data_ptr<float>(),
        rev_kl.data_ptr<float>(),
        ce_diff.data_ptr<float>(),
        ce_diff_grad.data_ptr<float>(),
        fwd_kl_mean,
        rev_kl_mean,
        grad_student.data_ptr<float>(),
        weighted_fkl.data_ptr<float>(),
        weighted_rkl.data_ptr<float>(),
        N,
        alpha
    );

    return {grad_student, weighted_fkl, weighted_rkl};
}
"""

_TRAINING_ABOMINATION_CPP = """
std::vector<torch::Tensor> abomination_pass1(
    torch::Tensor student,
    torch::Tensor teacher,
    torch::Tensor actual_bytes
);
std::vector<torch::Tensor> abomination_pass2(
    torch::Tensor student,
    torch::Tensor teacher,
    torch::Tensor actual_bytes,
    torch::Tensor fwd_kl,
    torch::Tensor rev_kl,
    torch::Tensor ce_diff,
    torch::Tensor ce_diff_grad,
    float fwd_kl_mean,
    float rev_kl_mean,
    float alpha
);
"""


def get_abomination_kernel() -> _AbominationKernel | None:
    if not _cuda_available:
        return None
    kernel = _compile_kernel_with_header(
        "fused_abomination",
        _TRAINING_ABOMINATION_CPP,
        _TRAINING_ABOMINATION_CUDA,
        ["abomination_pass1", "abomination_pass2"],
    )
    return None if kernel is None else cast(_AbominationKernel, kernel)


def fused_abomination_forward_backward(student_dists, teacher_dists, actual_bytes, alpha, entropy_weights=None):
    kernel = get_abomination_kernel()
    if kernel is None:
        return None, None

    student = student_dists.float().contiguous()
    teacher = teacher_dists.float().contiguous()
    actual_bytes_t = actual_bytes.int().contiguous()

    fwd_kl, rev_kl, student_ce, teacher_ce = kernel.abomination_pass1(student, teacher, actual_bytes_t)

    eps = 1e-8
    fwd_kl_mean = fwd_kl.mean().clamp(min=eps).item()
    rev_kl_mean = rev_kl.mean().clamp(min=eps).item()
    raw_diff = student_ce - teacher_ce
    ce_diff = torch.nn.functional.softplus(raw_diff, beta=5.0).clamp(max=5.0)
    ce_diff_grad = torch.sigmoid(5.0 * raw_diff)

    grad_student, weighted_fwd_per, weighted_rev_per = kernel.abomination_pass2(
        student, teacher, actual_bytes_t, fwd_kl, rev_kl, ce_diff, ce_diff_grad, fwd_kl_mean, rev_kl_mean, alpha
    )

    if entropy_weights is not None:
        N = fwd_kl.shape[0]
        w_sum = entropy_weights.sum().clamp(min=1e-8)
        scale = entropy_weights * N / w_sum
        grad_student = grad_student * scale.unsqueeze(1)
        weighted_fwd = (weighted_fwd_per * entropy_weights).sum() / w_sum
        weighted_rev = (weighted_rev_per * entropy_weights).sum() / w_sum
    else:
        weighted_fwd = weighted_fwd_per.mean()
        weighted_rev = weighted_rev_per.mean()
    total_loss = (weighted_fwd + weighted_rev) / 2

    return grad_student, {
        "train_loss": total_loss,
        "custom loss": total_loss,
        "CE loss": student_ce.mean().detach(),
        "kl_div": fwd_kl.mean().detach(),
        "reverse kl_div": rev_kl.mean().detach(),
        "weighted kl_div": weighted_fwd.detach(),
        "weighted rev. kl_div": weighted_rev.detach(),
        "teacher CE loss": teacher_ce.mean().detach(),
        "CE diff": ce_diff.mean().detach(),
    }


# ============================================================
# Training kernel: fused adaptive KL loss + gradient (single pass)
# ============================================================



_FUSED_TRAIN_ABOMINATION_CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ void process_depth_abomination_pass1(
    float* bins, float Z,
    const float* teacher_src, float* teacher_row, float* reduce_buf,
    int dist_idx, const int* actual_bytes,
    const uint8_t* byte_mask,
    float* fwd_kl_out, float* rev_kl_out, float* ce_out, float* tce_out,
    float* bins_global_out, float* Z_out,
    float* teacher_entropy_out,
    int tid
) {
    const float eps = 1e-8f;

    if (byte_mask[dist_idx] == 0) {
        if (tid == 0) {
            fwd_kl_out[dist_idx] = 0.0f;
            rev_kl_out[dist_idx] = 0.0f;
            ce_out[dist_idx] = 0.0f;
            tce_out[dist_idx] = 0.0f;
            Z_out[dist_idx] = 1.0f;
            teacher_entropy_out[dist_idx] = 0.0f;
        }
        if (tid < 256) {
            bins[tid] = 0.0f;
            bins_global_out[(long long)dist_idx * 256 + tid] = 0.0f;
        }
        __syncthreads();
        return;
    }

    if (tid < 256) {
        bins[tid] /= Z;
        teacher_row[tid] = teacher_src[tid];
    }
    __syncthreads();

    float fkl_b = 0.0f, rkl_b = 0.0f;
    if (tid < 256) {
        float S = bins[tid];
        float T = teacher_row[tid];
        fkl_b = T * (logf(T + eps) - logf(S + eps));
        rkl_b = S * (logf(S + eps) - logf(T + eps));
    }

    float fkl_sum = blockReduceSum(fkl_b);
    __syncthreads();
    float rkl_sum = blockReduceSum(rkl_b);
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
        fwd_kl_out[dist_idx] = fkl_sum;
        rev_kl_out[dist_idx] = rkl_sum;
        ce_out[dist_idx] = -logf(bins[actual_b] + eps);
        tce_out[dist_idx] = -logf(teacher_row[actual_b] + eps);
        Z_out[dist_idx] = Z;
    }

    if (tid < 256) {
        bins_global_out[(long long)dist_idx * 256 + tid] = bins[tid];
    }
    __syncthreads();
}

__device__ void process_depth_abomination_pass2(
    float* bins,
    const float* bins_global_in, float Z,
    const float* teacher_src, float* teacher_row, float* reduce_buf,
    int dist_idx, const int* actual_bytes,
    const uint8_t* byte_mask,
    const float* fwd_kl, const float* rev_kl, const float* ce_diff_arr,
    const float* ce_diff_grad_arr,
    float fwd_kl_mean, float rev_kl_mean,
    float* weighted_fwd, float* weighted_rev,
    float alpha, float inv_2N, int tid
) {
    const float eps = 1e-8f;

    if (byte_mask[dist_idx] == 0) {
        if (tid == 0) {
            weighted_fwd[dist_idx] = 0.0f;
            weighted_rev[dist_idx] = 0.0f;
        }
        if (tid < 256) bins[tid] = 0.0f;
        __syncthreads();
        return;
    }

    float S = 0.0f;
    if (tid < 256) {
        S = bins_global_in[(long long)dist_idx * 256 + tid];
        bins[tid] = S;
        teacher_row[tid] = teacher_src[tid];
    }
    __syncthreads();

    float F_i = fwd_kl[dist_idx];
    float R_i = rev_kl[dist_idx];
    float cd_i = ce_diff_arr[dist_idx];
    float cdg_i = ce_diff_grad_arr[dist_idx];
    int actual_b = actual_bytes[dist_idx];

    float ratio_F = F_i / fwd_kl_mean + 1.0f;
    float ratio_R = R_i / rev_kl_mean + 1.0f;
    float w_F = powf(ratio_F, alpha) + cd_i;
    float w_R = powf(ratio_R, alpha) + cd_i;

    if (tid == 0) {
        weighted_fwd[dist_idx] = F_i * w_F;
        weighted_rev[dist_idx] = R_i * w_R;
    }

    float grad_b = 0.0f;
    if (tid < 256) {
        float T = teacher_row[tid];
        float S_eps = S + eps;
        float log_S = logf(S_eps);
        float log_T = logf(T + eps);

        float dFdS = -T / S_eps;
        float dRdS = log_S + S / S_eps - log_T;

        float pw_F = alpha * powf(ratio_F, alpha - 1.0f) / fwd_kl_mean;
        float pw_R = alpha * powf(ratio_R, alpha - 1.0f) / rev_kl_mean;

        grad_b = inv_2N * (
            dFdS * (w_F + F_i * pw_F) +
            dRdS * (w_R + R_i * pw_R)
        );

        if (tid == actual_b && cd_i < 5.0f) {
            grad_b += inv_2N * (F_i + R_i) * cdg_i * (-1.0f / S_eps);
        }
    }

    float dot_prod = blockReduceSum(grad_b * S);
    if (tid == 0) reduce_buf[0] = dot_prod;
    __syncthreads();
    dot_prod = reduce_buf[0];
    __syncthreads();

    if (tid < 256) {
        bins[tid] = (grad_b - dot_prod) / Z;
    }
    __syncthreads();
}

__global__ __launch_bounds__(256, 3)
void fused_train_abomination_pass1_kernel(
    const __half* __restrict__ logits,
    const int* __restrict__ first_bytes,
    const int* __restrict__ second_bytes,
    const int* __restrict__ third_bytes,
    const int* __restrict__ byte_lens,
    const int* __restrict__ token_byte_seqs_flat,
    const int* __restrict__ target_byte_seqs,
    const int* __restrict__ target_byte_lens,
    const float* __restrict__ teacher_dists,
    const int* __restrict__ teacher_offsets,
    const int* __restrict__ actual_bytes,
    const uint8_t* __restrict__ byte_mask,
    float* __restrict__ fwd_kl_out,
    float* __restrict__ rev_kl_out,
    float* __restrict__ ce_out,
    float* __restrict__ tce_out,
    float* __restrict__ bins_out,
    float* __restrict__ Z_out,
    float* __restrict__ softmax_max_out,
    float* __restrict__ softmax_sum_out,
    float* __restrict__ teacher_entropy_out,
    int V, int max_byte_len, int N_total
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ float shared[];
    float* d0_warp_bins = shared;                // [8 * 256] per-warp bins
    float* d1_bins = d0_warp_bins + 8 * 256;     // [256]
    float* d2_bins = d1_bins + 256;
    float* reduce_buf = d2_bins + 256;
    float* teacher_row = reduce_buf + 16;
    float* extra_bins = teacher_row + 256;

    int my_num_depths = target_byte_lens[t];
    int my_offset = teacher_offsets[t];

    int target_b0 = target_byte_seqs[t * max_byte_len];
    int target_b1 = (my_num_depths >= 2) ? target_byte_seqs[t * max_byte_len + 1] : -1;
    int target_b2 = (my_num_depths >= 3) ? target_byte_seqs[t * max_byte_len + 2] : -1;

    const __half* row = logits + (long long)t * V;
    const half2* row_h2 = reinterpret_cast<const half2*>(row);
    int V2 = V / 2;

    // ============ PASS 1: Online softmax ============
    float m = -3.4e38f;
    float d = 0.0f;
    for (int i = tid; i < V2; i += 256) {
        half2 h2 = row_h2[i];
        float val0 = __half2float(h2.x);
        float val1 = __half2float(h2.y);
        float m_new = fmaxf(m, fmaxf(val0, val1));
        d = d * __expf(m - m_new) + __expf(val0 - m_new) + __expf(val1 - m_new);
        m = m_new;
    }
    blockReduceOnlineSoftmax(reduce_buf, m, d);
    float global_max = reduce_buf[0];
    float global_sum = reduce_buf[8];
    float logsumexp = global_max + logf(global_sum);
    __syncthreads();

    if (tid == 0) {
        softmax_max_out[t] = global_max;
        softmax_sum_out[t] = global_sum;
    }

    // ============ PASS 2: Scatter to d0/d1/d2 bins ============
    for (int i = tid; i < 8 * 256; i += 256) d0_warp_bins[i] = 0.0f;
    if (tid < 256) { d1_bins[tid] = 0.0f; d2_bins[tid] = 0.0f; }
    __syncthreads();

    for (int i = tid; i < V2; i += 256) {
        half2 h2 = row_h2[i];
        float val0 = __half2float(h2.x);
        float val1 = __half2float(h2.y);
        int v0 = 2 * i;
        int v1 = v0 + 1;

        float prob0 = __expf(val0 - logsumexp);
        int fb0 = first_bytes[v0];
        atomicAdd(&d0_warp_bins[wid * 256 + fb0], prob0);
        if (my_num_depths >= 2 && fb0 == target_b0 && byte_lens[v0] >= 2) {
            int sb = second_bytes[v0];
            atomicAdd(&d1_bins[sb], prob0);
            if (my_num_depths >= 3 && sb == target_b1 && byte_lens[v0] >= 3) {
                int tb = third_bytes[v0];
                atomicAdd(&d2_bins[tb], prob0);
            }
        }

        float prob1 = __expf(val1 - logsumexp);
        int fb1 = first_bytes[v1];
        atomicAdd(&d0_warp_bins[wid * 256 + fb1], prob1);
        if (my_num_depths >= 2 && fb1 == target_b0 && byte_lens[v1] >= 2) {
            int sb = second_bytes[v1];
            atomicAdd(&d1_bins[sb], prob1);
            if (my_num_depths >= 3 && sb == target_b1 && byte_lens[v1] >= 3) {
                int tb = third_bytes[v1];
                atomicAdd(&d2_bins[tb], prob1);
            }
        }
    }
    __syncthreads();

    // Reduce d0 across warps into first 256 slots
    {
        float sum = 0.0f;
        for (int w = 0; w < 8; w++) sum += d0_warp_bins[w * 256 + tid];
        d0_warp_bins[tid] = sum;
    }
    __syncthreads();
    float* d0_bins = d0_warp_bins;

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

    // ============ PASS 3: Forward metrics per depth ============
    process_depth_abomination_pass1(d0_bins, Z0, teacher_dists + (long long)my_offset * 256,
        teacher_row, reduce_buf, my_offset, actual_bytes, byte_mask,
        fwd_kl_out, rev_kl_out, ce_out, tce_out, bins_out, Z_out, teacher_entropy_out, tid);

    if (my_num_depths >= 2) {
        process_depth_abomination_pass1(d1_bins, Z1, teacher_dists + ((long long)my_offset + 1) * 256,
            teacher_row, reduce_buf, my_offset + 1, actual_bytes, byte_mask,
            fwd_kl_out, rev_kl_out, ce_out, tce_out, bins_out, Z_out, teacher_entropy_out, tid);
    }

    if (my_num_depths >= 3) {
        process_depth_abomination_pass1(d2_bins, Z2, teacher_dists + ((long long)my_offset + 2) * 256,
            teacher_row, reduce_buf, my_offset + 2, actual_bytes, byte_mask,
            fwd_kl_out, rev_kl_out, ce_out, tce_out, bins_out, Z_out, teacher_entropy_out, tid);
    }

    for (int dep = 3; dep < my_num_depths; dep++) {
        if (tid < 256) extra_bins[tid] = 0.0f;
        __syncthreads();

        for (int v = tid; v < V; v += 256) {
            if (byte_lens[v] <= dep) continue;
            if (first_bytes[v] != target_b0) continue;
            if (second_bytes[v] != target_b1) continue;
            if (third_bytes[v] != target_b2) continue;
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

        float zx = (tid < 256) ? extra_bins[tid] : 0.0f;
        float Zx = blockReduceSum(zx);
        if (tid == 0) reduce_buf[0] = fmaxf(Zx, 1e-30f);
        __syncthreads(); Zx = reduce_buf[0]; __syncthreads();

        process_depth_abomination_pass1(extra_bins, Zx, teacher_dists + ((long long)my_offset + dep) * 256,
            teacher_row, reduce_buf, my_offset + dep, actual_bytes, byte_mask,
            fwd_kl_out, rev_kl_out, ce_out, tce_out, bins_out, Z_out, teacher_entropy_out, tid);
    }
}

__global__ __launch_bounds__(256, 3)
void fused_train_abomination_pass2_kernel(
    const __half* __restrict__ logits,
    const int* __restrict__ first_bytes,
    const int* __restrict__ second_bytes,
    const int* __restrict__ third_bytes,
    const int* __restrict__ byte_lens,
    const int* __restrict__ token_byte_seqs_flat,
    const int* __restrict__ target_byte_seqs,
    const int* __restrict__ target_byte_lens,
    const float* __restrict__ teacher_dists,
    const int* __restrict__ teacher_offsets,
    const int* __restrict__ actual_bytes,
    const uint8_t* __restrict__ byte_mask,
    const float* __restrict__ bins_in,
    const float* __restrict__ Z_in,
    const float* __restrict__ fwd_kl,
    const float* __restrict__ rev_kl,
    const float* __restrict__ ce_diff_arr,
    const float* __restrict__ ce_diff_grad_arr,
    const float* __restrict__ softmax_max_in,
    const float* __restrict__ softmax_sum_in,
    float fwd_kl_mean,
    float rev_kl_mean,
    float alpha,
    float* __restrict__ grad_logits,
    float* __restrict__ weighted_fwd_out,
    float* __restrict__ weighted_rev_out,
    float* __restrict__ grad_unnorm_extra,
    int V, int max_byte_len, int N_total, int max_extra,
    int n_active
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* d0_bins = shared;
    float* d1_bins = shared + 256;
    float* d2_bins = d1_bins + 256;
    float* reduce_buf = d2_bins + 256;
    float* teacher_row = reduce_buf + 16;
    float* extra_bins = teacher_row + 256;

    int my_num_depths = target_byte_lens[t];
    int my_offset = teacher_offsets[t];

    int target_b0 = target_byte_seqs[t * max_byte_len];
    int target_b1 = (my_num_depths >= 2) ? target_byte_seqs[t * max_byte_len + 1] : -1;
    int target_b2 = (my_num_depths >= 3) ? target_byte_seqs[t * max_byte_len + 2] : -1;

    const __half* row = logits + (long long)t * V;
    const half2* row_h2 = reinterpret_cast<const half2*>(row);
    int V2 = V / 2;
    float* grad_row = grad_logits + (long long)t * V;

    float global_max = softmax_max_in[t];
    float global_sum = softmax_sum_in[t];
    float logsumexp = global_max + logf(global_sum);

    const float inv_2N = 1.0f / (2.0f * (float)n_active);

    // Initialize bins to zero (unused depths stay zero for pass 4)
    if (tid < 256) {
        d0_bins[tid] = 0.0f;
        d1_bins[tid] = 0.0f;
        d2_bins[tid] = 0.0f;
    }
    __syncthreads();

    // ============ Process each depth: compute grad_unnorm ============
    process_depth_abomination_pass2(d0_bins, bins_in, Z_in[my_offset],
        teacher_dists + (long long)my_offset * 256,
        teacher_row, reduce_buf, my_offset, actual_bytes, byte_mask,
        fwd_kl, rev_kl, ce_diff_arr, ce_diff_grad_arr, fwd_kl_mean, rev_kl_mean,
        weighted_fwd_out, weighted_rev_out,
        alpha, inv_2N, tid);

    if (my_num_depths >= 2) {
        process_depth_abomination_pass2(d1_bins, bins_in, Z_in[my_offset + 1],
            teacher_dists + ((long long)my_offset + 1) * 256,
            teacher_row, reduce_buf, my_offset + 1, actual_bytes, byte_mask,
            fwd_kl, rev_kl, ce_diff_arr, ce_diff_grad_arr, fwd_kl_mean, rev_kl_mean,
            weighted_fwd_out, weighted_rev_out,
            alpha, inv_2N, tid);
    }

    if (my_num_depths >= 3) {
        process_depth_abomination_pass2(d2_bins, bins_in, Z_in[my_offset + 2],
            teacher_dists + ((long long)my_offset + 2) * 256,
            teacher_row, reduce_buf, my_offset + 2, actual_bytes, byte_mask,
            fwd_kl, rev_kl, ce_diff_arr, ce_diff_grad_arr, fwd_kl_mean, rev_kl_mean,
            weighted_fwd_out, weighted_rev_out,
            alpha, inv_2N, tid);
    }

    for (int dep = 3; dep < my_num_depths; dep++) {
        process_depth_abomination_pass2(extra_bins, bins_in, Z_in[my_offset + dep],
            teacher_dists + ((long long)my_offset + dep) * 256,
            teacher_row, reduce_buf, my_offset + dep, actual_bytes, byte_mask,
            fwd_kl, rev_kl, ce_diff_arr, ce_diff_grad_arr, fwd_kl_mean, rev_kl_mean,
            weighted_fwd_out, weighted_rev_out,
            alpha, inv_2N, tid);

        if (tid < 256) {
            grad_unnorm_extra[((long long)t * max_extra + (dep - 3)) * 256 + tid] = extra_bins[tid];
        }
        __syncthreads();
    }

    // ============ PASS 4: Backward through scatter + softmax ============
    for (int i = tid; i < V2; i += 256) {
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

std::vector<torch::Tensor> fused_train_abomination_pass1(
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
    int N_total
) {
    const int T1 = logits.size(0);
    const int V = logits.size(1);
    const int max_byte_len = target_byte_seqs.size(1);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
    auto fwd_kl_out = torch::empty({N_total}, opts_f32);
    auto rev_kl_out = torch::empty({N_total}, opts_f32);
    auto ce_out = torch::empty({N_total}, opts_f32);
    auto tce_out = torch::empty({N_total}, opts_f32);
    auto bins_out = torch::empty({N_total, 256}, opts_f32);
    auto Z_out = torch::empty({N_total}, opts_f32);
    auto softmax_max_out = torch::empty({T1}, opts_f32);
    auto softmax_sum_out = torch::empty({T1}, opts_f32);
    auto teacher_entropy_out = torch::empty({N_total}, opts_f32);

    TORCH_CHECK(V % 2 == 0, "V must be even for half2 vectorized loads");
    const int shared_mem = (8*256 + 256 + 256 + 16 + 256 + 256) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_train_abomination_pass1_kernel<<<T1, 256, shared_mem, stream>>>(
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
        fwd_kl_out.data_ptr<float>(),
        rev_kl_out.data_ptr<float>(),
        ce_out.data_ptr<float>(),
        tce_out.data_ptr<float>(),
        bins_out.data_ptr<float>(),
        Z_out.data_ptr<float>(),
        softmax_max_out.data_ptr<float>(),
        softmax_sum_out.data_ptr<float>(),
        teacher_entropy_out.data_ptr<float>(),
        V, max_byte_len, N_total
    );

    return {fwd_kl_out, rev_kl_out, ce_out, tce_out, bins_out, Z_out, softmax_max_out, softmax_sum_out, teacher_entropy_out};
}

std::vector<torch::Tensor> fused_train_abomination_pass2(
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
    torch::Tensor bins_in,
    torch::Tensor Z_in,
    torch::Tensor fwd_kl,
    torch::Tensor rev_kl,
    torch::Tensor ce_diff,
    torch::Tensor ce_diff_grad,
    torch::Tensor softmax_max,
    torch::Tensor softmax_sum,
    float fwd_kl_mean,
    float rev_kl_mean,
    int N_total,
    int n_active,
    float alpha
) {
    const int T1 = logits.size(0);
    const int V = logits.size(1);
    const int max_byte_len = target_byte_seqs.size(1);
    const int max_extra = (max_byte_len > 3) ? (max_byte_len - 3) : 0;

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
    auto grad_logits = torch::zeros({T1, V}, opts_f32);
    auto weighted_fwd = torch::empty({N_total}, opts_f32);
    auto weighted_rev = torch::empty({N_total}, opts_f32);

    auto grad_extra = (max_extra > 0)
        ? torch::empty({T1, max_extra, 256}, opts_f32)
        : torch::empty({0}, opts_f32);

    TORCH_CHECK(V % 2 == 0, "V must be even for half2 vectorized loads");
    const int shared_mem = (256 + 256 + 256 + 16 + 256 + 256) * sizeof(float);  // 5184 bytes

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_train_abomination_pass2_kernel<<<T1, 256, shared_mem, stream>>>(
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
        bins_in.data_ptr<float>(),
        Z_in.data_ptr<float>(),
        fwd_kl.data_ptr<float>(),
        rev_kl.data_ptr<float>(),
        ce_diff.data_ptr<float>(),
        ce_diff_grad.data_ptr<float>(),
        softmax_max.data_ptr<float>(),
        softmax_sum.data_ptr<float>(),
        fwd_kl_mean,
        rev_kl_mean,
        alpha,
        grad_logits.data_ptr<float>(),
        weighted_fwd.data_ptr<float>(),
        weighted_rev.data_ptr<float>(),
        grad_extra.data_ptr<float>(),
        V, max_byte_len, N_total, max_extra,
        n_active
    );

    return {grad_logits, weighted_fwd, weighted_rev};
}

"""

_FUSED_TRAIN_ABOMINATION_CPP = """
std::vector<torch::Tensor> fused_train_abomination_pass1(
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
    int N_total
);
std::vector<torch::Tensor> fused_train_abomination_pass2(
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
    torch::Tensor bins_in,
    torch::Tensor Z_in,
    torch::Tensor fwd_kl,
    torch::Tensor rev_kl,
    torch::Tensor ce_diff,
    torch::Tensor ce_diff_grad,
    torch::Tensor softmax_max,
    torch::Tensor softmax_sum,
    float fwd_kl_mean,
    float rev_kl_mean,
    int N_total,
    int n_active,
    float alpha
);
"""


def get_fused_train_abomination_kernel() -> _FusedTrainAbominationKernel | None:
    if not _cuda_available:
        return None
    kernel = _compile_kernel_with_header(
        "fused_train_abomination",
        _FUSED_TRAIN_ABOMINATION_CPP,
        _FUSED_TRAIN_ABOMINATION_CUDA,
        ["fused_train_abomination_pass1", "fused_train_abomination_pass2"],
    )
    return None if kernel is None else cast(_FusedTrainAbominationKernel, kernel)



def fused_train_forward_backward_abomination(logits, token_ids, byte_vocab, teacher_dists_flat,
                                              actual_bytes_flat, teacher_offsets,
                                              target_byte_lens, alpha, entropy_weighting=False,
                                              byte_mask=None):
    kernel = get_fused_train_abomination_kernel()
    if kernel is None:
        return None, None

    T = logits.shape[0]
    T1 = T - 1

    next_tokens = token_ids[1:]
    next_byte_lens = byte_vocab.token_byte_lens[next_tokens]
    next_byte_seqs = byte_vocab.token_byte_seqs[next_tokens]
    N_total = int(teacher_offsets[-1].item()) + int(target_byte_lens[-1].item())

    if byte_mask is None:
        byte_mask = torch.ones(N_total, dtype=torch.uint8, device=logits.device)
        n_active = N_total
    else:
        byte_mask = byte_mask.to(torch.uint8).contiguous()
        n_active = int(byte_mask.sum().item())
    if n_active == 0:
        return None, None

    logits_h = logits[:T1].half().contiguous()
    fb = byte_vocab.first_bytes_i32.contiguous()
    sb = byte_vocab.second_bytes_i32.contiguous()
    tb = byte_vocab.third_bytes_i32.contiguous()
    bl = byte_vocab.byte_lens_i32.contiguous()
    tbs = byte_vocab.token_byte_seqs_i32.contiguous()
    nbs = next_byte_seqs.int().contiguous()
    nbl = next_byte_lens.int().contiguous()
    td = teacher_dists_flat.float().contiguous()
    to_ = teacher_offsets.int().contiguous()
    ab = actual_bytes_flat.int().contiguous()

    # Two-pass approach to avoid inter-block sync deadlock when T1 > max concurrent blocks
    pass1_results = kernel.fused_train_abomination_pass1(
        logits_h, fb, sb, tb, bl, tbs, nbs, nbl, td, to_, ab, byte_mask, N_total,
    )
    fwd_kl, rev_kl, ce, tce, bins, Z_vals, softmax_max, softmax_sum, teacher_entropy = pass1_results

    eps = 1e-8
    fwd_kl_mean = (fwd_kl.sum() / n_active).clamp(min=eps).item()
    rev_kl_mean = (rev_kl.sum() / n_active).clamp(min=eps).item()
    raw_diff = ce - tce
    ce_diff = torch.nn.functional.softplus(raw_diff, beta=5.0).clamp(max=5.0)
    ce_diff_grad = torch.sigmoid(5.0 * raw_diff)

    pass2_results = kernel.fused_train_abomination_pass2(
        logits_h, fb, sb, tb, bl, tbs, nbs, nbl, td, to_, ab, byte_mask,
        bins, Z_vals, fwd_kl, rev_kl, ce_diff, ce_diff_grad,
        softmax_max, softmax_sum,
        fwd_kl_mean, rev_kl_mean, N_total, n_active, alpha,
    )
    grad_logits, weighted_fwd_per, weighted_rev_per = pass2_results

    if entropy_weighting:
        ew = teacher_entropy / 5.545177  # normalize by log(256)
        ew = ew * byte_mask.float()

        ew_sum = ew.sum().clamp(min=1e-8)
        weighted_fwd = (weighted_fwd_per * ew).sum() / ew_sum
        weighted_rev = (weighted_rev_per * ew).sum() / ew_sum

        # Per-token average entropy weight for logit grad scaling
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
        weighted_fwd = weighted_fwd_per.sum() / n_active
        weighted_rev = weighted_rev_per.sum() / n_active

    total_loss = (weighted_fwd + weighted_rev) / 2

    return grad_logits, {
        "train_loss": total_loss,
        "custom loss": total_loss,
        "CE loss": (ce.sum() / n_active).detach(),
        "kl_div": (fwd_kl.sum() / n_active).detach(),
        "reverse kl_div": (rev_kl.sum() / n_active).detach(),
        "weighted kl_div": weighted_fwd.detach(),
        "weighted rev. kl_div": weighted_rev.detach(),
        "teacher CE loss": (tce.sum() / n_active).detach(),
        "CE diff": (ce_diff.sum() / n_active).detach(),
    }
