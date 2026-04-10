from kernels.compiler import _compile_kernel, _compile_kernel_with_header, _cuda_available


_TRAINING_AKL_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_akl_kernel(
    const float* __restrict__ student,
    const float* __restrict__ teacher,
    const int*   __restrict__ actual_bytes,
    float* __restrict__ grad_student,
    float* __restrict__ out_akl,
    float* __restrict__ out_ce,
    float* __restrict__ out_tce,
    float* __restrict__ out_fkl,
    float* __restrict__ out_rkl,
    float* __restrict__ out_w,
    int N,
    float alpha
) {
    const int pos = blockIdx.x;
    const int b = threadIdx.x;
    if (pos >= N || b >= 256) return;

    const float eps = 1e-8f;
    const float inv_N = 1.0f / (float)N;
    const int base = pos * 256;

    float S = student[base + b];
    float T = teacher[base + b];
    float S_eps = S + eps;
    float T_eps = T + eps;
    float log_S = logf(S_eps);
    float log_T = logf(T_eps);

    float fkl_b = T * (log_T - log_S);
    float rkl_b = S * (log_S - log_T);

    // === Phase 1: reduce fkl, rkl ===
    __shared__ float shared_f[256];

    float fkl_sum = blockReduceSum(fkl_b);
    if (b == 0) shared_f[0] = fkl_sum;
    __syncthreads();
    float fkl_pos = shared_f[0];
    __syncthreads();

    float rkl_sum = blockReduceSum(rkl_b);
    if (b == 0) shared_f[0] = rkl_sum;
    __syncthreads();
    float rkl_pos = shared_f[0];
    __syncthreads();

    // === Phase 2: parallel rank scan for head/tail classification ===
    shared_f[b] = T;
    __syncthreads();

    int my_rank = 0;
    for (int j = 0; j < 256; j++) {
        float other_T = shared_f[j];
        if (other_T > T || (other_T == T && j < b))
            my_rank++;
    }
    __syncthreads();

    // Write sorted values (descending) into shared_f
    shared_f[my_rank] = T;
    __syncthreads();

    // Inclusive prefix sum on sorted teacher values
    float pval = shared_f[b];
    for (int offset = 1; offset < 256; offset <<= 1) {
        __syncthreads();
        float tmp = (b >= offset) ? shared_f[b - offset] : 0.0f;
        __syncthreads();
        shared_f[b] = pval + tmp;
        pval = shared_f[b];
    }
    __syncthreads();

    // Head = bytes whose cumulative sum position <= 0.5
    int is_head = (my_rank == 0 || shared_f[my_rank] <= 0.5f) ? 1 : 0;
    __syncthreads();

    // === Phase 3: compute adaptive weight w ===
    float gap = fabsf(T - S);
    float head_gap_b = gap * (float)is_head;
    float tail_gap_b = gap * (float)(1 - is_head);

    float head_gap = blockReduceSum(head_gap_b);
    if (b == 0) shared_f[0] = head_gap;
    __syncthreads();
    head_gap = shared_f[0];
    __syncthreads();

    float tail_gap = blockReduceSum(tail_gap_b);
    if (b == 0) shared_f[0] = tail_gap;
    __syncthreads();
    tail_gap = shared_f[0];

    float denom = head_gap + tail_gap + eps;
    float w = tail_gap / denom;

    // === Phase 4: per-position loss and outputs ===
    int actual_b = actual_bytes[pos];

    if (b == 0) {
        float akl = (1.0f - w) * fkl_pos + w * rkl_pos;
        out_akl[pos] = akl;
        out_ce[pos] = -logf(student[base + actual_b] + eps);
        out_tce[pos] = -logf(teacher[base + actual_b] + eps);
        out_fkl[pos] = fkl_pos;
        out_rkl[pos] = rkl_pos;
        out_w[pos] = w;
    }

    // === Phase 5: gradient ===
    // d(loss)/d(S_b) has three terms from: (1-w)*fkl, w*rkl, and d(w)/d(S_b)*(rkl-fkl)
    float dFdS = -T / S_eps;
    float dRdS = log_S + S / S_eps - log_T;

    float grad = inv_N * (
        (1.0f - w) * dFdS
        + w * dRdS
    );

    if (b == actual_b) {
        grad += alpha * inv_N * (-1.0f / S_eps);
    }

    grad_student[base + b] = grad;
}

std::vector<torch::Tensor> fused_akl(
    torch::Tensor student,
    torch::Tensor teacher,
    torch::Tensor actual_bytes,
    float alpha
) {
    int N = student.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(student.device());
    auto grad_student = torch::empty({N, 256}, opts);
    auto out_akl = torch::empty({N}, opts);
    auto out_ce = torch::empty({N}, opts);
    auto out_tce = torch::empty({N}, opts);
    auto out_fkl = torch::empty({N}, opts);
    auto out_rkl = torch::empty({N}, opts);
    auto out_w = torch::empty({N}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_akl_kernel<<<N, 256, 0, stream>>>(
        student.data_ptr<float>(),
        teacher.data_ptr<float>(),
        actual_bytes.data_ptr<int>(),
        grad_student.data_ptr<float>(),
        out_akl.data_ptr<float>(),
        out_ce.data_ptr<float>(),
        out_tce.data_ptr<float>(),
        out_fkl.data_ptr<float>(),
        out_rkl.data_ptr<float>(),
        out_w.data_ptr<float>(),
        N, alpha
    );

    return {grad_student, out_akl, out_ce, out_tce, out_fkl, out_rkl, out_w};
}
"""

_TRAINING_AKL_CPP = """
std::vector<torch::Tensor> fused_akl(
    torch::Tensor student,
    torch::Tensor teacher,
    torch::Tensor actual_bytes,
    float alpha
);
"""


def get_akl_kernel():
    if not _cuda_available:
        return None
    return _compile_kernel_with_header(
        "fused_akl",
        _TRAINING_AKL_CPP,
        _TRAINING_AKL_CUDA,
        ["fused_akl"],
    )



def fused_akl_forward_backward(student_dists, teacher_dists, actual_bytes, alpha, entropy_weights=None):
    kernel = get_akl_kernel()
    if kernel is None:
        return None, None

    results = kernel.fused_akl(
        student_dists.float().contiguous(),
        teacher_dists.float().contiguous(),
        actual_bytes.int().contiguous(),
        alpha,
    )
    grad_student, loss_akl, loss_ce, loss_tce, fkl, rkl, w = results

    if entropy_weights is not None:
        N = loss_akl.shape[0]
        w_sum = entropy_weights.sum().clamp(min=1e-8)
        scale = entropy_weights * N / w_sum
        grad_student = grad_student * scale.unsqueeze(1)
        kl_loss = (loss_akl * entropy_weights).sum() / w_sum
        ce_loss = (loss_ce * entropy_weights).sum() / w_sum
    else:
        kl_loss = loss_akl.mean()
        ce_loss = loss_ce.mean()
    total_loss = kl_loss + alpha * ce_loss

    return grad_student, {
        "train_loss": total_loss,
        "custom loss": total_loss,
        "CE loss": loss_ce.mean().detach(),
        "kl_div": fkl.mean().detach(),
        "reverse kl_div": rkl.mean().detach(),
        "teacher CE loss": loss_tce.mean().detach(),
        "adaptive_weight": w.mean().detach(),
    }


# ============================================================
# Fully fused training kernel: softmax + scatter + skew_kl + backward
# ============================================================



_FUSED_TRAIN_AKL_CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ void process_depth_akl(
    float* bins, float Z,
    const float* teacher_src, float* teacher_row, float* reduce_buf,
    int dist_idx, const int* actual_bytes,
    float* loss_per_dist, float* ce_per_dist, float* tce_per_dist,
    float* fkl_per_dist, float* rkl_per_dist, float* w_per_dist,
    float inv_N, float alpha, int tid
) {
    const float eps = 1e-8f;

    // Normalize bins and load teacher
    if (tid < 256) {
        bins[tid] /= Z;
        teacher_row[tid] = teacher_src[tid];
    }
    __syncthreads();

    // Compute per-byte fkl, rkl
    float fkl_b = 0.0f, rkl_b = 0.0f;
    float S = 0.0f, T_val = 0.0f;
    if (tid < 256) {
        S = bins[tid];
        T_val = teacher_row[tid];
        float S_eps = S + eps;
        float T_eps = T_val + eps;
        float log_S = logf(S_eps);
        float log_T = logf(T_eps);
        fkl_b = T_val * (log_T - log_S);
        rkl_b = S * (log_S - log_T);
    }

    // Reduce fkl
    float fkl_sum = blockReduceSum(fkl_b);
    if (tid == 0) reduce_buf[0] = fkl_sum;
    __syncthreads();
    float fkl_pos = reduce_buf[0];
    __syncthreads();

    // Reduce rkl
    float rkl_sum = blockReduceSum(rkl_b);
    if (tid == 0) reduce_buf[0] = rkl_sum;
    __syncthreads();
    float rkl_pos = reduce_buf[0];
    __syncthreads();

    // Parallel threshold scan for head/tail classification
    // Each thread computes its rank (# of values strictly greater)
    int my_rank = 0;
    float my_T = (tid < 256) ? teacher_row[tid] : 0.0f;
    if (tid < 256) {
        for (int j = 0; j < 256; j++) {
            float other_T = teacher_row[j];
            if (other_T > my_T || (other_T == my_T && j < tid))
                my_rank++;
        }
    }
    __syncthreads();

    // Write sorted values (descending) into teacher_row
    if (tid < 256) teacher_row[my_rank] = my_T;
    __syncthreads();

    // Inclusive prefix sum on sorted teacher values
    float pval = (tid < 256) ? teacher_row[tid] : 0.0f;
    for (int offset = 1; offset < 256; offset <<= 1) {
        __syncthreads();
        float tmp = (tid >= offset && tid < 256) ? teacher_row[tid - offset] : 0.0f;
        __syncthreads();
        if (tid < 256) {
            pval += tmp;
            teacher_row[tid] = pval;
        }
    }
    __syncthreads();

    // Head = bytes whose cumulative sum position <= 0.5
    int is_head = (tid < 256) ? ((my_rank == 0 || teacher_row[my_rank] <= 0.5f) ? 1 : 0) : 0;
    __syncthreads();

    // Reload original teacher from global memory (sort destroyed teacher_row)
    if (tid < 256) {
        teacher_row[tid] = teacher_src[tid];
        T_val = teacher_row[tid];
    }
    __syncthreads();

    // Compute head_gap, tail_gap
    float gap = 0.0f;
    if (tid < 256) gap = fabsf(T_val - S);
    float head_gap_b = gap * (float)is_head;
    float tail_gap_b = gap * (float)(1 - is_head);

    float head_gap = blockReduceSum(head_gap_b);
    if (tid == 0) reduce_buf[0] = head_gap;
    __syncthreads();
    head_gap = reduce_buf[0];
    __syncthreads();

    float tail_gap = blockReduceSum(tail_gap_b);
    if (tid == 0) reduce_buf[0] = tail_gap;
    __syncthreads();
    tail_gap = reduce_buf[0];
    __syncthreads();

    float denom = head_gap + tail_gap + eps;
    float w = tail_gap / denom;

    // Write per-dist outputs
    int actual_b = actual_bytes[dist_idx];
    if (tid == 0) {
        float akl = (1.0f - w) * fkl_pos + w * rkl_pos;
        loss_per_dist[dist_idx] = akl;
        ce_per_dist[dist_idx] = -logf(bins[actual_b] + eps);
        tce_per_dist[dist_idx] = -logf(teacher_row[actual_b] + eps);
        fkl_per_dist[dist_idx] = fkl_pos;
        rkl_per_dist[dist_idx] = rkl_pos;
        w_per_dist[dist_idx] = w;
    }
    __syncthreads();

    // Compute gradient per byte
    float grad_b = 0.0f;
    if (tid < 256) {
        float S_eps = S + eps;
        float log_S = logf(S_eps);
        float log_T = logf(T_val + eps);

        float dFdS = -T_val / S_eps;
        float dRdS = log_S + S / S_eps - log_T;

        grad_b = inv_N * (
            (1.0f - w) * dFdS
            + w * dRdS
        );

        if (tid == actual_b) {
            grad_b += alpha * inv_N * (-1.0f / S_eps);
        }
    }

    // Apply normalization Jacobian: grad_unnorm = (grad - dot(grad, q)) / Z
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
void fused_train_akl_kernel(
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
    float* __restrict__ grad_logits,
    float* __restrict__ loss_per_dist,
    float* __restrict__ ce_per_dist,
    float* __restrict__ tce_per_dist,
    float* __restrict__ fkl_per_dist,
    float* __restrict__ rkl_per_dist,
    float* __restrict__ w_per_dist,
    float* __restrict__ grad_unnorm_extra,
    int V, int max_byte_len, int N_total, int max_extra,
    float alpha
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ float shared[];
    float* d0_warp_bins = shared;                // [8 * 256] per-warp bins
    float* d1_bins = d0_warp_bins + 8 * 256;     // [256]
    float* d2_bins = d1_bins + 256;              // [256]
    float* reduce_buf = d2_bins + 256;           // [16]
    float* teacher_row = reduce_buf + 16;        // [256]
    float* extra_bins = teacher_row + 256;       // [256]
    int num_tiles = (V / 2 + 255) / 256;
    float* tile_maxima = extra_bins + 256;        // [num_tiles]

    int my_num_depths = target_byte_lens[t];
    int my_offset = teacher_offsets[t];

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

        // Online softmax update — invalid threads contribute -inf (no effect)
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
    const float inv_N = 1.0f / (float)N_total;

    process_depth_akl(d0_bins, Z0, teacher_dists + (long long)my_offset * 256,
        teacher_row, reduce_buf, my_offset, actual_bytes,
        loss_per_dist, ce_per_dist, tce_per_dist,
        fkl_per_dist, rkl_per_dist, w_per_dist,
        inv_N, alpha, tid);

    if (my_num_depths >= 2) {
        process_depth_akl(d1_bins, Z1, teacher_dists + ((long long)my_offset + 1) * 256,
            teacher_row, reduce_buf, my_offset + 1, actual_bytes,
            loss_per_dist, ce_per_dist, tce_per_dist,
            fkl_per_dist, rkl_per_dist, w_per_dist,
            inv_N, alpha, tid);
    }

    if (my_num_depths >= 3) {
        process_depth_akl(d2_bins, Z2, teacher_dists + ((long long)my_offset + 2) * 256,
            teacher_row, reduce_buf, my_offset + 2, actual_bytes,
            loss_per_dist, ce_per_dist, tce_per_dist,
            fkl_per_dist, rkl_per_dist, w_per_dist,
            inv_N, alpha, tid);
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

        process_depth_akl(extra_bins, Zx, teacher_dists + ((long long)my_offset + dep) * 256,
            teacher_row, reduce_buf, my_offset + dep, actual_bytes,
            loss_per_dist, ce_per_dist, tce_per_dist,
            fkl_per_dist, rkl_per_dist, w_per_dist,
            inv_N, alpha, tid);

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

std::vector<torch::Tensor> fused_train_akl(
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
    int N_total,
    float alpha
) {
    const int T1 = logits.size(0);
    const int V = logits.size(1);
    TORCH_CHECK(V % 2 == 0, "Vocab size must be even for half2 vectorization");
    const int max_byte_len = target_byte_seqs.size(1);
    const int max_extra = (max_byte_len > 3) ? (max_byte_len - 3) : 0;

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
    auto grad_logits = torch::zeros({T1, V}, opts_f32);
    auto loss_akl = torch::empty({N_total}, opts_f32);
    auto loss_ce = torch::empty({N_total}, opts_f32);
    auto loss_tce = torch::empty({N_total}, opts_f32);
    auto loss_fkl = torch::empty({N_total}, opts_f32);
    auto loss_rkl = torch::empty({N_total}, opts_f32);
    auto loss_w = torch::empty({N_total}, opts_f32);

    auto grad_extra = (max_extra > 0)
        ? torch::empty({T1, max_extra, 256}, opts_f32)
        : torch::empty({0}, opts_f32);

    // Shared: d0_warp[8*256] + d1[256] + d2[256] + reduce[16] + teacher[256] + extra[256] + tile_maxima[num_tiles]
    int num_tiles = (V + 511) / 512;
    const int shared_mem = (8*256 + 256 + 256 + 16 + 256 + 256 + num_tiles) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_train_akl_kernel<<<T1, 256, shared_mem, stream>>>(
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
        grad_logits.data_ptr<float>(),
        loss_akl.data_ptr<float>(),
        loss_ce.data_ptr<float>(),
        loss_tce.data_ptr<float>(),
        loss_fkl.data_ptr<float>(),
        loss_rkl.data_ptr<float>(),
        loss_w.data_ptr<float>(),
        grad_extra.data_ptr<float>(),
        V, max_byte_len, N_total, max_extra,
        alpha
    );

    return {grad_logits, loss_akl, loss_ce, loss_tce, loss_fkl, loss_rkl, loss_w};
}
"""

_FUSED_TRAIN_AKL_CPP = """
std::vector<torch::Tensor> fused_train_akl(
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
    int N_total,
    float alpha
);
"""


def get_fused_train_akl_kernel():
    if not _cuda_available:
        return None
    return _compile_kernel_with_header(
        "fused_train_akl",
        _FUSED_TRAIN_AKL_CPP,
        _FUSED_TRAIN_AKL_CUDA,
        ["fused_train_akl"],
    )



def fused_train_forward_backward_akl(logits, token_ids, byte_vocab, teacher_dists_flat,
                                      actual_bytes_flat, teacher_offsets,
                                      target_byte_lens, alpha):
    kernel = get_fused_train_akl_kernel()
    if kernel is None:
        return None, None

    T = logits.shape[0]
    T1 = T - 1

    next_tokens = token_ids[1:]
    next_byte_lens = byte_vocab.token_byte_lens[next_tokens]
    next_byte_seqs = byte_vocab.token_byte_seqs[next_tokens]
    N_total = int(teacher_offsets[-1].item()) + int(target_byte_lens[-1].item())

    results = kernel.fused_train_akl(
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
        N_total, alpha,
    )

    grad_logits, loss_akl, loss_ce, loss_tce, loss_fkl, loss_rkl, loss_w = results

    kl_loss = loss_akl.mean()
    ce_loss = loss_ce.mean()
    total_loss = kl_loss + alpha * ce_loss

    return grad_logits, {
        "train_loss": total_loss,
        "custom loss": total_loss,
        "CE loss": ce_loss.detach(),
        "kl_div": loss_fkl.mean().detach(),
        "reverse kl_div": loss_rkl.mean().detach(),
        "teacher CE loss": loss_tce.mean().detach(),
        "adaptive_weight": loss_w.mean().detach(),
    }

