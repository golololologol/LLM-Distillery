from kernels.compiler import _compile_kernel_with_header, _cuda_available


_INFERENCE_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ __launch_bounds__(256, 3)
void fused_inference_byte_marginalize_kernel(
    const __half* __restrict__ logits,
    const int* __restrict__ first_bytes,
    const int* __restrict__ second_bytes,
    const int* __restrict__ third_bytes,
    const int* __restrict__ byte_lens,
    const int* __restrict__ token_byte_seqs_flat,
    const int* __restrict__ target_byte_seqs,
    const int* __restrict__ target_byte_lens,
    const long long* __restrict__ byte_offsets,
    __half* __restrict__ output,
    const int V,
    const int max_byte_len
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ float shared[];
    float* d0_warp_bins = shared;
    float* d1_bins = d0_warp_bins + 8 * 256;
    float* d2_bins = d1_bins + 256;
    float* extra_bins = d2_bins + 256;
    float* reduce_buf = extra_bins + 256;

    int my_num_depths = target_byte_lens[t];
    long long my_offset = byte_offsets[t];

    int target_b0 = target_byte_seqs[t * max_byte_len];
    int target_b1 = (my_num_depths >= 2) ? target_byte_seqs[t * max_byte_len + 1] : -1;
    int target_b2 = (my_num_depths >= 3) ? target_byte_seqs[t * max_byte_len + 2] : -1;

    const __half* row = logits + (long long)t * V;
    const half2* row_h2 = reinterpret_cast<const half2*>(row);
    int V2 = V / 2;

    // ============ Zero bins ============
    for (int i = tid; i < 8 * 256; i += 256) d0_warp_bins[i] = 0.0f;
    if (tid < 256) { d1_bins[tid] = 0.0f; d2_bins[tid] = 0.0f; }
    __syncthreads();

    // ============ Phase 1: Fused online softmax + C=0 frame scatter ============
    float softmax_max = -3.4e38f;
    float softmax_denom = 0.0f;
    for (int i = tid; i < V2; i += 256) {
        half2 h2 = row_h2[i];
        float val0 = __half2float(h2.x);
        float val1 = __half2float(h2.y);
        int v0 = 2 * i;
        int v1 = v0 + 1;

        float m_new = fmaxf(softmax_max, val0);
        softmax_denom = softmax_denom * __expf(softmax_max - m_new) + __expf(val0 - m_new);
        softmax_max = m_new;
        m_new = fmaxf(softmax_max, val1);
        softmax_denom = softmax_denom * __expf(softmax_max - m_new) + __expf(val1 - m_new);
        softmax_max = m_new;

        float e0 = __expf(fminf(val0, 80.0f));
        float e1 = __expf(fminf(val1, 80.0f));

        atomicAdd(&d0_warp_bins[wid * 256 + first_bytes[v0]], e0);
        atomicAdd(&d0_warp_bins[wid * 256 + first_bytes[v1]], e1);

        if (my_num_depths >= 2 && first_bytes[v0] == target_b0 && byte_lens[v0] >= 2) {
            int sb = second_bytes[v0];
            atomicAdd(&d1_bins[sb], e0);
            if (my_num_depths >= 3 && sb == target_b1 && byte_lens[v0] >= 3) {
                atomicAdd(&d2_bins[third_bytes[v0]], e0);
            }
        }
        if (my_num_depths >= 2 && first_bytes[v1] == target_b0 && byte_lens[v1] >= 2) {
            int sb = second_bytes[v1];
            atomicAdd(&d1_bins[sb], e1);
            if (my_num_depths >= 3 && sb == target_b1 && byte_lens[v1] >= 3) {
                atomicAdd(&d2_bins[third_bytes[v1]], e1);
            }
        }
    }
    blockReduceOnlineSoftmax(reduce_buf, softmax_max, softmax_denom);
    float global_max = reduce_buf[0];
    float global_sum = reduce_buf[8];
    float logsumexp = global_max + logf(global_sum);
    __syncthreads();

    // ============ Phase 2: Correct bins + normalize + write output ============
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

    // d0: normalize and write
    float norm_d0 = (tid < 256) ? d0_warp_bins[tid] : 0.0f;
    float Z0 = blockReduceSum(norm_d0);
    if (tid == 0) reduce_buf[0] = fmaxf(Z0, 1e-30f);
    __syncthreads(); Z0 = reduce_buf[0]; __syncthreads();
    if (tid < 256) {
        output[((long long)my_offset) * 256 + tid] = __float2half(d0_warp_bins[tid] / Z0);
    }

    // d1: normalize and write
    if (my_num_depths >= 2) {
        float norm_d1 = (tid < 256) ? d1_bins[tid] : 0.0f;
        float Z1 = blockReduceSum(norm_d1);
        if (tid == 0) reduce_buf[0] = fmaxf(Z1, 1e-30f);
        __syncthreads(); Z1 = reduce_buf[0]; __syncthreads();
        if (tid < 256) {
            output[((long long)(my_offset + 1)) * 256 + tid] = __float2half(d1_bins[tid] / Z1);
        }
    }

    // d2: normalize and write
    if (my_num_depths >= 3) {
        float norm_d2 = (tid < 256) ? d2_bins[tid] : 0.0f;
        float Z2 = blockReduceSum(norm_d2);
        if (tid == 0) reduce_buf[0] = fmaxf(Z2, 1e-30f);
        __syncthreads(); Z2 = reduce_buf[0]; __syncthreads();
        if (tid < 256) {
            output[((long long)(my_offset + 2)) * 256 + tid] = __float2half(d2_bins[tid] / Z2);
        }
    }

    // ============ Phase 3: Sequential d3+ loop ============
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
            float val = __half2float(logits[(long long)t * V + v]);
            float prob = __expf(val - logsumexp);
            int scatter_byte = token_byte_seqs_flat[voff + dep];
            atomicAdd(&extra_bins[scatter_byte], prob);
        }
        __syncthreads();

        float zx_local = (tid < 256) ? extra_bins[tid] : 0.0f;
        float Zx = blockReduceSum(zx_local);
        if (tid == 0) reduce_buf[0] = fmaxf(Zx, 1e-30f);
        __syncthreads();
        Zx = reduce_buf[0];
        __syncthreads();

        if (tid < 256) {
            output[((long long)(my_offset + dep)) * 256 + tid] = __float2half(extra_bins[tid] / Zx);
        }
        __syncthreads();
    }
}

void fused_inference_byte_marginalize(
    torch::Tensor logits,
    torch::Tensor first_bytes,
    torch::Tensor second_bytes,
    torch::Tensor third_bytes,
    torch::Tensor byte_lens,
    torch::Tensor token_byte_seqs,
    torch::Tensor target_byte_seqs,
    torch::Tensor target_byte_lens,
    torch::Tensor byte_offsets,
    torch::Tensor output,
    int64_t max_byte_len
) {
    const int T1 = logits.size(0);
    const int V = logits.size(1);
    const int block_size = 256;
    const int shared_mem = (8 * 256 + 256 + 256 + 256 + 16) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_inference_byte_marginalize_kernel<<<T1, block_size, shared_mem, stream>>>(
        reinterpret_cast<const __half*>(logits.data_ptr<at::Half>()),
        first_bytes.data_ptr<int>(),
        second_bytes.data_ptr<int>(),
        third_bytes.data_ptr<int>(),
        byte_lens.data_ptr<int>(),
        token_byte_seqs.data_ptr<int>(),
        target_byte_seqs.data_ptr<int>(),
        target_byte_lens.data_ptr<int>(),
        byte_offsets.data_ptr<long long>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        V,
        (int)max_byte_len
    );
}
"""

_INFERENCE_CPP_SOURCE = """
void fused_inference_byte_marginalize(
    torch::Tensor logits,
    torch::Tensor first_bytes,
    torch::Tensor second_bytes,
    torch::Tensor third_bytes,
    torch::Tensor byte_lens,
    torch::Tensor token_byte_seqs,
    torch::Tensor target_byte_seqs,
    torch::Tensor target_byte_lens,
    torch::Tensor byte_offsets,
    torch::Tensor output,
    int64_t max_byte_len
);
"""


def get_inference_kernel():
    if not _cuda_available:
        return None
    return _compile_kernel_with_header(
        "fused_inference_byte_marginalize",
        _INFERENCE_CPP_SOURCE,
        _INFERENCE_CUDA_SOURCE,
        ["fused_inference_byte_marginalize"],
    )

