#include <math.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

__device__ __forceinline__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

// Softmax Kernel (in-place)
__global__ void standard_softmax_kernel(
    half* __restrict__ S,
    int N,
    int total_rows)
{
    int warp_id_in_block = threadIdx.x >> 5;                           // divide by 32
    int lane             = threadIdx.x & 31;

    int warps_per_block  = blockDim.x >> 5;
    int global_row = blockIdx.x * warps_per_block + warp_id_in_block;  // one warp for each row

    if (global_row >= total_rows) return;

    int bh_index = global_row / N;                                     // batch index in B * H
    int row_in_bh = global_row - bh_index * N;

    half* row_ptr = S + \
        (size_t)bh_index * (size_t)N * (size_t)N + \
        (size_t)row_in_bh * (size_t)N;

    // 1. max
    float vmax = -CUDART_INF_F;
    for (int c = lane; c < N; c += 32) {
        vmax = fmaxf(vmax, __half2float(row_ptr[c]));
    }
    vmax = warp_reduce_max(vmax);
    vmax = __shfl_sync(0xffffffffu, vmax, 0);                          // broadcast vmax (lane 0 has reduced result)

    // 2. compute sum of exponentials
    float vsum = 0.0f;
    for (int c = lane; c < N; c += 32) {
        float ex = __expf(__half2float(row_ptr[c]) - vmax);
        row_ptr[c] = __float2half_rn(ex);
        vsum += ex;
    }
    vsum = warp_reduce_sum(vsum);
    vsum = __shfl_sync(0xffffffffu, vsum, 0);

    // 3. normalize
    float inv = 1.0f / vsum;
    for (int c = lane; c < N; c += 32) {
        float ex = __half2float(row_ptr[c]);
        row_ptr[c] = __float2half_rn(ex * inv);
    }
}

// Compute S = scale * QK^T via column-major CuBLAS
cublasStatus_t launch_standard_attention_score(
    cublasHandle_t handle,
    const half* dQ,
    const half* dK,
    half* dS,
    int N,
    int d,
    float scale,
    long long stride_qk,
    long long stride_s,
    int batch_count)
{
    const float beta = 0.0f;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_16F;
    return cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, d,
        &scale,
        dK, CUDA_R_16F, d, stride_qk,
        dQ, CUDA_R_16F, d, stride_qk,
        &beta,
        dS, CUDA_R_16F, N, stride_s,
        batch_count, 
        computeType,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void launch_standard_softmax(
    half* dS,
    int N,
    int batch_count,
    int warps_per_block,
    cudaStream_t stream)
{
    int total_rows = batch_count * N;
    int threads = warps_per_block * 32;
    int blocks  = (total_rows + warps_per_block - 1) / warps_per_block;
    standard_softmax_kernel<<<blocks, threads, 0, stream>>>(dS, N, total_rows);
}

// Compute O = PV via column-major CuBLAS
cublasStatus_t launch_standard_attention_value(
    cublasHandle_t handle,
    const half* dP,
    const half* dV,
    half* dO,
    int N,
    int d,
    long long stride_p,
    long long stride_vo,
    int batch_count)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_16F;
    return cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d, N, N,
        &alpha,
        dV, CUDA_R_16F, d, stride_vo,
        dP, CUDA_R_16F, N, stride_p,
        &beta,
        dO, CUDA_R_16F, d, stride_vo,
        batch_count, 
        computeType,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
