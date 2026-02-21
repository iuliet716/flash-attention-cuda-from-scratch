#include <math.h>

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
    float* __restrict__ S,
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

    float* row_ptr = S + \
        (size_t)bh_index * (size_t)N * (size_t)N + \
        (size_t)row_in_bh * (size_t)N;

    // 1. max
    float vmax = -CUDART_INF_F;
    for (int c = lane; c < N; c += 32) {
        vmax = fmaxf(vmax, row_ptr[c]);
    }
    vmax = warp_reduce_max(vmax);
    vmax = __shfl_sync(0xffffffffu, vmax, 0);                          // broadcast vmax (lane 0 has reduced result)

    // 2. compute sum of exponentials
    float vsum = 0.0f;
    for (int c = lane; c < N; c += 32) {
        float ex = expf(row_ptr[c] - vmax);
        row_ptr[c] = ex;
        vsum += ex;
    }
    vsum = warp_reduce_sum(vsum);
    vsum = __shfl_sync(0xffffffffu, vsum, 0);

    // 3. normalize
    float inv = 1.0f / vsum;
    for (int c = lane; c < N; c += 32) {
        row_ptr[c] *= inv;
    }
}

// Compute S = scale * QK^T via column-major CuBLAS
cublasStatus_t launch_standard_attention_score(
    cublasHandle_t handle,
    const float* dQ,
    const float* dK,
    float* dS,
    int N,
    int d,
    float scale,
    long long stride_qk,
    long long stride_s,
    int batch_count)
{
    const float beta = 0.0f;
    return cublasSgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, d,
        &scale,
        dK, d, stride_qk,
        dQ, d, stride_qk,
        &beta,
        dS, N, stride_s,
        batch_count);
}

void launch_standard_softmax(
    float* dS,
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
    const float* dP,
    const float* dV,
    float* dO,
    int N,
    int d,
    long long stride_p,
    long long stride_vo,
    int batch_count)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d, N, N,
        &alpha,
        dV, d, stride_vo,
        dP, N, stride_p,
        &beta,
        dO, d, stride_vo,
        batch_count);
}
