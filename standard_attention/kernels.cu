#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>
#include <math.h>

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
    int N)
{
    int warp_id_in_block = threadIdx.x >> 5;  // divide by 32
    int lane             = threadIdx.x & 31;

    int warps_per_block  = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id_in_block; // one warp for each row

    if (row >= N) return;

    // 1. max
    float vmax = -CUDART_INF_F;
    for (int c = lane; c < N; c += 32) {
        float v = S[row * N + c];
        vmax = fmaxf(vmax, v);
    }
    vmax = warp_reduce_max(vmax);

    // broadcast vmax (lane 0 has reduced result)
    vmax = __shfl_sync(0xffffffffu, vmax, 0);

    // 2. compute sum of exponentials
    float vsum = 0.0f;
    for (int c = lane; c < N; c += 32) {
        float exp = expf(S[row * N + c] - vmax);
        S[row * N + c] = exp;
        vsum += exp;
    }
    vsum = warp_reduce_sum(vsum);
    vsum = __shfl_sync(0xffffffffu, vsum, 0);

    // 3. normalize
    float inv = 1.0f / vsum;
    for (int c = lane; c < N; c += 32) {
        S[row * N + c] *= inv;
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
    float scale)
{
    const float beta = 0.0f;
    return cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, d,
        &scale, dK, d, dQ, d, 
        &beta, dS, N);
}

void launch_standard_softmax(
    float* dS,
    int N,
    int warps_per_block,
    cudaStream_t stream)
{
    int threads = warps_per_block * 32;
    int blocks  = (N + warps_per_block - 1) / warps_per_block;
    standard_softmax_kernel<<<blocks, threads, 0, stream>>>(dS, N);
}

// Compute O = PV via column-major CuBLAS
cublasStatus_t launch_standard_attention_value(
    cublasHandle_t handle,
    const float* dP,
    const float* dV,
    float* dO,
    int N,
    int d)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d, N, N,
        &alpha, dV, d, dP, N,
        &beta, dO, d);
}
