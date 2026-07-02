#include "kernels.cuh"

#include <math.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// cuBLAS is column-major, our tensors are row-major.
// row-major product C = A * B is computed as the column-major product C^T = B^T * A^T.

// S = scale * QK^T
// row-major: S (N x N) = Q (N x d) * K^T (d x N)
// col-major: S^T = K * Q^T
// - cuBLAS GEMM arguments: A = K buffer (op T), B = Q buffer (op N)
cublasStatus_t gemm_qk(
    cublasHandle_t handle,
    const float* dQ, const float* dK, float* dS,
    int N, int d, float scale, int batch_count)
{
    const float beta = 0.0f;
    return cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, d,
        &scale,
        dK, d, (long long)N * d,
        dQ, d, (long long)N * d,
        &beta,
        dS, N, (long long)N * N,
        batch_count);
}

__global__ void warp_softmax_kernel(
    float* __restrict__ S,
    int N,
    int total_rows)
{
    // one warp for softmax by row
    // consecutive lanes read consecutive elements at each iteration,
    // so accesses are coalesced (each lane strides by 32 across iterations)
    const int warps_per_block = blockDim.x / 32;
    const int warp = blockIdx.x * warps_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= total_rows) return;

    float* row = S + (size_t)warp * N;

    // max by row: per-lane partial max, then warp reduction
    float max_val = -CUDART_INF_F;
    for (int j = lane; j < N; j += 32) {
        max_val = fmaxf(max_val, row[j]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }

    // exp(x - max) written in place, per-lane partial sum, then warp reduction
    float sum = 0.0f;
    for (int j = lane; j < N; j += 32) {
        float e = __expf(row[j] - max_val);
        row[j] = e;
        sum += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // normalization
    const float inv_sum = 1.0f / sum;
    for (int j = lane; j < N; j += 32) {
        row[j] *= inv_sum;
    }
}

void launch_warp_softmax(
    float* dS, int N, int batch_count, cudaStream_t stream)
{
    const int total_rows = batch_count * N;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = (total_rows + warps_per_block - 1) / warps_per_block;
    warp_softmax_kernel<<<blocks, threads, 0, stream>>>(dS, N, total_rows);
}

// O = PV
// row-major: O (N x d) = P (N x N) * V (N x d)
// col-major: O^T = V^T * P^T
// - cuBLAS arguments: A = V buffer (op N), B = P buffer (op N)
cublasStatus_t gemm_pv(
    cublasHandle_t handle,
    const float* dP, const float* dV, float* dO,
    int N, int d, int batch_count)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d, N, N,
        &alpha,
        dV, d, (long long)N * d,
        dP, N, (long long)N * N,
        &beta,
        dO, d, (long long)N * d,
        batch_count);
}
