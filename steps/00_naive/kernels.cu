#include <math.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void naive_qk_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int N,
    int d,
    float scale)
{
    // one thread for each element in score matrix  
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // K row (K^T col) 
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // Q row
    const int batch = blockIdx.z;

    if (row >= N || col >= N) return;

    // batch offset
    const float* Qb = Q + (size_t)batch * N * d;
    const float* Kb = K + (size_t)batch * N * d;
    float* Sb = S + (size_t)batch * N * N;

    // S = scale * QK^T
    float acc = 0.0f;
    for (int k = 0; k < d; ++k) {
        acc += Qb[row * d + k] * Kb[col * d + k];
    }

    Sb[row * N + col] = acc * scale;
}

__global__ void naive_softmax_kernel(
    float* __restrict__ S,
    int N,
    int total_rows)
{
    // one thread for softmax by row
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= total_rows) return;

    float* row = S + (size_t)r * N;

    // max by row
    float max_val = -CUDART_INF_F;
    for (int j = 0; j < N; ++j) {
        max_val = fmaxf(max_val, row[j]);
    }

    // sum of exp(x - max)
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += __expf(row[j] - max_val);
    }

    // normalization
    const float inv_sum = 1.0f / sum;
    for (int j = 0; j < N; ++j) {
        float p = __expf(row[j] - max_val) * inv_sum;
        row[j] = p;
    }
}

__global__ void naive_pv_kernel(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int d)
{
    // one thread for each element in output matrixs
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // d
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // N
    const int batch = blockIdx.z;

    if (row >= N || col >= d) return;

    // batch offset
    const float* Pb = P + (size_t)batch * N * N;
    const float* Vb = V + (size_t)batch * N * d;
    float* Ob = O + (size_t)batch * N * d;

    // O = PV
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) {
        acc += Pb[row * N + k] * Vb[k * d + col];
    }

    Ob[row * d + col] = acc;
}

void launch_naive_qk(
    const float* dQ, const float* dK, float* dS,
    int N, int d, float scale, int batch_count, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y,
              batch_count);
    naive_qk_kernel<<<grid, block, 0, stream>>>(dQ, dK, dS, N, d, scale);
}

void launch_naive_softmax(
    float* dS, int N, int batch_count, cudaStream_t stream)
{
    const int total_rows = batch_count * N;
    const int threads = 256;
    const int blocks = (total_rows + threads - 1) / threads;
    naive_softmax_kernel<<<blocks, threads, 0, stream>>>(dS, N, total_rows);
}

void launch_naive_pv(
    const float* dP, const float* dV, float* dO,
    int N, int d, int batch_count, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((d + block.x - 1) / block.x,
              (N + block.y - 1) / block.y,
              batch_count);
    naive_pv_kernel<<<grid, block, 0, stream>>>(dP, dV, dO, N, d);
}