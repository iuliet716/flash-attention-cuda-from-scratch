#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <stdio.h>

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

// Kernel 1: Calculate QK^T and materialize scores in HBM
template<int BM=16, int BN=16, int BK=16>
__global__ void standard_attention_score_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    float* __restrict__ S, // Attention Scores stored in HBM (N x N)
    int N, int d, float scale) 
{
    // Q, K tiles in shared memory (16 x 16)
    __shared__ float Qsh[BM][BK];
    __shared__ float Ksh[BN][BK];

    int row0 = blockIdx.y * BM;
    int col0 = blockIdx.x * BN;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = row0 + ty;
    int col = col0 + tx;

    float score = 0.0f;

    for (int k0 = 0; k0 < d; k0 += BK) {
        // Load Q tile: Qsh[ty][k] = Q[row, k0 + k]
        if (row < N && (k0 + tx) < d) {
            Qsh[ty][tx] = Q[row * d + (k0 + tx)];
        } else {
            Qsh[ty][tx] = 0.0f;
        }

        // Load K tile: Ksh[col_in_tile][k] = K[col, k0 + k]
        if (col < N && (k0 + ty) < d) {
            Ksh[tx][ty] = K[col * d + (k0 + ty)];
        } else {
            Ksh[tx][ty] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            score += Qsh[ty][kk] * Ksh[tx][kk];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        S[row * N + col] = score * scale;
    }
}

// Kernel 2: Softmax Kernel (in-place)
__global__ void standard_softmax_kernel(
    float* __restrict__ S,
    int N) 
{
    int warp_id_in_block = threadIdx.x >> 5;  // divide by 32
    int lane             = threadIdx.x & 31;

    int warps_per_block  = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id_in_block;

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

// Kernel 3: Write Output (QK^T * V)
template<int BM=16, int BN=16, int BK=16>
__global__ void standard_attention_value_kernel(
    const float* __restrict__ P, 
    const float* __restrict__ V, 
    float* __restrict__ O, 
    int N, int d) 
{
    __shared__ float Psh[BM][BK];
    __shared__ float Vsh[BN][BK];

    int row0 = blockIdx.y * BM;
    int col0 = blockIdx.x * BN; 

    int ty = threadIdx.y; 
    int tx = threadIdx.x; 

    int row = row0 + ty;
    int col = col0 + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Load P tile: Psh[ty][k] = P[row, k0 + k]
        if (row < N && (k0 + tx) < N) {
            Psh[ty][tx] = P[row * N + (k0 + tx)];
        } else {
            Psh[ty][tx] = 0.0f;
        }

        // Load V tile: Vsh[col_in_tile][k] = V[k0 + k, col]
        if (col < d && (k0 + ty) < N) {
            Vsh[tx][ty] = V[(k0 + ty) * d + col];
        } else {
            Vsh[tx][ty] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += Psh[ty][kk] * Vsh[tx][kk];
        }

        __syncthreads();
    }

    if (row < N && col < d) {
        O[row * d + col] = acc;
    }
}

template __global__ void standard_attention_score_kernel<16,16,16>(
    const float*, const float*, float*, int, int, float
);

template __global__ void standard_attention_value_kernel<16,16,16>(
    const float*, const float*, float*, int, int
);

void launch_standard_attention_score(
    const float* dQ, const float* dK, float* dS,
    int N, int d, float scale, cudaStream_t stream
){
    dim3 block(16, 16);                      // (BN, BM)
    dim3 grid((N + 15)/16, (N + 15)/16);
    standard_attention_score_kernel<16,16,16><<<grid, block, 0, stream>>>(dQ, dK, dS, N, d, scale);
}

void launch_standard_softmax(
    float* dS, int N, int warps_per_block, cudaStream_t stream
){
    int threads = warps_per_block * 32;
    int blocks  = (N + warps_per_block - 1) / warps_per_block;
    standard_softmax_kernel<<<blocks, threads, 0, stream>>>(dS, N);
}

void launch_standard_attention_value(
    const float* dP, const float* dV, float* dO,
    int N, int d, cudaStream_t stream
){
    dim3 block(16, 16);                      // (BN, BM)
    dim3 grid((d + 15)/16, (N + 15)/16);
    standard_attention_value_kernel<16,16,16><<<grid, block, 0, stream>>>(dP, dV, dO, N, d);
}
