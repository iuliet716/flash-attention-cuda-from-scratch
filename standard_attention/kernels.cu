#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <stdio.h>

// Kernel 1: QK^T + materialize scores in HBM
// In practice, Standard Attention separates MatMul and Softmax into distinct kernels.
// For comparison, we explicitly materialize the intermediate N x N matrix in HBM.
__global__ void standard_attention_score_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    float* __restrict__ P, // Attention Scores (N x N) stored in HBM
    int N, int d, float scale) 
{
    // Each thread computes a single element P[row, col] of the attention matrix.
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Query Index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Key Index

    if (row < N && col < N) {
        float score = 0.0f;
        // Dot Product Q[row]. K[col]
        for (int k = 0; k < d; ++k) {
            score += Q[row * d + k] * K[col * d + k];
        }
        score *= scale;
                
        // True softmax requires row-wise reductions.
        // To keep this kernel simple, we only write out the scaled dot-product scores.
        P[row * N + col] = score;
    }
}

// Kernel 2: Softmax Kernel (in-place)
// Reads S from HBM, Writes P to HBM
__global__ void standard_softmax_kernel(
    float* __restrict__ S, // Input scores stored in the P buffer; overwritten
    int N) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        // 1. find max value
        float max_val = -CUDART_INF_F;
        for (int c = 0; c < N; ++c) {
            float val = S[row * N + c];
            if (val > max_val) max_val = val;
        }

        // 2. compute sum of exponentials
        float sum = 0.0f;
        for (int c = 0; c < N; ++c) {
            float exp_val = expf(S[row * N + c] - max_val);
            S[row * N + c] = exp_val;
            sum += exp_val;
        }

        // 3. normalize
        for (int c = 0; c < N; ++c) {
            S[row * N + c] /= sum; // Now S becomes Softmax(S)
        }
    }
}

// Kernel 3: P * V
// Reads P and V from HBM, Writes O
__global__ void standard_attention_value_kernel(
    const float* __restrict__ P, 
    const float* __restrict__ V, 
    float* __restrict__ O, 
    int N, int d) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < N && col < d) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            // P[row, k] * V[k, col]
            acc += P[row * N + k] * V[k * d + col];
        }
        O[row * d + col] = acc;
    }
}
