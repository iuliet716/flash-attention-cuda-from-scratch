#pragma once
#include <cuda_runtime.h>

void launch_standard_attention_score(
    const float* dQ,
    const float* dK,
    float* dS,
    int N,
    int d,
    float scale,
    cudaStream_t stream = 0
);

void launch_standard_softmax(
    float* dS,
    int N,
    int warps_per_block = 4,   // default: 4 warps = 128 threads
    cudaStream_t stream = 0
);

void launch_standard_attention_value(
    const float* dP,
    const float* dV,
    float* dO,
    int N,
    int d,
    cudaStream_t stream = 0
);
