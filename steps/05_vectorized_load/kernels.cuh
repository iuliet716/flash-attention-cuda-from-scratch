#pragma once

#include <cuda_runtime.h>

// maximum head dimension (must be a multiple of 4 for float4 loads)
constexpr int FUSED_D_MAX = 128;

void launch_fused_attention(
    const float* dQ,
    const float* dK,
    const float* dV,
    float* dO,
    int N,
    int d,
    float scale,
    int batch_count,
    cudaStream_t stream);