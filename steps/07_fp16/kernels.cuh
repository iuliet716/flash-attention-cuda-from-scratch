#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// maximum head dimension (multiple of 64 for 8-chunk XOR swizzle)
constexpr int FUSED_D_MAX = 128;

void launch_fused_attention(
    const __half* dQ,
    const __half* dK,
    const __half* dV,
    __half* dO,
    int N,
    int d,
    float scale,
    int batch_count,
    cudaStream_t stream);