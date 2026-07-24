#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// maximum head dimension (must be a multiple of 64: 16-byte chunks are
// 8 halves, and the XOR swizzle permutes chunks in groups of 8)
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
