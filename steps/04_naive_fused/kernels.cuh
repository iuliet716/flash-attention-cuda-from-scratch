#pragma once

#include <cuda_runtime.h>

// Maximum head dimension supported by the fused kernel
// (per-row output accumulator lives in registers: D_MAX / 32 floats per lane).
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
