#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// maximum head dimension (must be a multiple of 32: wmma tiles are 16 wide
// and the head dimension is split across 2 warp column groups for PV)
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
