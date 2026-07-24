#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Supported head dimensions. The kernel keeps its accumulators in
// registers, so array sizes must be compile-time constants: the kernel is
// specialized per head dim (currently 64 and 128) and per mask mode.
constexpr int FUSED_D_MAX = 128;

void launch_fused_attention(
    const __half* dQ,
    const __half* dK,
    const __half* dV,
    __half* dO,
    int N,
    int d,
    float scale,
    bool is_causal,
    int batch_count,
    cudaStream_t stream);
