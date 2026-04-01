#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_flash_attention_forward(
    const half* dQ,
    const half* dK,
    const half* dV,
    half* dO,
    int N,
    int d,
    float scale,
    int batch_count,
    cudaStream_t stream
);