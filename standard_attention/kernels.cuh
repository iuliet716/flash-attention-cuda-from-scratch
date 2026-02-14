#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

cublasStatus_t launch_standard_attention_score(
    cublasHandle_t handle,
    const float* dQ,
    const float* dK,
    float* dS,
    int N,
    int d,
    float scale
);

void launch_standard_softmax(
    float* dS,
    int N,
    int warps_per_block = 4,   // default: 4 warps = 128 threads
    cudaStream_t stream = 0
);

cublasStatus_t launch_standard_attention_value(
    cublasHandle_t handle,
    const float* dP,
    const float* dV,
    float* dO,
    int N,
    int d
);

