#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

cublasStatus_t launch_standard_attention_score(
    cublasHandle_t handle,
    const float* dQ,
    const float* dK,
    float* dS,
    int N,
    int d,
    float scale,
    long long stride_qk,
    long long stride_s,
    int batch_count
);

void launch_standard_softmax(
    float* dS,
    int N,
    int batch_count,
    int warps_per_block = 4,   // default: 4 warps = 128 threads
    cudaStream_t stream
);

cublasStatus_t launch_standard_attention_value(
    cublasHandle_t handle,
    const float* dP,
    const float* dV,
    float* dO,
    int N,
    int d,
    long long stride_p,
    long long stride_vo,
    int batch_count
);

