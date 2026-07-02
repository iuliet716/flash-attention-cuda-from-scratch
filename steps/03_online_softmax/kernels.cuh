#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

cublasStatus_t gemm_qk(
    cublasHandle_t handle,
    const float* dQ,
    const float* dK,
    float* dS,
    int N,
    int d,
    float scale,
    int batch_count);

void launch_online_softmax(
    float* dS,
    int N,
    int batch_count,
    cudaStream_t stream);

cublasStatus_t gemm_pv(
    cublasHandle_t handle,
    const float* dP,
    const float* dV,
    float* dO,
    int N,
    int d,
    int batch_count);
