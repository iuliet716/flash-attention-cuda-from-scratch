#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_naive_qk(
    const float* dQ,
    const float* dK,
    float* dS,
    int N,
    int d,
    float scale,
    int batch_count,
    cudaStream_t stream);

void launch_naive_softmax(
    float* dS,
    int N,
    int batch_count,
    cudaStream_t stream);

void launch_naive_pv(
    const float* dP,
    const float* dV,
    float* dO,
    int N,
    int d,
    int batch_count,
    cudaStream_t stream);
