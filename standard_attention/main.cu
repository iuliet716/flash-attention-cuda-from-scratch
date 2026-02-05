#include "kernels.cuh"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    // Example sizes
    const int N = 256;       // sequence length
    const int d = 768;       // head dim
    const float scale = 1.0f / std::sqrt((float)d);

    // Host buffers
    const size_t bytesQKV = (size_t)N * d * sizeof(float);
    const size_t bytesS   = (size_t)N * N * sizeof(float);
    const size_t bytesO   = (size_t)N * d * sizeof(float);

    float* hQ = (float*)std::malloc(bytesQKV);
    float* hK = (float*)std::malloc(bytesQKV);
    float* hV = (float*)std::malloc(bytesQKV);
    float* hO = (float*)std::malloc(bytesO);

    if (!hQ || !hK || !hV || !hO) {
        std::fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize Q, K, V
    for (int i = 0; i < N * d; ++i) {
        hQ[i] = 0.001f * (float)(i % 97);
        hK[i] = 0.001f * (float)(i % 89);
        hV[i] = 0.001f * (float)(i % 83);
    }

    // Device buffers
    float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dS=nullptr, *dO=nullptr;

    checkCuda(cudaMalloc(&dQ, bytesQKV), "cudaMalloc dQ");
    checkCuda(cudaMalloc(&dK, bytesQKV), "cudaMalloc dK");
    checkCuda(cudaMalloc(&dV, bytesQKV), "cudaMalloc dV");
    checkCuda(cudaMalloc(&dS, bytesS),   "cudaMalloc dS (NxN)");
    checkCuda(cudaMalloc(&dO, bytesO),   "cudaMalloc dO");

    checkCuda(cudaMemcpy(dQ, hQ, bytesQKV, cudaMemcpyHostToDevice), "H2D Q");
    checkCuda(cudaMemcpy(dK, hK, bytesQKV, cudaMemcpyHostToDevice), "H2D K");
    checkCuda(cudaMemcpy(dV, hV, bytesQKV, cudaMemcpyHostToDevice), "H2D V");

    // Clear buffers
    checkCuda(cudaMemset(dS, 0, bytesS), "memset S");
    checkCuda(cudaMemset(dO, 0, bytesO), "memset O");

    // 1. S = QK^T * scale
    launch_standard_attention_score(dQ, dK, dS, N, d, scale);

    // 2. S = softmax(S)   (in-place)
    // warps_per_block = 4 (128 threads)
    launch_standard_softmax(dS, N, 4);

    // 3. O = PV
    launch_standard_attention_value(dS, dV, dO, N, d);

    // Sync and Error Check
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "device sync");

    // Copy output from device to host
    checkCuda(cudaMemcpy(hO, dO, bytesO, cudaMemcpyDeviceToHost), "D2H O");

    // Free
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dS);
    cudaFree(dO);

    std::free(hQ);
    std::free(hK);
    std::free(hV);
    std::free(hO);

    return 0;
}
