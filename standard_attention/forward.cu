#include "kernels.cuh"

#include <torch/extension.h>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error (") + msg + "): " + cudaGetErrorString(e));
    }
}

static void checkCublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("CuBLAS error (") + msg + "): status=" + std::to_string((int)s));
    }
}

torch::Tensor standard_attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();

    const int B = static_cast<int>(q_contig.size(0));
    const int H = static_cast<int>(q_contig.size(1));
    const int N = static_cast<int>(q_contig.size(2));
    const int d = static_cast<int>(q_contig.size(3));
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const int batch_count = B * H;

    auto options = q_contig.options();
    auto scores = torch::empty({batch_count, N, N}, options);
    auto out = torch::empty({B, H, N, d}, options);

    c10::cuda::CUDAGuard device_guard(q_contig.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    checkCublas(cublasSetStream(cublas_handle, stream.stream()), "cublasSetStream");

    const float* dQ = q_contig.data_ptr<float>();
    const float* dK = k_contig.data_ptr<float>();
    const float* dV = v_contig.data_ptr<float>();
    float* dS = scores.data_ptr<float>();
    float* dO = out.data_ptr<float>();

    const long long stride_qkv = (long long)N * (long long)d;
    const long long stride_s   = (long long)N * (long long)N;

    // Calculate Attention Score
    checkCublas(launch_standard_attention_score(cublas_handle, dQ, dK, dS, N, d, scale, stride_qkv, stride_s, batch_count), "QK^T SGEMM");
    launch_standard_softmax(dS, N, batch_count, 4, stream.stream());
    checkCuda(cudaGetLastError(), "softmax kernel launch");
    checkCublas(launch_standard_attention_value(cublas_handle, dS, dV, dO, N, d, stride_s, stride_qkv, batch_count), "PV SGEMM");

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &standard_attention_forward, "standard attention forward (CUDA)");
}