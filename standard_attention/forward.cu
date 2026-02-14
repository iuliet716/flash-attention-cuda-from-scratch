#include "kernels.cuh"

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

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

static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        auto s = cublasCreate(&handle);
        if (s != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
    }
    return handle;
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

    const int N = static_cast<int>(q_contig.size(0));
    const int d = static_cast<int>(q_contig.size(1));
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    auto options = q_contig.options();
    auto s = torch::empty({N, N}, options);
    auto o = torch::empty({N, d}, options);

    c10::cuda::CUDAGuard device_guard(q_contig.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    cublasHandle_t cublas_handle = get_cublas_handle();
    checkCublas(cublasSetStream(cublas_handle, stream.stream()), "cublasSetStream");

    const float* dQ = q_contig.data_ptr<float>();
    const float* dK = k_contig.data_ptr<float>();
    const float* dV = v_contig.data_ptr<float>();
    float* dS = s.data_ptr<float>();
    float* dO = o.data_ptr<float>();

    // Calculate Attention Score
    checkCublas(launch_standard_attention_score(cublas_handle, dQ, dK, dS, N, d, scale), "QK^T SGEMM");
    launch_standard_softmax(dS, N, 4, stream.stream());
    checkCublas(launch_standard_attention_value(cublas_handle, dS, dV, dO, N, d), "PV SGEMM");

    checkCuda(cudaGetLastError(), "kernel launch");

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &standard_attention_forward, "standard attention forward (CUDA)");
}