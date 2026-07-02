#include "kernels.cuh"

#include <cmath>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error (") + msg + "): " + cudaGetErrorString(e));
    }
}

static void checkCublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error (") + msg + "): " + cublasGetStatusString(s));
    }
}

torch::Tensor attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    auto q_contig = q.to(torch::kFloat32).contiguous();
    auto k_contig = k.to(torch::kFloat32).contiguous();
    auto v_contig = v.to(torch::kFloat32).contiguous();

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
    auto stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    checkCublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");

    const float* dQ = q_contig.data_ptr<float>();
    const float* dK = k_contig.data_ptr<float>();
    const float* dV = v_contig.data_ptr<float>();
    float* dS       = scores.data_ptr<float>();
    float* dO       = out.data_ptr<float>();

    checkCublas(gemm_qk(handle, dQ, dK, dS, N, d, scale, batch_count), "gemm_qk");
    launch_online_softmax(dS, N, batch_count, stream.stream());
    checkCuda(cudaGetLastError(), "launch_online_softmax");
    checkCublas(gemm_pv(handle, dS, dV, dO, N, d, batch_count), "gemm_pv");

    return out;
}
