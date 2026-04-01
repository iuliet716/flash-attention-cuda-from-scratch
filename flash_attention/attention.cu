#include "kernels.cuh"

#include <cmath>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error (") + msg + "): " + cudaGetErrorString(e));
    }
}

torch::Tensor flash_attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    auto q_contig = q.to(torch::kFloat16).contiguous();
    auto k_contig = k.to(torch::kFloat16).contiguous();
    auto v_contig = v.to(torch::kFloat16).contiguous();

    const int B = static_cast<int>(q_contig.size(0));
    const int H = static_cast<int>(q_contig.size(1));
    const int N = static_cast<int>(q_contig.size(2));
    const int d = static_cast<int>(q_contig.size(3));
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const int batch_count = B * H;

    auto options = q_contig.options();
    auto out = torch::empty({B, H, N, d}, options);

    c10::cuda::CUDAGuard device_guard(q_contig.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const half* dQ = (const half*)(q_contig.data_ptr<at::Half>());
    const half* dK = (const half*)(k_contig.data_ptr<at::Half>());
    const half* dV = (const half*)(v_contig.data_ptr<at::Half>());
    half* dO = (half*)out.data_ptr<at::Half>();

    launch_flash_attention_forward(dQ, dK, dV, dO, N, d, scale, batch_count, stream);
    checkCuda(cudaGetLastError(), "flash attention forward kernel launch");

    return out;
}