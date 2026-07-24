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
    TORCH_CHECK(d <= FUSED_D_MAX, "head dim must be <= ", FUSED_D_MAX, ", got ", d);
    // float4 loads move 4 floats at a time
    TORCH_CHECK(d % 4 == 0, "head dim must be a multiple of 4, got ", d);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const int batch_count = B * H;

    // fused kernel: no N x N scores buffer in HBM anymore
    auto out = torch::empty({B, H, N, d}, q_contig.options());

    c10::cuda::CUDAGuard device_guard(q_contig.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    launch_fused_attention(
        q_contig.data_ptr<float>(),
        k_contig.data_ptr<float>(),
        v_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        N, d, scale, batch_count, stream.stream());
    checkCuda(cudaGetLastError(), "launch_fused_attention");

    return out;
}
