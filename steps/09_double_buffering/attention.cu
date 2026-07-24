#include "kernels.cuh"

#include <cmath>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
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

    // FP16 storage: inputs are converted to half, and the output is
    // returned in half (same convention as PyTorch SDPA FlashAttention)
    auto q_contig = q.to(torch::kHalf).contiguous();
    auto k_contig = k.to(torch::kHalf).contiguous();
    auto v_contig = v.to(torch::kHalf).contiguous();

    const int B = static_cast<int>(q_contig.size(0));
    const int H = static_cast<int>(q_contig.size(1));
    const int N = static_cast<int>(q_contig.size(2));
    const int d = static_cast<int>(q_contig.size(3));
    TORCH_CHECK(d <= FUSED_D_MAX, "head dim must be <= ", FUSED_D_MAX, ", got ", d);
    // wmma tiles are 16 wide and the PV product splits d across 2 warp
    // column groups, so each warp's slice (d/2) must be a multiple of 16
    TORCH_CHECK(d % 32 == 0, "head dim must be a multiple of 32, got ", d);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const int batch_count = B * H;

    auto out = torch::empty({B, H, N, d}, q_contig.options());

    c10::cuda::CUDAGuard device_guard(q_contig.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    launch_fused_attention(
        reinterpret_cast<const __half*>(q_contig.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_contig.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_contig.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        N, d, scale, batch_count, stream.stream());
    checkCuda(cudaGetLastError(), "launch_fused_attention");

    return out;
}
