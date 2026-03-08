#include <torch/extension.h>

torch::Tensor standard_attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &standard_attention_forward, "standard attention forward (CUDA)");
}