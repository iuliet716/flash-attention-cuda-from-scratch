#include <torch/extension.h>

torch::Tensor attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool is_causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // is_causal defaults to false so the standard (non-causal) benchmark
    // path calls this module exactly like every earlier step
    m.def("forward", &attention_forward, "attention forward",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("is_causal") = false);
}
