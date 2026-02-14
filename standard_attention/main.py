import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils import cpp_extension

CUDA_LIB64 = "/usr/local/cuda/lib64"

def build_cuda():
    return cpp_extension.load(
        name="standard_attention_cuda",
        sources=[
            "standard_attention/forward.cu",
            "standard_attention/kernels.cu",
        ],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[
            f"-L{CUDA_LIB64}",
            "-lcublas",
            "-lcudart",
        ],
        verbose=True,
    )

def pytorch_math_self_attention(q, k, v):
    S, D = q.shape

    # (1, 1, S, D)
    q = q.unsqueeze(0).unsqueeze(0)
    k = k.unsqueeze(0).unsqueeze(0)
    v = v.unsqueeze(0).unsqueeze(0)

    with sdpa_kernel([SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    # (S, D)
    return out.squeeze(0).squeeze(0)

# build CUDA
cuda_extension = build_cuda()

# declare Q, K, V
seq_len, dim = 256, 768

q = torch.randn(seq_len, dim, dtype=torch.float32).cuda()
k = torch.randn(seq_len, dim, dtype=torch.float32).cuda()
v = torch.randn(seq_len, dim, dtype=torch.float32).cuda()

# warm-up
for _ in range(3):
    _ = cuda_extension.forward(q, k, v)
    _ = pytorch_math_self_attention(q, k, v)
torch.cuda.synchronize()

# calculate Self-Attention
o_torch_math = pytorch_math_self_attention(q, k, v)
o_cuda = cuda_extension.forward(q, k, v)
torch.cuda.synchronize()

# diff
diff = (o_torch_math - o_cuda).abs()
max_abs = diff.max().item()
mean_abs = diff.mean().item()

denom = o_torch_math.abs().clamp_min(1e-8)
max_rel = (diff / denom).max().item()
mean_rel = (diff / denom).mean().item()

print(f"max_abs_diff : {max_abs:e}")
print(f"mean_abs_diff: {mean_abs:e}")
print(f"max_rel_diff : {max_rel:e}")
print(f"mean_rel_diff: {mean_rel:e}")

atol = 1e-4
rtol = 1e-3
print("allclose:", torch.allclose(o_torch_math, o_cuda, atol=atol, rtol=rtol))