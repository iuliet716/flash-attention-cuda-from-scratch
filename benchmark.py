import math
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils import cpp_extension

CUDA_LIB64 = "/usr/local/cuda/lib64"

def build_standard_attention_cuda():
    return cpp_extension.load(
        name="standard_attention_cuda",
        sources=[
            "standard_attention/bindings.cpp",
            "standard_attention/attention.cu",
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

def build_flash_attention_cuda():
    return cpp_extension.load(
        name="flash_attention_cuda",
        sources=[
            "flash_attention/bindings.cpp",
            "flash_attention/attention.cu",
            "flash_attention/kernels.cu",
        ],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[
            f"-L{CUDA_LIB64}",
            "-lcublas",
            "-lcudart",
        ],
        verbose=True,
    )

def pytorch_ops_attention(q, k, v):
    # FP16
    q, k, v = (x.half() for x in (q, k, v))

    D = q.shape[-1]

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    softmax_scores = torch.softmax(scores, dim=-1)
    out = softmax_scores @ v
    return out

def pytorch_math_attention(q, k, v):
    # FP16
    q, k, v = (x.half() for x in (q, k, v))

    with sdpa_kernel([SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    return out

def diff_report(name, reference, other):
    # FP32
    diff = (reference.float() - other.float()).abs()

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    denom = reference.abs().clamp_min(1e-4)
    max_rel = (diff / denom).max().item()
    mean_rel = (diff / denom).mean().item()

    print(f"\n[{name}]")
    print(f"max_abs_diff  : {max_abs:e}")
    print(f"mean_abs_diff : {mean_abs:e}")
    print(f"max_rel_diff  : {max_rel:e}")
    print(f"mean_rel_diff : {mean_rel:e}")

    atol = 1e-3
    rtol = 1e-2
    print("allclose      :", torch.allclose(reference, other, atol=atol, rtol=rtol))

def bench(function, iters=200, warm_up=5):
    # warm-up
    for _ in range(warm_up):
        _ = function()
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        starter.record()
        _ = function()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    times_t = torch.tensor(times, device="cpu")
    return {
        "mean_ms": times_t.mean().item(),
        "median_ms": times_t.median().item(),
        "p95_ms": times_t.kthvalue(int(0.95 * len(times)) if int(0.95 * len(times)) > 0 else 1).values.item(),
        "min_ms": times_t.min().item(),
        "max_ms": times_t.max().item(),
        "iters": iters,
    }

# build CUDA
standard_attention_cuda_extension = build_standard_attention_cuda()
flash_attention_cuda_extension = build_flash_attention_cuda()

# declare Q, K, V
batch = 8
num_heads = 16
seq_len = 256
dim = 64

assert dim % num_heads == 0
head_dim = dim // num_heads

q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()
k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()
v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()

# calculate Attention
o_torch_ops = pytorch_ops_attention(q, k, v)
o_torch_math = pytorch_math_attention(q, k, v)
o_standard_cuda = standard_attention_cuda_extension.forward(q, k, v)
o_flash_cuda = flash_attention_cuda_extension.forward(q, k, v)
torch.cuda.synchronize()

# diff
diff_report("torch_ops vs o_stnadard_cuda", o_torch_ops, o_standard_cuda)
diff_report("torch_math vs o_standard_cuda",  o_torch_math, o_standard_cuda)
diff_report("o_standard_cuda vs o_flash_cuda",  o_flash_cuda, o_standard_cuda)

ops_stats  = bench(lambda: pytorch_ops_attention(q, k, v))
math_stats = bench(lambda: pytorch_math_attention(q, k, v))
standard_cuda_stats = bench(lambda: standard_attention_cuda_extension.forward(q, k, v))
flash_cuda_stats    = bench(lambda: flash_attention_cuda_extension.forward(q, k, v))

print("\n=== Speed (ms) ===")
print(f"torch_ops     : mean {ops_stats['mean_ms']:.4f} | median {ops_stats['median_ms']:.4f} | p95 {ops_stats['p95_ms']:.4f} | min {ops_stats['min_ms']:.4f}")
print(f"torch_math    : mean {math_stats['mean_ms']:.4f} | median {math_stats['median_ms']:.4f} | p95 {math_stats['p95_ms']:.4f} | min {math_stats['min_ms']:.4f}")
print(f"standard_CUDA : mean {standard_cuda_stats['mean_ms']:.4f} | median {standard_cuda_stats['median_ms']:.4f} | p95 {standard_cuda_stats['p95_ms']:.4f} | min {standard_cuda_stats['min_ms']:.4f}")
print(f"flash_CUDA : mean {flash_cuda_stats['mean_ms']:.4f} | median {flash_cuda_stats['median_ms']:.4f} | p95 {flash_cuda_stats['p95_ms']:.4f} | min {flash_cuda_stats['min_ms']:.4f}")

print("\n=== Speed-up (mean) ===")
print(f"standard_CUDA is faster than torch_ops by {ops_stats['mean_ms'] / standard_cuda_stats['mean_ms']:.2f}x")
print(f"standard_CUDA is faster than torch_math by {math_stats['mean_ms'] / standard_cuda_stats['mean_ms']:.2f}x")
print(f"flash_CUDA is faster than standard_CUDA by {standard_cuda_stats['mean_ms'] / flash_cuda_stats['mean_ms']:.2f}x")