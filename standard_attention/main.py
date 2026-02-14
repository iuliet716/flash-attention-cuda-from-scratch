import math
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

def pytorch_ops_self_attention(q, k, v):
    S, D = q.shape

    scores = (q @ k.transpose(0, 1)) / math.sqrt(D)
    softmax_scores = torch.softmax(scores, dim=-1)
    out = softmax_scores @ v
    
    return out

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

def diff_report(name, reference, other):
    diff = (reference - other).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    denom = reference.abs().clamp_min(1e-8)
    max_rel = (diff / denom).max().item()
    mean_rel = (diff / denom).mean().item()

    print(f"\n[{name}]")
    print(f"max_abs_diff  : {max_abs:e}")
    print(f"mean_abs_diff : {mean_abs:e}")
    print(f"max_rel_diff  : {max_rel:e}")
    print(f"mean_rel_diff : {mean_rel:e}")

    atol = 1e-4
    rtol = 1e-3
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
cuda_extension = build_cuda()

# declare Q, K, V
seq_len, dim = 256, 768

q = torch.randn(seq_len, dim, dtype=torch.float32).cuda()
k = torch.randn(seq_len, dim, dtype=torch.float32).cuda()
v = torch.randn(seq_len, dim, dtype=torch.float32).cuda()

# calculate Self-Attention
o_torch_ops = pytorch_ops_self_attention(q, k, v)
o_torch_math = pytorch_math_self_attention(q, k, v)
o_cuda = cuda_extension.forward(q, k, v)
torch.cuda.synchronize()

# diff
diff_report("torch_ops vs o_cuda", o_torch_ops, o_cuda)
diff_report("torch_math vs o_cuda",  o_torch_math, o_cuda)

ops_stats  = bench(lambda: pytorch_ops_self_attention(q, k, v))
math_stats = bench(lambda: pytorch_math_self_attention(q, k, v))
cuda_stats = bench(lambda: cuda_extension.forward(q, k, v))

print("\n=== Speed (ms) ===")
print(f"torch_ops  : mean {ops_stats['mean_ms']:.4f} | median {ops_stats['median_ms']:.4f} | p95 {ops_stats['p95_ms']:.4f} | min {ops_stats['min_ms']:.4f}")
print(f"torch_math : mean {math_stats['mean_ms']:.4f} | median {math_stats['median_ms']:.4f} | p95 {math_stats['p95_ms']:.4f} | min {math_stats['min_ms']:.4f}")
print(f"CUDA       : mean {cuda_stats['mean_ms']:.4f} | median {cuda_stats['median_ms']:.4f} | p95 {cuda_stats['p95_ms']:.4f} | min {cuda_stats['min_ms']:.4f}")

print("\n=== CUDA Speed-up (mean) ===")
print(f"CUDA is faster than torch_ops by {ops_stats['mean_ms'] / cuda_stats['mean_ms']:.2f}x")
print(f"CUDA is faster than torch_math by {math_stats['mean_ms'] / cuda_stats['mean_ms']:.2f}x")
