#!/usr/bin/env python3
"""Benchmark
Usage:
    python benchmark.py                       # default shape
    python benchmark.py -B 8 -H 16 -N 4096 -d 64
    python benchmark.py --steps 00 01         # only selected steps
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
STEPS_DIR = ROOT / "steps"

# Source extensions that make up a step's extension module.
SOURCE_SUFFIXES = (".cu", ".cpp")

# the README benchmark table.
TECHNIQUES = {
    "00": "Naive Standard Attention (Baseline)",
    "01": "cuBLAS GEMM",
    "02": "Warp-reduction Softmax",
    "03": "Online Softmax",
    "04": "Naive Fused Attention (SRAM Tiling)",
    "05": "Coalescing + Vectorized Load",
    "06": "Bank Conflict Avoidance (Swizzling)",
    "07": "Half-Precision (FP16)",
    "08": "WMMA TensorCore",
    "09": "Double Buffering",
}


# Pre-defined (B, H, N, d) shape sets. Each entry is (label, B, H, N, d).
SHAPE_PRESETS = {
    # small : tiny shapes for quick sanity / correctness checks on any GPU.
    "small": [
        ("tiny", 1, 4, 128, 64),
        ("small", 2, 8, 256, 64),
        ("medium", 4, 8, 512, 64),
    ],
    "llm": [
        # GPT-2 small: n_head=12, n_positions=1024, d_head=768/12=64
        ("gpt2-small", 8, 12, 1024, 64),

        # LLaMA-2 7B-style: hidden_size=4096, n_head=32, d_head=128
        ("llama2-7b-2k", 4, 32, 2048, 128),
        ("llama2-7b-4k", 1, 32, 4096, 128),
    ],
}


def step_prefix(name):
    """Numeric step id, e.g. '00_naive' -> '00'."""
    return name.split("_", 1)[0]


def technique(name):
    return TECHNIQUES.get(step_prefix(name), name)


def discover_steps(selected=None):
    """Return [(name, path), ...] for step dirs that contain buildable sources."""
    steps = []
    for path in sorted(p for p in STEPS_DIR.iterdir() if p.is_dir()):
        if selected is not None:
            # Match on the numeric prefix (e.g. "00") or the full dir name.
            prefix = path.name.split("_", 1)[0]
            if prefix not in selected and path.name not in selected:
                continue
        sources = [p for p in sorted(path.iterdir()) if p.suffix in SOURCE_SUFFIXES]
        if not sources:
            print(f"  [skip] {path.name}: no source files yet")
            continue
        steps.append((path.name, path, sources))
    return steps


def build_step(name, sources, verbose=False):
    """JIT-compile a step into an importable extension module."""
    module_name = "flash_attn_" + name
    return load(
        name=module_name,
        sources=[str(s) for s in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )


def reference_attention(q, k, v):
    """FP32 reference: scale * QK^T -> softmax -> PV."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float())


@torch.no_grad()
def benchmark(fn, q, k, v, warmup, iters):
    """Return median latency in milliseconds over ``iters`` runs."""
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn(q, k, v)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))  # ms
    times.sort()
    return times[len(times) // 2]


def resolve_shapes(args):
    """Build the list of (label, B, H, N, d) shapes to benchmark."""
    # Explicit -B/-H/-N/-d override everything else with a single custom shape.
    if any(v is not None for v in (args.batch, args.heads, args.seqlen, args.headdim)):
        B = args.batch if args.batch is not None else 8
        H = args.heads if args.heads is not None else 16
        N = args.seqlen if args.seqlen is not None else 4096
        d = args.headdim if args.headdim is not None else 64
        return [("custom", B, H, N, d)]

    presets = ["small", "llm"] if args.preset == "all" else [args.preset]
    shapes = []
    for name in presets:
        shapes.extend(SHAPE_PRESETS[name])
    return shapes


def run_shape(modules, label, B, H, N, d, args, device):
    """Run all built modules on one shape and print its benchmark table."""
    print(f"\n{'=' * 70}")
    print(f"Shape [{label}]: B={B}, H={H}, N={N}, d={d}  (dtype=float32)")
    print(f"{'=' * 70}")

    torch.manual_seed(args.seed)
    q = torch.randn(B, H, N, d, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, d, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, d, device=device, dtype=torch.float32)
    ref = reference_attention(q, k, v)

    results = []  # (name, time_ms, ok)
    for name, module in modules:
        if module is None:  # build failed earlier
            results.append((name, None, False))
            continue
        try:
            out = module.forward(q, k, v)
            ok = torch.allclose(out.float(), ref, rtol=args.rtol, atol=args.atol)
            max_err = (out.float() - ref).abs().max().item()
            t_ms = benchmark(module.forward, q, k, v, args.warmup, args.iters)
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {name} run error: {exc}")
            results.append((name, None, False))
            continue
        status = "PASS" if ok else f"FAIL (max_err={max_err:.2e})"
        print(f"  {name:<28} {status:<22} {t_ms:.3f} ms")
        results.append((name, t_ms, ok))

    # Reference implementations: plain-PyTorch attention and SDPA FlashAttention.
    eager_ms = benchmark(reference_attention, q, k, v, args.warmup, args.iters)
    print(f"  {'PyTorch matmul + softmax':<28} {'(reference)':<22} {eager_ms:.3f} ms")
    # SDPA's FlashAttention backend only supports FP16/BF16, so measure it on
    # half-precision inputs with the flash kernel explicitly selected.
    qh, kh, vh = (t.half() for t in (q, k, v))
    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
        sdpa_ms = benchmark(
            F.scaled_dot_product_attention, qh, kh, vh, args.warmup, args.iters
        )
    print(f"  {'PyTorch SDPA (FP16 flash)':<28} {'(reference)':<22} {sdpa_ms:.3f} ms")

    refs = [
        ("PyTorch matmul + softmax", eager_ms),
        ("PyTorch SDPA FlashAttention", sdpa_ms),
    ]
    print_table(results, sdpa_ms, refs, B, H, N, d, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["small", "llm", "all"],
        default="small",
        help="Shape preset to sweep (default: small). Ignored if -B/-H/-N/-d given.",
    )
    parser.add_argument("-B", "--batch", type=int, default=None)
    parser.add_argument("-H", "--heads", type=int, default=None)
    parser.add_argument("-N", "--seqlen", type=int, default=None)
    parser.add_argument("-d", "--headdim", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        nargs="*",
        default=None,
        help="Step prefixes/names to run (default: all). e.g. --steps 00 01",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose build logs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this benchmark requires a GPU.")

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    steps = discover_steps(args.steps)
    if not steps:
        raise SystemExit("No buildable steps found under steps/.")

    # Build every step extension once, then reuse across all shapes.
    modules = []  # (name, module_or_None)
    for name, _path, sources in steps:
        print(f"Building {name} ...")
        try:
            modules.append((name, build_step(name, sources, verbose=args.verbose)))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [fail] build error: {exc}")
            modules.append((name, None))

    for label, B, H, N, d in resolve_shapes(args):
        run_shape(modules, label, B, H, N, d, args, device)


def fmt_time(t_ms):
    """Format the latency as a milliseconds string for the README table."""
    return f"{t_ms:.3f} ms"


def attention_flops(B, H, N, d):
    """FLOPs for one attention forward pass.

    Two matmuls dominate: scores = Q @ K^T and out = probs @ V, each
    2*N*N*d multiply-adds per (batch, head). Softmax is negligible.
    """
    return 2 * (2 * B * H * N * N * d)


def tflops(B, H, N, d, t_ms):
    """Achieved TFLOP/s for a shape given its latency in milliseconds."""
    return attention_flops(B, H, N, d) / (t_ms * 1e-3) / 1e12


def print_table(results, sdpa_ms, refs, B, H, N, d, label=""):
    """Print a markdown benchmark table matching the README format."""
    baseline_ms = next((t for _, t, _ in results if t is not None), None)

    tag = f" [{label}]" if label else ""
    print(f"\nBenchmark{tag}  (B = {B}, H = {H}, N = {N}, d = {d})\n")
    print(
        "| Step | Technique | Latency | Speedup vs. prev. | Speedup vs. Baseline "
        "| TFLOPS* | Speed vs. PyTorch SDPA FlashAttention* (%) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")

    failed = []
    prev_ms = None
    for name, t_ms, ok in results:
        step = step_prefix(name)
        tech = technique(name)
        if t_ms is None:
            print(f"| {step} | {tech} | N/A | N/A | N/A | N/A | N/A |")
            continue
        if not ok:
            failed.append(step)
        is_baseline = prev_ms is None  # first timed row is the baseline
        sp_prev = "N/A" if is_baseline else f"{prev_ms / t_ms:.2f}x"
        sp_base = "N/A" if is_baseline else f"{baseline_ms / t_ms:.2f}x"
        tf = f"{tflops(B, H, N, d, t_ms):.1f}"
        pct = f"{sdpa_ms / t_ms * 100:.1f} %"
        print(
            f"| {step} | {tech} | {fmt_time(t_ms)} | {sp_prev} | {sp_base} "
            f"| {tf} | {pct} |"
        )
        prev_ms = t_ms

    for ref_label, t_ms in refs:
        sp_base = f"{baseline_ms / t_ms:.2f}x" if baseline_ms else "N/A"
        tf = f"{tflops(B, H, N, d, t_ms):.1f}"
        pct = f"{sdpa_ms / t_ms * 100:.1f} %"
        print(
            f"| -- | {ref_label} | {fmt_time(t_ms)} | N/A | {sp_base} "
            f"| {tf} | {pct} |"
        )

    if failed:
        print(f"\nCorrectness check FAILED for step(s): {', '.join(failed)}")


if __name__ == "__main__":
    main()
