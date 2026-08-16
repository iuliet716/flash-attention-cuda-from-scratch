"""Build and benchmark the attention implementations under ``steps/``.

Examples:
    python benchmark.py
    python benchmark.py --preset llm
    python benchmark.py -B 8 -H 16 -N 4096 -d 64
    python benchmark.py --causal
"""

import argparse
import math
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
STEPS_DIR = ROOT / "steps"

# Pre-defined (B, H, N, d) shape sets. Each entry is (label, B, H, N, d).
PRESETS = {
    "small": (1, 4, 128, 64),
    "gpt2-small": (8, 12, 1024, 64),
    "llama2-7b-2k": (4, 32, 2048, 128),
    "llama2-7b-4k": (1, 32, 4096, 128),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("-B", "--batch", type=int)
    parser.add_argument("-H", "--heads", type=int)
    parser.add_argument("-N", "--seqlen", type=int)
    parser.add_argument("-d", "--headdim", type=int)
    parser.add_argument("--steps", nargs="+", help="step prefixes or directory names")
    parser.add_argument(
        "--causal", action="store_true", help="benchmark causal step 11 only"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_shape(args):
    shape = list(PRESETS[args.preset])
    overrides = (args.batch, args.heads, args.seqlen, args.headdim)
    return tuple(
        value if value is not None else shape[i]
        for i, value in enumerate(overrides)
    )


def discover_steps(selected):
    steps = [path for path in sorted(STEPS_DIR.iterdir()) if path.is_dir()]
    if not selected:
        return steps

    matches = []
    for value in selected:
        match = next(
            (
                path
                for path in steps
                if path.name == value or path.name.split("_", 1)[0] == value
            ),
            None,
        )
        if match is None:
            raise SystemExit(f"step not found: {value}")
        if match not in matches:
            matches.append(match)
    return sorted(matches)


def build_step(step, verbose):
    sources = sorted(
        str(path) for path in step.iterdir() if path.suffix in (".cu", ".cpp")
    )
    if not sources:
        raise RuntimeError("no .cu or .cpp sources")

    return load(
        name=f"flash_attn_{step.name}",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )


def tolerances(prefix):
    if prefix >= 7:
        return 1e-2, 1e-2
    return 1e-4, 1e-5


@torch.inference_mode()
def low_precision_reference(inputs, causal=False):
    """Compute a low-precision PyTorch baseline for the FA-style error check."""
    q, k, v = inputs
    scale = math.sqrt(q.shape[-1])

    # Match FlashAttention's reference test idea: keep FP16 arithmetic and
    # use a mathematically equivalent operation order for the PyTorch baseline.
    scores = torch.matmul(q, (k / scale).transpose(-2, -1))

    if causal:
        seqlen_q = q.shape[-2]
        seqlen_k = k.shape[-2]
        mask = torch.ones(
            (seqlen_q, seqlen_k),
            device=scores.device,
            dtype=torch.bool,
        ).triu(diagonal=1)
        scores.masked_fill_(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def validate(
    output,
    reference,
    rtol,
    atol,
    fa_reference=None,
    pytorch_reference=None,
):
    output_fp32 = output.float()

    allclose_ok = torch.allclose(
        output_fp32,
        reference,
        rtol=rtol,
        atol=atol,
    )

    if fa_reference is None or pytorch_reference is None:
        return allclose_ok

    custom_error = (output - fa_reference).abs().max().item()
    pytorch_error = (pytorch_reference - fa_reference).abs().max().item()

    flash_ok = custom_error <= 2.0 * pytorch_error

    return allclose_ok and flash_ok


@torch.inference_mode()
def median_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]

    for start, end in events:
        start.record()
        fn()
        end.record()

    torch.cuda.synchronize()

    return statistics.median(
        start.elapsed_time(end) for start, end in events
    )


def tflops(batch, heads, seqlen, headdim, latency_ms, causal=False):
    key_count = seqlen * (seqlen + 1) // 2 if causal else seqlen * seqlen
    flops = 4 * batch * heads * key_count * headdim
    return flops / latency_ms / 1e9


def print_results(results, sdpa_ms, shape, causal):
    batch, heads, seqlen, headdim = shape
    mode = "causal" if causal else "non-causal"

    print(f"\nB={batch}, H={heads}, N={seqlen}, d={headdim}, mode={mode}\n")
    print("| Step | Correctness | Latency | TFLOPS | vs. prev. | % SDPA |")
    print("|---|---:|---:|---:|---:|---:|")

    previous_ms = None

    for name, latency_ms, correct in results:
        status = "PASS" if correct else "FAIL"
        speedup = (
            "-" if previous_ms is None else f"{previous_ms / latency_ms:.2f}x"
        )

        print(
            f"| {name} | {status} | {latency_ms:.3f} ms | "
            f"{tflops(*shape, latency_ms, causal):.1f} | "
            f"{speedup} | "
            f"{sdpa_ms / latency_ms * 100:.1f}% |"
        )

        previous_ms = latency_ms

    print(
        f"| PyTorch SDPA (FP16) | reference | {sdpa_ms:.3f} ms | "
        f"{tflops(*shape, sdpa_ms, causal):.1f} | - | 100.0% |"
    )


@torch.inference_mode()
def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this benchmark")
    
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")

    shape = resolve_shape(args)

    selected = args.steps or (["11"] if args.causal else None)
    steps = discover_steps(selected)

    if args.causal and any(
        step.name.split("_", 1)[0] != "11" for step in steps
    ):
        raise SystemExit("--causal can only benchmark step 11")

    print(f"Device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(0)

    fp32_inputs = tuple(
        torch.randn(*shape, device="cuda") for _ in range(3)
    )
    fp16_inputs = tuple(tensor.half() for tensor in fp32_inputs)

    prefixes = [int(step.name.split("_", 1)[0]) for step in steps]
    needs_fp32 = any(prefix < 7 for prefix in prefixes)
    needs_fp16 = any(prefix >= 7 for prefix in prefixes)

    fp32_reference = None
    fp16_input_reference = None
    fa_reference = None
    fp16_pytorch_reference = None

    # Use the Math backend for high-precision correctness references.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        if needs_fp32:
            fp32_reference = F.scaled_dot_product_attention(
                *fp32_inputs,
                is_causal=args.causal,
            )

        if needs_fp16:
            # Preserve the values seen by the FP16 kernel, then compute the
            # high-precision reference in FP32.
            fp16_input_reference = F.scaled_dot_product_attention(
                *(tensor.float() for tensor in fp16_inputs),
                is_causal=args.causal,
            )

    if needs_fp16:
        # FlashAttention-style comparison uses the high-precision result
        # rounded back to the kernel's output dtype.
        fa_reference = fp16_input_reference.half()

        fp16_pytorch_reference = low_precision_reference(
            fp16_inputs,
            causal=args.causal,
        )

    # PyTorch FlashAttention is used only as the performance baseline.
    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
        sdpa_ms = median_ms(
            lambda: F.scaled_dot_product_attention(
                *fp16_inputs,
                is_causal=args.causal,
            ),
            args.warmup,
            args.iters,
        )

    results = []
    errors = []

    for step in steps:
        print(f"Building {step.name} ...")

        try:
            module = build_step(step, args.verbose)
            prefix = int(step.name.split("_", 1)[0])

            if prefix >= 7:
                inputs = fp16_inputs
                reference = fp16_input_reference
                step_fa_reference = fa_reference
                pytorch_reference = fp16_pytorch_reference
            else:
                inputs = fp32_inputs
                reference = fp32_reference
                step_fa_reference = None
                pytorch_reference = None

            rtol, atol = tolerances(prefix)

            def forward():
                if args.causal:
                    return module.forward(*inputs, True)
                return module.forward(*inputs)

            output = forward()

            correct = validate(
                output,
                reference,
                rtol,
                atol,
                fa_reference=step_fa_reference,
                pytorch_reference=pytorch_reference,
            )

            latency_ms = median_ms(
                forward,
                args.warmup,
                args.iters,
            )

            results.append(
                (
                    step.name,
                    latency_ms,
                    correct,
                )
            )

        except Exception as error:
            # Keep benchmarking the remaining learning steps.
            message = str(error).strip() or type(error).__name__
            message = message.splitlines()[-1]
            errors.append((step.name, message))

    if not results:
        for name, message in errors:
            print(f"[skipped] {name}: {message}")
        raise SystemExit("no step completed successfully")

    print_results(
        results,
        sdpa_ms,
        shape,
        args.causal,
    )

    for name, message in errors:
        print(f"[skipped] {name}: {message}")


if __name__ == "__main__":
    main()