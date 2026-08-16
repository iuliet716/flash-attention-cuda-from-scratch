"""Build and benchmark the attention implementations under ``steps/``.

Examples:
    python benchmark.py
    python benchmark.py --preset small
    python benchmark.py -B 8 -H 16 -N 4096 -d 64
    python benchmark.py --flush-l2 --fast-math
"""

import argparse
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
STEPS_DIR = ROOT / "steps"

# Pre-defined (B, H, N, d) shape sets. Each entry is (label, B, H, N, d).
PRESETS = {
    "small": (1, 4, 128, 64),
    "llm": (8, 16, 4096, 64),
    "gpt2-small": (8, 12, 1024, 64),
    "llama2-7b-2k": (4, 32, 2048, 128),
    "llama2-7b-4k": (1, 32, 4096, 128),
}

DEFAULT_PRESET = "llm"

# Below this latency the CUDA-event window is dominated by host-side dispatch
# (pybind, tensor allocation, argument checks) instead of the kernel itself.
OVERHEAD_BOUND_MS = 0.10

# Validation cases. ``logit_scale`` multiplies Q, which widens the score
# distribution: the peaked case forces the running-max updates and accumulator
# rescaling of the online-softmax kernels to actually fire.
VALIDATION_CASES = (
    ("uniform", 1.0),
    ("peaked", 4.0),
)

# The case whose tensors are used for the timed runs.
TIMING_CASE = "uniform"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    parser.add_argument("-B", "--batch", type=int)
    parser.add_argument("-H", "--heads", type=int)
    parser.add_argument("-N", "--seqlen", type=int)
    parser.add_argument("-d", "--headdim", type=int)
    parser.add_argument("--steps", nargs="+", help="step prefixes or directory names")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--fast-math",
        action="store_true",
        help="compile the custom kernels with --use_fast_math (off by default: "
             "the cuBLAS/SDPA baselines are prebuilt without it)",
    )
    parser.add_argument(
        "--flush-l2",
        action="store_true",
        help="evict L2 between timed iterations so repeated runs do not read "
             "a warm cache",
    )
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


def step_dtype(prefix):
    return "FP16" if prefix >= 7 else "FP32"


def step_track(prefix):
    """Steps 00-03 are the unfused track, 04+ the fused FlashAttention track."""
    return "A" if prefix < 4 else "B"


def build_step(step, verbose, fast_math):
    sources = sorted(
        str(path) for path in step.iterdir() if path.suffix in (".cu", ".cpp")
    )
    if not sources:
        raise RuntimeError("no .cu or .cpp sources")

    cuda_cflags = ["-O3"]
    if fast_math:
        cuda_cflags.append("--use_fast_math")

    return load(
        name=f"flash_attn_{step.name}",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_cflags,
        verbose=verbose,
    )


def tolerances(prefix, reference):
    """Tolerances scaled by the magnitude of the reference output.

    A fixed ``atol`` is meaningless here: attention over N keys averages V, so
    the output magnitude shrinks roughly as 1/sqrt(N). At N=4096 the old fixed
    1e-2 was about as large as the signal itself, which let a nearly arbitrary
    FP16 result pass ``allclose``.
    """
    magnitude = reference.abs().max().item()

    if prefix >= 7:
        # Never looser than the original 1e-2, and never absurdly tight.
        return 1e-2, min(1e-2, max(1e-2 * magnitude, 1e-5))

    # FP32 track keeps its original floor and only loosens for large outputs.
    return 1e-4, max(1e-5, 1e-5 * magnitude)


@torch.inference_mode()
def low_precision_reference(inputs):
    """Compute a low-precision PyTorch baseline for the FA-style error check."""
    q, k, v = inputs
    scale = math.sqrt(q.shape[-1])

    # Match FlashAttention's reference test idea: keep FP16 arithmetic and
    # use a mathematically equivalent operation order for the PyTorch baseline.
    scores = torch.matmul(q, (k / scale).transpose(-2, -1))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


@dataclass
class Case:
    """One validation input set and its references."""

    name: str
    fp32_inputs: Tuple[torch.Tensor, ...]
    fp16_inputs: Tuple[torch.Tensor, ...]
    fp32_reference: Optional[torch.Tensor] = None
    fp16_reference: Optional[torch.Tensor] = None
    fa_reference: Optional[torch.Tensor] = None
    torch_fp16_reference: Optional[torch.Tensor] = None


@torch.inference_mode()
def build_case(name, shape, logit_scale, seed, needs_fp32, needs_fp16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (
        torch.randn(*shape, device="cuda", generator=generator)
        for _ in range(3)
    )
    if logit_scale != 1.0:
        q = q * logit_scale

    fp32_inputs = (q, k, v)
    fp16_inputs = tuple(tensor.half() for tensor in fp32_inputs)

    case = Case(name=name, fp32_inputs=fp32_inputs, fp16_inputs=fp16_inputs)

    # Use the Math backend for high-precision correctness references.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        if needs_fp32:
            case.fp32_reference = F.scaled_dot_product_attention(*fp32_inputs)

        if needs_fp16:
            # Preserve the values seen by the FP16 kernel, then compute the
            # high-precision reference in FP32.
            case.fp16_reference = F.scaled_dot_product_attention(
                *(tensor.float() for tensor in fp16_inputs)
            )

    if needs_fp16:
        # FlashAttention-style comparison uses the high-precision result
        # rounded back to the kernel's output dtype.
        case.fa_reference = case.fp16_reference.half()
        case.torch_fp16_reference = low_precision_reference(fp16_inputs)

    return case


@torch.inference_mode()
def validate_case(output, case, prefix):
    if prefix >= 7:
        reference = case.fp16_reference
    else:
        reference = case.fp32_reference

    rtol, atol = tolerances(prefix, reference)

    if not torch.allclose(output.float(), reference, rtol=rtol, atol=atol):
        return False

    if prefix < 7:
        return True

    custom_error = (output - case.fa_reference).abs().max().item()
    pytorch_error = (
        (case.torch_fp16_reference - case.fa_reference).abs().max().item()
    )

    return custom_error <= 2.0 * pytorch_error


@dataclass
class Timing:
    median: float
    minimum: float
    maximum: float

    @property
    def spread_pct(self):
        if self.median == 0.0:
            return 0.0
        return (self.maximum - self.minimum) / self.median * 100.0


def make_flush_buffer():
    """A buffer large enough that writing it evicts the whole L2."""
    properties = torch.cuda.get_device_properties(0)
    l2_bytes = getattr(properties, "L2_cache_size", 0) or (128 << 20)
    return torch.empty(2 * l2_bytes, dtype=torch.uint8, device="cuda")


@torch.inference_mode()
def time_op(fn, warmup, iters, flush_buffer=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]

    for start, end in events:
        # Enqueued before the start event, so the eviction is not timed.
        if flush_buffer is not None:
            flush_buffer.zero_()
        start.record()
        fn()
        end.record()

    torch.cuda.synchronize()

    samples = sorted(start.elapsed_time(end) for start, end in events)

    return Timing(
        median=statistics.median(samples),
        minimum=samples[0],
        maximum=samples[-1],
    )


def tflops(batch, heads, seqlen, headdim, latency_ms):
    flops = 4 * batch * heads * seqlen * seqlen * headdim
    return flops / latency_ms / 1e9


@dataclass
class StepResult:
    name: str
    prefix: int
    dtype: str
    track: str
    timing: Timing
    sdpa_timing: Timing
    correct: bool
    failures: Tuple[str, ...]


def format_vs_prev(result, previous):
    """Only chain the speedup when the two rows are actually comparable.

    A ratio across a track boundary (03 -> 04), across a dtype change
    (06 -> 07), or across a step that failed to build attributes the delta to
    the wrong cause, so those print as ``-``.
    """
    if previous is None:
        return "-"
    if result.prefix != previous.prefix + 1:
        return "-"
    if result.dtype != previous.dtype:
        return "-"
    if result.track != previous.track:
        return "-"
    return f"{previous.timing.median / result.timing.median:.2f}x"


def print_results(results, shape, torch_fp32_timing, sdpa_cold, sdpa_drift_pct):
    batch, heads, seqlen, headdim = shape

    print(f"\nB={batch}, H={heads}, N={seqlen}, d={headdim}\n")
    print(
        "| Step | dtype | Correctness | Latency | Spread | TFLOPS | "
        "vs. prev. | % SDPA |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")

    previous = None

    for result in results:
        if result.correct:
            status = "PASS"
        else:
            status = "FAIL (" + ", ".join(result.failures) + ")"

        print(
            f"| {result.name} | {result.dtype} | {status} | "
            f"{result.timing.median:.3f} ms | "
            f"{result.timing.spread_pct:.1f}% | "
            f"{tflops(*shape, result.timing.median):.1f} | "
            f"{format_vs_prev(result, previous)} | "
            f"{result.sdpa_timing.median / result.timing.median * 100:.1f}% |"
        )

        previous = result

    print(
        "\n`% SDPA` uses a PyTorch SDPA (FP16) measurement taken immediately "
        "after each step,\nso both sides see the same clock and thermal state. "
        "FP32 steps are still compared\nagainst an FP16 baseline. "
        "`vs. prev.` is blank across track, dtype, and skipped-step "
        "boundaries.\n`Spread` is (max - min) / median over the timed "
        "iterations."
    )

    print("\nReferences (single measurement, taken after the steps above)\n")
    print("| Reference | dtype | Latency | TFLOPS |")
    print("|---|---|---:|---:|")

    if torch_fp32_timing is not None:
        print(
            f"| PyTorch matmul + softmax | FP32 | "
            f"{torch_fp32_timing.median:.3f} ms | "
            f"{tflops(*shape, torch_fp32_timing.median):.1f} |"
        )

    print(
        f"| PyTorch SDPA FlashAttention (cold) | FP16 | "
        f"{sdpa_cold.median:.3f} ms | "
        f"{tflops(*shape, sdpa_cold.median):.1f} |"
    )

    print(
        f"\nSDPA baseline drift over the run: {sdpa_drift_pct:+.1f}% "
        "(cold start -> last paired measurement)."
    )


@torch.inference_mode()
def measure_torch_fp32_reference(inputs, warmup, iters, flush_buffer):
    """Unfused FP32 matmul + softmax, the naive PyTorch reference."""
    q, k, v = inputs
    scale = 1.0 / math.sqrt(q.shape[-1])

    def run():
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    return time_op(run, warmup, iters, flush_buffer)


def warn_if_overhead_bound(timing):
    if timing.median >= OVERHEAD_BOUND_MS:
        return

    print(
        f"\nWARNING: the SDPA baseline runs in {timing.median:.3f} ms, below "
        f"the {OVERHEAD_BOUND_MS:.2f} ms\n"
        "         threshold where host-side dispatch dominates the CUDA-event "
        "window.\n"
        "         The custom steps pay pybind dispatch, argument checks and "
        "output\n"
        "         allocation inside that window while SDPA does not, so these "
        "numbers\n"
        "         measure overhead, not kernel throughput. Use a larger shape "
        "(e.g.\n"
        "         --preset llm) for a meaningful comparison."
    )


@torch.inference_mode()
def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this benchmark")

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")

    # Pin TF32 off so the cuBLAS handle shared with step 01 cannot silently
    # switch to tensor-core math while the FP32 reference stays true FP32.
    torch.backends.cuda.matmul.allow_tf32 = False

    shape = resolve_shape(args)
    steps = discover_steps(args.steps)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(
        f"Config: warmup={args.warmup}, iters={args.iters}, "
        f"fast_math={args.fast_math}, flush_l2={args.flush_l2}, "
        f"allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
    )

    flush_buffer = make_flush_buffer() if args.flush_l2 else None

    prefixes = [int(step.name.split("_", 1)[0]) for step in steps]
    needs_fp32 = any(prefix < 7 for prefix in prefixes)
    needs_fp16 = any(prefix >= 7 for prefix in prefixes)

    cases = {
        name: build_case(
            name,
            shape,
            logit_scale,
            seed,
            needs_fp32,
            needs_fp16,
        )
        for seed, (name, logit_scale) in enumerate(VALIDATION_CASES)
    }
    timing_case = cases[TIMING_CASE]

    def run_sdpa():
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            return time_op(
                lambda: F.scaled_dot_product_attention(*timing_case.fp16_inputs),
                args.warmup,
                args.iters,
                flush_buffer,
            )

    # Cold baseline, kept only to report how far the GPU drifts during the run.
    sdpa_cold = run_sdpa()
    warn_if_overhead_bound(sdpa_cold)

    results = []
    errors = []
    sdpa_last = sdpa_cold

    for step in steps:
        print(f"Building {step.name} ...")

        try:
            module = build_step(step, args.verbose, args.fast_math)
            prefix = int(step.name.split("_", 1)[0])

            failures = []
            for name, _ in VALIDATION_CASES:
                case = cases[name]
                inputs = case.fp16_inputs if prefix >= 7 else case.fp32_inputs
                if not validate_case(module.forward(*inputs), case, prefix):
                    failures.append(name)

            timing_inputs = (
                timing_case.fp16_inputs if prefix >= 7 else timing_case.fp32_inputs
            )

            timing = time_op(
                lambda: module.forward(*timing_inputs),
                args.warmup,
                args.iters,
                flush_buffer,
            )

            # Paired baseline: measured right after the step, under the same
            # clock and thermal state.
            sdpa_last = run_sdpa()

            results.append(
                StepResult(
                    name=step.name,
                    prefix=prefix,
                    dtype=step_dtype(prefix),
                    track=step_track(prefix),
                    timing=timing,
                    sdpa_timing=sdpa_last,
                    correct=not failures,
                    failures=tuple(failures),
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

    torch_fp32_timing = None
    try:
        torch_fp32_timing = measure_torch_fp32_reference(
            timing_case.fp32_inputs,
            args.warmup,
            args.iters,
            flush_buffer,
        )
    except torch.cuda.OutOfMemoryError:
        print(
            "[skipped] PyTorch matmul + softmax reference: "
            "out of memory for the N x N scores buffer"
        )

    sdpa_drift_pct = (sdpa_last.median / sdpa_cold.median - 1.0) * 100.0

    print_results(
        results,
        shape,
        torch_fp32_timing,
        sdpa_cold,
        sdpa_drift_pct,
    )

    for name, message in errors:
        print(f"[skipped] {name}: {message}")


if __name__ == "__main__":
    main()