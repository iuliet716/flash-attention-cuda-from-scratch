"""Shared infrastructure for the per-step measurement scripts.

Each ``bench_stepXX.py`` script measures the metrics that step's doc will
discuss and renders the figures for it. This module provides:

- module building (reuses the JIT helpers from the root ``benchmark.py``)
- latency timing (CUDA events, adaptive iteration count)
- per-kernel timing via ``torch.profiler`` + QK / Softmax / PV phase split
- analytic HBM-traffic and FLOP models
- GPU peak numbers (spec + measured) for %-of-peak and roofline charts
- result JSON writing (``benchmarks/results/``) and figure styling/saving
  (``docs/assets/``)
"""

import datetime
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark as bench  # noqa: E402  (root benchmark.py)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ASSETS_DIR = ROOT / "docs" / "assets"

# Default shape: the README benchmark table shape.
SHAPE = dict(B=8, H=16, N=4096, d=64)
# N sweep used by the scaling charts (unfused steps OOM above 4096 at B8·H16).
SWEEP_NS = [256, 512, 1024, 2048, 4096]

DTYPE_BYTES = 4  # every step 00-04 runs in FP32


# --------------------------------------------------------------------------
# Build / inputs
# --------------------------------------------------------------------------

def build_module(prefix):
    """JIT-build one step by numeric prefix ('00'..'04') -> (name, module)."""
    steps = bench.discover_steps([prefix])
    if not steps:
        raise SystemExit(f"step {prefix} not found under steps/")
    name, _path, sources = steps[0]
    print(f"building {name} ...", flush=True)
    return name, bench.build_step(name, sources)


def make_qkv(B, H, N, d, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    return q, k, v


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def bench_ms(fn, *args, warmup=5, iters=None, budget_ms=2500,
             min_iters=3, max_iters=50):
    """Median latency in ms. If ``iters`` is None, pick it so the measurement
    stays within ``budget_ms`` (slow naive kernels get fewer iterations)."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if iters is None:
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        est = max(start.elapsed_time(end), 1e-3)
        iters = int(max(min_iters, min(max_iters, budget_ms / est)))

    times = []
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], iters


def kernel_timeline(fn, *args, iters=6, warmup=3):
    """Profile ``fn`` and return the averaged per-call kernel timeline:
    [(kernel_name, ms), ...] in launch order within one call."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn(*args)
        torch.cuda.synchronize()

    evts = [e for e in prof.events()
            if e.device_type == DeviceType.CUDA and e.self_device_time_total > 0]
    evts.sort(key=lambda e: e.time_range.start)

    if not evts:
        return []
    if len(evts) % iters != 0:
        # irregular launch pattern: fall back to name-aggregated averages
        agg = {}
        for e in evts:
            agg.setdefault(e.key, 0.0)
            agg[e.key] += e.self_device_time_total
        return [(name, us / iters / 1000.0) for name, us in agg.items()]

    k = len(evts) // iters  # kernels per call
    per_call = []
    for pos in range(k):
        es = [evts[i * k + pos] for i in range(iters)]
        ms = sum(e.self_device_time_total for e in es) / iters / 1000.0
        per_call.append((es[0].key, ms))
    return per_call


PHASES = ["QK matmul", "Softmax", "PV matmul", "Other"]


def phase_breakdown(timeline):
    """Fold a per-call kernel timeline into QK / Softmax / PV phases.

    GEMM kernels are classified by position: the attention pipeline always
    runs QK -> softmax -> PV, so a matmul kernel before the softmax kernel
    is QK and one after it is PV (cuBLAS kernel names carry no role)."""
    phases = {p: 0.0 for p in PHASES}
    seen_softmax = False
    for name, ms in timeline:
        n = name.lower()
        if "softmax" in n:
            phases["Softmax"] += ms
            seen_softmax = True
        elif "qk" in n:
            phases["QK matmul"] += ms
        elif "pv" in n:
            phases["PV matmul"] += ms
        elif any(t in n for t in ("gemm", "cutlass", "magma")):
            phases["PV matmul" if seen_softmax else "QK matmul"] += ms
        else:
            phases["Other"] += ms
    return phases


def softmax_ms(timeline):
    return sum(ms for name, ms in timeline if "softmax" in name.lower())


# --------------------------------------------------------------------------
# Analytic models (ideal traffic: each operand read once, output written once)
# --------------------------------------------------------------------------

# HBM passes over the N x N score matrix per softmax variant (reads, writes):
# naive  (00/01): max pass R, exp-sum pass R, normalize pass R+W  -> (3, 1)
# warp   (02)   : max pass R, exp pass R+W, normalize pass R+W    -> (3, 2)
# online (03)   : fused max+sum pass R, normalize pass R+W        -> (2, 1)
SOFTMAX_PASSES = {"naive": (3, 1), "warp": (3, 2), "online": (2, 1)}


def bytes_qk(B, H, N, d):
    """read Q, K; write S."""
    return DTYPE_BYTES * B * H * (2 * N * d + N * N)


def bytes_pv(B, H, N, d):
    """read P, V; write O."""
    return DTYPE_BYTES * B * H * (N * N + 2 * N * d)


def bytes_softmax(B, H, N, variant):
    r, w = SOFTMAX_PASSES[variant]
    return DTYPE_BYTES * B * H * N * N * (r + w)


def bytes_unfused(B, H, N, d, variant):
    """Total ideal HBM traffic of the 3-kernel pipeline."""
    return (bytes_qk(B, H, N, d) + bytes_softmax(B, H, N, variant)
            + bytes_pv(B, H, N, d))


def bytes_fused_ideal(B, H, N, d):
    """Fused kernel, perfect tiling: read Q, K, V once; write O once."""
    return DTYPE_BYTES * B * H * 4 * N * d


def flops_total(B, H, N, d):
    return bench.attention_flops(B, H, N, d)  # 4*B*H*N^2*d


def flops_gemm(B, H, N, d):
    """One of the two matmuls (QK or PV)."""
    return 2 * B * H * N * N * d


def tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def gbs(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


# --------------------------------------------------------------------------
# GPU peaks
# --------------------------------------------------------------------------

# Spec-sheet peaks keyed by a substring of the CUDA device name.
GPU_SPECS = {
    "RTX 5090": {"fp32_tflops": 104.8, "hbm_gbs": 1792.0},
}


def gpu_peaks():
    """Spec peaks for the current device (None fields if the GPU is unknown)."""
    name = torch.cuda.get_device_name(0)
    for key, spec in GPU_SPECS.items():
        if key in name:
            return dict(spec, gpu=name, source="spec sheet")
    return {"fp32_tflops": None, "hbm_gbs": None, "gpu": name, "source": "unknown"}


def measure_copy_gbs(mib=1024):
    """Achievable HBM bandwidth via a device-to-device copy (read + write)."""
    x = torch.empty(mib * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    ms, _ = bench_ms(lambda: y.copy_(x), warmup=5, iters=30)
    out = gbs(2 * x.nbytes, ms)
    del x, y
    return out


def measure_sgemm_tflops(n=8192):
    """Achievable FP32 compute via a large cuBLAS SGEMM (torch.matmul)."""
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)
    ms, _ = bench_ms(lambda: a @ b, warmup=3, iters=10)
    out = tflops(2 * n ** 3, ms)
    del a, b
    return out


# --------------------------------------------------------------------------
# References (same protocol as benchmark.py)
# --------------------------------------------------------------------------

def reference_latencies(q, k, v):
    """(eager matmul+softmax ms, SDPA FP16 flash ms)."""
    eager_ms, _ = bench_ms(bench.reference_attention, q, k, v)
    qh, kh, vh = (t.half() for t in (q, k, v))
    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
        sdpa_ms, _ = bench_ms(F.scaled_dot_product_attention, qh, kh, vh)
    return eager_ms, sdpa_ms


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

def machine_meta():
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "vram_gib": round(p.total_memory / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date": datetime.date.today().isoformat(),
    }


def save_json(step, data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"step{step}.json"
    with open(path, "w") as f:
        json.dump({"meta": machine_meta(), **data}, f, indent=2)
    print(f"results  -> {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# Figure styling (light-mode reference palette)
# --------------------------------------------------------------------------

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# categorical slots (fixed order — never cycled)
C1_BLUE = "#2a78d6"
C2_AQUA = "#1baf7a"
C3_YELLOW = "#eda100"
C4_GREEN = "#008300"
C5_VIOLET = "#4a3aa7"

PHASE_COLORS = {
    "QK matmul": C1_BLUE,
    "Softmax": C2_AQUA,
    "PV matmul": C3_YELLOW,
    "Other": MUTED,
}
# softmax variants keep one hue per entity across every chart (steps 02/03)
VARIANT_COLORS = {"naive": C1_BLUE, "warp": C2_AQUA, "online": C3_YELLOW}
STEP_COLORS = {"00": C1_BLUE, "01": C2_AQUA, "02": C3_YELLOW,
               "03": C4_GREEN, "04": C5_VIOLET}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 1.0,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK2,
})


def new_fig(w=7.0, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    return fig, ax


def style_axes(ax, grid_axis="y"):
    """Recessive chrome: hairline grid behind the data, no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.tick_params(length=0)


def n_log_axis(ax, ns, axis="x"):
    """Log2 axis ticked exactly at the swept N values."""
    labels = [f"{n // 1024}K" if n >= 1024 else str(n) for n in ns]
    if axis == "x":
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, labels)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_yscale("log", base=2)
        ax.set_yticks(ns, labels)
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def plot_series(ax, xs, ys, color, label=None):
    """2px line, >=8px markers with a surface ring."""
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label,
            zorder=3, solid_capstyle="round")


def end_label(ax, x, y, text, color=INK2, dx=6, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", fontsize=8.5, color=color)


def ref_line(ax, y, text, orientation="h"):
    """A named reference level (peak bandwidth, peak FLOPS, ...)."""
    if orientation == "h":
        ax.axhline(y, color=BASELINE, linewidth=1, zorder=1)
        ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=8, color=MUTED)
    else:
        ax.axvline(y, color=BASELINE, linewidth=1, zorder=1)
        ax.annotate(text, (y, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points",
                    va="top", fontsize=8, color=MUTED)


def stacked_hbars(ax, rows, phases=PHASES, colors=PHASE_COLORS, height=0.5):
    """rows = [(row_label, {phase: ms})] top-to-bottom. 2px surface gaps
    between segments; total labeled at the bar end."""
    ys = range(len(rows))
    for y, (label, parts) in zip(ys, rows):
        left = 0.0
        for phase in phases:
            w = parts.get(phase, 0.0)
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=height, color=colors[phase],
                    edgecolor=SURFACE, linewidth=1.5, zorder=3,
                    label=phase if y == 0 else None)
            left += w
        end_label(ax, left, y, f"{left:.1f} ms")
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.invert_yaxis()
    style_axes(ax, grid_axis="x")


def save_fig(fig, name, footnote=None):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if footnote:
        # below the figure bottom edge; bbox_inches="tight" expands to fit
        fig.text(0.01, -0.03, footnote, fontsize=7.5, color=MUTED, va="top")
    path = ASSETS_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"figure   -> {path.relative_to(ROOT)}")


def shape_tag(B=None, H=None, N=None, d=None):
    s = SHAPE if B is None else dict(B=B, H=H, N=N, d=d)
    return f"B={s['B']} H={s['H']} N={s['N']} d={s['d']}, FP32"
