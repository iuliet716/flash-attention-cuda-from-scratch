#!/usr/bin/env python3
"""Step 04 (Naive Fused Attention, SRAM tiling) measurements.

Metrics for the step-04 doc:
- end-to-end latency of steps 00-04 + PyTorch references: fusion alone is
  a regression (slower than steps 01-03)
- what fusion buys anyway:
  * HBM traffic model — the N x N score matrix never touches HBM
  * measured peak GPU memory vs. N — unfused OOMs where fused keeps going
  * latency vs. N past the unfused OOM wall
- roofline: fusion moves attention from the bandwidth-bound region into the
  compute-bound region, but the naive fused kernel sits far below the roof
  (bank conflicts, scalar FMAs -> steps 05+)

Figures -> docs/assets/04_*.png, raw numbers -> benchmarks/results/step04.json
"""

import numpy as np
import torch

import common as cm

STEPS = ["00", "01", "02", "03", "04"]
SOFTMAX_VARIANT = {"00": "naive", "01": "naive", "02": "warp",
                   "03": "online", "04": None}
BIG_NS = [1024, 2048, 4096, 8192, 16384]


def try_run(fn, *args):
    """Run fn, returning ('ok', result) or ('oom', None)."""
    try:
        return "ok", fn(*args)
    except torch.cuda.OutOfMemoryError:
        return "oom", None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return "oom", None
        raise


def peak_mem_gib(module, B, H, N, d):
    """Peak allocator memory of one forward, inputs included. None on OOM."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def run():
        q, k, v = cm.make_qkv(B, H, N, d)
        module.forward(q, k, v)
        torch.cuda.synchronize()

    status, _ = try_run(run)
    torch.cuda.empty_cache()
    if status == "oom":
        return None
    return torch.cuda.max_memory_allocated() / 2**30


def latency_or_oom(module, B, H, N, d, warmup=2):
    torch.cuda.empty_cache()
    status, qkv = try_run(cm.make_qkv, B, H, N, d)
    if status == "ok":
        status, _ = try_run(module.forward, *qkv)
    if status == "oom":
        del qkv
        torch.cuda.empty_cache()
        return None
    ms, _ = cm.bench_ms(module.forward, *qkv, warmup=warmup)
    del qkv
    torch.cuda.empty_cache()
    return ms


def main():
    mods = {step: cm.build_module(step)[1] for step in STEPS}
    B, H, N, d = cm.SHAPE["B"], cm.SHAPE["H"], cm.SHAPE["N"], cm.SHAPE["d"]
    peaks = cm.gpu_peaks()
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {peaks['gpu']} ({vram_gib:.1f} GiB)")

    # ---- end-to-end latency, all steps + references ---------------------
    q, k, v = cm.make_qkv(B, H, N, d)
    e2e = {}
    for step in STEPS:
        e2e[step], _ = cm.bench_ms(mods[step].forward, q, k, v)
        print(f"step {step}: {e2e[step]:9.3f} ms")
    eager_ms, sdpa_ms = cm.reference_latencies(q, k, v)
    print(f"PyTorch eager: {eager_ms:.3f} ms · SDPA FP16 flash: {sdpa_ms:.3f} ms")
    del q, k, v
    torch.cuda.empty_cache()

    # ---- latency + peak memory vs. N (fused survives the OOM wall) ------
    lat_sweep = {"03": {}, "04": {}}
    mem_sweep = {"03": {}, "04": {}}
    for step in ("03", "04"):
        for n in BIG_NS:
            mem_sweep[step][n] = peak_mem_gib(mods[step], B, H, n, d)
            lat_sweep[step][n] = (latency_or_oom(mods[step], B, H, n, d)
                                  if mem_sweep[step][n] is not None else None)
            m = mem_sweep[step][n]
            t = lat_sweep[step][n]
            print(f"step {step} N={n:<6} "
                  f"{'OOM' if t is None else f'{t:10.3f} ms · {m:6.2f} GiB'}")

    # ---- models for traffic + roofline ----------------------------------
    flops = cm.flops_total(B, H, N, d)
    step_bytes = {
        s: (cm.bytes_fused_ideal(B, H, N, d) if s == "04"
            else cm.bytes_unfused(B, H, N, d, SOFTMAX_VARIANT[s]))
        for s in STEPS
    }
    roofline = {
        s: {"ai": flops / step_bytes[s], "tflops": cm.tflops(flops, e2e[s])}
        for s in STEPS
    }
    traffic_ratio = step_bytes["03"] / step_bytes["04"]
    print(f"traffic model at N={N}: unfused {step_bytes['03'] / 2**30:.1f} GiB "
          f"vs fused {step_bytes['04'] / 2**30:.2f} GiB ({traffic_ratio:.0f}x)")

    # =====================================================================
    # Figure 1: end-to-end latency by step (log scale)
    # =====================================================================
    fig, ax = cm.new_fig(7.4, 3.8)
    rows = [(f"{s} · {cm.bench.TECHNIQUES[s]}", e2e[s], cm.STEP_COLORS[s])
            for s in STEPS]
    rows += [("PyTorch matmul + softmax", eager_ms, cm.MUTED),
             ("PyTorch SDPA flash (FP16)", sdpa_ms, cm.MUTED)]
    for y, (label, ms, color) in enumerate(rows):
        ax.barh(y, ms, height=0.55, color=color,
                edgecolor=cm.SURFACE, linewidth=1.5, zorder=3)
        cm.end_label(ax, ms, y, f"{ms:.1f} ms")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    cm.style_axes(ax, grid_axis="x")
    ax.set_xlabel("Latency (ms, log)")
    ax.set_title("Step 04 · Fusing alone is a regression — tiling ≠ fast yet")
    cm.save_fig(fig, "04_latency_by_step.png",
                footnote=f"{cm.shape_tag()} · median latency")

    # =====================================================================
    # Figure 2: HBM traffic model vs. N (unfused vs. fused)
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 4.4)
    model_ns = [512, 1024, 2048, 4096, 8192, 16384]
    unfused = [cm.bytes_unfused(B, H, n, d, "online") / 2**30 for n in model_ns]
    fused = [cm.bytes_fused_ideal(B, H, n, d) / 2**30 for n in model_ns]
    cm.plot_series(ax, model_ns, unfused, cm.STEP_COLORS["03"],
                   label="Unfused pipeline (step 03) — O(N²) scores in HBM")
    cm.plot_series(ax, model_ns, fused, cm.STEP_COLORS["04"],
                   label="Fused ideal (step 04) — Q, K, V, O only")
    cm.end_label(ax, model_ns[-1], unfused[-1], f"{unfused[-1]:,.0f} GiB")
    cm.end_label(ax, model_ns[-1], fused[-1], f"{fused[-1]:.0f} GiB")
    i = model_ns.index(N)
    ax.annotate(f"{traffic_ratio:.0f}× less traffic at N={N // 1024}K",
                (N, fused[i]), xytext=(0, -18), textcoords="offset points",
                ha="center", fontsize=9, color=cm.INK2)
    cm.n_log_axis(ax, model_ns)
    ax.set_yscale("log")
    cm.style_axes(ax, grid_axis="both")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Ideal HBM traffic (GiB, log)")
    ax.set_title("Step 04 · Fusion removes the N² score traffic (analytic model)")
    ax.legend(loc="upper left")
    cm.save_fig(fig, "04_hbm_traffic_model.png",
                footnote=f"B={B} H={H} d={d}, FP32 · each operand counted once per pass")

    # =====================================================================
    # Figure 3: measured peak memory vs. N (OOM wall)
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 4.4)
    for step, label in [("03", "Unfused (step 03)"), ("04", "Fused (step 04)")]:
        pts = [(n, mem_sweep[step][n]) for n in BIG_NS
               if mem_sweep[step][n] is not None]
        xs, ys = zip(*pts)
        cm.plot_series(ax, xs, ys, cm.STEP_COLORS[step], label=label)
        cm.end_label(ax, xs[-1], ys[-1], f"{ys[-1]:.2f} GiB")
        for x, y in pts:
            # WSL2 unified memory lets allocations exceed VRAM by spilling
            # to host RAM over PCIe instead of failing
            if y > vram_gib:
                ax.annotate("exceeds VRAM →\nspills to host RAM", (x, y),
                            xytext=(-10, -4), textcoords="offset points",
                            ha="right", va="top", fontsize=8.5, color=cm.INK2)
    oom_ns = [n for n in BIG_NS if mem_sweep["03"][n] is None]
    if oom_ns:
        need = (B * H * oom_ns[0] ** 2 * cm.DTYPE_BYTES) / 2**30
        ax.annotate(f"OOM at N={oom_ns[0] // 1024}K\n(scores alone need {need:,.0f} GiB)",
                    (oom_ns[0], vram_gib), xytext=(0, -30),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=cm.INK2)
        ax.plot([oom_ns[0]], [vram_gib], marker="x", markersize=9,
                markeredgewidth=2, color=cm.STEP_COLORS["03"], zorder=3)
    cm.ref_line(ax, vram_gib, f"VRAM {vram_gib:.0f} GiB")
    cm.n_log_axis(ax, BIG_NS)
    ax.set_yscale("log")
    cm.style_axes(ax, grid_axis="both")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Peak GPU memory (GiB, log)")
    ax.set_title("Step 04 · No N×N matrix, no OOM wall")
    ax.legend(loc="upper left")
    cm.save_fig(fig, "04_peak_memory_vs_n.png",
                footnote=f"B={B} H={H} d={d}, FP32 · torch.cuda.max_memory_allocated, "
                         "inputs included")

    # =====================================================================
    # Figure 4: latency vs. N past the OOM wall
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 4.4)
    for step, label in [("03", "Unfused (step 03)"), ("04", "Fused (step 04)")]:
        pts = [(n, lat_sweep[step][n]) for n in BIG_NS
               if lat_sweep[step][n] is not None]
        xs, ys = zip(*pts)
        cm.plot_series(ax, xs, ys, cm.STEP_COLORS[step], label=label)
        cm.end_label(ax, xs[-1], ys[-1], f"{ys[-1]:,.0f} ms")
        for x, y in pts:
            if (mem_sweep[step][x] or 0) > vram_gib:
                ax.annotate("spilled to host RAM", (x, y),
                            xytext=(-10, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=8.5, color=cm.INK2)
    if oom_ns:
        last_ok = max(n for n in BIG_NS if lat_sweep["03"][n] is not None)
        ax.annotate("unfused: OOM →", (last_ok, lat_sweep["03"][last_ok]),
                    xytext=(10, 6), textcoords="offset points",
                    fontsize=9, color=cm.INK2)
    cm.n_log_axis(ax, BIG_NS)
    ax.set_yscale("log")
    cm.style_axes(ax, grid_axis="both")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Latency (ms, log)")
    ax.set_title("Step 04 · Slower per token, but it keeps running")
    ax.legend(loc="upper left")
    cm.save_fig(fig, "04_latency_vs_n.png",
                footnote=f"B={B} H={H} d={d}, FP32 · log-log")

    # =====================================================================
    # Figure 5: roofline, steps 00-04
    # =====================================================================
    fig, ax = cm.new_fig(7.2, 4.6)
    ai_axis = np.geomspace(1, 2048, 128)
    roof = np.minimum(ai_axis * peaks["hbm_gbs"] / 1000.0, peaks["fp32_tflops"])
    ax.plot(ai_axis, roof, color=cm.BASELINE, linewidth=1.4, zorder=1)
    knee = 1000.0 * peaks["fp32_tflops"] / peaks["hbm_gbs"]
    ax.annotate(f"HBM {peaks['hbm_gbs']:.0f} GB/s", (knee / 12, roof[0] * 4),
                fontsize=8, color=cm.MUTED, rotation=38,
                rotation_mode="anchor")
    ax.annotate(f"FP32 peak {peaks['fp32_tflops']:.0f} TFLOPS",
                (ai_axis[-1], peaks["fp32_tflops"]), xytext=(-4, 5),
                textcoords="offset points", ha="right",
                fontsize=8, color=cm.MUTED)
    # per-point label placement: 02/03 nearly coincide, 04 hugs the right edge
    offsets = {"00": (7, -3, "left"), "01": (7, -3, "left"),
               "02": (-9, -3, "right"), "03": (7, -3, "left"),
               "04": (-9, -3, "right")}
    for step in STEPS:
        pt = roofline[step]
        ax.plot(pt["ai"], pt["tflops"], marker="o", markersize=9,
                color=cm.STEP_COLORS[step], markeredgecolor=cm.SURFACE,
                markeredgewidth=1.5, zorder=3, linestyle="none",
                label=f"{step} · {cm.bench.TECHNIQUES[step]}")
        dx, dy, ha = offsets[step]
        ax.annotate(step, (pt["ai"], pt["tflops"]), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=8.5, color=cm.INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    cm.style_axes(ax, grid_axis="both")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte, model)")
    ax.set_ylabel("Achieved TFLOPS (log)")
    ax.set_title("Step 04 · Fusion moves attention off the memory roof — "
                 "now the kernel is the problem")
    ax.legend(loc="upper left", fontsize=8)
    cm.save_fig(fig, "04_roofline.png",
                footnote=f"{cm.shape_tag()} · AI = 4·B·H·N²·d FLOPs / ideal "
                         "HBM bytes per step — L2 caching can lift a point "
                         "slightly above the model roof")

    cm.save_json("04", {
        "shape": cm.SHAPE,
        "e2e_ms": e2e,
        "reference_ms": {"pytorch_eager": eager_ms, "pytorch_sdpa_fp16": sdpa_ms},
        "latency_sweep_ms": {s: {str(n): lat_sweep[s][n] for n in BIG_NS}
                             for s in lat_sweep},
        "peak_memory_gib": {s: {str(n): mem_sweep[s][n] for n in BIG_NS}
                            for s in mem_sweep},
        "traffic_model_bytes": step_bytes,
        "roofline": roofline,
        "vram_gib": vram_gib,
        "peaks": peaks,
    })


if __name__ == "__main__":
    main()
