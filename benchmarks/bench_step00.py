#!/usr/bin/env python3
"""Step 00 (Naive Standard Attention) measurements.

Metrics for the step-00 doc:
- where the time goes: per-kernel latency (QK / Softmax / PV)
- how it scales: per-phase latency vs. sequence length N (O(N^2) growth)
- how far from the hardware: % of peak FP32 compute and % of peak HBM
  bandwidth per kernel (neither is utilized -> naive kernels, memory-bound
  algorithm)

Figures -> docs/assets/00_*.png, raw numbers -> benchmarks/results/step00.json
"""

import common as cm


def main():
    _, mod = cm.build_module("00")
    B, H, N, d = cm.SHAPE["B"], cm.SHAPE["H"], cm.SHAPE["N"], cm.SHAPE["d"]
    peaks = cm.gpu_peaks()
    print(f"GPU: {peaks['gpu']}")

    # ---- default-shape latency + per-kernel breakdown -------------------
    q, k, v = cm.make_qkv(B, H, N, d)
    total_ms, iters = cm.bench_ms(mod.forward, q, k, v)
    timeline = cm.kernel_timeline(mod.forward, q, k, v)
    phases = cm.phase_breakdown(timeline)
    print(f"total {total_ms:.3f} ms (median of {iters})")
    for p in cm.PHASES:
        if phases[p] > 0:
            print(f"  {p:<10} {phases[p]:9.3f} ms  ({phases[p] / total_ms:5.1%})")
    del q, k, v

    # ---- N sweep: per-phase latency -------------------------------------
    sweep = {}
    for n in cm.SWEEP_NS:
        q, k, v = cm.make_qkv(B, H, n, d)
        sweep[n] = cm.phase_breakdown(cm.kernel_timeline(mod.forward, q, k, v))
        del q, k, v
        print(f"  N={n:<5} " + "  ".join(
            f"{p}: {sweep[n][p]:.3f} ms" for p in cm.PHASES if sweep[n][p] > 0))

    # ---- % of peak per kernel -------------------------------------------
    copy_gbs = cm.measure_copy_gbs()
    sgemm_tflops = cm.measure_sgemm_tflops()
    print(f"measured copy bandwidth {copy_gbs:.0f} GB/s, "
          f"cuBLAS SGEMM {sgemm_tflops:.1f} TFLOPS")

    kernel_model = {  # phase -> (flops, ideal bytes)
        "QK matmul": (cm.flops_gemm(B, H, N, d), cm.bytes_qk(B, H, N, d)),
        "Softmax": (5 * B * H * N * N, cm.bytes_softmax(B, H, N, "naive")),
        "PV matmul": (cm.flops_gemm(B, H, N, d), cm.bytes_pv(B, H, N, d)),
    }
    utilization = {}
    for phase, (flops, nbytes) in kernel_model.items():
        ms = phases[phase]
        utilization[phase] = {
            "ms": ms,
            "tflops": cm.tflops(flops, ms),
            "gbs": cm.gbs(nbytes, ms),
            "pct_fp32_peak": 100 * cm.tflops(flops, ms) / peaks["fp32_tflops"],
            "pct_hbm_peak": 100 * cm.gbs(nbytes, ms) / peaks["hbm_gbs"],
        }

    # =====================================================================
    # Figure 1: kernel latency breakdown at the default shape
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 2.9)
    rows = [p for p in cm.PHASES if phases[p] > 0]
    for y, p in enumerate(rows):
        ax.barh(y, phases[p], height=0.5, color=cm.PHASE_COLORS[p],
                edgecolor=cm.SURFACE, linewidth=1.5, zorder=3)
        cm.end_label(ax, phases[p], y,
                     f"{phases[p]:.1f} ms · {phases[p] / total_ms:.0%}")
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    cm.style_axes(ax, grid_axis="x")
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Step 00 · Naive attention — where the time goes")
    cm.save_fig(fig, "00_kernel_breakdown.png",
                footnote=f"{cm.shape_tag()} · per-kernel device time "
                         f"(torch.profiler) · total {total_ms:.1f} ms")

    # =====================================================================
    # Figure 2: per-phase latency vs. N (log-log, ~N^2 growth)
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 4.4)
    ns = cm.SWEEP_NS
    # nudge the two colliding end labels (softmax and PV nearly coincide)
    dys = {"QK matmul": 0, "Softmax": 6, "PV matmul": -6}
    for p in ["QK matmul", "Softmax", "PV matmul"]:
        ys = [sweep[n][p] for n in ns]
        cm.plot_series(ax, ns, ys, cm.PHASE_COLORS[p], label=p)
        cm.end_label(ax, ns[-1], ys[-1], f"{ys[-1]:.0f} ms", dy=dys[p])
    # N^2 slope guide, offset below the QK line so it stays visible
    guide = [0.4 * sweep[ns[0]]["QK matmul"] * (n / ns[0]) ** 2 for n in ns]
    ax.plot(ns, guide, color=cm.BASELINE, linewidth=1.2, zorder=1)
    ax.annotate("∝ N²", (ns[-2], guide[-2]), xytext=(4, -12),
                textcoords="offset points", fontsize=8.5, color=cm.MUTED)
    cm.n_log_axis(ax, ns)
    ax.set_yscale("log")
    cm.style_axes(ax, grid_axis="both")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Kernel latency (ms)")
    ax.set_title("Step 00 · Every kernel scales as N²")
    ax.legend(loc="upper left")
    cm.save_fig(fig, "00_latency_vs_n.png",
                footnote=f"B={B} H={H} d={d}, FP32 · log-log")

    # =====================================================================
    # Figure 3: % of hardware peak per kernel
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 4.0)
    kernels = list(kernel_model)
    xs = range(len(kernels))
    w = 0.32
    for off, key, label, color in [
        (-w / 2, "pct_fp32_peak", f"FP32 compute (peak {peaks['fp32_tflops']:.0f} TFLOPS)", cm.C1_BLUE),
        (w / 2, "pct_hbm_peak", f"HBM bandwidth (peak {peaks['hbm_gbs']:.0f} GB/s)", cm.C2_AQUA),
    ]:
        vals = [utilization[p][key] for p in kernels]
        ax.bar([x + off for x in xs], vals, width=w, color=color,
               edgecolor=cm.SURFACE, linewidth=1.5, zorder=3, label=label)
        for x, val in zip(xs, vals):
            ax.annotate(f"{val:.1f}%" if val >= 0.1 else "<0.1%",
                        (x + off, val), xytext=(0, 3),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=cm.INK2)
    cm.ref_line(ax, 100, "hardware peak = 100%")
    ax.set_ylim(0, 112)
    ax.set_xticks(list(xs), kernels)
    cm.style_axes(ax, grid_axis="y")
    ax.set_ylabel("% of peak")
    ax.set_title("Step 00 · Naive kernels use neither the FLOPs nor the bandwidth")
    ax.legend(loc="upper left")
    cm.save_fig(fig, "00_pct_of_peak.png",
                footnote=f"{cm.shape_tag()} · ideal-traffic model "
                         "(each operand read once, output written once)")

    cm.save_json("00", {
        "shape": cm.SHAPE,
        "total_ms": total_ms,
        "phase_ms": phases,
        "sweep_phase_ms": {str(n): sweep[n] for n in ns},
        "utilization": utilization,
        "peaks": peaks,
        "measured_copy_gbs": copy_gbs,
        "measured_sgemm_tflops": sgemm_tflops,
        "kernel_timeline": [(n, round(ms, 4)) for n, ms in timeline],
    })


if __name__ == "__main__":
    main()
