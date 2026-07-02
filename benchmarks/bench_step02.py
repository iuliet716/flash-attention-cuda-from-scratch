#!/usr/bin/env python3
"""Step 02 (Warp-reduction Softmax) measurements.

Metrics for the step-02 doc:
- softmax kernel latency vs. N: one thread per row (step 01) vs. one warp
  per row with coalesced access (step 02)
- effective softmax bandwidth vs. the HBM peak: the warp version turns the
  softmax into a properly memory-bound kernel
- end-to-end kernel breakdown step 01 vs. step 02

Figures -> docs/assets/02_*.png, raw numbers -> benchmarks/results/step02.json
"""

import common as cm

VARIANT_OF = {"01": "naive", "02": "warp"}
LABEL = {"naive": "Naive (1 thread / row)", "warp": "Warp (32 lanes / row)"}


def main():
    _, mod01 = cm.build_module("01")
    _, mod02 = cm.build_module("02")
    B, H, N, d = cm.SHAPE["B"], cm.SHAPE["H"], cm.SHAPE["N"], cm.SHAPE["d"]
    peaks = cm.gpu_peaks()
    copy_gbs = cm.measure_copy_gbs()
    print(f"GPU: {peaks['gpu']} · copy bandwidth {copy_gbs:.0f} GB/s")

    # ---- softmax kernel latency across N --------------------------------
    ns = [n for n in cm.SWEEP_NS if n >= 512]
    softmax_sweep = {"naive": {}, "warp": {}}
    breakdown = {}
    total_ms = {}
    for step, mod in [("01", mod01), ("02", mod02)]:
        variant = VARIANT_OF[step]
        for n in ns:
            q, k, v = cm.make_qkv(B, H, n, d)
            timeline = cm.kernel_timeline(mod.forward, q, k, v)
            softmax_sweep[variant][n] = cm.softmax_ms(timeline)
            if n == N:
                breakdown[step] = cm.phase_breakdown(timeline)
                total_ms[step], _ = cm.bench_ms(mod.forward, q, k, v)
            del q, k, v
        print(f"step {step} softmax ms: " + "  ".join(
            f"N={n}: {softmax_sweep[variant][n]:.3f}" for n in ns))

    speedup = softmax_sweep["naive"][N] / softmax_sweep["warp"][N]
    print(f"softmax speedup at N={N}: {speedup:.1f}x")

    # "useful" bandwidth: the minimum traffic any softmax needs (read the
    # N x N matrix once, write it once), divided by the measured time.
    useful_bytes = cm.DTYPE_BYTES * B * H * N * N * 2
    useful_gbs = {v: cm.gbs(useful_bytes, softmax_sweep[v][N])
                  for v in ("naive", "warp")}
    model_gbs = {v: cm.gbs(cm.bytes_softmax(B, H, N, v), softmax_sweep[v][N])
                 for v in ("naive", "warp")}

    # =====================================================================
    # Figure 1: softmax kernel latency vs. N
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 4.4)
    for variant in ("naive", "warp"):
        ys = [softmax_sweep[variant][n] for n in ns]
        cm.plot_series(ax, ns, ys, cm.VARIANT_COLORS[variant],
                       label=LABEL[variant])
        cm.end_label(ax, ns[-1], ys[-1], f"{ys[-1]:.1f} ms")
    ax.annotate(f"{speedup:.0f}× at N={N // 1024}K",
                (N, softmax_sweep["warp"][N]), xytext=(-8, -16),
                textcoords="offset points", ha="right",
                fontsize=9, color=cm.INK2)
    cm.n_log_axis(ax, ns)
    ax.set_yscale("log")
    cm.style_axes(ax, grid_axis="both")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Softmax kernel latency (ms)")
    ax.set_title("Step 02 · One warp per row: softmax kernel latency")
    ax.legend(loc="upper left")
    cm.save_fig(fig, "02_softmax_latency_vs_n.png",
                footnote=f"B={B} H={H} d={d}, FP32 · log-log")

    # =====================================================================
    # Figure 2: effective (useful) softmax bandwidth vs. HBM peak
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 4.0)
    xs = range(2)
    for x, variant in zip(xs, ("naive", "warp")):
        val = useful_gbs[variant]
        ax.bar(x, val, width=0.34, color=cm.VARIANT_COLORS[variant],
               edgecolor=cm.SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{val:.0f} GB/s · {100 * val / peaks['hbm_gbs']:.0f}% of peak",
                    (x, val), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=cm.INK2)
    cm.ref_line(ax, peaks["hbm_gbs"], f"HBM peak {peaks['hbm_gbs']:.0f} GB/s")
    cm.ref_line(ax, copy_gbs, f"measured D2D copy {copy_gbs:.0f} GB/s")
    ax.set_ylim(0, peaks["hbm_gbs"] * 1.12)
    ax.set_xticks(list(xs), [LABEL[v] for v in ("naive", "warp")])
    cm.style_axes(ax, grid_axis="y")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title("Step 02 · Coalesced warp softmax approaches the memory roof")
    cm.save_fig(fig, "02_softmax_bandwidth.png",
                footnote=f"{cm.shape_tag()} · useful traffic = N×N matrix "
                         "read once + written once")

    # =====================================================================
    # Figure 3: end-to-end kernel breakdown, step 01 vs. step 02
    # =====================================================================
    fig, ax = cm.new_fig(7.2, 3.0)
    cm.stacked_hbars(ax, [
        ("01 · naive softmax", breakdown["01"]),
        ("02 · warp softmax", breakdown["02"]),
    ])
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Step 02 · End-to-end: the softmax cost collapses")
    # the horizontal band between the two bars is the only free space
    ax.legend(ncols=3, loc="center right")
    cm.save_fig(fig, "02_kernel_breakdown.png",
                footnote=f"{cm.shape_tag()} · per-kernel device time (torch.profiler)")

    cm.save_json("02", {
        "shape": cm.SHAPE,
        "softmax_sweep_ms": {v: {str(n): softmax_sweep[v][n] for n in ns}
                             for v in softmax_sweep},
        "softmax_speedup_at_default_n": speedup,
        "useful_bandwidth_gbs": useful_gbs,
        "model_bandwidth_gbs": model_gbs,
        "phase_ms": breakdown,
        "total_ms": total_ms,
        "measured_copy_gbs": copy_gbs,
        "peaks": peaks,
    })


if __name__ == "__main__":
    main()
