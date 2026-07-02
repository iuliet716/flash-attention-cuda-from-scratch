#!/usr/bin/env python3
"""Step 03 (Online Softmax) measurements.

Metrics for the step-03 doc:
- softmax kernel latency vs. N: naive / warp / online. Online softmax is
  roughly on par with the warp version — the point of this step is not
  speed but removing the separate max pass, which is what makes the
  fusion in step 04 possible.
- HBM traffic model per variant: passes over the N x N score matrix
  (3R+1W naive, 3R+2W warp, 2R+1W online)
- end-to-end latency 01 vs. 02 vs. 03 (recorded in the JSON)

Figures -> docs/assets/03_*.png, raw numbers -> benchmarks/results/step03.json
"""

import common as cm

VARIANT_OF = {"01": "naive", "02": "warp", "03": "online"}
LABEL = {
    "naive": "Naive (1 thread / row)",
    "warp": "Warp, 3-pass",
    "online": "Online, single-pass (m, s)",
}
PASS_TAG = {"naive": "3R + 1W", "warp": "3R + 2W", "online": "2R + 1W"}


def main():
    mods = {step: cm.build_module(step)[1] for step in ("01", "02", "03")}
    B, H, N, d = cm.SHAPE["B"], cm.SHAPE["H"], cm.SHAPE["N"], cm.SHAPE["d"]
    peaks = cm.gpu_peaks()
    print(f"GPU: {peaks['gpu']}")

    ns = [n for n in cm.SWEEP_NS if n >= 512]
    softmax_sweep = {}
    total_ms = {}
    for step, mod in mods.items():
        variant = VARIANT_OF[step]
        softmax_sweep[variant] = {}
        for n in ns:
            q, k, v = cm.make_qkv(B, H, n, d)
            timeline = cm.kernel_timeline(mod.forward, q, k, v)
            softmax_sweep[variant][n] = cm.softmax_ms(timeline)
            if n == N:
                total_ms[step], _ = cm.bench_ms(mod.forward, q, k, v)
            del q, k, v
        print(f"step {step} ({variant}) softmax ms: " + "  ".join(
            f"N={n}: {softmax_sweep[variant][n]:.3f}" for n in ns))
    print("end-to-end ms: " + "  ".join(
        f"step {s}: {total_ms[s]:.3f}" for s in total_ms))

    traffic_gib = {v: cm.bytes_softmax(B, H, N, v) / 2**30
                   for v in ("naive", "warp", "online")}

    # =====================================================================
    # Figure 1: softmax kernel latency vs. N (three variants)
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 4.4)
    dys = {"naive": 0, "warp": -6, "online": 6}  # warp/online labels collide
    for variant in ("naive", "warp", "online"):
        ys = [softmax_sweep[variant][n] for n in ns]
        cm.plot_series(ax, ns, ys, cm.VARIANT_COLORS[variant],
                       label=LABEL[variant])
        cm.end_label(ax, ns[-1], ys[-1], f"{ys[-1]:.1f} ms", dy=dys[variant])
    cm.n_log_axis(ax, ns)
    ax.set_yscale("log")
    cm.style_axes(ax, grid_axis="both")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Softmax kernel latency (ms)")
    ax.set_title("Step 03 · Online softmax matches the warp version in one pass")
    ax.legend(loc="upper left")
    cm.save_fig(fig, "03_softmax_latency_vs_n.png",
                footnote=f"B={B} H={H} d={d}, FP32 · log-log")

    # =====================================================================
    # Figure 2: HBM traffic model per softmax variant
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 3.8)
    variants = ["naive", "warp", "online"]
    for x, variant in enumerate(variants):
        val = traffic_gib[variant]
        ax.bar(x, val, width=0.34, color=cm.VARIANT_COLORS[variant],
               edgecolor=cm.SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{val:.0f} GiB · {PASS_TAG[variant]}",
                    (x, val), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=cm.INK2)
    ax.set_xticks(range(len(variants)), [LABEL[v] for v in variants])
    cm.style_axes(ax, grid_axis="y")
    ax.set_ylabel("HBM traffic over the N×N scores (GiB)")
    ax.set_title("Step 03 · Passes over the score matrix (analytic model)")
    cm.save_fig(fig, "03_softmax_traffic_model.png",
                footnote=f"{cm.shape_tag()} · R/W passes × B·H·N²·4 bytes — "
                         "the single (m, s) pass is what enables fusion in step 04")

    cm.save_json("03", {
        "shape": cm.SHAPE,
        "softmax_sweep_ms": {v: {str(n): softmax_sweep[v][n] for n in ns}
                             for v in softmax_sweep},
        "traffic_model_gib": traffic_gib,
        "traffic_passes": PASS_TAG,
        "total_ms": total_ms,
        "peaks": peaks,
    })


if __name__ == "__main__":
    main()
