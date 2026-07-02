#!/usr/bin/env python3
"""Step 01 (cuBLAS GEMM) measurements.

Metrics for the step-01 doc:
- kernel breakdown step 00 vs. step 01: the two GEMMs collapse, the
  untouched naive softmax becomes the dominant cost (Amdahl's law)
- achieved GEMM TFLOPS: hand-written naive matmul vs. cuBLAS, against the
  FP32 peak and a large-SGEMM measured ceiling

Figures -> docs/assets/01_*.png, raw numbers -> benchmarks/results/step01.json
"""

import common as cm


def main():
    _, mod00 = cm.build_module("00")
    _, mod01 = cm.build_module("01")
    B, H, N, d = cm.SHAPE["B"], cm.SHAPE["H"], cm.SHAPE["N"], cm.SHAPE["d"]
    peaks = cm.gpu_peaks()
    print(f"GPU: {peaks['gpu']}")

    q, k, v = cm.make_qkv(B, H, N, d)
    results = {}
    for step, mod in [("00", mod00), ("01", mod01)]:
        total_ms, _ = cm.bench_ms(mod.forward, q, k, v)
        phases = cm.phase_breakdown(cm.kernel_timeline(mod.forward, q, k, v))
        results[step] = {"total_ms": total_ms, "phase_ms": phases}
        print(f"step {step}: {total_ms:8.3f} ms | " + "  ".join(
            f"{p}: {phases[p]:.3f}" for p in cm.PHASES if phases[p] > 0))

    sgemm_tflops = cm.measure_sgemm_tflops()
    gemm_flops = cm.flops_gemm(B, H, N, d)
    gemm_tflops = {
        step: {
            "QK matmul": cm.tflops(gemm_flops, results[step]["phase_ms"]["QK matmul"]),
            "PV matmul": cm.tflops(gemm_flops, results[step]["phase_ms"]["PV matmul"]),
        }
        for step in results
    }
    softmax_share = {
        step: results[step]["phase_ms"]["Softmax"] / results[step]["total_ms"]
        for step in results
    }
    print(f"softmax share of runtime: step00 {softmax_share['00']:.1%} "
          f"-> step01 {softmax_share['01']:.1%}")

    # =====================================================================
    # Figure 1: stacked kernel breakdown, step 00 vs. step 01
    # =====================================================================
    fig, ax = cm.new_fig(7.2, 3.0)
    cm.stacked_hbars(ax, [
        ("00 · naive kernels", results["00"]["phase_ms"]),
        ("01 · cuBLAS GEMM", results["01"]["phase_ms"]),
    ])
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Step 01 · cuBLAS collapses the GEMMs — softmax is now the bottleneck")
    ax.legend(ncols=3, loc="lower right")
    cm.save_fig(fig, "01_kernel_breakdown_vs_step00.png",
                footnote=f"{cm.shape_tag()} · per-kernel device time (torch.profiler) · "
                         f"softmax share {softmax_share['00']:.0%} → {softmax_share['01']:.0%}")

    # =====================================================================
    # Figure 2: achieved GEMM TFLOPS, naive vs. cuBLAS
    # =====================================================================
    fig, ax = cm.new_fig(7.0, 4.2)
    groups = ["QK matmul", "PV matmul"]
    xs = range(len(groups))
    w = 0.32
    for off, step, label, color in [
        (-w / 2, "00", "Naive kernel (step 00)", cm.MUTED),
        (w / 2, "01", "cuBLAS SGEMM (step 01)", cm.C1_BLUE),
    ]:
        vals = [gemm_tflops[step][g] for g in groups]
        ax.bar([x + off for x in xs], vals, width=w, color=color,
               edgecolor=cm.SURFACE, linewidth=1.5, zorder=3, label=label)
        for x, val in zip(xs, vals):
            ax.annotate(f"{val:.1f}", (x + off, val), xytext=(0, 3),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=cm.INK2)
    cm.ref_line(ax, peaks["fp32_tflops"],
                f"FP32 peak {peaks['fp32_tflops']:.0f} TFLOPS")
    cm.ref_line(ax, sgemm_tflops,
                f"measured cuBLAS SGEMM (8192³) {sgemm_tflops:.0f} TFLOPS")
    ax.set_ylim(0, peaks["fp32_tflops"] * 1.12)
    ax.set_xticks(list(xs), groups)
    cm.style_axes(ax, grid_axis="y")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title("Step 01 · Achieved GEMM throughput")
    ax.legend(loc="upper left")
    cm.save_fig(fig, "01_gemm_tflops.png",
                footnote=f"{cm.shape_tag()} · 2·B·H·N²·d FLOPs per GEMM")

    cm.save_json("01", {
        "shape": cm.SHAPE,
        "steps": results,
        "gemm_tflops": gemm_tflops,
        "softmax_share": softmax_share,
        "measured_sgemm_tflops": sgemm_tflops,
        "peaks": peaks,
    })


if __name__ == "__main__":
    main()
