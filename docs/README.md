# Implementation notes & benchmarks

Step-by-step notes for the techniques listed in the [root README](../README.md).
Each document explains the motivation, the key idea, the implementation, and the
metrics measured on the target GPU.

| Step | Document | One-line takeaway |
|---|---|---|
| 00 | [Naive Standard Attention](00_naive.md) | Every kernel scales as N² and uses neither the FLOPs nor the bandwidth |
| 01 | [cuBLAS GEMM](01_cublas.md) | The GEMMs collapse (1.8 → 32 TFLOPS) and softmax becomes the bottleneck |
| 02 | [Warp-reduction Softmax](02_warp_softmax.md) | One warp per row + coalescing pushes softmax to 75 % of the memory roof |
| 03 | [Online Softmax](03_online_softmax.md) | Same speed as step 02, but a single pass — the key that unlocks fusion |
| 04 | [Naive Fused Attention (SRAM tiling)](04_naive_fused.md) | 81× less HBM traffic and no OOM wall, yet 11× slower — kernel quality is now the problem |

## Measurement environment

All numbers in these documents were measured with the scripts in
[`benchmarks/`](../benchmarks/) (raw values in `benchmarks/results/*.json`):

- NVIDIA GeForce RTX 5090 32 GB (sm_120, 170 SMs), WSL2
- PyTorch 2.7.0 + CUDA 12.8, FP32
- Default shape `B=8, H=16, N=4096, d=64`; median latency after warm-up
- Peaks used for utilization: 104.8 TFLOPS FP32, 1792 GB/s HBM (spec sheet);
  measured ceilings: 1513 GB/s device-to-device copy, ~67 TFLOPS cuBLAS SGEMM (8192³)
- Per-kernel times come from `torch.profiler` CUDA events

To regenerate every figure (`docs/assets/`) and JSON:

```bash
source env.sh
benchmarks/run_all.sh        # or python benchmarks/bench_step0X.py individually
```

> Note (WSL2): allocations larger than VRAM do not fail with OOM — unified
> memory spills to host RAM over PCIe and the kernel slows down by ~1000×.
> The step-04 charts mark where this happens.
