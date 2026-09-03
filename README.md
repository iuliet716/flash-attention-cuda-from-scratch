# Overview
This project implements FlashAttention from scratch in CUDA.  
The implementation starts from naive attention and is progressively optimized step by step.

Each optimization step is preserved separately to show how individual design affects performance.

[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) provides the official implementation.


# Key Results
- **3.035 ms** latency and **181.1 TFLOPS**
- **82.3% of paired PyTorch SDPA FlashAttention performance**


# Benchmark
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)

NVIDIA RTX 5090 32GB  

Median of 50 iterations after 10 warm-up runs; fast math enabled; TF32 disabled; L2 flush disabled.

(B, H, N, d) = (8, 16, 4096, 64)

TFLOPS reports effective algorithmic attention throughput:

$$
\mathrm{TFLOPS}_{\mathrm{effective}} =
\frac{4 B H N^2 d}{\mathrm{latency}}.
$$

Counts QKᵀ and PV FLOPs; not hardware-counter FLOPs.

Correctness is checked against PyTorch SDPA Math on uniform and peaked inputs:

* FP32: `rtol = 1e-4`, `atol = 1e-5`
* FP16: `rtol = 1e-2`, `atol = 1e-2`

FP16 kernels additionally require a max-error bound within 2× the PyTorch FP16 reference error.


## Track A: Unfused kernel

| Step | Technique | dtype | Latency | Speedup | TFLOPS | %&nbsp;SDPA |
|---|---|---|---|---|---|---|
| 00 | Naive Standard Attention | FP32 | 261.236 ms | - | 2.1 | 0.9% |
| 01 | cuBLAS GEMM | FP32 | 67.794 ms | 3.85x | 8.1 | 3.6% |
| 02 | Warp-reduction Softmax | FP32 | 30.709 ms | 2.21x | 17.9 | 8.0% |
| 03 | Online Softmax | FP32 | 31.754 ms | 0.97x | 17.3 | 7.7% |

## Track B: Fused FlashAttention Kernel

| Step | Technique | dtype | Latency | Speedup | TFLOPS | %&nbsp;SDPA |
|---|---|---|---|---|---|---|
| 04 | Naive Fused Attention<br>(SRAM Tiling) | FP32 | 333.742&nbsp;ms | - | 1.6 | 0.7% |
| 05 | Coalescing + Vectorized Load | FP32 | 121.008&nbsp;ms | 2.76x | 4.5 | 2.0% |
| 06 | Bank Conflict Avoidance (Swizzling) | FP32 | 64.687 ms | 1.87x | 8.5 | 3.9% |
| 07 | Half-Precision (FP16) | FP16 | 59.268 ms | 1.09x | 9.3 | 4.2% |
| 08 | WMMA Tensor Cores | FP16 | 39.471 ms | 1.50x | 13.9 | 6.2% |
| 09 | Split-Q Warp Partitioning | FP16 | 32.298 ms | 1.22x | 17.0 | 7.6% |
| 10 | [Warp-Owned Register Dataflow](./docs/10_register_dataflow.md) | FP16 | **3.035 ms** | 10.64x | **181.1** | **82.3%** |

The large Step 09 → Step 10 transition is analyzed with profiler counters in
[Nsight Compute Analysis — Step 10](./docs/ncu/10_register_dataflow.md).

## Reference

| Reference | dtype | Latency | Effective TFLOPS |
|---|---|---:|---:|
| PyTorch matmul + softmax | FP32 | 39.590 ms | 13.9 |
| PyTorch SDPA FlashAttention (initial reference) | FP16 | 2.421 ms | 227.0 |

> `% SDPA` uses a paired PyTorch SDPA FlashAttention measurement taken immediately after each custom-kernel measurement.  
> The Reference table reports a separate initial SDPA measurement, so the `2.421 ms` value is not the denominator for every `% SDPA` entry.


# Kernel Design Highlights

See [docs](./docs) for detailed design notes.

### SRAM Tiling
Process Q/K/V in tiles to avoid materializing the full attention matrix in HBM.

### Online Softmax
Compute numerically stable softmax incrementally across K/V tiles.

### Coalesced & Vectorized Memory Access
Improve global-memory efficiency with aligned and vectorized loads.

### Shared-Memory Swizzling
Reduce shared-memory bank conflicts during tiled matrix operations.

### Tensor Core Acceleration
Use FP16 WMMA operations for QKᵀ and PV matrix multiplication.

### Split-Q Warp Partitioning
Give each warp its own 16-row Q/O slice while the block shares the K/V tile.

### Warp-local Register Dataflow
Keep Q, S/P, O, and softmax state in the owner warp's registers instead of shared memory.


# Limitations & Roadmap

## Current Limitations
- Forward pass only
- FP16 fused kernel
- Head dimension limited to 64 and 128
- Non-causal self-attention only
- No support for variable-length sequences

## Roadmap
- [ ] Backward pass
- [ ] Causal attention
- [ ] BF16 support
- [ ] Support for additional head dimensions
- [ ] Further kernel optimization


# Get Started
```bash
$ source env.sh         # Run this command if you need the environment for RTX 5090
$ python benchmark.py
  # python benchmark.py --preset llm
  # python benchmark.py -B 8 -H 16 -N 4096 -d 64
```

Nsight Compute profiling methodology and reproduction commands are documented in
[docs/ncu/README.md](./docs/ncu/README.md).


