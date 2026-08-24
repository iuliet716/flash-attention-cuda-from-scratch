# Overview
This project implements FlashAttention from scratch in CUDA.  
The implementation starts from naive attention and is progressively optimized step by step.

Each optimization step is preserved separately to show how individual design affects performance.

[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) provides the official implementation.


# Key Results
- **3.273 ms** latency and **168.0 TFLOPS**
- **76.8% of PyTorch SDPA FlashAttention performance**


# Benchmark
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)

NVIDIA RTX 5090 32GB  

Median of 50 iterations after 10 warm-up runs; fast math enabled; TF32 disabled; L2 flush disabled.

(B, H, N, d) = (8, 16, 4096, 64)

## Track A: Unfused kernel

| Step | Technique | dtype | Latency | vs. prev. | TFLOPS |
|---|---|---|---|---|---|
| 00 | Naive Standard Attention | FP32 | 253.136 ms | - | 2.2 |
| 01 | cuBLAS GEMM | FP32 | 64.376 ms | 3.93x | 8.5 |
| 02 | Warp-reduction Softmax | FP32 | 28.690 ms | 2.24x | 19.2 |
| 03 | Online Softmax | FP32 | 30.071 ms | 0.95x | 18.3 |

## Track B: Fused FlashAttention Kernel

| Step | Technique | dtype | Latency | vs. prev. | TFLOPS | vs. SDPA |
|---|---|---|---|---|---|---|
| 04 | Naive Fused Attention (SRAM Tiling) | FP32 | 326.611 ms | - | 1.7 | 0.7% |
| 05 | Coalescing + Vectorized Load | FP32 | 120.939 ms | 2.70x | 4.5 | 2.0% |
| 06 | Bank Conflict Avoidance (Swizzling) | FP32 | 64.736 ms | 1.87x | 8.5 | 3.9% |
| 07 | Half-Precision (FP16) | FP16 | 57.156 ms | 1.13x | 9.6 | 4.4% |
| 08 | WMMA Tensor Cores | FP16 | 37.782 ms | 1.51x | 14.6 | 6.5% |
| 09 | Double Buffering | FP16 | 23.542 ms | 1.60x | 23.4 | 10.5% |
| 10 | Register-Resident Accumulators | FP16 | **3.273 ms** | 7.19x | **168.0** | **76.8%** |

> All reported kernels pass correctness checks against PyTorch SDPA.

> % SDPA is measured against PyTorch SDPA FlashAttention immediately after each kernel run.


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

### Double Buffering
Overlap tile loading with computation to reduce memory stalls.

### Register-Resident Accumulation
Keep output accumulators in registers to minimize shared-memory traffic.


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


