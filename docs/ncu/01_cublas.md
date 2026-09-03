# Nsight Compute Analysis — Step 01: cuBLAS GEMM

This document contains the detailed Nsight Compute analysis for [Step 01: cuBLAS GEMM](../01_cublas.md).

The goal is to measure how replacing the naive GEMMs changes the hardware bottlenecks.

## Profiling setup

Representative workload:

```text
GPU: NVIDIA GeForce RTX 5090
dtype: FP32

B = 8
H = 16
N = 4096
d = 64
```

Step 01 contains three operations:

```text
cuBLAS QKᵀ
naive_softmax_kernel
cuBLAS PV
```

The two matrix multiplications use cuBLAS.

Softmax remains unchanged from Step 00.

## Overview

| Kernel       | SM Throughput | L1/TEX |    L2 |  DRAM | Main signal                                      |
| ------------ | ------------: | -----: | ----: | ----: | ------------------------------------------------ |
| cuBLAS `QKᵀ` |         24.9% |  29.1% | 45.5% | 22.3% | naive L1/TEX saturation removed                  |
| Softmax      |          0.9% |  16.4% | 44.7% | 13.6% | latency- and serialization-limited execution     |
| cuBLAS `PV`  |         25.2% |  29.0% | 45.5% | 22.3% | similar optimized GEMM behavior                  |

---

## QKᵀ

Step 00 showed:
```text
L1/TEX Throughput  99.6%
DRAM Throughput     2.6%
```

The naive kernel saturated the L1/TEX path because of inefficient memory access.

With cuBLAS:
```text
SM Throughput       24.9%
L1/TEX Throughput   29.1%
L2 Throughput       45.5%
DRAM Throughput     22.3%
```

L1/TEX pressure drops substantially as cuBLAS uses tiled computation and better data reuse.

Nsight Compute also shows improved scheduling behavior compared with the naive `QKᵀ` kernel:

| Metric                     |  Step 00 `QKᵀ` |  Step 01 cuBLAS `QKᵀ` |
| -------------------------- | -------------: | --------------------: |
| Eligible warps / scheduler |           0.35 |                  0.56 |
| Instruction issue interval |    13.6 cycles |             ~4 cycles |
| Theoretical occupancy      |           100% |                  ~58% |

Although theoretical occupancy is lower, more eligible warps and a shorter issue interval indicate more effective execution.

## PV

PV shows similar behavior to QKᵀ:
```text
SM Throughput       25.2%
L1/TEX Throughput   29.0%
L2 Throughput       45.5%
DRAM Throughput     22.3%
```

## Conclusion

cuBLAS removes the main inefficiencies of the naive GEMMs.

Softmax remains unchanged and becomes the next clear kernel-level bottleneck.

Beyond that, the $O(N^2)$ intermediate-memory cost remains and is addressed in later fused-attention steps.
