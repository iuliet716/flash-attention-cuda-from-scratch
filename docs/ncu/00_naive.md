# Nsight Compute Analysis — Step 00: Naive Attention

This document contains the detailed Nsight Compute analysis for [Step 00: Naive Standard Attention](../00_naive.md).

The goal is to identify the hardware bottlenecks of the baseline implementation before applying any optimization.

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

The baseline consists of three CUDA kernels:

```text
naive_qk_kernel
naive_softmax_kernel
naive_pv_kernel
```

A full Nsight Compute metric set was collected for each kernel.

## Overview

| Kernel  | SM Throughput | L1/TEX |    L2 |  DRAM | Main signal                                                 |
| ------- | ------------: | -----: | ----: | ----: | ----------------------------------------------------------- |
| `QKᵀ`   |         22.5% |  99.6% | 10.4% |  2.6% | L1/TEX saturation and inefficient global-memory accesses    |
| Softmax |          1.0% |  18.4% | 37.2% | 12.8% | highly serialized execution and very low eligible-warp rate |
| `PV`    |         90.0% |  90.4% | 15.2% |  9.1% | high utilization with significant memory-dependency stalls  |

The three kernels are limited by different parts of the execution pipeline.

There is therefore no single bottleneck that explains the entire baseline.

---

## QKᵀ

`QKᵀ` reaches:

```text
SM Throughput       22.5%
L1/TEX Throughput   99.6%
L2 Throughput       10.4%
DRAM Throughput      2.6%
```

L1/TEX throughput is nearly saturated while DRAM throughput remains low,  
indicating that raw DRAM bandwidth is not the bottleneck.

Nsight Compute also reports:
> Only 4 of 32 bytes per sector are utilized.

This indicates inefficient memory transactions.

Therefore, QKᵀ is likely limited by inefficient memory accesses and L1/TEX pressure, rather than DRAM bandwidth.

## Softmax 

Softmax reaches:

```text
SM Throughput        1.0%
L1/TEX Throughput   18.4%
L2 Throughput       37.2%
DRAM Throughput     12.8%
```

Nsight Compute also reports approximately:
> Eligible warps / scheduler ≈ 0.02  
> Cycles between issued instructions ≈ 234

Each thread processes an entire row sequentially through `max`, `sum-exp`, and `normalization` passes.  

Together, these results indicate that the kernel is primarily limited by latency and serialized execution.

## PV

PV reaches:

```text
SM Throughput       90.0%
L1/TEX Throughput   90.4%
L2 Throughput       15.2%
DRAM Throughput      9.1%
```text

Unlike QKᵀ and Softmax, both SM and L1/TEX utilization are high.

However, Nsight Compute reports approximately:
> Long scoreboard ≈ 58%

indicating significant memory-dependency stalls.

PV is better utilized than the other baseline kernels, but it can still be improved with tiled GEMM and better latency hiding.

## Conclusion

The baseline is not simply DRAM-bandwidth bound.

Its performance is limited by inefficient memory access, low parallelism, long dependency chains, and poor data reuse.

These observations motivate the following steps: optimized GEMM, warp-level softmax, and eventually tiled fused attention.
