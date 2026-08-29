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
| Softmax      |          0.9% |  16.4% | 44.7% | 13.6% | serialized execution and very few eligible warps |
| cuBLAS `PV`  |         25.2% |  29.0% | 45.5% | 22.3% | similar optimized GEMM behavior                  |

The largest change from Step 00 appears in the matrix-multiplication kernels.

---

## QKᵀ

### Throughput

The cuBLAS `QKᵀ` kernel reaches approximately:

```text
SM Throughput       24.9%
L1/TEX Throughput   29.1%
L2 Throughput       45.5%
DRAM Throughput     22.3%
```

This is very different from the naive `QKᵀ` kernel.

Step 00 reached:

```text
L1/TEX Throughput   99.6%
DRAM Throughput      2.6%
```

The naive kernel saturated the global-load/L1-TEX path.

With cuBLAS, that behavior disappears.

### Improved data reuse

Lower L1/TEX utilization is not itself the optimization goal.

The important difference is that cuBLAS no longer generates the pathological access pattern seen in Step 00.

The GEMM implementation tiles the computation and reuses data more effectively.

This reduces redundant memory-path pressure.

### Remaining limits

The kernel still does not saturate peak compute throughput.

Nsight Compute reports approximately:

```text
Eligible warps / scheduler ≈ 0.56
Instruction issue interval ≈ 4 cycles
Theoretical occupancy      ≈ 58%
```

The occupancy limit is partly associated with register usage.

Memory latency also remains visible in the stall behavior.

The kernel is therefore much more efficient than Step 00, but it is not purely compute-bound.

---

## Softmax

The softmax implementation is unchanged from Step 00.

With the GEMMs optimized, softmax becomes the next clear optimization target.

---

## PV

The cuBLAS `PV` kernel shows behavior similar to `QKᵀ`:

```text
SM Throughput       25.2%
L1/TEX Throughput   29.0%
L2 Throughput       45.5%
DRAM Throughput     22.3%
```

The scalar dot-product kernel from Step 00 has been replaced by a tiled GEMM implementation.

This removes the need to manually optimize the original scalar `PV` kernel.

The remaining limitations are mainly related to latency, occupancy, and the shape of this FP32 workload.

---

## Is Step 01 memory-bound?

There is still no single bottleneck for the entire attention pipeline.

| Operation    | Characterization                                          |
| ------------ | --------------------------------------------------------- |
| cuBLAS `QKᵀ` | optimized GEMM with remaining latency and resource limits |
| Softmax      | latency and serialization limited                         |
| cuBLAS `PV`  | optimized GEMM with similar latency and resource limits   |

The severe memory-access problem of the naive GEMMs has been removed.

This does not eliminate the larger memory cost of standard attention.

---

## O(N²) intermediate traffic remains

cuBLAS changes how efficiently each GEMM is executed.

It does not change the attention dataflow.

The pipeline still performs:

```text
QKᵀ
 ↓
write N × N scores
 ↓
Softmax
 ↓
read / update N × N matrix
 ↓
PV
 ↓
read N × N probabilities
```

The full attention matrix is still materialized in device memory.

Its size grows as:

$$
O(N^2).
$$

Nsight Compute can show the memory-system behavior of each kernel.

The algorithmic $O(N^2)$ traffic comes from the dataflow itself.

This remains even after the individual GEMMs are optimized.

---

## Overall

Step 01 removes the main weaknesses of the naive matrix-multiplication kernels:

```text
naive scalar GEMM
        ↓
cuBLAS tiled GEMM
        ↓
better reuse and memory behavior
```

The profiler now exposes the next bottleneck more clearly:

```text
one thread per softmax row
        ↓
serialized execution
        ↓
very low instruction-issue efficiency
```

At the same time, the full $(N \times N)$ attention matrix is still stored between operations.

Step 02 address the softmax kernel.  
Later fused steps address the larger IO problem by avoiding repeated materialization of the attention matrix.
