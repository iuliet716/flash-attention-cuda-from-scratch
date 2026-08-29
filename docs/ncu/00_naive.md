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

### Throughput

`QKᵀ` reaches:

```text
SM Throughput       22.5%
L1/TEX Throughput   99.6%
L2 Throughput       10.4%
DRAM Throughput      2.6%
```

The striking result is the combination of almost fully saturated L1/TEX throughput with very low DRAM throughput.

This is **not a DRAM-bandwidth bottleneck**.

Instead, the global-memory access path reaches its effective limit much earlier in the hierarchy.

### Inefficient global-memory accesses

Each thread computes one score:

$$
S_{ij} =
\sum_k Q_{ik}K_{jk}.
$$

The naive mapping provides no explicit shared-memory reuse.

Different threads repeatedly consume the same `Q` values, while the `K` access pattern produces inefficient memory transactions.

Nsight Compute explicitly reports:

> Only 4 of 32 bytes per sector are utilized.

It also reports approximately **66% excessive sectors** for the affected accesses.

This means that substantially more memory sectors are transferred through the load path than the useful payload actually requires.

### Why L1/TEX reaches 99.6%

The high L1/TEX number should therefore not be interpreted as evidence that the kernel is efficiently reusing cache.

Instead, the memory path is heavily exercised by inefficient and redundant accesses.

The kernel reaches:

```text
L1/TEX  ~99.6%
DRAM     ~2.6%
SM      ~22.5%
```

so the bottleneck occurs well before raw DRAM bandwidth becomes saturated.

Nsight Compute also identifies LG-throttle related stalls, consistent with pressure on the global-load path.

### Bottleneck classification

It is reasonable to classify `QKᵀ` as **memory-system limited**, more specifically limited around the global-load/L1-TEX path.

It should not be described simply as **DRAM-bandwidth bound**.

The optimization target is therefore:

```text
better access pattern
        +
explicit on-chip reuse
        ↓
fewer redundant global loads
```

rather than merely attempting to increase DRAM bandwidth.

---

## Softmax

### Throughput

Softmax shows:

```text
SM Throughput        1.0%
L1/TEX Throughput   18.4%
L2 Throughput       37.2%
DRAM Throughput     12.8%
```

Neither the compute pipelines nor DRAM bandwidth are close to saturation.

The problem comes primarily from the execution strategy.

### One thread per row

Each thread processes one entire row sequentially:

```text
max pass
   ↓
sum(exp) pass
   ↓
normalization pass
```

For `N=4096`, this creates a long serial dependency chain inside each thread.

There is no warp-level cooperation for the row reduction.

### Scheduler behavior

Nsight Compute reports approximately:

```text
Eligible warps / scheduler ≈ 0.02
Cycles between issued instructions ≈ 234
```

Very few warps are ready to issue instructions at any given time.

This indicates that the kernel is dominated by latency and serialization rather than by peak arithmetic throughput or DRAM bandwidth.

### Bottleneck classification

The primary problem is therefore **insufficient parallelism and instruction-issue efficiency**.

The direct optimization is to distribute one row across an entire warp:

```text
one thread
processes N elements

        ↓

one warp
cooperatively processes N elements
```

with warp-level reductions for max and sum.

This directly motivates Step 02's warp-level softmax.

---

## PV

### Throughput

`PV` reaches approximately:

```text
SM Throughput       90.0%
L1/TEX Throughput   90.4%
L2 Throughput       15.2%
DRAM Throughput      9.1%
```

Unlike the previous two kernels, both SM and L1/TEX utilization are high.

The `(32, 8)` thread-block layout also gives `V` a more favorable access pattern across the warp.

### Remaining stalls

Despite the high throughput, Nsight Compute reports substantial long-scoreboard stalls.

Approximately:

```text
Long scoreboard ≈ 58%
```

of the cycles between issued instructions are associated with these memory dependencies.

A long scoreboard stall generally means that a warp is waiting for a memory dependency whose data has not yet become available.

The kernel still computes each output through a long scalar dot-product loop:

```cuda
for (int k = 0; k < N; ++k) {
    acc += Pb[row * N + k] * Vb[k * d + col];
}
```

without shared-memory tiling or explicit reuse.

### Bottleneck classification

`PV` is significantly better utilized than the naive `QKᵀ` and Softmax kernels, but it still leaves optimization opportunities in:

* data reuse
* memory-latency hiding
* tiled matrix multiplication

This motivates replacing the scalar implementation with an optimized GEMM path.

---

## Is the baseline memory-bound?

The answer depends on the kernel.

| Kernel  | Characterization                                                |
| ------- | --------------------------------------------------------------- |
| `QKᵀ`   | memory-system limited, particularly the global-load/L1-TEX path |
| Softmax | latency/serialization limited                                   |
| `PV`    | high utilization with significant memory-dependency stalls      |

Therefore, describing the entire baseline as simply **DRAM-bandwidth bound** would be inaccurate.

A more precise conclusion is:

> The naive attention implementation is strongly limited by memory behavior and data movement, but the exact bottleneck differs across kernels.

In particular, the `QKᵀ` result demonstrates that poor memory behavior can become a bottleneck even when DRAM bandwidth itself is far from saturated.

---

## Overall bottleneck

The baseline exposes three independent inefficiencies:

### QKᵀ

```text
redundant / inefficient global loads
        ↓
L1/TEX saturation
        ↓
low compute utilization
```

### Softmax

```text
one thread per row
        ↓
long serial dependency chain
        ↓
very few eligible warps
```

### PV

```text
scalar dot-product loop
        ↓
memory dependencies
        ↓
long-scoreboard stalls
```

These results show why optimizing attention requires more than simply increasing arithmetic throughput.

The larger problem is how data moves through the GPU memory hierarchy and how much intermediate state is materialized between operations.
