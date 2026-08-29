# Nsight Compute Analysis — Step 02: Warp-reduction Softmax

This document contains the detailed Nsight Compute analysis for [Step 02: Warp-reduction Softmax](../02_warp_softmax.md).

The goal is to measure how replacing the single-thread-per-row softmax with a warp-level implementation changes the hardware bottlenecks.

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

Step 02 contains three operations:

```text
cuBLAS QKᵀ
warp_softmax_kernel
cuBLAS PV
```

The two matrix multiplications remain unchanged from Step 01.

Only the softmax kernel is replaced.

## Overview

| Kernel       | SM Throughput | L1/TEX |    L2 |  DRAM | Main signal                                             |
| ------------ | ------------: | -----: | ----: | ----: | ------------------------------------------------------- |
| cuBLAS `QKᵀ` |         24.1% |  29.0% | 45.2% | 22.5% | similar optimized GEMM behavior to Step 01              |
| Warp softmax |          4.2% |   8.9% | 20.5% | 35.1% | high occupancy but global-memory instruction throttling |
| cuBLAS `PV`  |         25.2% |  29.0% | 45.7% | 22.4% | similar optimized GEMM behavior                         |

The major change from Step 01 appears in the softmax kernel.

---

## Warp softmax

### Throughput

The warp-level softmax reaches approximately:

```text
SM Throughput        4.2%
L1/TEX Throughput    8.9%
L2 Throughput       20.5%
DRAM Throughput     35.1%
```

Step 01's single-thread-per-row softmax reached approximately:

```text
SM Throughput        0.9%
L1/TEX Throughput   16.4%
L2 Throughput       44.7%
DRAM Throughput     13.6%
```

SM throughput increases from about 0.9% to 4.2%.

The kernel now exposes substantially more parallel execution, although compute throughput remains low.

### Coalesced memory access

The previous implementation assigned one thread to an entire row.

Step 02 instead assigns one warp to each row:

```text
lane 0  → row[0],  row[32], row[64], ...
lane 1  → row[1],  row[33], row[65], ...
...
lane 31 → row[31], row[63], row[95], ...
```

Adjacent lanes therefore access adjacent elements.

Nsight Compute reports zero excessive global sectors for the warp-softmax kernel.

This confirms that the lane mapping removes the inefficient global-memory access pattern.

The remaining problem is therefore not poor coalescing.

---

## Occupancy

The kernel reaches approximately:

```text
Theoretical occupancy   100%
Achieved occupancy       96.1%
Active warps/scheduler   11.64
```

Occupancy is no longer the limiting factor.

There are enough resident warps available to hide latency in principle.

However, residency alone does not mean those warps are ready to issue instructions.

---

## Instruction-issue efficiency

Nsight Compute reports only approximately:

```text
Eligible warps / scheduler   0.06
Instruction issue interval   ~75 cycles
```

Despite having more than 11 active warps per scheduler, almost none are eligible to issue an instruction in a typical cycle.

This explains why SM throughput remains low even though occupancy is high.

The limiting factor has moved from insufficient parallelism to the execution and memory-access behavior of the algorithm.

---

## LG throttle

The dominant scheduler stall is LG throttle.

Nsight Compute reports that each warp spends roughly:

```text
~630 cycles per issued instruction
```

waiting on the local/global-memory instruction path.

This accounts for approximately 72% of the cycles between issued instructions.

LG throttle indicates that warps frequently wait for the instruction queue serving local/global-memory operations.

The kernel therefore generates memory instructions faster than this path can accept them.

This does not imply that peak DRAM bandwidth itself is saturated.

Measured DRAM throughput is only about 35.1% of peak.

The limitation is better described as memory-path or memory-instruction pressure rather than pure DRAM-bandwidth saturation.

---

## Why does softmax still generate so much traffic?

Warp-level reduction improves how each row is processed.

It does not change the conventional softmax dataflow.

For each score row, the kernel still performs three phases:

```text
1. read scores to compute max

2. read scores
   compute exp
   write exp
   compute sum

3. read exp
   normalize
   write probabilities
```

Therefore, the full $(N \times N)$ score matrix is accessed repeatedly.

For the profiled configuration:

$$
BHN^2 =
8 \times 16 \times 4096^2
$$

FP32 score elements require:

$$
8 \text{ GiB}
$$

for one complete attention matrix.

The three softmax phases therefore correspond to approximately:

```text
max reduction       :  8 GiB read

exp + sum           :  8 GiB read
                     + 8 GiB write

normalization       :  8 GiB read
                     + 8 GiB write
-----------------------------------
logical traffic       40 GiB
```

This is logical score-matrix traffic inside softmax alone.

Actual DRAM traffic can differ because of caching and other memory-system effects.

---

## Overall

Step 02 removes two major weaknesses of the previous softmax implementation:

```text
one thread per row
        ↓
one warp per row
        ↓
parallel reduction
+ coalesced access
```

The profile confirms that occupancy and memory coalescing are no longer the primary problems.

The remaining limitation is more fundamental:

```text
materialized N × N matrix
        ↓
multiple full-matrix passes
        ↓
frequent global load/store instructions
        ↓
memory-path stalls
```

Step 03 introduces online softmax.

Later fused steps address the larger IO problem by avoiding repeated materialization and transfer of the attention matrix.
