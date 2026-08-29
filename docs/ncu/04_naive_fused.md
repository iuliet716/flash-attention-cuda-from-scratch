# Nsight Compute Analysis — Step 04: Naive Fused Attention

This document contains the detailed Nsight Compute analysis for [Step 04: Naive Fused Attention](../04_naive_fused.md).

The goal is to measure how fusing $QK^\top$, softmax, and $PV$ changes the memory behavior and to identify the bottlenecks of the first tiled implementation.

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

Unlike the previous steps, Step 04 executes the complete attention operation inside a single kernel:

```text
fused_attention_kernel
```

The full $N \times N$ score and probability matrices are no longer materialized in global memory.

## Overview

| Metric                     | Value |
| -------------------------- | ----: |
| SM Throughput              | 27.2% |
| Memory Throughput          | 95.0% |
| L1/TEX Throughput          | 95.1% |
| L2 Throughput              |  5.2% |
| DRAM Throughput            | 0.18% |
| Theoretical Occupancy      | 83.3% |
| Achieved Occupancy         | 83.0% |
| Eligible warps / scheduler |  0.54 |

The kernel is dominated by the on-chip memory path rather than DRAM bandwidth.

L1/TEX throughput reaches about 95%, while DRAM throughput remains below 1%,  
indicating that the primary pressure comes from **on-chip memory accesses**.

---

## Shared-memory bank conflicts

The strongest signal in the profile is the shared-memory access pattern.

Nsight Compute reports an average:

```text id="kjmwvv"
11.3-way bank conflict
```

across shared-memory loads.

The underlying metrics show how much additional work this creates:

```text id="yd4igw"
Shared load requests       12,884,901,888
Shared load wavefronts    146,092,967,568
Bank conflicts            133,208,065,680
```

The difference between shared-load wavefronts and requests is exactly the reported number of bank conflicts:

```text id="8fp12p"
146,092,967,568
- 12,884,901,888
= 133,208,065,680
```

Ideally, a conflict-free shared-memory request can be serviced with a single wavefront.

Here, bank conflicts force each logical load request to be split into multiple wavefronts, serializing the shared-memory accesses.

As a result, the kernel generates far more on-chip memory transactions than necessary.

This explains why L1/TEX throughput reaches approximately 95% while DRAM throughput remains below 1%.

---

## Why do the bank conflicts occur?

Each warp computes one query row.

For $QK^\top$, lane `lane` repeatedly reads:

```cuda
Qtile[warp * d + k]
Ktile[lane * d + k]
```

for the same `k`.

The straightforward row-major shared-memory layout does not consider how these addresses map onto shared-memory banks.

As the warp progresses through the dot product, many lane accesses map onto overlapping bank patterns.

The result is serialization inside shared memory rather than fully parallel bank accesses.

The fused algorithm has therefore reduced HBM traffic while introducing a severe on-chip access inefficiency.

---

## Scheduler efficiency

Occupancy itself is high:

```text
Theoretical occupancy       83.3%
Achieved occupancy          83.0%
Active warps / SM           39.83
Active warps / scheduler     9.96
```

The achieved occupancy is almost identical to the theoretical value.

This means insufficient warp residency is not the main problem.

However, Nsight Compute reports only:

```text
Eligible warps / scheduler    0.54
```

and an instruction is issued only about once every:

```text
7.2 cycles
```

on each scheduler.

Approximately 86% of scheduler cycles have no eligible warp available to issue.

The kernel therefore has many resident warps, but most of them are waiting rather than ready to execute.

---

## MIO pipeline pressure

The dominant warp stall is MIO throttle.

Nsight Compute reports approximately:

```text
50.3 cycles
```

of MIO-throttle stall per issued instruction.

This accounts for about:

```text
70.6%
```

of the average:

```text
71.3 warp cycles / issued instruction
```

MIO throttle occurs when the memory-input/output instruction queue cannot accept additional operations.

This pipeline serves operations including shared-memory accesses and special-function instructions.

For Step 04, the large amount of conflicted shared-memory traffic is the dominant signal.

The scalar access pattern also requires many individual memory instructions.

Nsight Compute therefore recommends reducing the number of memory operations with fewer, wider accesses.

This directly motivates the vectorized loads introduced in Step 05.

---

## Occupancy limits

The launch configuration uses:

```text
256 threads / block
8 warps / block

Registers / thread          40
Dynamic shared memory       18,432 bytes / block
```

Nsight Compute reports:

```text
Block limit — registers      6
Block limit — warps          6
Block limit — shared memory  5
```

Shared-memory allocation is therefore the tightest residency constraint.

Five resident blocks correspond to:

```text
5 blocks
× 8 warps / block
= 40 warps / SM
```

which matches the measured theoretical occupancy of approximately 83.3%.

However, achieved occupancy is already approximately 83.0%, so increasing occupancy alone would not solve the main problem.

The more important issue is making the resident warps spend less time stalled on shared-memory operations.

---

## What happened to the HBM bottleneck?

Step 04 fundamentally changes the attention dataflow.

Previous implementations materialized the complete attention matrix:

```text
QKᵀ
 ↓
N × N score matrix in HBM
 ↓
softmax
 ↓
N × N probability matrix in HBM
 ↓
PV
```

Step 04 instead performs:

```text
Q/K/V tiles
    ↓
shared memory
    ↓
QKᵀ tile
    ↓
online softmax
    ↓
PV accumulation
    ↓
next K/V tile
```

The $N \times N$ intermediate matrix no longer travels through HBM.

The profile confirms this change:

```text
DRAM Throughput    0.18%
```

while L1/TEX throughput reaches approximately 95%.

This is an important transition.

The original large external-memory traffic has been removed,  
but the first fused implementation does not yet use the on-chip memory system efficiently.

---

## Overall

Step 04 successfully introduces the core IO-aware attention structure:

```text
materialized N × N matrix
            ↓ removed

tiled K/V
+ online softmax
+ fused PV accumulation
```

Nsight Compute confirms that DRAM bandwidth is no longer the limiting resource.

Instead, the naive fused kernel exposes a new bottleneck:

```text
row-major shared-memory layout
        ↓
severe bank conflicts
        ↓
excessive shared-memory wavefronts
        ↓
L1/TEX saturation
        ↓
MIO-throttle stalls
        ↓
low instruction-issue efficiency
```

This also explains why fusion alone is not sufficient for high performance.

Step 05 reduces memory-instruction pressure with `float4` vectorized accesses.

Step 06 then directly addresses the shared-memory bank conflicts by introducing a swizzled shared-memory layout.
