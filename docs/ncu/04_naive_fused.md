# Nsight Compute Analysis — Step 04: Naive Fused Attention

This document contains the detailed Nsight Compute analysis for [Step 04: Naive Fused Attention](../04_naive_fused.md).

The goal is to identify the bottlenecks of the first tiled fused-attention kernel.

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
| L1/TEX Throughput          | 95.1% |
| L2 Throughput              |  5.2% |
| DRAM Throughput            | 0.18% |
| Achieved Occupancy         | 83.0% |
| Eligible warps / scheduler |  0.54 |
| Shared-load bank conflicts | 10.3x |

DRAM throughput is very low after fusion, while L1/TEX throughput is nearly saturated.

The bottleneck has therefore shifted from intermediate HBM traffic to inefficient on-chip data movement.

## Shared-memory bank conflicts

Nsight Compute reports approximately:

Shared load requests       12.9 B
Shared load wavefronts    146.1 B
Bank conflicts            133.2 B

or roughly:

10.3 shared-memory bank conficts / load request

For $QK^\top$, lanes access:

`Ktile[lane * d + k]`

With `d = 64`, adjacent lanes access rows separated by 64 floats, causing many accesses to hit the same shared-memory banks.

Serialization increases shared-memory traffic, driving high L1/TEX utilization despite low DRAM throughput.

## Scheduler efficiency

Occupancy remains relatively high:

Theoretical occupancy               83.3%
Achieved occupancy                  83.0%
Eligible warps / scheduler           0.54
Instruction issue interval    ~7.2 cycles

The kernel therefore has enough resident warps, **but few are ready to issue instructions.**

MIO-throttle stalls dominate, indicating heavy shared-memory instruction pressure.

This is consistent with the heavy shared-memory instruction pressure caused by the conflicted accesses.

## Effect of fusion

Compared with the optimized unfused baseline:

| Step | HBM read | HBM write | Total |
| ---- | -------: | --------: | ----: |
| 02   | ~13.6 GB |  ~34.7 GB | ~48.3 GB |
| 04   | ~1.38 GB |  ~0.13 GB | ~1.50 GB |

the $N \times N$ intermediate matrices no longer pass through HBM.

However, fusion alone is not enough for high performance.  

The kernel still uses scalar matrix computation and an inefficient shared-memory layout, unlike the optimized GEMMs used in the preceding steps.

## Conclusion

Step 04 removes the full attention-matrix materialization and shifts the bottleneck on-chip.

The main remaining issues are:

* memory-instruction pressure from scalar accesses
* severe shared-memory bank conflicts

Step 05 reduces memory-instruction pressure with vectorized accesses,  
and Step 06 addresses the bank conflicts with a swizzled shared-memory layout.
