# Nsight Compute Analysis — Step 02: Warp-reduction Softmax

This document contains the detailed Nsight Compute analysis for [Step 02: Warp-reduction Softmax](../02_warp_softmax.md).

The goal is to measure how warp-level softmax changes the hardware bottlenecks.

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

The matrix multiplications remain unchanged from Step 01.

## Overview

Softmax kernel comparison:

| Metric                     | Step 01 | Step 02 |
| -------------------------- | ------: | ------: |
| SM Throughput              |    0.9% |    4.2% |
| L1/TEX Throughput          |   16.4% |    8.9% |
| L2 Throughput              |   44.7% |   20.5% |
| DRAM Throughput            |   13.6% |   35.1% |
| Eligible warps / scheduler |   ~0.02 |   ~0.06 |

## Warp softmax

Step 01 used one thread per row, while Step 02 assigns one warp to each row.

Adjacent lanes therefore access adjacent elements, producing coalesced global memory accesses.

Nsight Compute reports zero excessive global sectors, confirming that the inefficient access pattern from the previous implementation has been removed.

SM throughput increases from approximately:
```text
Step 01     0.9%
Step 02     4.2%
```

showing the benefit of distributing each row across 32 lanes.

However, scheduling efficiency remains low:
```text
Theoretical occupancy          100%
Achieved occupancy             96.1%
Active warps / scheduler       11.64
Eligible warps / scheduler      0.06
Instruction issue interval   ~75 cycles
```

Despite high occupancy, very few resident warps are ready to issue instructions.

The dominant scheduler stall is **LG throttle**, indicating pressure on the local/global-memory instruction path.

DRAM throughput reaches only 35.1%, so the kernel is not saturating raw DRAM bandwidth.  
Instead, frequent global-memory instructions create issue pressure, reflected in the dominant LG-throttle stalls.

## Conclusion

Warp-level softmax improves row-wise parallelism and global memory access efficiency.

However, the kernel still makes multiple passes over the materialized $N \times N$ score matrix,  
keeping global-memory instruction pressure high and contributing to LG-throttle stalls.

Step 03 introduces online softmax, preparing for later fused steps that eliminate this intermediate matrix.
