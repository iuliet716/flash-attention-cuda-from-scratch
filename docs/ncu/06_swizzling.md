# Nsight Compute Analysis — Step 06: Bank Conflict Avoidance (Swizzling)

This document contains the detailed Nsight Compute analysis for [Step 06: Bank Conflict Avoidance (Swizzling)](../06_swizzling.md).

The goal is to measure how XOR-swizzling the K tile reduces shared-load bank conflicts and identify the remaining bottlenecks.

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

The attention algorithm, tile sizes, and thread mapping are unchanged from Step 05.

## Overview

| Metric | Step 05 | Step 06 |
| --- | ---: | ---: |
| SM Throughput | 47.5% | 93.4% |
| L1/TEX Throughput | 84.7% | 93.5% |
| L2 Throughput | 14.2% | 27.9% |
| DRAM Throughput | 0.27% | 0.45% |
| Theoretical Occupancy | 50.0% | 66.7% |
| Achieved Occupancy | 49.9% | 66.5% |
| Eligible warps / scheduler | 0.53 | 1.51 |
| Shared-load wavefronts | 40.80B | 10.74B |
| Shared-load bank conflicts | 30.07B | 0.25M |
| Shared-store bank conflicts | 105.64M | 205.12M |
| MIO-throttle stall / issued inst. | 11.2 cycles | 6.8 cycles |
| Warp cycles / issued inst. | 24.7 cycles | 16.0 cycles |
| Instruction-issue interval | ~4.1 cycles | ~2.0 cycles |

Swizzling removes the dominant shared-load conflicts and improves instruction issue efficiency.

## Effect of swizzling

Step 05 reads the same `float4` column from different K rows during $QK^\top$, repeatedly accessing the same bank groups.

Step 06 distributes these accesses across banks with:

```cuda
k4[k ^ (lane % 8)]
```

Shared-load bank conflicts fall by more than 99.999%, and shared-load wavefronts fall by 73.7%.

The logical loads are unchanged, but fewer wavefronts are needed to serve them.  
This reduces MIO-throttle stalls and leaves more warps ready to issue instructions.

Swizzled addressing also increases the executed instruction count by approximately 5%.  
FP32 instruction counts remain unchanged, so the extra work is in addressing rather than attention arithmetic.

## Occupancy increase

Register usage decreases in this build:

| Metric | Step 05 | Step 06 |
| --- | ---: | ---: |
| Registers / thread | 65 | 64 |
| Allocated registers / thread | 72 | 64 |
| Resident blocks / SM (register limit) | 3 | 4 |

With 256 threads per block, crossing this allocation boundary raises theoretical occupancy from 50.0% to 66.7%.

The additional resident warps help hide the remaining stalls.  
This is a compiler code-generation effect specific to these kernels, rather than an inherent benefit of XOR swizzling.

## Remaining bottlenecks

Shared-store bank conflicts increase from 105.64M to 205.12M, while load conflicts are nearly eliminated.

The trade-off is favorable: K is written once per tile and read repeatedly during $QK^\top$.  
The added store conflicts are much smaller than the 30.07B load conflicts removed.

MIO throttle remains the largest stall category, accounting for approximately 42.4% of warp cycles per issued instruction.

L1/TEX throughput reaches 93.5%, while DRAM throughput remains only 0.45%.  
The roofline reports approximately 8% of peak FP32 performance, so high SM throughput does not imply saturated arithmetic pipelines.

The remaining bottleneck is the on-chip memory/MIO path.

## Conclusion

Step 06 removes the dominant shared-load conflicts, reducing MIO stalls and improving scheduler efficiency.

Smaller shared-store conflicts and heavy on-chip data movement remain.

Step 07 reduces Q/K/V/O storage to FP16 while retaining FP32 accumulation.
