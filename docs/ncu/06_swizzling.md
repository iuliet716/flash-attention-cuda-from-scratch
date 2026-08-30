# Nsight Compute Analysis — Step 06: Bank Conflict Avoidance (Swizzling)

This document contains the detailed Nsight Compute analysis for [Step 06: Bank Conflict Avoidance (Swizzling)](../06_swizzling.md).

The goal is to measure whether XOR-swizzling the K tile removes the shared-load bank conflicts identified in Step 05 and to identify the bottlenecks that remain.

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

The complete attention operation runs inside:

```text
fused_attention_kernel
```

The attention algorithm, tile sizes, and thread mapping are unchanged from Step 05.

Step 06 changes only the K-tile address mapping in shared memory:

```text
row-major K[r][c4]
        ↓
K[r][c4 ^ (r % 8)]
```

## Overview

| Metric | Value |
| --- | ---: |
| SM Throughput | 93.4% |
| Memory Throughput | 93.4% |
| L1/TEX Throughput | 93.5% |
| L2 Throughput | 27.9% |
| DRAM Throughput | 0.45% |
| Theoretical Occupancy | 66.7% |
| Achieved Occupancy | 66.5% |
| Eligible warps / scheduler | 1.51 |

The dominant shared-load conflict from Step 05 is effectively eliminated.

| Metric | Step 05 | Step 06 |
| --- | ---: | ---: |
| SM Throughput | 47.5% | 93.4% |
| L1/TEX Throughput | 84.8% | 93.5% |
| Shared-load bank conflicts | 30.07B | 0.25M |
| Shared-load wavefronts | 40.80B | 10.74B |
| Shared-store bank conflicts | 105.64M | 205.12M |
| MIO-throttle stall / issued inst. | 11.2 cycles | 6.8 cycles |
| Warp cycles / issued inst. | 24.7 cycles | 16.0 cycles |
| Instruction-issue interval | ~4.1 cycles | ~2.0 cycles |
| Eligible warps / scheduler | 0.53 | 1.51 |
| Achieved Occupancy | 49.9% | 66.5% |

The result is not simply higher cache throughput.  
The swizzle removes serialization from the repeatedly executed K-load path,  
allowing the same mathematical work to move through the SM much more efficiently.

---

## Dominant shared-load conflicts removed

Step 05 stored K in a row-major shared-memory layout.  
During each $QK^\top$ dot-product iteration, lanes accessed the same `float4` column from different rows, mapping those addresses to repeated bank groups.

Nsight Compute reported:

```text
Shared-load bank conflicts

Step 05    30,065,151,075
Step 06           247,843
```

The conflict count falls by more than 99.999%.

Relative to the total Step 06 shared-load wavefront count, the remaining conflicts are negligible:

```text
247,843 / 10,737,666,083
≈ 0.0023%
```

The number of shared-load wavefronts falls at the same time:

```text
Step 05    40,802,569,315
Step 06    10,737,666,083
```

This is a reduction of approximately 73.7%.

The source still performs the same logical Q and K `float4` loads.  
The reduction occurs because the shared-memory hardware no longer needs to serialize the K accesses into many additional wavefronts.

This directly validates the purpose of the XOR layout.

---

## Residual shared-store conflicts

Removing the K-load conflicts does not make every shared-memory access conflict-free.

Nsight Compute now flags the shared-store path:

```text
Shared-store requests          268,697,600
Shared-store wavefronts      1,279,905,797
Shared-store bank conflicts    205,115,397
Average conflict                   4.8-way
Conflict share of wavefronts        16.03%
```

The shared-store conflict count rises from approximately 105.6 million in Step 05 to 205.1 million in Step 06.  
This is consistent with the extra address permutation used while writing the swizzled K tile.

The trade-off is strongly favorable because the K tile is written once and then read repeatedly by the Q warps during $QK^\top$.

The swizzle removes approximately 30.07 billion load conflicts while adding about 99.5 million store conflicts.  
Across all shared operations, the total bank-conflict count falls from approximately 30.17 billion to 0.21 billion.

The remaining store pattern is therefore worth noting, but it does not invalidate the optimization.

---

## MIO pipeline pressure

MIO throttle remains the largest warp-stall category, but it decreases after removing the shared-load conflicts:

```text
MIO-throttle stall / issued instruction

Step 05    11.2 cycles
Step 06     6.8 cycles
```

The overall latency between issued instructions also falls:

```text
Warp cycles / issued instruction

Step 05    24.7 cycles
Step 06    16.0 cycles
```

In Step 06, MIO throttle accounts for approximately:

```text
42.4%
```

of the average cycles between issued instructions, compared with about 45.4% in Step 05.

The MIO path remains important because the kernel still performs a large number of shared-memory and special-function operations.  
The lower stall duration shows that removing bank-conflict serialization makes that path substantially easier to feed.

---

## Scheduler efficiency

More warps are ready to issue in Step 06:

```text
Active warps / scheduler

Step 05    5.99
Step 06    7.97
```

```text
Eligible warps / scheduler

Step 05    0.53
Step 06    1.51
```

The proportion of cycles with at least one eligible warp rises from:

```text
24.2% → 50.0%
```

The scheduler consequently improves from issuing approximately one instruction every 4.1 cycles to one every 2.0 cycles.

Two effects contribute:

* each warp spends less time stalled on conflicting shared loads
* the compiled kernel has more resident warps available to hide latency

This raises issue-slot utilization from 24.2% to 50.0% and the reported SM throughput from 47.5% to 93.4%.

---

## Occupancy increase

The Step 06 launch configuration remains:

```text
256 threads / block
8 warps / block

Dynamic shared memory    18,432 bytes / block
```

However, compiler-generated register usage crosses an allocation boundary:

```text
Registers / thread

Step 05    65 requested, 72 allocated
Step 06    64 requested, 64 allocated
```

The register residency limit therefore changes from three to four blocks per SM:

```text
Step 05    3 blocks × 8 warps = 24 warps / SM
Step 06    4 blocks × 8 warps = 32 warps / SM
```

Nsight Compute reports:

```text
Theoretical Occupancy    50.0% → 66.7%
Achieved Occupancy       49.9% → 66.5%
```

This additional residency helps the scheduler hide the stalls that remain.

The occupancy increase should not be described as an inherent consequence of XOR swizzling.  
It is a code-generation effect of these compiled kernels, but it contributes to the measured Step 05-to-Step 06 improvement.

---

## Addressing overhead

The swizzle adds XOR and address-calculation work on both the K store and K load paths.

The total executed instruction count increases:

```text
Step 05    57,375,719,424
Step 06    60,260,876,288
```

This is an increase of approximately 5.0%.

The floating-point instruction counts are unchanged:

```text
Fused FP32 instructions        11,878,268,928
Non-fused FP32 instructions     3,088,580,608
```

The attention arithmetic is therefore the same.  
Step 06 performs slightly more address-generation work, but the cost is much smaller than the serialization removed from the shared-load path.

---

## High on-chip throughput is not a DRAM bottleneck

The memory hierarchy shows:

```text
L1/TEX Throughput    93.5%
L2 Throughput        27.9%
L2 hit rate          99.5%
DRAM Throughput      0.45%
```

L1/TEX throughput is higher than in Step 05 even though bank conflicts are lower.  
This does not mean the swizzle made the access pattern worse.

The kernel now serves the same logical shared-memory operands in much less time,  
so the on-chip path sustains a higher rate of useful work rather than spending cycles serializing conflicting K loads.

The high reported SM throughput also does not imply that FP32 arithmetic is saturated.  
Nsight Compute's roofline section reports only approximately 8% of peak FP32 performance,  
and its compute analysis states that the arithmetic pipelines remain under-utilized.

Step 06 should therefore be described as limited by the on-chip memory/MIO path, not by DRAM bandwidth or peak FP32 throughput.

---

## Overall

Step 06 changes the K-tile address mapping while preserving the attention algorithm and floating-point work:

```text
row-major K layout
        ↓
XOR-swizzled float4 columns
        ↓
shared-load conflicts effectively eliminated
        ↓
73.7% fewer shared-load wavefronts
        ↓
lower MIO stalls and more eligible warps
        ↓
higher issue efficiency
```

The profile also reveals the remaining limitations:

```text
residual shared-store conflicts
        +
high on-chip memory/MIO utilization
        +
FP32 storage and conventional FP32 matmul loops
```

Step 07 changes Q, K, V, and O storage to FP16 while retaining FP32 accumulation. Step 08 then introduces Tensor Core execution.
