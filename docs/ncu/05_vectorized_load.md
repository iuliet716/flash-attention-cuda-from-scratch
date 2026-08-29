# Nsight Compute Analysis — Step 05: Coalescing + Vectorized Load

This document contains the detailed Nsight Compute analysis for [Step 05: Coalescing + Vectorized Load](../05_vectorized_load.md).

The goal is to measure how replacing scalar Q/K/V accesses with `float4` loads changes memory-instruction pressure and to identify the bottlenecks that remain.

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

As in Step 04, the complete attention operation runs inside:

```text
fused_attention_kernel
```

The attention algorithm, tile sizes, and thread mapping are unchanged.

Step 05 changes only the access granularity:

```text
scalar float loads
        ↓
16-byte float4 loads
```

## Overview

| Metric                     | Value |
| -------------------------- | ----: |
| SM Throughput              | 47.5% |
| Memory Throughput          | 84.7% |
| L1/TEX Throughput          | 84.7% |
| L2 Throughput              | 14.2% |
| DRAM Throughput            | 0.27% |
| Theoretical Occupancy      | 50.0% |
| Achieved Occupancy         | 49.9% |
| Eligible warps / scheduler |  0.53 |

The kernel remains limited by the on-chip memory path rather than DRAM bandwidth.

However, vectorization substantially reduces the pressure observed in Step 04.

| Metric                              | Step 04 | Step 05 |
| ----------------------------------- | ------: | ------: |
| SM Throughput                       |   27.2% |   47.5% |
| L1/TEX Throughput                   |   95.1% |   84.7% |
| Shared-load requests                |  12.88B |   6.44B |
| Shared-load wavefronts              | 146.09B |  40.80B |
| Shared-load bank conflicts          | 133.21B |  30.07B |
| Average shared-load bank conflict   | 11.3-way | 6.3-way |
| MIO-throttle stall / issued inst.   | 50.3 cycles | 11.2 cycles |
| Warp cycles / issued inst.          | 71.3 cycles | 24.7 cycles |
| Instruction-issue interval          | ~7.2 cycles | ~4.1 cycles |

The profile shows that fewer, wider accesses improve execution efficiency even though the remaining shared-memory layout is still conflict-prone.

---

## Global-memory access efficiency

Q, K, and V are now loaded as aligned `float4` values.

Across a warp, adjacent lanes access adjacent 16-byte chunks:

```text
lane 0  → bytes   0–15
lane 1  → bytes  16–31
...
lane 31 → bytes 496–511
```

Nsight Compute reports:

```text
Average bytes / global-load sector    32 / 32
Excessive global sectors                    0
```

This confirms that the global loads fully utilize each 32-byte sector.

The improvement is therefore not caused by eliminating an uncoalesced access pattern.  
The accesses are both coalesced and wider: each instruction moves four FP32 values instead of one.

Vectorization reduces the number of memory instructions required to transfer the same logical Q/K/V data.

It does not reduce the amount of input data required by the attention algorithm.

---

## Reduced shared-memory request pressure

Step 05 also uses `float4` when reading Q and K from shared memory during the $QK^\top$ dot product.

The aggregate shared-load request count falls from:

```text
Step 04    12,884,901,888
Step 05     6,442,450,944
```

This is an exact 2× reduction across the complete kernel.

The total number of shared-load wavefronts falls even more:

```text
Step 04    146,092,967,568
Step 05     40,802,569,315
```

and the reported bank-conflict count decreases from:

```text
Step 04    133,208,065,680
Step 05     30,065,151,075
```

The vectorized kernel therefore presents substantially less work to the shared-memory pipeline.

This is the main hardware-level explanation for the Step 05 speedup.

---

## MIO pipeline pressure

MIO throttle remains the largest warp-stall category, but its severity is much lower than in Step 04.

Nsight Compute reports approximately:

```text
MIO-throttle stall / issued instruction

Step 04    50.3 cycles
Step 05    11.2 cycles
```

The overall latency between issued instructions also falls:

```text
Warp cycles / issued instruction

Step 04    71.3 cycles
Step 05    24.7 cycles
```

In Step 05, MIO throttle accounts for approximately:

```text
45.4%
```

of the average cycles between issued instructions, compared with about 70.6% in Step 04.

The same on-chip memory path remains the dominant bottleneck,  
but vectorization makes each warp spend much less time waiting for that path to accept new operations.

---

## Scheduler efficiency

Step 05 has fewer active warps than Step 04:

```text
Active warps / scheduler

Step 04    9.96
Step 05    5.99
```

Nevertheless, the number of eligible warps remains almost unchanged:

```text
Eligible warps / scheduler

Step 04    0.54
Step 05    0.53
```

Reducing per-warp stalls allows a much larger fraction of the smaller resident-warp pool to become ready to issue.

The scheduler consequently improves from issuing approximately one instruction every 7.2 cycles to one every 4.1 cycles.

This raises SM throughput from about 27.2% to 47.5%.

---

## Occupancy trade-off

Vectorization increases register usage:

```text
Registers / thread

Step 04    40
Step 05    65
```

Registers are allocated at a granularity that results in:

```text
72 allocated registers / thread
```

The Step 05 launch configuration is:

```text
256 threads / block
8 warps / block

Dynamic shared memory       18,432 bytes / block
```

Nsight Compute reports the following residency limits:

```text
Block limit — registers      3
Block limit — shared memory  5
Block limit — warps          6
```

Registers are now the tightest constraint.

Three resident blocks correspond to:

```text
3 blocks
× 8 warps / block
= 24 warps / SM
```

which produces 50% theoretical occupancy. Achieved occupancy is nearly identical at 49.9%.

Despite this decrease from approximately 83% occupancy in Step 04, Step 05 is substantially faster.

The result illustrates that occupancy is a means of hiding latency, not a performance objective by itself.  
Reducing the latency and instruction pressure experienced by each warp can outweigh having fewer resident warps.

---

## Remaining shared-memory bank conflicts

Vectorization reduces the amount of shared-memory work, but it does not redesign the row-major shared-memory layout.

Nsight Compute still reports:

```text
Average shared-load bank conflict     6.3-way
Shared-load requests                  6,442,450,944
Shared-load wavefronts               40,802,569,315
Bank conflicts                       30,065,151,075
```

Approximately 73.7% of the shared-load wavefronts are associated with bank conflicts.

The accesses are wider and less frequent, but many of them still serialize inside shared memory because their addresses map poorly across banks.

This explains why L1/TEX throughput remains high at approximately 84.7% and why MIO throttle is still the dominant stall reason.

Step 06 changes the shared-memory layout with swizzling to address this remaining bottleneck directly.

---

## DRAM is still not the bottleneck

The fused kernel still avoids materializing the $N \times N$ attention matrix in global memory.

The memory hierarchy shows:

```text
L1/TEX Throughput    84.7%
L2 Throughput        14.2%
L2 hit rate          99.4%
DRAM Throughput      0.27%
```

The high L2 hit rate and negligible DRAM utilization show that repeated tile accesses are largely served on-chip.

Step 05 should therefore not be described as DRAM-bandwidth-bound.

Its performance is limited by shared-memory execution and memory-instruction throughput inside the SM.

---

## Overall

Step 05 makes the same fused attention algorithm more efficient by changing the access granularity:

```text
scalar loads
    ↓
coalesced float4 loads
    ↓
fewer memory instructions
    ↓
fewer shared-memory requests and wavefronts
    ↓
lower MIO-throttle stalls
    ↓
higher instruction-issue and SM throughput
```

The profile also reveals the next bottleneck:

```text
row-major shared-memory layout
        ↓
remaining 6.3-way bank conflicts
        ↓
high L1/TEX and MIO pressure
```

Step 06 addresses this problem with shared-memory swizzling.
