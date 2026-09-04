# Nsight Compute Analysis — Step 05: Coalescing + Vectorized Load

This document contains the detailed Nsight Compute analysis for [Step 05: Coalescing + Vectorized Load](../05_vectorized_load.md).

The goal is to measure how float4 vectorization changes memory-instruction pressure and identify the remaining bottlenecks.

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

The attention algorithm, tile sizes, and thread mapping are unchanged from Step 04.

## Overview

| Metric                               |     Step 04 |     Step 05 |
| ------------------------------------ | ----------: | ----------: |
| SM Throughput                        |       27.2% |       47.5% |
| L1/TEX Throughput                    |       95.1% |       84.7% |
| L2 Throughput                        |        5.2% |       14.2% |
| DRAM Throughput                      |       0.18% |       0.27% |
| Theoretical Occupancy                |       83.3% |       50.0% |
| Achieved Occupancy                   |       83.0% |       49.9% |
| Eligible warps / scheduler           |        0.54 |        0.53 |
| Shared-load requests                 |      12.88B |       6.44B |
| Shared-load wavefronts               |     146.09B |      40.80B |
| Shared-load bank conflicts           |     133.21B |      30.07B |
| Shared-load wavefronts / request     |        11.3 |         6.3 |
| Shared-load bank conflicts / request |        10.3 |         4.7 |
| MIO-throttle stall / issued inst.    | 50.3 cycles | 11.2 cycles |
| Warp cycles / issued inst.           | 71.3 cycles | 24.7 cycles |
| Instruction-issue interval           | ~7.2 cycles | ~4.1 cycles |

Vectorization substantially reduces pressure on the on-chip memory path.

## Effect of vectorization

Q, K, and V tile loads use aligned float4 accesses, and Q/K reads from shared memory during $QK^\top$ are vectorized as well.

Nsight Compute reports:
```text
Average bytes / global-load sector    32 / 32
Excessive global sectors                    0
```

The global loads therefore fully utilize each memory sector, while moving more data per instruction.

Shared-memory pressure also falls:
```text
Shared-load requests

Step 04    12.88 B
Step 05     6.44 B
```
```text
Shared-load wavefronts

Step 04   146.09 B
Step 05    40.80 B
```

Fewer shared-memory requests and wavefronts reduce pressure on the MIO pipeline,  
so warps spend less time waiting to issue memory instructions.

As a result, these reductions lower MIO-throttle stalls from 50.3 to 11.2 cycles,  
improving the issue interval from 7.2 to 4.1 cycles and SM throughput from 27.2% to 47.5%.

## Occupancy trade-off

Vectorization increases register usage:
```text
Registers / thread

Step 04       40
Step 05       65
```

Higher register usage per thread increases the register requirement per block,  
reducing the number of resident blocks and warps; as a result, achieved occupancy drops from 83.0% to 49.9%.


However, eligible warps per scheduler remain nearly unchanged:
```text
Step 04    0.54
Step 05    0.53
```

Despite fewer resident warps, reduced MIO stalls leave a larger fraction of them ready to issue,
so eligible warps per scheduler remain nearly unchanged.

This shows that lower occupancy does not necessarily reduce performance when each resident warp executes more efficiently.

## Remaining shared-memory bank conflicts

The shared-memory layout itself is unchanged, so significant bank conflicts remain:

```text
Shared-load requests       6.44 B
Shared-load wavefronts    40.80 B
Bank conflicts            30.07 B
```

This corresponds to approximately:
```text
Shared-load wavefronts / request     6.3
Bank conflicts / request             4.7
```

## Conclusion

Step 05 further reduces on-chip memory pressure after fusion has already removed most intermediate HBM traffic.

The remaining bottleneck is shared-memory execution rather than DRAM bandwidth, motivating the swizzled layout in Step 06.
