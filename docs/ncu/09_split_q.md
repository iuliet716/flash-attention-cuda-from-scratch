# Nsight Compute Analysis — Step 09: Split-Q Warp Partitioning

This document contains the Nsight Compute analysis for
[Step 09: Split-Q Warp Partitioning](../09_split_q.md).

## Profiling setup

```text
GPU: NVIDIA GeForce RTX 5090
dtype: FP16 operands with FP32 accumulation

B = 8
H = 16
N = 4096
d = 64
```

Both $QK^\top$ and $PV$ use WMMA.

| Configuration | Step 08 | Step 09 |
| --- | ---: | ---: |
| Q rows / block (`BR`) | 16 | 64 |
| K/V rows / tile (`BC`) | 64 | 64 |
| Warps / block | 4 | 4 |
| Softmax | Warp 0 only | All four warps |
| K/V staging | Separate buffers | Reused buffer |

Step 09 assigns each warp its own query rows.  
The comparison includes changes to warp mapping, tile size, synchronization, and staging.

## Overview

| Metric | Step 08 | Step 09 |
| --- | ---: | ---: |
| Dynamic shared memory / block | 33,472 B | 62,208 B |
| Achieved occupancy | 16.6% | 8.33% |
| Active warps / scheduler | 2.00 | 1.00 |
| Eligible warps / scheduler | 0.12 | 0.11 |
| Issue Active (%) | 10.9% | 10.9% |
| Tensor-pipe utilization | 2.90% | 3.58% |

Barrier stalls decrease, but the larger shared-memory footprint halves warp residency.  
Instruction issue and Tensor-pipe utilization remain low.

## Softmax distribution and barrier stalls

Each warp now performs softmax for its own query rows.

| Stall reason / issued instruction | Step 08 | Step 09 |
| --- | ---: | ---: |
| Barrier | 8.88 cycles | 0.16 cycles |
| Short scoreboard | 3.91 cycles | 4.29 cycles |
| Long scoreboard | 2.06 cycles | 1.34 cycles |
| Wait | 1.96 cycles | 1.81 cycles |
| Math-pipe throttle | 0.27 cycles | 0.21 cycles |
| MIO throttle | 0.03 cycles | 0.05 cycles |

Barrier stalls fall substantially, consistent with the more balanced work distribution and revised synchronization.

[Nsight Compute’s documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)
suggests shared memory dependencies may contribute to the dominant short-scoreboard stalls.

## Global loads and K/V reuse

| Metric | Step 08 | Step 09 |
| --- | ---: | ---: |
| Global-load requests | 67,239,936 | 16,908,288 |
| Global-load sectors | 1,075,838,976 | 270,532,608 |
| Global-store requests | 1,048,576 | 1,048,576 |
| Global-store sectors | 2,097,152 | 2,097,152 |

Increasing `BR` from 16 to 64 allows each K/V tile to serve four times as many query rows.

Global-load requests and sectors fall to approximately one quarter.  
Q-load work and output-store counts remain unchanged.

## Shared-memory footprint and occupancy

The larger tiles raise shared-memory usage to 62,208 bytes per block.  
Shared memory limits residency to one four-warp block per SM, reducing achieved occupancy from 16.6% to 8.33%.

Active warps per scheduler fall from 2.00 to 1.00.  
With fewer warps available to hide dependency stalls, Issue Active remains at 10.9%.

## Shared-memory access and bank conflicts

| Shared-memory metric | Loads | Stores |
| --- | ---: | ---: |
| Requests | 578,813,952 | 292,716,544 |
| Wavefronts | 6,585,571,501 | 2,826,469,880 |
| Bank conflicts | 5,570,549,933 | 2,415,919,608 |
| Wavefronts / request | 11.4 | 9.7 |

Wavefronts per request include the work required by access width and instruction type.  
These ratios are not bank conflict multipliers.

Softmax still assigns one lane to each row.  
The S and P row strides are 256 and 128 bytes, respectively,  
so adjacent active lanes accessing the same column map to different addresses in the same bank.

These conflicts are consistent with shared memory dependencies contributing to short-scoreboard stalls.

DRAM throughput remains low at 0.70%, making shared memory access and limited warp availability the stronger optimization signals.

## Conclusion

Step 09 reduces barrier stalls and repeated K/V loads,  
but shared-memory usage and access patterns still limit instruction issue.

Step 10 moves intermediate state into registers to reduce these costs.  
