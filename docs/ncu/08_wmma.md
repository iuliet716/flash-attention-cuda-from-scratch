# Nsight Compute Analysis — Step 08: WMMA Tensor Cores

This document contains the detailed Nsight Compute analysis for
[Step 08: WMMA Tensor Cores](../08_wmma.md).

The goal is to verify Tensor Core execution and identify what limits utilization.

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
The online-softmax state remains FP32, while P is stored in FP16.

The implementation also changes tile geometry and shared-memory layout:

| Configuration | Step 07 | Step 08 |
| --- | ---: | ---: |
| Q rows / block (`BR`) | 8 | 16 |
| K/V rows / tile (`BC`) | 32 | 64 |
| Warps / block | 8 | 4 |
| K layout | XOR-swizzled | Padded regular |
| Shared-memory tiles | Q, K, V | Q, K, V, S, P, O |

The comparison measures the complete WMMA redesign, including changes to tiling, warp mapping, and intermediate storage.

## Overview

| Metric | Step 07 | Step 08 |
| --- | ---: | ---: |
| SM Throughput | 94.3% | 11.3% |
| L1/TEX Throughput | 94.3% | 54.7% |
| L2 Throughput | 15.6% | 6.38% |
| DRAM Throughput | 0.31% | 0.39% |
| Tensor-pipe utilization | 0.0% | 2.90% |
| FMA-pipe utilization | 32.8% | 2.29% |
| Dynamic shared memory / block | 9,216 B | 33,472 B |
| Theoretical occupancy | 66.7% | 16.7% |
| Achieved occupancy | 66.5% | 16.6% |
| Active warps / scheduler | 7.97 | 2.00 |
| Eligible warps / scheduler | 2.26 | 0.12 |
| Issue Active (%) | 67.1% | 10.9% |
| Warp cycles / issued inst. | 11.89 | 18.44 |

Tensor Core execution is present, but low warp availability and synchronization stalls limit instruction issue.

The memory throughput summary reflects activity on the L1/shared-memory path; DRAM utilization remains low.

Utilization percentages alone do not establish a speedup or slowdown.  
Runtime comparisons require kernel-duration or benchmark measurements.

## Tensor Core execution

The generated SASS contains:

```text
HMMA.16816.F32
```

Together with the increase in Tensor-pipe utilization from 0.0% to 2.90%,  
this confirms that the WMMA path executes Tensor Core instructions.

FMA-pipe utilization falls from 32.8% to 2.29% as both matrix multiplications move away from scalar FP32 loops.

Tensor-pipe utilization remains low, consistent with the limited instruction issue shown by the scheduler metrics.

## Global-memory traffic and tile reuse

| Metric | Step 07 | Step 08 |
| --- | ---: | ---: |
| Global-load requests | 134,348,800 | 67,239,936 |
| Global-load sectors | 2,149,580,800 | 1,075,838,976 |
| Global-store requests | 1,048,576 | 1,048,576 |
| Global-store sectors | 2,097,152 | 2,097,152 |

Increasing `BR` from 8 to 16 halves the number of query blocks.  
Each loaded K/V tile serves twice as many query rows, approximately halving global-load requests and sectors.

Global-store counts remain unchanged with the same output shape and scalar store pattern.

The load reduction comes from greater tile reuse within the WMMA redesign.

## Shared-memory footprint and occupancy

| Metric | Step 07 | Step 08 |
| --- | ---: | ---: |
| Dynamic shared memory / block | 9,216 B | 33,472 B |
| Warps / block | 8 | 4 |
| Block limit — shared memory | 10 | 2 |
| Block limit — registers | 4 | 9 |
| Block limit — warps | 6 | 12 |
| Theoretical occupancy | 66.7% | 16.7% |
| Achieved occupancy | 66.5% | 16.6% |

Step 08 adds shared-memory S, P, and O tiles, moves the softmax state to shared memory, and pads Q/K/V rows.

For `d = 64`, the allocation is:

| Allocation | Bytes |
| --- | ---: |
| FP16 Q, K, V with row padding | 23,040 |
| FP32 S | 4,096 |
| FP16 P | 2,048 |
| FP32 O accumulator | 4,096 |
| FP32 m, l, alpha | 192 |
| Total | 33,472 |

Shared memory limits residency to two blocks per SM.  
At four warps per block, this gives eight resident warps out of a maximum of 48, or 16.7% theoretical occupancy.

Achieved occupancy closely matches this limit at 16.6%.

## Scheduler availability and barrier stalls

Active warps per scheduler fall from 7.97 to 2.00, while eligible warps fall from 2.26 to 0.12.

Issue Active decreases from 67.1% to 10.9%.  
The recorded scheduler summary reports no eligible warp in 89.1% of cycles.

The largest stall category is barrier waiting:

| Stall reason / issued instruction | Step 08 |
| --- | ---: |
| Barrier | 8.88 cycles |
| Short scoreboard | 3.91 cycles |
| Long scoreboard | 2.06 cycles |
| Wait | 1.96 cycles |
| Math-pipe throttle | 0.27 cycles |
| MIO throttle | 0.03 cycles |

Barrier stalls account for approximately 48% of the reported 18.44 warp cycles per issued instruction.

The softmax section assigns one lane to each query row in warp 0:

```cuda
if (warp == 0 && lane < BR) {
    const int r = lane;
    ...
}
__syncthreads();
```

Only 16 lanes perform softmax, and warps 1–3 wait at the block-wide barrier.

The recorded source correlation places 1,533,512 of 1,588,233 barrier samples (96.6%) at the instruction immediately following this barrier.
This identifies the softmax phase as the main source of inter-warp waiting.

The sample share describes where barrier stalls were observed;  
it is not a fraction of total kernel runtime.

## Shared-memory access and bank conflicts

Step 08 replaces XOR swizzling with 16-half padding for Q/K/V rows.  
At `d = 64`, this changes their stride from 128 to 160 bytes,  
shifting successive row starts by 32 bytes modulo the 128-byte bank cycle.

S, P, and O remain unpadded.

| Shared-memory metric | Loads | Stores |
| --- | ---: | ---: |
| Requests | 578,813,952 | 343,080,960 |
| Wavefronts | 6,658,735,609 | 3,109,553,189 |
| Bank conflicts | 5,643,714,041 | 2,497,643,557 |
| Wavefronts / request | 11.5 | 9.1 |

Step 07 reports 115,412 shared-load conflicts and 83,305,196 shared-store conflicts.

The wavefronts-per-request ratios describe total servicing work.  
They include the requirements of the access width and instruction type,  
so they should not be interpreted directly as bank-conflict multipliers.

The recorded source correlation groups excessive wavefronts as follows:

| SASS access group | Excessive wavefronts | Share |
| --- | ---: | ---: |
| Scalar `LDS` | 4.094B | 50.4% |
| Scalar `STS.U16` | 2.013B | 24.8% |
| WMMA `LDSM` | 1.342B | 16.5% |
| Other `LDS`/`STS` | 0.671B | 8.3% |

The scalar `LDS` and `STS.U16` groups occur mainly in softmax, which reads S and writes P.

| Tile | Row stride |
| --- | ---: |
| S | 64 FP32 values = 256 bytes |
| P | 64 FP16 values = 128 bytes |

With one lane per row, adjacent active lanes access the same column across different rows.  
Both strides map those accesses to different addresses in the same bank.

Q/K/V padding does not change this S/P access pattern.  
The increase in conflicts reflects the combined changes to layout, thread mapping, intermediate tiles, and instruction types.

## DRAM traffic

| Metric | Step 08 |
| --- | ---: |
| L1/TEX Throughput | 54.7% |
| L2 Throughput | 6.38% |
| L2 hit rate | 99.1% |
| DRAM Throughput | 0.39% |
| DRAM reads | 312.7 MB |
| DRAM writes | 61.0 MB |

The high L2 hit rate indicates substantial on-chip reuse.  
DRAM bandwidth is far from saturated in this profile.

The stronger optimization signals are limited warp availability, softmax-related barrier waiting, and shared-memory serialization.

## Conclusion

| Change | Observed result |
| --- | --- |
| WMMA for both matrix multiplications | Tensor-pipe utilization reaches 2.90% |
| More query rows per block | Global-load work approximately halves |
| Larger shared-memory footprint and fewer warps per block | Achieved occupancy falls to 16.6% |
| Warp-0-only softmax | Barrier stalls dominate |
| New shared-memory access patterns | Substantial load and store conflicts |

Step 08 establishes the Tensor Core path, but work partitioning and shared-memory access limit its utilization.

Step 09 assigns each warp its own query rows across $QK^\top$, softmax, and $PV$ to address the softmax work imbalance.
