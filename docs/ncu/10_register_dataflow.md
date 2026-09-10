# Nsight Compute Analysis — Step 10: Warp-Owned Register Dataflow

This document contains the Nsight Compute analysis for
[Step 10: Warp-Owned Register Dataflow](../10_register_dataflow.md).

## Profiling setup

```text
GPU: NVIDIA GeForce RTX 5090
dtype: FP16 operands with FP32 accumulation

B = 8
H = 16
N = 4096
d = 64
```

Both steps use `BR = 64`, `BC = 64`, and four warps per block.  
Each warp owns 16 query rows, and both matrix multiplications use Tensor Cores.

Step 10 moves Q, S/P, O, and softmax state into registers.  
It also changes the MMA interface, softmax lane mapping, head-dimension specialization, and K/V staging.

## Benchmark result

The [repository benchmark](../../README.md#benchmark) reports:

| Metric | Step 09 | Step 10 |
| --- | ---: | ---: |
| Latency | 32.298 ms | 3.035 ms |
| Effective TFLOPS | 17.0 | 181.1 |

This is a **10.64x speedup**.

Timing uses CUDA events, with 10 warm-up runs and the median of 50 iterations.  
Fast math is enabled, TF32 is disabled, and L2 flushing is disabled.

These timings come from the benchmark, separately from the NCU counters below.  
Effective TFLOPS is calculated from the same algorithmic FLOP count and latency; it is not independent evidence of the speedup.

## Shared-memory accesses and bank conflicts

In Step 09, the same warp produces and consumes its S/P/O state, but those values still pass through shared memory.  
Step 10 removes those intermediate shared arrays.

| Metric | Step 09 | Step 10 |
| --- | ---: | ---: |
| Shared-load requests | 578.8M | 402.7M |
| Shared-store requests | 292.7M | 16.8M |
| Shared-load wavefronts | 6.59B | 806.6M |
| Shared-store wavefronts | 2.83B | 79.5M |
| Shared-load bank conflicts | 5.57B | 404.0M |
| Shared-store bank conflicts | 2.42B | 12.4M |

Shared-store requests fall by approximately **17.4x** because the kernel no longer stores Q/S/P/O and softmax state in shared memory.

Load requests fall more modestly because K/V operands still need to be read.  
However, the wavefronts required to serve those loads decrease substantially.

Summing load and store wavefronts gives approximately **9.41B → 0.886B**, or **10.6x fewer wavefronts**.

These counters show both fewer shared memory operations and less bank-conflict overhead.  
They provide direct evidence that the shared memory work has decreased.

Wavefronts per request reflect both access requirements and bank conflicts.  
They should not be interpreted as bank-conflict multipliers.

The wavefront reduction is not itself a prediction of the latency speedup.

## Shared-memory capacity and warp residency

| Metric | Step 09 | Step 10 |
| --- | ---: | ---: |
| Dynamic shared memory / block | 62,208 B | 20,480 B |
| Registers / thread | 72 | 124 |
| Maximum resident blocks / SM | 1 | 4 |
| Theoretical occupancy | 8.3% | 33.3% |
| Achieved occupancy | 8.33% | 32.53% |

Removing shared Q/S/P/O and softmax state reduces the shared allocation by **67.1%**.

Step 09 is limited to one four-warp block per SM by shared memory.  
Step 10 permits four blocks, or 16 resident warps per SM.

For Step 10, NCU reports both register and shared-memory residency limits of four blocks per SM.  
The register allocation is 128 registers per thread.

Despite the higher register usage, local-memory loads and stores are both zero in this `d = 64` profile.  
The removed shared-memory accesses have therefore not been replaced by spill traffic.

## Dependency stalls and instruction issue

| Metric | Step 09 | Step 10 |
| --- | ---: | ---: |
| Short-scoreboard stall / issued inst. | 4.29 cycles | 0.42 cycles |
| Active warps / scheduler | 1.00 | 3.89 |
| Eligible warps / scheduler | 0.11 | 0.51 |
| Issue Active (%) | 10.9% | 33.2% |
| Tensor-pipe utilization | 3.58% | 40.87% |

[Nsight Compute's documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)
associates short-scoreboard stalls with MIO dependencies, commonly shared memory operations.

The reduction in shared accesses and bank conflicts supports shared-memory dependencies as a major contributor to the Step 09 stalls.

Short-scoreboard stalls fall substantially while the scheduler has almost four resident warps available instead of one.  
Eligible warps and Issue Active increase accordingly.

These changes support two complementary effects:

- Fewer intermediate shared-memory dependencies.
- More resident warps available to hide remaining dependencies.

Both steps already use Tensor Cores.  
The higher Tensor-pipe utilization is consistent with the surrounding dataflow allowing matrix operations to execute more frequently.

Stall values are normalized per issued instruction, and the instruction stream also changes.  
The 4.29 → 0.42 reduction should not be interpreted as a tenfold reduction in total stall time.

Likewise, higher Tensor-pipe utilization is an outcome of the improved execution schedule, not an independent speedup factor.

## Softmax work distribution

Step 09 assigns one lane to each row.  
That lane scans all 64 scores, and only 16 lanes per warp participate.

Step 10 distributes each row across four lanes, with 16 scores per lane per row.  
Each four-lane group handles two rows, so all 32 lanes participate.

This mapping is directly visible in the implementation.  
It reduces the sequential work for each row and uses shuffle reductions instead of shared score reads.

However, the aggregate NCU counters do not isolate the time saved by this change.  
A separate comparison would be needed to measure its individual contribution.

## Global-memory work

Global-load requests remain close: **16.91M → 17.30M**.

Both steps retain the same Q tile size and K/V reuse structure.  
Step 10 therefore does not introduce another major reduction in logical K/V loading.

Global-load requests are not DRAM-byte counts.  
Step 10 reports a 97.5% L2 hit rate and 4.09% DRAM throughput, so many accesses are served on chip.

Together with the implementation and shared-memory counters, these observations support an improvement primarily in on-chip execution.  
The reported profile does not indicate DRAM-bandwidth saturation.

## Remaining stalls

| Stall reason / issued instruction | Step 09 | Step 10 |
| --- | ---: | ---: |
| Math-pipe throttle | 0.21 cycles | 4.75 cycles |
| MIO throttle | 0.05 cycles | 1.48 cycles |
| Long scoreboard | 1.34 cycles | 1.21 cycles |
| Barrier | 0.16 cycles | 0.78 cycles |
| Short scoreboard | 4.29 cycles | 0.42 cycles |

Math-pipe throttle becomes the largest stall category.  
This indicates execution-pipeline contention, but does not identify the Tensor pipeline alone as the bottleneck.

Barrier stalls do not decrease in this comparison.  
The large speedup should therefore not be explained simply as less synchronization.

K/V bank conflicts, register pressure, and synchronous single-stage loading also remain.

## Conclusion

The measured changes support the following explanation for the 10.64x benchmark improvement:

- Removing shared intermediate state sharply reduces shared-memory operations and bank-conflict overhead.
- The smaller shared allocation increases resident blocks from one to four per SM.
- Lower dependency stalls and more available warps accompany higher instruction issue and Tensor-pipe utilization.
- The revised softmax mapping reduces sequential work per row without introducing register spills in this profile.

These effects are coupled and their ratios should not be multiplied together.

The counters strongly support the register-dataflow redesign as the overall explanation.  
They do not establish how much of the speedup comes from each individual change.  
That attribution would require controlled comparisons of intermediate implementations.
