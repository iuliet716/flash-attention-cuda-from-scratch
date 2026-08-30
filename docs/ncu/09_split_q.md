# Nsight Compute Analysis — Step 09: Split-Q Warp Partitioning

This document contains the detailed Nsight Compute analysis for [Step 09: Split-Q Warp Partitioning](../09_split_q.md).

The goal is to verify whether distributing Q rows across the warps removes the scheduling imbalance seen in Step 08, and to identify the next bottleneck after that change.

## Profiling setup

Representative workload:

```text
GPU: NVIDIA GeForce RTX 5090
dtype: FP16 operands with FP32 WMMA accumulation

B = 8
H = 16
N = 4096
d = 64
```

The complete attention operation runs inside:

```text
fused_attention_kernel
```

Both matrix multiplications remain on the Tensor Core path:

```text
QK^T    FP16 x FP16 → FP32 accumulator
PV      FP16 x FP16 → FP32 accumulator
```

The main change from Step 08 is the warp partition:

| Configuration          |          Step 08 |           Step 09 |
| ---------------------- | ---------------: | ----------------: |
| Q rows / block (`BR`)  |               16 |                64 |
| K/V rows / tile (`BC`) |               64 |                64 |
| Warps / block          |                4 |                 4 |
| Q rows / warp          |   shared 16 rows |     owned 16 rows |
| Softmax                |      warp 0 only |         all warps |
| K/V staging            | separate K and V | reused K/V buffer |

Step 08 assigns all four warps to different column slices of the same Q rows.

Step 09 instead assigns each warp its own 16-row Q/O slice:

```text
warp 0 : rows  0:16
warp 1 : rows 16:32
warp 2 : rows 32:48
warp 3 : rows 48:64
```

The comparison therefore measures both the new warp partition and the larger Q tile.

## Overview

| Metric                        |  Step 08 |  Step 09 |
| ----------------------------- | -------: | -------: |
| SM Throughput                 |    11.3% |    12.0% |
| L1/TEX Throughput             |    54.7% |    63.3% |
| L2 Throughput                 |    6.38% |    3.69% |
| DRAM Throughput               |    0.39% |    0.70% |
| Tensor-pipe utilization       |    2.90% |    3.58% |
| Dynamic shared memory / block | 33,472 B | 62,208 B |
| Theoretical Occupancy         |    16.7% |     8.3% |
| Achieved Occupancy            |    16.6% |    8.33% |
| Active warps / scheduler      |     2.00 |     1.00 |
| Eligible warps / scheduler    |     0.12 |     0.11 |
| Issue-active cycles           |    10.9% |    10.9% |

The split-Q mapping removes the dominant Step 08 barrier problem.

However, the larger Q tile increases the shared-memory allocation enough to reduce residency from two blocks to one block per SM.

The resulting profile is therefore:

```text
split-Q ownership
        ↓
warp-0-only softmax removed
        ↓
barrier stall largely disappears
        ↓
larger shared-memory footprint
        ↓
only four resident warps / SM
        ↓
conflict-heavy shared-memory accesses
        ↓
short-scoreboard stalls dominate
```

The optimization fixes one scheduling problem, but the shared-memory dataflow becomes the next limit.

---

## Split-Q removes the warp-0 softmax bottleneck

The clearest improvement is visible in the warp-stall profile.

Step 08 performs softmax only in warp 0:

```cuda
if (warp == 0 && lane < BR) {
    ...
}
```

Warps 1–3 then wait at the following block-wide synchronization.

Step 09 instead distributes the rows:

```cuda
if (lane < ROWS_PER_WARP) {
    const int r = r0 + lane;
    ...
}
```

All four warps therefore execute softmax for their own rows.

The effect is visible directly in the stall metrics:

| Stall reason / issued instruction |     Step 08 |     Step 09 |
| --------------------------------- | ----------: | ----------: |
| Barrier                           | 8.88 cycles | 0.16 cycles |
| Short scoreboard                  | 3.91 cycles | 4.29 cycles |
| Long scoreboard                   | 2.06 cycles | 1.34 cycles |
| Wait                              | 1.96 cycles | 1.81 cycles |
| Math-pipe throttle                | 0.27 cycles | 0.21 cycles |

Barrier stalls fall from the dominant Step 08 category to a minor cost.

This confirms that the split-Q work partition addresses the intended scheduling imbalance.

The main stall is now:

```text
Short scoreboard    4.29 cycles / issued instruction
```

Nsight Compute reports that this represents approximately **47.6%** of the average 9.0 warp cycles between issued instructions.

The profiler identifies shared-memory operations as the primary likely source of these short-scoreboard dependencies.

The bottleneck has therefore shifted from waiting for another warp to waiting on the warp's own shared-memory accesses.

---

## Larger Q tiles reduce repeated global loads

Step 09 increases:

```text
BR = 16 → 64
```

while keeping:

```text
BC = 64
```

A K/V tile can therefore be reused across four times as many query rows before the block finishes.

Nsight Compute shows the corresponding reduction in global-load work:

| Metric                |       Step 08 |     Step 09 |
| --------------------- | ------------: | ----------: |
| Global-load requests  |    67,239,936 |  16,908,288 |
| Global-load sectors   | 1,075,838,976 | 270,532,608 |
| Global-store requests |     1,048,576 |   1,048,576 |
| Global-store sectors  |     2,097,152 |   2,097,152 |

Global-load requests and sectors fall by approximately **4x**.

The reduction is slightly smaller than an exact factor of four because the Q tensor still has to be loaded once regardless of the Q-block size.

The main reduction comes from K/V:

```text
fewer Q blocks
        ↓
fewer complete scans of K/V
        ↓
more Q rows reuse each K/V load
```

The output tensor has the same size, so global-store requests and sectors remain unchanged.

This verifies the expected K/V-reuse advantage of increasing `BR`.

---

## Shared-memory growth reduces occupancy

The larger Q tile requires substantially more shared memory.

Nsight Compute reports:

```text
Dynamic shared memory / block    62,208 B
Registers / thread                     72
Warps / block                           4
```

The residency limits are:

```text
Block limit — shared memory      1
Block limit — registers          7
Block limit — warps             12
Block limit — blocks            24
```

Shared memory is therefore the active limit.

Only one block can reside on each SM:

```text
1 block / SM
x 4 warps / block
= 4 warps / SM
```

The RTX 5090 supports 48 resident warps per SM, giving:

```text
4 / 48
= 8.3% theoretical occupancy
```

Nsight Compute reports:

```text
Theoretical Occupancy       8.33%
Achieved Occupancy          8.33%

Theoretical active warps    4.00 / SM
Achieved active warps       4.00 / SM
```

The measured occupancy therefore closely matches the shared-memory-limited theoretical value.

For comparison, Step 08 allows two four-warp blocks per SM:

```text
Step 08    8 resident warps / SM    16.7%
Step 09    4 resident warps / SM     8.3%
```

The K/V staging-buffer reuse prevents an even larger shared-memory allocation, but the larger `BR` still halves the available warp residency.

---

## Scheduler utilization remains low

Lower occupancy directly reduces the number of warps available to each scheduler.

Nsight Compute reports:

```text
Active warps / scheduler      1.00
Eligible warps / scheduler    0.11
Issued warps / scheduler      0.11
```

A scheduler has only one active warp on average.

That warp is eligible to issue in only approximately 11% of cycles:

```text
One or more eligible    11.08%
No eligible             88.92%
```

The profiler summarizes this as approximately:

```text
one instruction every 9.0 cycles
```

Step 08 had:

```text
Active warps / scheduler      2.00
Eligible warps / scheduler    0.12
```

Step 09 removes the explicit warp-0 softmax imbalance, but reducing residency to one active warp per scheduler removes most of the remaining latency-hiding opportunity.

If that warp stalls on a shared-memory dependency, there is usually no second resident warp available on the same scheduler to issue instead.

This explains why issue activity remains approximately unchanged even though the dominant barrier stall has been removed.

---

## Shared-memory bank conflicts remain severe

The split-Q mapping gives each warp independent rows, but the current implementation still stores S, P, O, and softmax state in shared memory.

Nsight Compute reports:

| Shared-memory metric     |         Loads |        Stores |
| ------------------------ | ------------: | ------------: |
| Requests                 |   578,813,952 |   292,716,544 |
| Wavefronts               | 6,585,571,501 | 2,826,469,880 |
| Bank conflicts           | 5,570,549,933 | 2,415,919,608 |
| Average conflict         |      11.4-way |       9.7-way |
| Conflict wavefront share |         84.6% |         85.5% |

The load access pattern therefore remains almost as conflict-heavy as Step 08.

Across all shared-memory accesses, Nsight Compute reports:

```text
Total wavefronts          9,411,526,656
Excessive wavefronts      7,985,954,816
```

Approximately **85%** of the shared-memory wavefronts are excessive.

The split-Q mapping itself does not repair these layouts.

The kernel still materializes:

```text
S       : FP32 shared memory
P       : FP16 shared memory
O       : FP32 shared memory
m, l, α : FP32 shared memory
```

even though each of these rows now has a clear owner warp.

As a result, the new ownership structure is not yet translated into a warp-local dataflow.

The large bank-conflict cost is consistent with short scoreboard becoming the dominant stall reason.

---

## Short scoreboard becomes the main stall

The warp-stall distribution is now:

| Stall reason / issued instruction |     Step 09 |
| --------------------------------- | ----------: |
| Short scoreboard                  | 4.29 cycles |
| Wait                              | 1.81 cycles |
| Long scoreboard                   | 1.34 cycles |
| Math-pipe throttle                | 0.21 cycles |
| Barrier                           | 0.16 cycles |
| MIO throttle                      | 0.05 cycles |

The important transition is:

```text
Step 08

barrier
    ↓
warp 0 performs softmax
while other warps wait


Step 09

short scoreboard
    ↓
owner warp waits on
shared-memory dependencies
```

Nsight Compute explicitly points to shared-memory operations as the typical cause of the short-scoreboard stall in this profile and recommends reducing shared-memory bank conflicts or keeping frequently accessed values in registers.

This matches the structure of the implementation.

The split-Q mapping means S/P/O and the softmax state are no longer fundamentally shared between warps.

Their continued shared-memory representation therefore creates avoidable on-chip traffic and dependencies.

---

## Tensor Cores remain lightly utilized

Both matrix multiplications still execute through WMMA.

Tensor-pipe utilization changes from:

```text
Step 08    2.90%
Step 09    3.58%
```

The increase is small, and Tensor Core utilization remains low.

The profile also reports that all compute pipelines are underutilized.

This is consistent with the scheduler metrics:

```text
Active warps / scheduler       1.00
Eligible warps / scheduler     0.11
Issue-active cycles          ~10.9%
```

The Tensor Core instructions themselves are not the current limiting resource.

Instead, the Tensor pipeline frequently waits because the single resident warp per scheduler is stalled elsewhere in the kernel.

Improving Tensor Core utilization therefore requires improving the surrounding dataflow rather than simply adding more WMMA operations.

---

## DRAM is not the bottleneck

The memory hierarchy shows:

```text
L1/TEX Throughput     63.3%
L2 Throughput         3.69%
L2 hit rate           95.0%
DRAM Throughput       0.70%
```

Measured DRAM traffic is approximately:

```text
DRAM reads     373.0 MB
DRAM writes    171.1 MB
```

Peak DRAM bandwidth is nowhere close to saturation.

The high top-level memory throughput is instead associated primarily with the L1/shared-memory path.

Step 09 is therefore not DRAM-bandwidth-bound.

The relevant memory bottleneck is on chip:

```text
shared-memory traffic
        +
bank conflicts
        +
short-scoreboard dependencies
```

This is consistent with the profiler's memory and warp-stall diagnostics.

---

## Overall

Step 09 successfully changes the warp-level work partition:

```text
Step 08

same 16 Q rows
      ↓
warps split column work
      ↓
warp 0 performs softmax
      ↓
large barrier stall


Step 09

64 Q rows
      ↓
16 rows owned by each warp
      ↓
all warps perform
QK^T → softmax → PV
```

Nsight Compute verifies the intended scheduling improvement:

```text
Barrier stall

8.88 → 0.16 cycles / issued instruction
```

The larger Q tile also increases K/V reuse:

```text
Global-load requests

67.24M → 16.91M
```

However, that larger tile raises dynamic shared memory to 62,208 bytes per block and reduces theoretical occupancy to 8.3%.

At the same time, the existing S/P/O shared-memory dataflow remains highly conflicted:

```text
shared loads      11.4-way average conflict
shared stores      9.7-way average conflict
excess wavefronts                      ~85%
```

The dominant stall therefore moves from barrier synchronization to short-scoreboard dependencies:

```text
split-Q fixes warp imbalance
        ↓
shared-memory dataflow becomes exposed
        ↓
4.29-cycle short-scoreboard stall
```

Step 09 establishes independent warp ownership, but the data belonging to each warp still travels through shared memory.

The next step exploits this ownership by keeping the warp-local Q, S/P, O, and online-softmax state in registers,  
reducing both shared-memory traffic and synchronization requirements.
