# Nsight Compute Analysis — Step 10: Warp-Owned Register Dataflow

This document contains the detailed Nsight Compute analysis for [Step 10: Warp-Owned Register Dataflow](../10_register_dataflow.md).

The goal is to verify whether moving the warp-owned Q, S/P, O, and online-softmax state from shared memory into registers removes the shared-memory bottleneck exposed in Step 09, and to identify the next limiting factor.

## Profiling setup

Representative workload:

```text
GPU: NVIDIA GeForce RTX 5090
dtype: FP16 operands with FP32 MMA accumulation

B = 8
H = 16
N = 4096
d = 64
```

The complete attention operation runs inside:

```text
fused_attention_kernel<64>
```

Both matrix multiplications use Tensor Core MMA:

```text
QK^T    FP16 × FP16 → FP32 accumulator
PV      FP16 × FP16 → FP32 accumulator
```

Step 10 preserves the Step 09 split-Q partition:

```text
warp 0 : Q/O rows  0:16
warp 1 : Q/O rows 16:32
warp 2 : Q/O rows 32:48
warp 3 : Q/O rows 48:64
```

The main difference is the storage and execution model.

| State / design        | Step 09           | Step 10               |
| --------------------- | ----------------- | --------------------- |
| Q                     | shared memory     | registers             |
| S                     | shared memory     | registers             |
| P                     | shared memory     | S registers reused    |
| O                     | shared memory     | registers             |
| `m`, `l`, `alpha`     | shared memory     | registers             |
| K                     | shared memory     | shared memory         |
| V                     | reused K/V buffer | shared memory         |
| Tensor Core interface | WMMA              | explicit `mma.sync`   |
| Softmax row mapping   | one lane / row    | four-lane group / row |
| K/V stages            | 1                 | 1                     |

The comparison therefore measures a broader dataflow redesign rather than register caching alone.

## Overview

| Metric                        |     Step 09 |     Step 10 |
| ----------------------------- | ----------: | ----------: |
| SM Throughput                 |       12.0% |       72.0% |
| L1/TEX Throughput             |       63.3% |       76.1% |
| L2 Throughput                 |       3.69% |      40.55% |
| DRAM Throughput               |       0.70% |       4.09% |
| Tensor-pipe utilization       |       3.58% |      40.87% |
| Dynamic shared memory / block |    62,208 B |    20,480 B |
| Registers / thread            |          72 |         124 |
| Theoretical Occupancy         |        8.3% |       33.3% |
| Achieved Occupancy            |       8.33% |      32.53% |
| Active warps / scheduler      |        1.00 |        3.89 |
| Eligible warps / scheduler    |        0.11 |        0.51 |
| Issue-active cycles           |       10.9% |       33.2% |
| Short scoreboard              | 4.29 cycles | 0.42 cycles |
| Math-pipe throttle            | 0.21 cycles | 4.75 cycles |

The profile changes qualitatively:

```text
Step 09

split-Q ownership
        ↓
large shared Q/S/P/O state
        ↓
one block / SM
        ↓
conflict-heavy shared-memory accesses
        ↓
short-scoreboard dependencies


Step 10

register-local Q/S/P/O state
        ↓
much smaller shared allocation
        ↓
four blocks / SM
        ↓
more resident and eligible warps
        ↓
much denser Tensor Core execution
        ↓
math-pipe throttle becomes visible
```

This is the expected transition for the register-dataflow redesign.

---

## Why the performance change is so large

The large performance improvement observed after Step 09 is consistent with a similarly large change in the kernel's execution profile.

No single profiler metric explains the improvement,  
but several independent measurements show that the dominant Step 09 shared-memory dependency chain is largely removed.

The strongest Nsight Compute changes are:

| Metric                      |     Step 09 |     Step 10 |        Change |
| --------------------------- | ----------: | ----------: | ------------: |
| Shared wavefronts           |     ~9.41 B |     ~0.87 B |  ~10.8x fewer |
| Excessive shared wavefronts |     ~7.99 B |     ~0.40 B |  ~19.8x fewer |
| Shared-store requests       |     292.7 M |      16.8 M |  ~17.4x fewer |
| Short scoreboard            | 4.29 cycles | 0.42 cycles |  ~10.2x lower |
| Dynamic shared memory       |    62,208 B |    20,480 B | ~3.0x smaller |
| Theoretical occupancy       |        8.3% |       33.3% |   4.0x higher |
| Active warps / scheduler    |        1.00 |        3.89 |  3.89x higher |
| Eligible warps / scheduler  |        0.11 |        0.51 |  ~4.7x higher |
| Issue-active cycles         |       10.9% |       33.2% |  ~3.0x higher |
| Tensor-pipe utilization     |       3.58% |      40.87% | ~11.4x higher |

Most notably, total shared-memory wavefronts fall by approximately **10.8x** and short-scoreboard stalls by approximately **10.2x**.

These are the two clearest measurements showing that the heavy shared-memory dataflow exposed in Step 09 has largely disappeared.

At the same time:

```text
theoretical occupancy
8.3% → 33.3%

eligible warps / scheduler
0.11 → 0.51

issue-active cycles
10.9% → 33.2%

Tensor-pipe utilization
3.58% → 40.87%
```

all increase substantially.

These ratios are not independent speedup factors and should not be multiplied together.

For example, higher occupancy gives the scheduler more warps with which to hide dependencies,  
while removing those dependencies makes the additional resident warps more useful.

Likewise, higher Tensor-pipe utilization is partly the result of performing the same Tensor Core work within a much denser execution schedule.

Instead, the metrics provide several views of the same transition:

```text
shared-memory round trips removed
        ↓
far fewer shared wavefronts
        ↓
far fewer shared-memory dependencies
        ↓
higher block residency
        ↓
more eligible warps
        ↓
more instructions can issue
        ↓
Tensor Core work executes more densely
```

The magnitude of these changes also makes an order-of-magnitude performance improvement plausible without relying on a benchmark latency inside this profiler document.

In particular:

```text
shared wavefronts       ~10.8x fewer
short scoreboard        ~10.2x lower
```

show that the dominant Step 09 cost changes by roughly an order of magnitude at the profiler level as well.

This should not be interpreted as a one-to-one relationship between either metric and overall latency.

The kernel changes several coupled execution properties simultaneously.

There is also an important optimization that is not captured by the shared-memory counters alone.

Step 09 performs softmax with:

```text
one lane → 64 score values
```

Step 10 distributes each row across:

```text
four lanes → 16 score values per lane
```

followed by shuffle reductions.

The per-lane sequential score loop is therefore substantially shorter.

The large improvement is consequently consistent with the combined effect of:

```text
register-resident attention state
        +
far less shared-memory traffic
        +
far fewer short-scoreboard dependencies
        +
higher warp residency
        +
greater scheduler availability
        +
shorter softmax dependency chains
```

Rather than one isolated optimization producing the entire gain,  
Step 10 removes several mutually reinforcing limits that are all present in Step 09.

---

## Shared-memory capacity no longer dominates residency

Step 09 requires:

```text
Dynamic shared memory / block    62,208 B
```

and shared memory restricts the kernel to:

```text
1 block / SM
```

Step 10 removes the Q, S, P, O, and online-softmax arrays from shared memory.

Only the K/V tiles remain.

Nsight Compute reports:

```text
Dynamic shared memory / block    20,480 B
```

The register requirement increases:

```text
Registers / thread                   124
Allocated registers / thread         128
```

The occupancy limits become:

```text
Block limit — registers          4
Block limit — shared memory      4
Block limit — warps             12
Block limit — blocks            24
```

Registers and shared memory therefore jointly limit the kernel to four blocks per SM.

Each block contains four warps:

```text
4 blocks / SM
×
4 warps / block
=
16 active warps / SM
```

The hardware maximum is 48 resident warps per SM, so:

```text
16 / 48
=
33.3%
```

theoretical occupancy.

Nsight Compute reports:

```text
Theoretical Occupancy    33.3%
Achieved Occupancy       32.5%
```

For comparison:

```text
Step 09     4 warps / SM     8.3%
Step 10    16 warps / SM    33.3%
```

The register-resident implementation therefore increases register pressure,  
but the reduction in shared-memory capacity is large enough to increase available warp residency by approximately four times.

---

## Shared-memory traffic falls sharply

Step 09 materializes:

```text
Q
S
P
O
m
l
alpha
```

in shared memory.

Step 10 removes all of these shared arrays.

The remaining shared-memory traffic is primarily the K/V staging and Tensor Core operand path.

### Shared loads

Nsight Compute reports:

| Metric                     |  Step 09 | Step 10 |
| -------------------------- | -------: | ------: |
| Shared-load requests       |  578.8 M | 402.7 M |
| Shared-load wavefronts     |   6.59 B | 806.6 M |
| Shared-load bank conflicts |   5.57 B | 404.0 M |
| Average conflict           | 11.4-way | 2.0-way |

The request count decreases, but the much larger change is in the number of wavefronts required to serve those requests.

Step 09 needs more than 6.5 billion load wavefronts because the accesses are heavily conflicted.

Step 10 requires approximately 807 million.

The average load conflict consequently falls from:

```text
11.4-way
```

to:

```text
2.0-way
```

### Shared stores

The shared-store path changes even more strongly:

| Metric                      | Step 09 | Step 10 |
| --------------------------- | ------: | ------: |
| Shared-store requests       | 292.7 M |  16.8 M |
| Shared-store wavefronts     |  2.83 B |  79.5 M |
| Shared-store bank conflicts |  2.42 B |  12.4 M |
| Average conflict            | 9.7-way | 4.7-way |

The request reduction is a direct consequence of removing shared materialization of the attention state.

Step 09 repeatedly writes values such as:

```text
Q
S
P
O
softmax state
```

to shared memory.

Step 10 primarily writes the staged K/V data.

### Total shared-memory wavefronts

Nsight Compute reports:

```text
Step 09

Total shared wavefronts       ~9.41 B
Excessive wavefronts          ~7.99 B
```

Step 10 reports:

```text
Total shared wavefronts       ~0.87 B
Excessive wavefronts          ~0.40 B
```

The total wavefront count therefore falls by approximately:

```text
10.8x
```

and the excessive component falls by approximately:

```text
19.8x
```

The important improvement is not only a better conflict ratio.

Most of the shared-memory operations that generated those conflicts no longer exist.

---

## Short scoreboard largely disappears

The dominant Step 09 stall is:

```text
Short scoreboard
4.29 cycles / issued instruction
```

The profiler associates this profile primarily with dependencies on shared-memory operations.

Step 10 reports:

```text
Short scoreboard
0.42 cycles / issued instruction
```

This is approximately a tenfold reduction.

The change matches the implementation directly.

Step 09 follows:

```text
Tensor Core accumulator
      ↓
shared S
      ↓
load S
      ↓
shared P
      ↓
load P
      ↓
shared O
      ↓
load O
```

Step 10 follows:

```text
Tensor Core accumulator
      ↓
register S/P
      ↓
register O
```

Q and the online-softmax state are also register resident.

The owner warp therefore waits far less frequently for its own intermediate shared-memory accesses.

This is the clearest profiler evidence that the main Step 09 dependency has been addressed.

---

## Softmax has less sequential work per lane

The shared-memory reduction alone does not describe the complete Step 10 change.

Step 09 maps one lane to each row:

```text
lane
 ↓
score 0
 ↓
score 1
 ↓
...
 ↓
score 63
```

Each participating lane therefore processes all 64 columns sequentially for both the maximum and exponential sum.

Step 10 follows the explicit MMA register layout.

One row is distributed across four lanes:

```text
lane 0 ─┐
lane 1 ─┼─ one row
lane 2 ─┤
lane 3 ─┘
```

Each lane handles approximately:

```text
64 / 4
=
16 score values
```

and the partial values are combined with shuffle reductions.

Conceptually:

```text
Step 09

64-value scalar scan


Step 10

16-value local scan
       +
4-lane reduction
```

This shortens the row-local dependency chain and allows all 32 lanes to participate in useful softmax work.

It is therefore another important contributor to the changed execution profile.

---

## Higher residency improves scheduler utilization

Step 09 reports:

```text
Active warps / scheduler      1.00
Eligible warps / scheduler    0.11
```

A scheduler has effectively one resident warp available.

When that warp waits on a shared-memory dependency, there is usually no alternative warp ready to issue.

Step 10 reports:

```text
Active warps / scheduler      3.89
Eligible warps / scheduler    0.51
```

The scheduler now has almost four resident warps on average and substantially more opportunities to find ready work.

Issue activity changes accordingly:

```text
Step 09    10.9%
Step 10    33.2%
```

This is roughly a threefold increase.

The eligible count remains much smaller than the active count:

```text
Active      3.89
Eligible    0.51
```

so latency hiding is still incomplete.

However, Step 10 combines two improvements that reinforce each other:

```text
fewer shared-memory dependencies
        +
more resident warps
```

The scheduler both encounters fewer stalls and has more alternatives when a stall does occur.

---

## Tensor Core utilization rises substantially

Both Step 09 and Step 10 already execute QK^T and PV on Tensor Cores.

The amount of attention matrix multiplication does not disappear in Step 10.

However, Tensor-pipe utilization changes from:

```text
Step 09     3.58%
Step 10    40.87%
```

Step 10 also reports approximately:

```text
41.31%
```

Tensor-pipeline utilization over active cycles.

The Tensor Cores were therefore present before Step 10, but they were poorly fed by the surrounding execution.

The transition is:

```text
Step 09

Tensor Core
    ↓
shared-memory round trip
    ↓
warp dependency
    ↓
Tensor Core


Step 10

Tensor Core
    ↓
register processing
    ↓
Tensor Core
```

Together with the higher scheduler availability, this allows Tensor instructions to execute much more densely.

The increase in Tensor utilization should not be interpreted as an independent speedup factor.

It is also a consequence of the same shorter and more efficient execution schedule.

The useful conclusion is that Step 10 does not gain by replacing Tensor Cores with a different amount of matrix multiplication.

It makes the existing Tensor Core work much easier to issue.

---

## SM throughput reflects denser useful execution

Overall SM throughput changes from:

```text
Step 09    12.0%
Step 10    72.0%
```

This accompanies:

```text
Theoretical occupancy
8.3% → 33.3%

Issue-active cycles
10.9% → 33.2%

Tensor-pipe utilization
3.58% → 40.87%
```

The three metrics are consistent.

Step 09 has Tensor Core instructions, but the SM spends much of its time unable to issue useful work because:

```text
few resident warps
        +
shared-memory dependencies
```

limit progress.

Step 10 provides more resident warps and removes most of those intermediate dependencies.

The same attention computation therefore occupies the execution pipelines much more densely.

---

## Global-load work remains nearly unchanged

The global-load request count is:

```text
Step 09    ~16.91 M
Step 10    ~17.30 M
```

The value is essentially unchanged at the scale of the overall redesign.

This is an important result.

Step 09 already obtains most of the global K/V reuse improvement by increasing the Q tile to:

```text
BR = 64
```

while retaining:

```text
BC = 64
```

Step 10 keeps those dimensions.

The large improvement therefore does not come from eliminating another full scan of K/V from HBM.

Instead:

```text
similar global tile traffic
        ↓
much less intermediate shared traffic
        ↓
register-local computation
```

The optimization is primarily an **on-chip dataflow improvement**.

---

## Register pressure does not cause spills

Moving Q, S/P, O, and the softmax state into registers raises register usage:

```text
Registers / thread

72 → 124
```

The hardware allocation is:

```text
128 registers / thread
```

This creates an explicit register occupancy limit of:

```text
4 blocks / SM
```

However, Nsight Compute reports:

```text
Local-memory loads     0
Local-memory stores    0
```

for the profiled kernel.

The additional register state therefore does not spill into local memory.

This is important when interpreting the optimization.

The reduced shared-memory traffic has not simply been replaced by hidden local-memory traffic.

For this `d = 64` profile, the intended state remains genuinely register resident.

---

## DRAM is not the bottleneck

Step 10 reports:

```text
L1/TEX Throughput     76.1%
L2 Throughput         40.5%
L2 hit rate           97.5%
DRAM Throughput        4.1%
```

Measured DRAM traffic is approximately:

```text
DRAM reads      201.4 MB
DRAM writes      79.6 MB
```

Peak DRAM bandwidth is nowhere close to saturation.

The kernel is therefore not HBM-bandwidth-bound.

The higher L1/TEX and L2 utilization primarily reflects the much denser execution of the kernel rather than DRAM saturation.

This is consistent with the unchanged global-load request count and the large reduction in intermediate shared-memory traffic.

---

## Remaining shared-memory conflicts belong to K/V

Register dataflow does not remove shared memory entirely.

The block still stages:

```text
K
V
```

because these operands are reused by all four Q-owning warps.

Nsight Compute therefore still reports approximately:

```text
Shared-load average conflict     2.0-way
Shared-store average conflict    4.7-way
```

and:

```text
~0.40 billion excessive shared wavefronts
```

The important difference is the scope of the problem.

Step 09 has shared traffic for:

```text
Q
K/V
S
P
O
m
l
alpha
```

Step 10 has shared traffic primarily for:

```text
K
V
```

The remaining shared-memory problem is therefore much narrower and can be treated as a K/V staging and operand-layout issue rather than a general attention-state dataflow issue.

---

## Math-pipe throttle becomes the main stall

The Step 10 stall profile is:

| Stall reason / issued instruction |     Step 10 |
| --------------------------------- | ----------: |
| Math-pipe throttle                | 4.75 cycles |
| MIO throttle                      | 1.48 cycles |
| Wait                              | 1.33 cycles |
| Long scoreboard                   | 1.21 cycles |
| Barrier                           | 0.78 cycles |
| Not selected                      | 0.55 cycles |
| Short scoreboard                  | 0.42 cycles |

The main transition is:

```text
Step 09

Short scoreboard        4.29
Math-pipe throttle      0.21


Step 10

Short scoreboard        0.42
Math-pipe throttle      4.75
```

The dominant problem has therefore changed.

Step 09 is primarily characterized by:

```text
warp waits for intermediate data
```

Step 10 is more often characterized by:

```text
warp has arithmetic ready
        ↓
required execution pipe is busy
```

This is the intended direction.

Removing the shared-memory dependency chain exposes the arithmetic execution itself as a more visible limit.

Math-pipe throttle should not be interpreted as proof that the Tensor pipeline alone is saturated.

Tensor utilization is approximately:

```text
40.9%
```

and the kernel contains several forms of arithmetic:

```text
Tensor Core MMA
softmax exponentials
max / sum reductions
FP16 conversions and packing
register arithmetic
```

The profile has therefore shifted toward a more compute-oriented execution regime rather than a single saturated Tensor Core pipeline.

---

## Overall

Step 09 establishes split-Q ownership:

```text
64 Q rows
   ↓
16 rows per warp
   ↓
QK^T → softmax → PV
```

but the corresponding state is still mostly stored in shared memory.

Step 10 converts that ownership into a local dataflow:

```text
Q registers
    ↓
QK^T
    ↓
S registers
    ↓
four-lane online softmax
    ↓
P in the same registers
    ↓
PV
    ↓
O registers
```

Only K and V remain block-shared.

Nsight Compute verifies the intended transition through several independent measurements:

```text
Dynamic shared memory
62,208 B → 20,480 B

Shared wavefronts
~9.41 B → ~0.87 B

Excessive shared wavefronts
~7.99 B → ~0.40 B

Short scoreboard
4.29 → 0.42 cycles

Theoretical occupancy
8.3% → 33.3%

Active warps / scheduler
1.00 → 3.89

Eligible warps / scheduler
0.11 → 0.51

Issue-active cycles
10.9% → 33.2%

Tensor-pipe utilization
3.58% → 40.87%
```

At the same time:

```text
Global-load requests
~16.91 M → ~17.30 M
```

remain essentially unchanged.

The improvement therefore comes primarily from the **on-chip execution path**, not another reduction in HBM tile traffic.

The additional register state also produces:

```text
Local-memory loads     0
Local-memory stores    0
```

so register spilling does not offset the benefit in the profiled configuration.

No single NCU metric should be treated as the isolated cause of the performance change.

The large improvement is supported by a consistent group of effects:

```text
remove shared attention-state materialization
        ↓
far fewer shared wavefronts
        ↓
far fewer short-scoreboard dependencies
        ↓
smaller shared-memory footprint
        ↓
higher warp residency
        ↓
more eligible warps and issue activity
        ↓
more efficient Tensor Core execution
```

The four-lane softmax mapping further shortens the row-local sequential dependency chain.

The dominant stall consequently moves from short-scoreboard dependencies to math-pipe throttle.

Step 10 therefore marks a clear change in the kernel's execution regime:

```text
Step 09

shared-memory-heavy
dependency-limited execution


Step 10

warp-local register dataflow
with substantially denser compute execution
```

The remaining optimization opportunities are now more specific:

* high register pressure
* residual K/V shared-memory conflicts
* synchronous single-stage K/V loading
* remaining scheduler stalls
* math-pipeline contention

Step 10 successfully removes the broad shared-memory dataflow bottleneck exposed by Step 09 and reveals the next layer of compute and K/V-staging limits.
