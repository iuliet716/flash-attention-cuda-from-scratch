# Nsight Compute Analysis — Step 08: WMMA Tensor Cores

This document contains the detailed Nsight Compute analysis for [Step 08: WMMA Tensor Cores](../08_wmma.md).

The goal is to verify that the WMMA implementation reaches the Tensor Core pipeline and to identify what limits its utilization.

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

Both matrix multiplications use WMMA:

```text
QK^T    FP16 x FP16 → FP32 accumulator
PV      FP16 x FP16 → FP32 accumulator
```

The online-softmax state remains FP32.

Step 08 also changes tile geometry, warp mapping, and shared-memory layout:

| Configuration          | Step 07 | Step 08 |
| ---------------------- | ------: | ------: |
| Q rows / block (`BR`)  |       8 |      16 |
| K/V rows / tile (`BC`) |      32 |      64 |
| Warps / block          |       8 |       4 |
| K layout               | XOR-swizzled | padded regular |
| Shared intermediates   | Q, K, V | Q, K, V, S, P, O |

The comparison with Step 07 therefore measures the complete WMMA-based redesign.  
It does not isolate the effect of Tensor Core instructions alone.

## Overview

| Metric                     | Value |
| -------------------------- | ----: |
| SM Throughput              | 11.3% |
| Memory Throughput          | 54.5% |
| L1/TEX Throughput          | 54.7% |
| L2 Throughput              | 6.38% |
| DRAM Throughput            | 0.39% |
| Tensor-pipe utilization    | 2.90% |
| Theoretical Occupancy      | 16.7% |
| Achieved Occupancy         | 16.6% |
| Active warps / scheduler   |  2.00 |
| Eligible warps / scheduler |  0.12 |
| Issue-active cycles        | 10.9% |

The kernel is not close to peak Tensor Core throughput or peak DRAM bandwidth.

Instead, the profile shows a latency and work-partition problem inside the SM:

```text
large shared-memory footprint
        ↓
few resident warps
        ↓
warp-0-only softmax
        ↓
long barrier waits
        ↓
very few eligible warps
        ↓
low Tensor-pipe utilization
```

The top-level 54.5% memory throughput is driven by the L1/shared-memory path, not by DRAM traffic.

---

## Tensor Cores are active, but lightly utilized

Step 07 performs both matrix multiplications with conventional FP32 multiply-add loops.

Step 08 replaces those loops with WMMA, and the generated SASS contains:

```text
HMMA.16816.F32
```

Nsight Compute reports:

| Pipeline utilization | Step 07 | Step 08 |
| -------------------- | ------: | ------: |
| Tensor               |    0.0% |    2.90% |
| FMA                  |   32.8% |    2.29% |

The nonzero Tensor value confirms that Tensor Core instructions execute.

However, 2.90% is still very low.  
It indicates presence of Tensor Core work, not saturation or efficient scheduling of that work.

The low value is consistent with the scheduler profile:

```text
Eligible warps / scheduler    0.12
Issue-active cycles          10.9%
No-eligible cycles           89.1%
```

The Tensor pipeline is idle during most cycles because too few warps are ready to issue.

---

## Tile reuse approximately halves global-load work

Global-load requests and sectors fall by approximately half:

| Metric              | Step 07 | Step 08 |
| ------------------- | ------: | ------: |
| Global-load requests | 134,348,800 | 67,239,936 |
| Global-load sectors  | 2,149,580,800 | 1,075,838,976 |
| Global-store requests | 1,048,576 | 1,048,576 |
| Global-store sectors  | 2,097,152 | 2,097,152 |

This reduction is not produced by WMMA itself.

Step 07 processes eight query rows per block:

```text
BR = 8
```

Step 08 processes sixteen:

```text
BR = 16
```

The number of query blocks is therefore halved, and each loaded K/V tile is reused across twice as many query rows.

The output shape does not change, so global-store requests and sectors remain unchanged.

This is a tile-geometry and data-reuse improvement that accompanies the WMMA conversion.

---

## Shared-memory growth limits occupancy

The per-block dynamic shared-memory footprint increases substantially:

| Metric                        | Step 07 | Step 08 |
| ----------------------------- | ------: | ------: |
| Dynamic shared memory / block | 9,216 B | 33,472 B |
| Warps / block                 |       8 |       4 |
| Shared-memory block limit     |      10 |       2 |
| Achieved occupancy            |   66.5% |   16.6% |

Step 07 stores only the FP16 Q, K, and V tiles in shared memory.  
Its online-softmax state and output accumulator remain in registers.

Step 08 adds:

```text
FP32 O tile
FP32 S tile
FP32 m, l, alpha
FP16 P tile
16-half padding for Q, K, and V rows
```

For `d = 64`, these allocations require 33,472 bytes per block.

Nsight Compute reports the Step 08 residency limits:

```text
Block limit — shared memory     2
Block limit — registers         9
Block limit — warps            12
```

Shared memory is therefore the active residency limit.

With four warps per block:

```text
2 blocks / SM
x 4 warps / block
= 8 warps / SM
```

The RTX 5090 can host 48 warps per SM, so eight resident warps correspond to 16.7% theoretical occupancy.  
Achieved occupancy closely matches it at 16.6%.

---

## Warp availability collapses

The lower residency leaves much less latency-hiding capacity:

| Scheduler metric              | Step 07 | Step 08 |
| ----------------------------- | ------: | ------: |
| Active warps / scheduler      |    7.97 |    2.00 |
| Eligible warps / scheduler    |    2.26 |    0.12 |
| Issue-active cycles           |   67.1% |   10.9% |
| Warp cycles / issued inst.    |   11.89 |   18.44 |

Step 08 has two active warps per scheduler on average, but only 0.12 are ready to issue in a typical cycle.

Nsight Compute reports that at least one warp is eligible in only 10.9% of cycles.  
The other 89.1% have no eligible warp, so the issue slot remains unused.

This explains why every compute pipeline remains underutilized even though WMMA has replaced the scalar matrix-multiplication loops.

---

## Warp-0-only softmax makes barrier stalls dominant

The largest warp-stall category changes from MIO throttle in Step 07 to barrier stalls in Step 08:

| Stall reason / issued instruction | Step 08 |
| --------------------------------- | ------: |
| Barrier                           | 8.88 cycles |
| Short scoreboard                  | 3.91 cycles |
| Long scoreboard                   | 2.06 cycles |
| Wait                              | 1.96 cycles |
| Math-pipe throttle                | 0.27 cycles |
| MIO throttle                      | 0.03 cycles |

Barrier stalls account for 48.1% of the average 18.44 warp cycles between issued instructions.

The implementation assigns the entire 16-row softmax tile to warp 0:

```cuda
if (warp == 0 && lane < BR) {
    const int r = lane;
    ...
}
```

Warps 1–3 do no softmax work and wait at the following block-wide barrier:

```cuda
__syncthreads();
```

Source-correlated sampling attributes 1,533,512 of 1,588,233 barrier samples, or 96.6%, to the instruction immediately following this barrier.

The dominant barrier cost is therefore a direct consequence of the Step 08 work partition:  
all four warps cooperate on WMMA, but only one warp performs the intervening softmax.

This directly motivates the next step, which divides the softmax rows across all warps.

---

## The padded layout is not conflict-free

Step 08 removes the Step 07 XOR swizzle and pads each Q, K, and V row by 16 half values.

For `d = 64`:

```text
logical Q/K/V row      128 bytes
padded Q/K/V stride    160 bytes
```

The 160-byte stride shifts consecutive Q/K/V row starts by 32 bytes relative to the 128-byte bank cycle.

However, S, P, and O are not padded, and the WMMA path introduces additional fragment loads and stores.

Nsight Compute reports:

| Shared-memory metric | Loads | Stores |
| -------------------- | ----: | -----: |
| Requests             | 578,813,952 | 343,080,960 |
| Wavefronts           | 6,658,735,609 | 3,109,553,189 |
| Bank conflicts       | 5,643,714,041 | 2,497,643,557 |
| Average conflict     | 11.5-way | 9.1-way |

For comparison, Step 07 reports only 115,412 shared-load conflicts and 83,305,196 shared-store conflicts.

This increase cannot be attributed to removing XOR swizzling alone.  
Step 08 changes the tile shapes, thread mapping, shared-memory contents, and instruction types at the same time.

Source correlation shows where the excessive shared wavefronts are concentrated:

| SASS access group | Excessive wavefronts | Share |
| ----------------- | --------------------: | ----: |
| Scalar `LDS`      | 4.094B | 50.4% |
| Scalar `STS.U16`  | 2.013B | 24.8% |
| WMMA `LDSM`       | 1.342B | 16.5% |
| Other `LDS`/`STS` | 0.671B |  8.3% |

The scalar `LDS` and `STS.U16` groups occur mainly in the softmax region that reads S and writes P.

Their row strides are:

```text
S row    64 FP32 values = 256 bytes
P row    64 FP16 values = 128 bytes
```

Both are multiples of the 128-byte bank cycle.

Because warp 0 assigns one active lane to each row, the lanes access the same column across different rows at the same time.  
Those addresses repeatedly map to the same banks.

The Q/K/V padding changes their row-start pattern, but it does not address this S/P access pattern.  
The profile therefore does not support describing the Step 08 shared-memory layout as conflict-free.

---

## DRAM is not the bottleneck

The memory hierarchy shows:

```text
L1/TEX Throughput    54.7%
L2 Throughput         6.38%
L2 hit rate          99.1%
DRAM Throughput       0.39%
```

Measured DRAM traffic is:

```text
DRAM reads     312.7 MB
DRAM writes     61.0 MB
```

The high L2 hit rate and negligible DRAM throughput show that repeated tile traffic remains on-chip.

Step 08 is therefore not DRAM-bandwidth-bound.  
It is limited by low warp availability, barrier synchronization, and conflict-heavy shared-memory accesses.

---

## Overall

Step 08 successfully moves both matrix multiplications onto Tensor Cores:

```text
scalar FP32 QK^T and PV
        ↓
FP16 WMMA operands
        ↓
FP32 Tensor Core accumulation
```

The profile confirms this transition:

```text
Tensor-pipe utilization    0.0% → 2.90%
```

It also shows why the Tensor pipeline remains lightly utilized:

```text
larger shared-memory footprint
        ↓
16.6% achieved occupancy
        ↓
warp-0-only softmax
        ↓
8.88-cycle barrier stall
        ↓
0.12 eligible warps / scheduler
        ↓
2.90% Tensor-pipe utilization
```

The approximately halved global-load work is a benefit of processing more query rows per block, not of WMMA alone.

The next step addresses the clearest scheduling problem by distributing the WMMA tiles and softmax rows across all warps.
