# Step 10. Warp-Owned Register Dataflow

## What this step implements

Step 09 assigned each warp its own 16 query rows across $QK^\top$, softmax, and $PV$.

However, the scores, softmax state, and output accumulator still passed through shared memory,  
even though the same warp produced and consumed them.

This step keeps those intermediate values in the owner warp's registers.  
Q also stays in registers across K/V tiles.

| Data | Step 09 | Step 10 |
| --- | --- | --- |
| Q | Shared memory | Registers |
| S/P | Separate shared tiles | Registers reused for S and P |
| O accumulator | Shared memory | Registers |
| Running maximum, normalization factor, rescale factor | Shared memory | Registers |
| K/V | Shared memory | Shared memory |

K and V remain in shared memory because all four warps reuse the same tile.

## Explicit MMA register layout

Step 09 uses WMMA fragments, whose per-thread element layout is opaque.  
The kernel stores the score fragment in shared memory so softmax can access its rows.

Step 10 uses an explicit Tensor Core instruction:

```cuda
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

Its defined register layout lets softmax directly consume the FP32 score accumulators.  
The resulting probabilities are then packed into FP16 operands for $PV$.

This removes the shared S/P tiles between the two matrix multiplications.

## Four-lane softmax

Each warp still owns 16 query rows.  
Within the warp, a four-lane group handles two rows:

```cuda
const int g = lane / 4;
const int tig = lane % 4;

const int r_lo = q_base + warp * 16 + g;
const int r_hi = r_lo + 8;
```

For each row, the four lanes hold different score columns.

| Softmax mapping | Step 09 | Step 10 |
| --- | ---: | ---: |
| Lanes participating per warp | 16 | 32 |
| Lanes per row | 1 | 4 |
| Scores per lane per row (`BC = 64`) | 64 | 16 |

Each lane computes partial maxima and sums from its registers.  
Shuffle reductions combine them within the four-lane group:

```cuda
for (int off = 1; off <= 2; off <<= 1) {
    mt[0] = fmaxf(mt[0], __shfl_xor_sync(0xffffffff, mt[0], off));
    mt[1] = fmaxf(mt[1], __shfl_xor_sync(0xffffffff, mt[1], off));
}
```

The exponential sum uses the same reduction pattern.

This shortens each row's sequential scan and avoids reading scores from shared memory.

## Register-resident accumulation

Q fragments are loaded once before the K/V loop:

```cuda
uint32_t qa[KK][4];
```

The output accumulator and running softmax state also remain live across tiles:

```cuda
float o[OB][4];
float m[2] = {-FLT_MAX, -FLT_MAX};
float l[2] = {0.0f, 0.0f};
```

Within each tile, the score array:

```cuda
float s[SB][4];
```

is overwritten with unnormalized exponentials.  
These values are packed into FP16 MMA operands without creating a shared P tile.

The previous output is rescaled when the running maximum changes,  
and the current $PV$ contribution is accumulated into the same output registers:

```cuda
o[jo][0] *= alpha[0];
o[jo][1] *= alpha[0];
o[jo][2] *= alpha[1];
o[jo][3] *= alpha[1];
```

After all K/V tiles, the output is divided by the running normalization factor and written to global memory.

No intermediate output is stored in shared memory.

## K/V staging and head dimension

The block configuration remains `BR = 64`, `BC = 64`, and four warps.

Removing shared Q/S/P/O state leaves enough capacity for separate K and V buffers:

```cuda
__half* Ks = smem;
__half* Vs = Ks + STAGES * BC * LDH;
```

Loading remains synchronous with `STAGES = 1`.  
There is no asynchronous copy/compute overlap.

The kernel is specialized for `D = 64` and `D = 128`, giving the compiler fixed-size register arrays and allowing loop unrolling.

## Nsight Compute summary

For `B=8, H=16, N=4096, d=64`:

| Metric | Step 09 | Step 10 |
| --- | ---: | ---: |
| Dynamic shared memory / block | 62,208 B | 20,480 B |
| Registers / thread | 72 | 124 |
| Achieved occupancy | 8.33% | 32.53% |
| Short-scoreboard stall / issued inst. | 4.29 cycles | 0.42 cycles |
| Eligible warps / scheduler | 0.11 | 0.51 |
| Issue Active (%) | 10.9% | 33.2% |
| Tensor-pipe utilization | 3.58% | 40.87% |

Shared memory accesses fall substantially, and the smaller allocation allows more resident warps despite higher register usage.

The profiled `d = 64` kernel reports no local-memory loads or stores.

The benchmark improvement includes register dataflow, softmax mapping, explicit MMA, compile-time specialization, and revised K/V staging.

Detailed performance analysis:

→ [Nsight Compute Analysis — Step 10](ncu/10_register_dataflow.md)

## Conclusion

Step 09 established which warp owns each query row.  
Step 10 keeps that row's intermediate state in the same warp's registers.

This removes shared-memory round trips between matrix multiplication, softmax, and output accumulation.

High register usage, residual K/V bank conflicts, and synchronous loading remain as optimization targets.
