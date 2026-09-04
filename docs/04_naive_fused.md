# Step 4. Naive Fused Attention (SRAM Tiling)

## What this step implements

Fuse $QK^\top$, softmax, and $PV$ into a **single CUDA kernel**.

Instead of materializing the $N \times N$ score and probability matrices in HBM,  
the kernel processes $K$ and $V$ in tiles using shared memory and immediately accumulates the output.

This introduces the core dataflow of FlashAttention:

$$
QK^\top
\rightarrow
\text{online softmax}
\rightarrow
PV
$$

without writing the intermediate attention matrix to HBM.

## Thread mapping

Each block processes **8 query rows** using 8 warps:

```cuda
constexpr int BR = 8;
constexpr int BC = 32;

const int warp = threadIdx.x / 32;
const int lane = threadIdx.x % 32;
const int row = blockIdx.x * BR + warp;
```

Each warp owns one query row.

Within a $K$ tile of 32 rows, each lane computes the score for one key:

```cuda
const int key = tile + lane;

float dot = 0.0f;
for (int k = 0; k < d; ++k) {
    dot += Qtile[warp * d + k] * Ktile[lane * d + k];
}

float s = dot * scale;
```

Therefore, one warp computes **32 attention scores at a time** without storing them as a global score matrix.

## SRAM tiling

The block stores one $Q$ tile and the current $K/V$ tiles in shared memory:

```text
Q tile: 8 x d
K tile: 32 x d
V tile: 32 x d
```

```cuda
extern __shared__ float smem[];

float* Qtile = smem;
float* Ktile = Qtile + BR * d;
float* Vtile = Ktile + BC * d;
```

The $Q$ tile is loaded once for the block, while $K$ and $V$ are loaded tile-by-tile:

```cuda
for (int tile = 0; tile < N; tile += BC) {
    // load K/V tile into shared memory

    // compute QK^T tile
    // update online softmax
    // accumulate partial output
}
```

Only the current tile needs to remain on-chip, so the full $N \times N$ attention matrix is never materialized.

## Online softmax with output accumulation

Step 03 maintained the running maximum and exponential sum.

The fused kernel extends this state with an output accumulator:

* $m$: running maximum
* $l$: running exponential sum
* $a$: unnormalized output accumulator

For scores $s_j$ in the current tile:

$$
m_{\text{new}} =
\max
\left(
m_{\text{old}},
\max_j s_j
\right)
$$

The previous state is rescaled to the new maximum:

$$
\alpha =
e^{m_{\text{old}}-m_{\text{new}}}
$$

and the current tile unnormalized weights are

$$
p_j =
e^{s_j-m_{\text{new}}}.
$$

The denominator and output accumulator can then be updated together:

$$
l_{\text{new}} =
\alpha l_{\text{old}}
+
\sum_j p_j
$$

$$
a_{\text{new}} =
\alpha a_{\text{old}}
+
\sum_j p_j V_j.
$$

In the kernel:

```cuda
const float alpha = __expf(m - m_new);

l = l * alpha + l_tile;
m = m_new;

for (int i = 0; i < ACC; ++i)
    acc[i] *= alpha;
```

The current $PV$ contribution is then accumulated directly into `acc`.

## Warp-level output accumulation

Each lane initially owns one probability value for the current $K/V$ tile.

`__shfl_sync()` broadcasts each probability across the warp:

```cuda
for (int c = 0; c < BC; ++c) {
    const float pc = __shfl_sync(0xffffffff, p, c);

    for (int i = 0; i < ACC; ++i) {
        const int k = lane + 32 * i;

        if (k < d)
            acc[i] += pc * Vtile[c * d + k];
    }
}
```

Output dimensions are distributed across lanes, and each lane keeps its partial output values in registers.

After all $K/V$ tiles have been processed, the accumulated numerator is normalized:

```cuda
const float inv_l = 1.0f / l;

Ob[row * d + k] = acc[i] * inv_l;
```

This is the first and only global-memory write for the output element.

## Why this implementation is still naive

This kernel establishes the **fused tiled attention structure**, but the matrix operations themselves remain simple:

* scalar global-memory loads
* scalar shared-memory accesses
* scalar dot-product loops for $QK^\top$
* scalar accumulation for $PV$
* no Tensor Cores
* no optimized shared-memory layout
* no double buffering or asynchronous copies

The attention matrix is no longer materialized in HBM, but the fused kernel is still limited by inefficient on-chip data movement.

## Nsight Compute summary

For `B=8, H=16, N=4096, d=64`:

| Metric                               | Value |
| ------------------------------------ | ----: |
| SM throughput                        | 27.2% |
| L1/TEX throughput                    | 95.1% |
| DRAM throughput                      | 0.18% |
| Eligible warps / scheduler           |  0.54 |
| Shared-load bank conflicts / request |  10.3 |

| Step | Total HBM traffic |
| ---- | ----------------: |
| 02   |          ~48.3 GB |
| 04   |           ~1.5 GB |

Fusion removes the HBM traffic for materializing the $N \times N$ intermediate attention matrices,  
while scalar matrix computation and severe shared-memory bank conflicts limit the fused kernel.

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 04](ncu/04_naive_fused.md)

> **Note**  
> DRAM throughput is low in both Step 00 and Step 04, but for different reasons.  
> inefficient global memory access in Step 00, versus reduced HBM traffic after fusion in Step 04.

## Conclusion

Step 04 introduces tiled fused attention, avoiding materialization of the full $(N \times N)$ attention matrix in HBM.

The following steps optimize how data moves through the fused kernel:

* Step 05 reduces memory-instruction pressure with vectorized loads.
* Step 06 addresses shared-memory bank conflicts with swizzling.
