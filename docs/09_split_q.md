# Step 09. Split-Q Warp Partitioning

## What this step implements

Step 08 introduced WMMA for both $QK^\top$ and $PV$, but only warp 0 performed softmax while the other warps waited.

Each warp computes attention scores, softmax, and output for **its own query rows**.

| Configuration | Step 08 | Step 09 |
| --- | ---: | ---: |
| Q rows / block (`BR`) | 16 | 64 |
| K/V rows / tile (`BC`) | 64 | 64 |
| Warps / block | 4 | 4 |
| Q rows / warp | Same 16 rows | Separate 16 rows |
| Softmax | Warp 0 only | All four warps |
| K/V staging | Separate buffers | Reused buffer |

Both matrix multiplications retain FP16 operands with FP32 accumulation.  
The online softmax formulation is unchanged.

## Split-Q warp mapping

The kernel assigns one WMMA row tile to each warp:

```cuda
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int WARPS = 4;
constexpr int ROWS_PER_WARP = BR / WARPS;

static_assert(
    ROWS_PER_WARP == WMMA_M,
    "each warp owns one WMMA row tile"
);

const int r0 = warp * ROWS_PER_WARP;
```

Each warp computes every score and output column for its assigned rows:

| Warp | Q/S/P/O rows within the block |
| --- | --- |
| 0 | `[0, 16)` |
| 1 | `[16, 32)` |
| 2 | `[32, 48)` |
| 3 | `[48, 64)` |

All warps read the same K/V tile, but their S/P/O slices do not overlap.

This follows the Q-row partitioning idea in [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf).  

## QKᵀ follows Q-row ownership

Each warp computes a `16 x 64` score slice using four `16 x 16` WMMA tiles.

The outer loop covers all K rows in the tile:

```cuda
for (int j = 0; j < BC; j += WMMA_N) {
    // Initialize the FP32 score fragment.
    ...
    for (int k = 0; k < d; k += WMMA_K) {
        wmma::load_matrix_sync(
            q_frag, Qs + r0 * ldh + k, ldh
        );
        wmma::load_matrix_sync(
            k_frag, KVsm + j * ldh + k, ldh
        );
        wmma::mma_sync(
            s_frag, q_frag, k_frag, s_frag
        );
    }

    wmma::store_matrix_sync(
        Ssm + r0 * BC + j,
        s_frag,
        BC,
        wmma::mem_row_major
    );
}
```

Q loads and S stores use the warp's row offset `r0`.  
K loads access the shared tile.

Only the owner warp consumes its S slice during softmax, so this transition uses warp-level synchronization:

```cuda
__syncwarp();
```

Block-wide synchronization is still required before reusing the shared K/V buffer.

## Softmax runs in every warp

Each warp assigns its first 16 lanes to its 16 query rows:

```cuda
if (lane < ROWS_PER_WARP) {
    const int r = r0 + lane;
    // Compute online softmax for row r.
    ...
}
```

Each active lane scans the 64 score columns of one row.  
This distributes softmax across all four warps while retaining the one-lane-per-row implementation.

The running maximum, normalization factor, and output-rescaling factor remain FP32.  
The unnormalized exponentials are stored in FP16 for $PV$:

```cuda
Ps[r * BC + c] = __float2half(p);
```

Final normalization is applied after all K/V tiles have been processed.

## PV uses the same row partition

Each warp rescales its previous output by the online-softmax factor, then adds the current $PV$ contribution.

The output-column loop covers the full head dimension:

```cuda
for (int j = 0; j < d; j += WMMA_N) {
    wmma::load_matrix_sync(
        o_frag,
        Osm + r0 * d + j,
        d,
        wmma::mem_row_major
    );

    // Accumulate P[r0:r0+16, :] @ V[:, j:j+16].
    ...

    wmma::store_matrix_sync(
        Osm + r0 * d + j,
        o_frag,
        d,
        wmma::mem_row_major
    );
}
```

The same warp owns each row's scores, softmax state, and output accumulator.  
No cross-warp reduction is needed for an output row.

## K/V reuse and staging

Increasing `BR` from 16 to 64 allows each loaded K/V tile to serve four times as many query rows.  
For the same sequence length, this reduces the number of query blocks that scan K/V.

The larger Q, S, P, and O tiles also increase shared-memory usage.  
To limit this growth, K and V reuse one staging buffer:

```cuda
constexpr int STAGES = 1;

__half* KVsm = Qs + BR * ldh;
```

Within each iteration:

1. The block loads K and synchronizes.
2. Each warp computes its score slice and softmax.
3. The block synchronizes before overwriting K with V.
4. The block loads V and synchronizes.
5. Each warp updates its output slice.
6. The block synchronizes before the next K load.

Loading remains synchronous and single-stage.

## Nsight Compute summary

For `B=8, H=16, N=4096, d=64`:

| Metric | Step 08 | Step 09 |
| --- | ---: | ---: |
| Barrier stall / issued inst. | 8.88 cycles | 0.16 cycles |
| Short-scoreboard stall / issued inst. | 3.91 cycles | 4.29 cycles |
| Global-load requests | 67.24M | 16.91M |
| Dynamic shared memory / block | 33,472 B | 62,208 B |
| Achieved occupancy | 16.6% | 8.33% |
| Eligible warps / scheduler | 0.12 | 0.11 |
| Issue Active (%) | 10.9% | 10.9% |

Distributing softmax is accompanied by a large reduction in barrier stalls.  
The larger Q tile also reduces repeated K/V loads.

However, shared memory now limits residency to one four-warp block per SM.  
Shared memory conflicts remain substantial, and short scoreboard becomes the largest stall category.

The comparison includes changes to warp mapping, tile size, synchronization, and K/V staging.

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 09](ncu/09_split_q.md)

## Conclusion

Step 09 assigns each warp its own query rows throughout the attention computation.  
Barrier stalls decrease, and the larger Q tile improves K/V reuse.

Shared memory usage and bank conflicts still limit occupancy and instruction issue.  
Independent row ownership provides a foundation for further optimization through register-based computation and reduced synchronization.

Step 10 moves warp-owned intermediate state into registers.
Later steps can build on this structure to overlap K/V loading with computation.
