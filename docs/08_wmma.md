# Step 8. WMMA Tensor Cores

## What this step implements

Step 07 reduced Q, K, V, and O to FP16, but the matrix multiplications were still computed with scalar CUDA-core operations.

This step replaces both $QK^\top$ and $PV$ with **WMMA operations on Tensor Cores**.

The online-softmax algorithm remains unchanged.  
Only the matrix-multiplication path is reorganized around WMMA tiles.

## WMMA tile layout

The kernel uses a `16 x 16 x 16` WMMA configuration:

```cuda
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
```

The attention tile is also changed to:

```cuda
constexpr int BR = 16;
constexpr int BC = 64;
constexpr int WARPS = 4;
```

Therefore, one block processes a score tile of:

```text
Q tile : 16 x d
K tile : 64 x d

S tile = QK^T : 16 x 64
```

The four warps split the 64 K rows into four `16 x 16` WMMA tiles:

```text
                          K rows

             0      16      32      48      64
             |-------|-------|-------|-------|

Q  16 rows   | warp0 | warp1 | warp2 | warp3 |
             | 16x16 | 16x16 | 16x16 | 16x16 |
             |-------|-------|-------|-------|

                        S : 16 x 64
```

Unlike Step 07, where each warp handled one query row, the warps now cooperate on the same **16-row Q tile**.

## QKᵀ with Tensor Cores

Each warp computes one `16 x 16` part of the score tile.

The FP32 accumulator fragment is initialized first:

```cuda
wmma::fragment<
    wmma::accumulator,
    WMMA_M, WMMA_N, WMMA_K,
    float
> s_frag;

wmma::fill_fragment(s_frag, 0.0f);
```

The head dimension is processed in chunks of 16:

```cuda
for (int k = 0; k < d; k += WMMA_K) {
    wmma::fragment<
        wmma::matrix_a,
        WMMA_M, WMMA_N, WMMA_K,
        __half,
        wmma::row_major
    > a_frag;

    wmma::fragment<
        wmma::matrix_b,
        WMMA_M, WMMA_N, WMMA_K,
        __half,
        wmma::col_major
    > b_frag;

    wmma::load_matrix_sync(a_frag, Qs + k, ldh);

    wmma::load_matrix_sync(
        b_frag,
        Ks + (warp * WMMA_N) * ldh + k,
        ldh
    );

    wmma::mma_sync(
        s_frag,
        a_frag,
        b_frag,
        s_frag
    );
}
```

Q is loaded as a row-major matrix.

K is physically stored row-major, but loaded as `col_major`, so the WMMA operation sees the corresponding $K^\top$ tile.

Conceptually:

```text
A : Q tile       16 x 16
B : K^T tile     16 x 16

       A x B
        ↓
S fragment       16 x 16
```

The multiplication uses FP16 operands while `s_frag` accumulates the result in FP32.

The resulting fragment is then stored in shared memory:

```cuda
wmma::store_matrix_sync(
    Ssm + warp * WMMA_N,
    s_frag,
    BC,
    wmma::mem_row_major
);
```

Four warps together produce the full `16 x 64` score tile.

## Online softmax remains FP32

WMMA accelerates the matrix multiplications, but softmax is still performed explicitly in FP32.

After $QK^\top$, warp 0 in each block assigns one lane to each of the 16 query rows:

```cuda
if (warp == 0 && lane < BR) {
    const int r = lane;

    float m_tile = -FLT_MAX;

    for (int c = 0; c < BC; ++c) {
        if (tile + c < N)
            m_tile = fmaxf(
                m_tile,
                Ssm[r * BC + c] * scale
            );
    }

    ...
}
```

The running maximum and normalization factor remain FP32:

```cuda
float* m_sm;
float* l_sm;
float* a_sm;
```

The exponential values are converted to FP16 when stored as the probability tile:

```cuda
const float p =
    (tile + c < N)
        ? __expf(Ssm[r * BC + c] * scale - m_new)
        : 0.0f;

Ps[r * BC + c] = __float2half(p);
```

This FP16 `P` tile can then be used directly as a WMMA operand for $PV$.

## PV with Tensor Cores

The second matrix multiplication,

$$
O \mathrel{+=} PV
$$

is also converted to WMMA.

The four warps divide the output dimensions:

```cuda
const int dw = d / WARPS;
const int c0 = warp * dw;
```

For example, with `d = 64`:

```text
warp 0 : O[:,  0:16]
warp 1 : O[:, 16:32]
warp 2 : O[:, 32:48]
warp 3 : O[:, 48:64]
```

Each warp loads an FP32 output fragment:

```cuda
wmma::fragment<
    wmma::accumulator,
    WMMA_M, WMMA_N, WMMA_K,
    float
> o_frag;

wmma::load_matrix_sync(
    o_frag,
    Osm + c0 + j,
    d,
    wmma::mem_row_major
);
```

Then `P` and `V` are processed in 16-row chunks:

```cuda
for (int k = 0; k < BC; k += WMMA_K) {
    wmma::fragment<
        wmma::matrix_a,
        WMMA_M, WMMA_N, WMMA_K,
        __half,
        wmma::row_major
    > p_frag;

    wmma::fragment<
        wmma::matrix_b,
        WMMA_M, WMMA_N, WMMA_K,
        __half,
        wmma::row_major
    > v_frag;

    wmma::load_matrix_sync(
        p_frag,
        Ps + k,
        BC
    );

    wmma::load_matrix_sync(
        v_frag,
        Vs + k * ldh + c0 + j,
        ldh
    );

    wmma::mma_sync(
        o_frag,
        p_frag,
        v_frag,
        o_frag
    );
}
```

Thus both matrix multiplications now use the same precision pattern:

```text
FP16 x FP16
     ↓
Tensor Core
     ↓
FP32 accumulation
```

## Padded shared-memory layout

Step 07 used an XOR-swizzled K layout.

This step instead uses a regular matrix layout with additional row padding:

```cuda
constexpr int SKEW = 16;

const int ldh = d + SKEW;
```

Since each element is FP16, 16 extra values add **32 bytes of padding** to every Q, K, and V row.

```text
logical row

|----------- d values -----------|

shared-memory row

|----------- d values -----------|-- 16 half padding --|
```

The padded stride is used by the WMMA loads:

```cuda
wmma::load_matrix_sync(a_frag, Qs + k, ldh);

wmma::load_matrix_sync(b_frag, ..., ldh);
```

This gives WMMA a regular strided layout and shifts consecutive Q, K, and V row starts across shared-memory banks.

However, the padding applies only to Q, K, and V.  
The S, P, and O tiles retain unpadded row strides, and WMMA introduces fragment loads and stores with their own bank-access patterns.

The explicit XOR indexing used in Step 06 and Step 07 is therefore no longer used, but the overall shared-memory path is not conflict-free.

> Since Step 08 also replaces XOR swizzling with a padded shared-memory layout and changes the tile configuration,  
> the measured speedup over Step 07 should be interpreted as the effect of the overall WMMA-based redesign rather than Tensor Cores alone.

## Nsight Compute summary

Nsight Compute confirms that Step 08 reaches the Tensor Core pipeline, but also shows that the Tensor Cores are lightly utilized.

| Metric                         |  Step 07  |  Step 08  |
| ------------------------------ | --------: | -------: |
| Tensor-pipe utilization        |    0.0%   |   2.90%  |
| Dynamic shared memory / block  |  9,216 B  | 33,472 B |
| Achieved occupancy             |   66.5%   |   16.6%  | 
| Eligible warps / scheduler     |    2.26   |    0.12  |
| Issue-active cycles            |   67.1%   |   10.9%  |

The nonzero Tensor-pipe value verifies that the generated kernel executes WMMA/HMMA instructions.  
It does not indicate that the Tensor Cores are saturated.

The larger `BR = 16` tile also lets each block reuse K and V across twice as many query rows as Step 07.  
Global-load requests and sectors therefore fall by approximately half,  
but this is a tile-reuse effect of the overall redesign rather than a Tensor Core effect alone.

The new shared-memory intermediates and padded Q/K/V tiles increase the per-block footprint enough to limit each SM to two blocks.  
With four warps per block, this leaves only eight resident warps per SM.

Barrier stalls then become the largest warp-stall reason at 8.88 cycles per issued instruction.  
The main imbalance is the warp-0-only softmax:  
the other three warps wait at the following block-wide barrier.

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 08](ncu/08_wmma.md)

## Head-dimension constraint

The output columns are divided across four warps:

```cuda
const int dw = d / WARPS;
```

Each WMMA output tile contains 16 columns, so the number of columns assigned to each warp must be divisible by 16.

Therefore:

```text
d / 4 must be divisible by 16

→ d must be divisible by 64
```

The implementation enforces this condition:

```cuda
TORCH_CHECK(
    d % 64 == 0,
    "head dim must be a multiple of 64"
);
```

With the current `FUSED_D_MAX`, this supports head dimensions 64 and 128.

## Remaining bottlenecks

WMMA removes the scalar dot-product loops from $QK^\top$ and $PV$, but intermediate tiles are still stored in shared memory:

```text
Q, K, V : FP16 shared memory
S       : FP32 shared memory
P       : FP16 shared memory
O       : FP32 shared memory
```

WMMA fragments are written back to shared memory before later stages consume them.

The profile shows two additional costs:

* the warp-0-only softmax leaves the other three warps waiting at a block-wide barrier
* the unpadded S, P, and O layouts, together with WMMA fragment accesses, produce substantial shared-memory bank conflicts

Nsight Compute reports an average 11.5-way conflict for shared loads and 9.1-way conflict for shared stores.  
These values describe the complete redesigned access pattern;  
they should not be attributed to the removal of XOR swizzling alone.

Step 08 therefore introduces Tensor Core computation, but does not yet keep the Tensor pipeline busy.  
The next step reorganizes the warp mapping so that the softmax rows and WMMA tiles are distributed across all warps.

## Conclusion

Step 08 establishes a mixed-precision Tensor Core path for both matrix multiplications:

```text
FP16 Q, K, P, V
        ↓
WMMA QK^T and PV
        ↓
FP32 accumulation
```

The online-softmax formulation remains unchanged.

The profile verifies that Tensor Core instructions are executed,  
while also showing that the overall WMMA redesign introduces a larger shared-memory footprint, low occupancy, warp-level work imbalance, and conflict-heavy shared accesses.

The measured change from Step 07 must therefore be interpreted as the combined effect of WMMA, new tile geometry, a different warp mapping, and a different shared-memory layout.
