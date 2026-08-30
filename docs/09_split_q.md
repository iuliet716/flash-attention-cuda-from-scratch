# Step 9. Split-Q Warp Partitioning

## What this step implements

Step 08 moved both $QK^\top$ and $PV$ onto Tensor Cores, but its warp mapping was still imbalanced.

All four warps cooperated on the same 16-row Q tile:

```text
Q rows 0:16
    ↓
warp 0 → score columns  0:16
warp 1 → score columns 16:32
warp 2 → score columns 32:48
warp 3 → score columns 48:64
```

After $QK^\top$, however, only warp 0 performed the online softmax for all 16 rows.

The other three warps waited at the following block-wide barrier.

This step changes the work partition so that each warp owns a separate **16-row Q/O slice** for the entire attention computation.

```cuda
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int WARPS = 4;

constexpr int ROWS_PER_WARP = BR / WARPS;

static_assert(
    ROWS_PER_WARP == WMMA_M,
    "each warp owns one WMMA row tile"
);
```

Therefore:

```text
warp 0 : Q/O rows  0:16
warp 1 : Q/O rows 16:32
warp 2 : Q/O rows 32:48
warp 3 : Q/O rows 48:64
```

K and V tiles remain shared across the block.

The arithmetic precision and online-softmax formulation are unchanged from Step 08.

## Split-Q warp mapping

Each warp determines the first row of its owned slice:

```cuda
const int r0 = warp * ROWS_PER_WARP;
```

With four warps:

```text
                     64 Q rows

             warp 0   rows  0:16
             warp 1   rows 16:32
             warp 2   rows 32:48
             warp 3   rows 48:64

                        │
                        │ shared K/V tile
                        ▼

                    64 K/V rows
```

The important change is the ownership direction.

Step 08 partitions work mainly across the column dimension of the same Q tile.

Step 09 partitions the Q rows themselves:

```text
Step 08

same Q rows
    ↓
warps split K / output-column work


Step 09

Q rows split across warps
    ↓
each warp owns its rows through
QK^T → softmax → PV
```

This gives each warp an independent output-row slice and removes the warp-0-only softmax structure.

## QKᵀ follows Q-row ownership

Each warp now computes all score columns for the 16 query rows it owns.

```cuda
for (int j = 0; j < BC; j += WMMA_N) {
    wmma::fragment<
        wmma::accumulator,
        WMMA_M, WMMA_N, WMMA_K,
        float
    > s_frag;

    wmma::fill_fragment(s_frag, 0.0f);

    for (int k = 0; k < d; k += WMMA_K) {
        ...
        wmma::load_matrix_sync(
            q_frag,
            Qs + r0 * ldh + k,
            ldh
        );

        wmma::load_matrix_sync(
            k_frag,
            KVsm + j * ldh + k,
            ldh
        );

        wmma::mma_sync(
            s_frag,
            q_frag,
            k_frag,
            s_frag
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

Conceptually:

```text
warp 0

Q[ 0:16, :] × K^T[:, 0:64]
        ↓
S[ 0:16, 0:64]


warp 1

Q[16:32, :] × K^T[:, 0:64]
        ↓
S[16:32, 0:64]

...
```

All warps read the same K tile, while each warp reads a different Q slice and writes a different S slice.

The score slices do not overlap.

Because the owner warp is also the only warp that consumes its S rows during softmax,  
a block-wide synchronization is not required between the local $QK^\top$ computation and that warp's softmax:

```cuda
__syncwarp();
```

The block-wide synchronization is deferred until the shared K buffer is about to be reused.

## Softmax is distributed across all warps

Step 08 assigns all softmax rows to warp 0:

```cuda
if (warp == 0 && lane < BR) {
    ...
}
```

Step 09 instead gives each warp the softmax rows corresponding to its Q slice:

```cuda
if (lane < ROWS_PER_WARP) {
    const int r = r0 + lane;

    ...
}
```

Each of the first 16 lanes handles one query row:

```text
warp 0 : lanes 0:15 → rows  0:16
warp 1 : lanes 0:15 → rows 16:32
warp 2 : lanes 0:15 → rows 32:48
warp 3 : lanes 0:15 → rows 48:64
```

Therefore all four warps perform useful softmax work concurrently.

The online-softmax state remains FP32:

```cuda
float* m_sm;
float* l_sm;
float* a_sm;
```

and the probability tile remains FP16:

```cuda
Ps[r * BC + c] = __float2half(p);
```

The numerical formulation is unchanged.

Only the ownership of the rows changes.

## PV follows the same row partition

Step 08 divides the output columns across warps.

For example, with `d = 64`:

```text
warp 0 : O[:,  0:16]
warp 1 : O[:, 16:32]
warp 2 : O[:, 32:48]
warp 3 : O[:, 48:64]
```

Step 09 removes this column partition.

Each warp instead computes every output column for its own 16 rows:

```cuda
for (int j = 0; j < d; j += WMMA_N) {
    wmma::fragment<
        wmma::accumulator,
        WMMA_M, WMMA_N, WMMA_K,
        float
    > o_frag;

    wmma::load_matrix_sync(
        o_frag,
        Osm + r0 * d + j,
        d,
        wmma::mem_row_major
    );

    ...

    wmma::store_matrix_sync(
        Osm + r0 * d + j,
        o_frag,
        d,
        wmma::mem_row_major
    );
}
```

The ownership is therefore consistent across the complete tile:

```text
warp
 │
 ├── Q rows
 │
 ├── S rows
 │
 ├── softmax rows
 │
 ├── P rows
 │
 └── O rows
```

No reduction between different warps is required to produce an output row.

This is the main structural advantage of the split-Q mapping.

## Relation to FlashAttention-2

This work partition follows the main within-block scheduling idea used by FlashAttention-2.

The original FlashAttention warp partition divides K/V-side work between warps, which requires communication of intermediate results.

FlashAttention-2 instead divides Q across warps while keeping K and V accessible to all warps.

Conceptually:

```text
K, V
shared by the block
     │
     ├────────┬────────┬────────┐
     ▼        ▼        ▼        ▼

   warp 0   warp 1   warp 2   warp 3
   Q 0:16   Q16:32   Q32:48   Q48:64
      │        │        │        │
      ▼        ▼        ▼        ▼
   O 0:16   O16:32   O32:48   O48:64
```

Step 09 adopts this **split-Q ownership direction**.

However, this implementation should not yet be considered a complete FlashAttention-2 dataflow.

The current kernel still stores several warp-local intermediates in shared memory:

```text
S       : FP32 shared memory
P       : FP16 shared memory
O       : FP32 shared memory
m, l, α : FP32 shared memory
```

The split-Q mapping creates independent warp ownership,  
but the implementation does not yet fully exploit that independence to keep the corresponding data in registers.

That is addressed in the next step.

## Larger Q tile improves K/V reuse

The Q tile also increases from:

```text
Step 08 : BR = 16
Step 09 : BR = 64
```

while:

```text
BC = 64
```

remains unchanged.

A loaded K/V tile is therefore reused across four times as many query rows.

For a fixed sequence length, increasing `BR` reduces the number of Q blocks that independently scan the K/V tiles.

Each loaded K/V tile is therefore reused across more Q rows before the block completes.

This is a second benefit of the redesign, separate from removing the warp-0-only softmax imbalance.

The Step 08 → Step 09 change should therefore be interpreted as the combined effect of:

```text
split-Q warp ownership
+
distributed softmax
+
larger BR
+
greater K/V reuse
+
changed shared-memory allocation
```

rather than the warp mapping alone.

## Reusing the K/V staging buffer

Increasing `BR` from 16 to 64 substantially increases the shared-memory footprint.

Keeping separate padded K and V tiles would make the allocation especially large for `d = 128`.

Step 09 therefore uses one shared buffer for both:

```cuda
__half* KVsm =
    Qs + BR * ldh;
```

K is loaded first:

```text
KVsm = K tile
    ↓
QK^T
    ↓
softmax
```

After all warps finish reading K, the block synchronizes:

```cuda
__syncthreads();
```

The same memory is then overwritten with V:

```text
KVsm = V tile
    ↓
PV
```

This is safe because K is no longer needed once the score tile has been produced.

The buffer reuse reduces the dynamic shared-memory requirement compared with keeping separate K and V staging regions.

This is buffer reuse, not double buffering.

The kernel still uses:

```cuda
constexpr int STAGES = 1;
```

and K/V loading remains synchronous.

## Nsight Compute summary

The split-Q mapping removes the most obvious warp-level imbalance from Step 08,  
but the larger tile introduces a different constraint.

For the profiled workload, Nsight Compute reports:

```text
Dynamic shared memory / block    62,208 B
Theoretical occupancy                8.3%
```

Shared memory is the active residency limit.

Only one four-warp block can therefore reside on an SM.

The profile also shows that shared-memory accesses remain heavily conflicted:

```text
Shared-load requests       578,813,952
Average load conflict         11.4-way

Shared-store requests      292,716,544
Average store conflict         9.7-way
```

Approximately 85% of the shared-memory wavefronts are excessive.

The largest profiler diagnostic is now a short-scoreboard dependency associated primarily with shared-memory operations:

```text
Short-scoreboard stall
≈ 4.3 cycles / issued instruction

≈ 47.6% of the average
9.0 warp cycles between issued instructions
```

Nsight Compute also reports that all compute pipelines remain underutilized.

This represents a different problem from Step 08.

Step 08 had an explicit warp-level scheduling imbalance:

```text
warp 0 → softmax
warps 1–3 → wait
```

Step 09 distributes that work:

```text
all four warps → softmax
```

but the kernel still moves the owned S, P, O, and softmax state through shared memory.

The larger Q tile also lowers residency to one block per SM.

The bottleneck therefore shifts toward the shared-memory dataflow and the low occupancy required to support it.

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 09](ncu/09_split_q.md)

## Remaining bottlenecks

Step 09 establishes a much cleaner warp ownership model,  
but the corresponding data is still mostly materialized in shared memory:

```text
Q       : FP16 shared memory
K / V   : FP16 shared staging buffer
S       : FP32 shared memory
P       : FP16 shared memory
O       : FP32 shared memory
m, l, α : FP32 shared memory
```

This produces several remaining costs:

* shared memory limits theoretical occupancy to 8.3%
* S/P/O accesses still generate substantial bank conflicts
* WMMA fragments are repeatedly stored to and loaded from shared memory
* softmax state is shared-memory resident even though each row has a single owner warp
* K/V loading remains synchronous and single-stage

The split-Q mapping means most of these values now have a natural warp-local owner.

Keeping them in shared memory therefore becomes increasingly unnecessary.

The next step exploits this property by moving the warp-owned attention state into registers and removing much of the intermediate shared-memory traffic.

## Conclusion

Step 09 changes the kernel from column-oriented warp cooperation to persistent Q-row ownership.

```text
Step 08

16 Q rows
   ↓
four warps cooperate
   ↓
warp 0 performs softmax
   ↓
block-wide synchronization


Step 09

64 Q rows
   ↓
16 rows per warp
   ↓
QK^T
   ↓
softmax
   ↓
PV
   ↓
same warp owns the rows throughout
```

This follows the split-Q work-partitioning direction of FlashAttention-2 while preserving the existing WMMA and online-softmax implementation.

The larger Q tile also increases K/V reuse, but most warp-local intermediates are still stored in shared memory.

Nsight Compute therefore shows that shared-memory capacity, bank conflicts, and short-scoreboard stalls remain important constraints.

Step 09 establishes the **ownership structure** needed for the next optimization.

Step 10 turns that ownership into a more local dataflow by keeping the corresponding intermediate state in registers.
