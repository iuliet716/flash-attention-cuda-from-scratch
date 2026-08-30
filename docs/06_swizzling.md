# Step 6. Bank Conflict Avoidance (Swizzling)

## What this step implements

Step 05 vectorized global and shared-memory accesses with `float4`, but the $K$ tile still used a regular row-major shared-memory layout.

During $QK^\top$, lanes in a warp read the same column from different rows of $K$.  
For the supported head dimensions, these addresses repeatedly map to the same shared-memory banks, causing **bank conflicts**.

This step changes the shared-memory layout of $K$ using an **XOR swizzle** while keeping the attention algorithm unchanged.

```cuda
__device__ __forceinline__ int swizzle(int r, int c4) {
    return c4 ^ (r % 8);
}
```

## Bank conflicts in the K tile

In Step 05, each lane reads one row of $K$:

```cuda
const float4* k4 =
    reinterpret_cast<const float4*>(Ktile + lane * d);

const float4 b = k4[k];
```

For a fixed `k`, all lanes access the same `float4` column from different rows.

```text
lane 0  -> K[0][k]
lane 1  -> K[1][k]
lane 2  -> K[2][k]
...
lane 31 -> K[31][k]
```

With `d = 64` or `128`, each row starts at a shared-memory address separated by a multiple of the 32-bank layout.

As a result, these accesses repeatedly target the same bank groups instead of being distributed across shared memory.

## Swizzled K layout

Instead of storing each `float4` chunk at its original column, the column index is permuted using the row index:

```cuda
const int rl = idx / d4;
const int c4 = idx % d4;

Ktile4[
    rl * d4 + swizzle(rl, c4)
] = Kb4[(size_t)r * d4 + c4];
```

For example, consider the same logical `float4` column `c4 = 0` across eight K rows.

Without swizzling, all values are stored at the same physical column:

```text
          physical float4 column
          0  1  2  3  4  5  6  7

row 0     X
row 1     X
row 2     X
row 3     X
row 4     X
row 5     X
row 6     X
row 7     X
```

With `swizzle(r, c4) = c4 ^ (r % 8)`,  
the same logical column is distributed across different physical shared-memory columns:

```text
          physical float4 column
          0  1  2  3  4  5  6  7

row 0     X
row 1        X
row 2           X
row 3              X
row 4                 X
row 5                    X
row 6                       X
row 7                          X
```



The logical data is unchanged; only its physical location in shared memory changes.

## Swizzled QKᵀ access

The same mapping is applied when reading $K$:

```cuda
const float4* k4 =
    reinterpret_cast<const float4*>(Ktile) + lane * d4;

for (int k = 0; k < d4; ++k) {
    const float4 a = q4[k];
    const float4 b = k4[swizzle(lane, k)];

    dot += a.x * b.x
         + a.y * b.y
         + a.z * b.z
         + a.w * b.w;
}
```

For a group of eight lanes, `lane % 8` maps the accesses to different `float4` positions.

Since one `float4` spans four 4-byte banks, eight distinct `float4` positions can cover all 32 shared-memory banks.

The swizzle therefore distributes the K accesses across banks instead of repeatedly mapping them to the same bank group.

## Why only K is swizzled

The problematic access pattern occurs when reading $K$ for $QK^\top$.

For each dot-product iteration:

* all lanes read the same Q value, which can use shared-memory broadcast
* lanes read different K rows at the same column, which causes the **conflict**
* V is read across output dimensions during $PV$, giving a different access pattern

Therefore, this step swizzles only the $K$ tile.

The Q and V layouts remain unchanged.

## Nsight Compute summary

Nsight Compute confirms that the XOR layout removes the dominant shared-load conflict from Step 05.

```text
Shared-load bank conflicts    30.07B →   0.25M
Shared-load wavefronts        40.80B →  10.74B
MIO-throttle stall              11.2 →    6.8 cycles / issued inst.
Instruction-issue interval      ~4.1 →   ~2.0 cycles
Eligible warps / scheduler       0.53 →   1.51
SM Throughput                   47.5% →  93.4%
```

The shared-load conflict count falls by more than 99.999%, while the number of shared-load wavefronts falls by approximately 73.7%.

Introducing the swizzled indexing also changes compiler code generation.

```cuda
// Step 05
k4[k]

// Step 06
k4[k ^ (lane % 8)]
```

In this build, register usage decreases from 65 to 64 registers per thread,  
crossing an allocation boundary and increasing achieved occupancy from 49.9% to 66.5%.
This is an implementation-specific side effect rather than a general property of XOR swizzling.

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 06](ncu/06_swizzling.md)

## Remaining bottlenecks

The dominant shared-load conflicts are removed, but the kernel is not yet compute-efficient.

In particular:

* MIO throttle remains the largest warp-stall category
* smaller shared-store bank-conflict cost remains
* Q, K, V, and O are still stored in FP32
* $QK^\top$ and $PV$ still use conventional FP32 multiply-add loops
* no Tensor Cores are used
* tile loading and computation are still serialized

The profile reports approximately 205.1 million shared-store bank conflicts, compared with only 0.25 million shared-load conflicts.  
This is far smaller than the 30.07 billion load conflicts removed by swizzling,  
but shared stores are now the main remaining bank-conflict issue reported by Nsight Compute.

## Conclusion

Step 06 changes the physical K-tile layout without changing the attention computation:

```text
row-major K tile
        ↓
XOR-swizzled float4 columns
        ↓
different bank group per lane
        ↓
shared-load conflicts effectively eliminated
        ↓
fewer wavefronts and lower MIO stalls
```

Step 07 changes the Q/K/V/O storage path to FP16 while retaining FP32 accumulation,  
reducing the amount of data moved through global and shared memory.
