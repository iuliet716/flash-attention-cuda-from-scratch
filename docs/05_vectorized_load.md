# Step 5. Coalescing + Vectorized Load

## What this step implements

Step 04 established the fused tiled attention dataflow, but Q, K, and V were still loaded one `float` at a time.

This step replaces scalar memory accesses with **`float4` vectorized loads**, moving four FP32 values with each 16-byte access.

The attention algorithm itself is unchanged:

$$
QK^\top
\rightarrow
\text{online softmax}
\rightarrow
PV
$$

The optimization focuses only on how data is moved from global memory to shared memory and accessed during the $QK^\top$ computation.

## Vectorized tile loading

In Step 04, Q, K, and V were copied into shared memory one scalar at a time:

```cuda
Qtile[idx] = Qb[...];
Ktile[idx] = Kb[...];
Vtile[idx] = Vb[...];
```

This step views the same memory as arrays of `float4`:

```cuda
const float4* Qb4 = reinterpret_cast<const float4*>(Qb);
const float4* Kb4 = reinterpret_cast<const float4*>(Kb);
const float4* Vb4 = reinterpret_cast<const float4*>(Vb);

float4* Qtile4 = reinterpret_cast<float4*>(Qtile);
float4* Ktile4 = reinterpret_cast<float4*>(Ktile);
float4* Vtile4 = reinterpret_cast<float4*>(Vtile);
```

The head dimension is therefore processed in groups of four elements:

```cuda
const int d4 = d / 4;
```

For the Q tile:

```cuda
for (int idx = threadIdx.x; idx < BR * d4; idx += blockDim.x) {
    const int r = q_base + idx / d4;

    if (r < N)
        Qtile4[idx] = Qb4[(size_t)r * d4 + idx % d4];
}
```

Adjacent threads load consecutive `float4` values, keeping memory access coalesced while loading four floats at once.

The K and V tiles use the same pattern:

```cuda
for (int idx = threadIdx.x; idx < BC * d4; idx += blockDim.x) {
    const int r = tile + idx / d4;
    const int c = idx % d4;
    const bool in = r < N;

    Ktile4[idx] = in ? Kb4[(size_t)r * d4 + c] : zero4;
    Vtile4[idx] = in ? Vb4[(size_t)r * d4 + c] : zero4;
}
```

The tiling strategy remains unchanged:

```text
Q tile:  8 x d
K tile: 32 x d
V tile: 32 x d
```

The only difference is how much memory is read or written at a time.

## Vectorized QKᵀ

The same vectorization is also applied when reading Q and K from shared memory.

Step 04 computed the dot product one scalar at a time:

```cuda
for (int k = 0; k < d; ++k) {
    dot += Qtile[warp * d + k]
         * Ktile[lane * d + k];
}
```

In this step, each iteration loads four values:

```cuda
const float4* q4 =
    reinterpret_cast<const float4*>(Qtile + warp * d);

const float4* k4 =
    reinterpret_cast<const float4*>(Ktile + lane * d);

float dot = 0.0f;

for (int k = 0; k < d4; ++k) {
    const float4 a = q4[k];
    const float4 b = k4[k];

    dot += a.x * b.x
         + a.y * b.y
         + a.z * b.z
         + a.w * b.w;
}
```

The arithmetic is still ordinary FP32 multiply-add operations.

Vectorization does **not** introduce Tensor Cores or change the mathematical computation.  
It reduces the number of load operations required to provide the same dot-product operands.

## Memory alignment

A `float4` occupies 16 bytes, so vectorized accesses require 16-byte-aligned addresses.

The dynamic shared-memory region is explicitly aligned:

```cuda
extern __shared__ __align__(16) float smem[];
```

The head dimension must also be divisible by four:

```cpp
TORCH_CHECK(
    d % 4 == 0,
    "head dim must be a multiple of 4, got ",
    d
);
```

With `d` divisible by four, each row remains 16-byte aligned for `float4` access.

## Nsight Compute summary

Nsight Compute confirms that the `float4` accesses reduce memory-instruction pressure substantially.

Compared with Step 04:

```text
Shared-load requests          12.88B →  6.44B
Shared-load wavefronts       146.09B → 40.80B
MIO-throttle stall             50.3 → 11.2 cycles / issued inst.
Instruction-issue interval      7.2 →  4.1 cycles
SM Throughput                  27.2% → 47.5%
```

Global loads use all 32 bytes of each memory sector, and Nsight Compute reports zero excessive global sectors.

Vectorization increases register usage from 40 to 65 registers per thread,  
reducing achieved occupancy from approximately 83% to 50%. Despite the lower occupancy,  
the kernel is faster because each resident warp spends much less time stalled on the memory-instruction path.

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 05](ncu/05_vectorized_load.md)

## Remaining bottlenecks

Although memory accesses are now vectorized, the matrix operations are still implemented with conventional FP32 arithmetic.

In particular:

* $QK^\top$ still uses explicit multiply-add loops
* $PV$ still uses scalar FP32 accumulation
* no Tensor Cores are used
* shared-memory access patterns are not yet optimized for bank conflicts
* tile loading and computation are still serialized

The profile still reports an average 6.3-way shared-load bank conflict, with approximately 73.7% of shared-load wavefronts associated with conflicts.

The next step addresses these **shared-memory bank conflicts** by changing the shared-memory layout with swizzling.

## Conclusion

Step 05 changes the access granularity without changing the fused attention algorithm:

```text
scalar loads
    ↓
coalesced float4 loads
    ↓
fewer memory instructions
    ↓
lower MIO pressure
```

However, vectorization reduces the amount of shared-memory work without fixing the row-major bank-mapping problem itself.

Step 06 introduces shared-memory swizzling to address the remaining bank conflicts.
