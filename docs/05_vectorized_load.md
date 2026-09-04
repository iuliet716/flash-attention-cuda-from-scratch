# Step 5. Vectorized Load

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

For `B=8, H=16, N=4096, d=64`:

| Metric                               |     Step 04 |     Step 05 |
| ------------------------------------ | ----------: | ----------: |
| SM throughput                        |       27.2% |       47.5% |
| L1/TEX throughput                    |       95.1% |       84.7% |
| Shared-load requests                 |      12.88B |       6.44B |
| Shared-load bank conflicts / request |        10.3 |         4.7 |
| MIO-throttle stall / issued inst.    | 50.3 cycles | 11.2 cycles |
| Achieved occupancy                   |       83.0% |       49.9% |

Vectorized accesses reduce shared-memory requests and MIO pressure, raising SM throughput.

Higher register usage lowers occupancy, but reduced MIO pressure keeps the kernel well utilized.

The shared-memory layout is still conflict-prone, making the remaining bank conflicts the next bottleneck.

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 05](ncu/05_vectorized_load.md)

## Conclusion

Step 05 replaces scalar accesses with float4 vectorization,  
reducing memory-instruction and MIO pressure without changing the fused attention algorithm.

This improves SM utilization, but the shared-memory layout still causes significant bank conflicts.

Step 06 addresses these conflicts with shared-memory swizzling.
