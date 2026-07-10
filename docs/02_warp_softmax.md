# Step 2. Warp-reduction Softmax

## What this step implements

In this step, we replace the naive row-wise softmax kernel with a **warp-level reduction softmax** kernel.

## What is Warp-level Reduction Softmax

A **warp** is a **group of 32 threads** that execute the same instruction at the same time on NVIDIA GPUs.  

A warp-level reduction softmax **assigns one warp per row** and performs both **reductions** — row max and exp-sum — using **warp shuffles**.  

## Wait, What's a Warp Shuffle

A **warp shuffle** is an instruction that **lets threads in the same warp read each other's register values directly**,  
**without going through shared or global memory**.  

In this implementation, we use `__shfl_xor_sync()` to perform the reductions.  
```cuda
T __shfl_xor_sync(unsigned mask, T var, int laneMask);

// mask — which lanes participate; 0xFFFFFFFF means all 32 lanes.
// var — the register value this lane contributes.
// laneMask — each lane receives var from lane laneId ^ laneMask.
```

Because XOR is symmetric, lanes exchange values in pairs:  
with laneMask = 16, lane 0 swaps with lane 16, lane 1 with lane 17, and so on — every lane both sends and receives in one instruction.  

Calling it repeatedly with laneMask = 16, 8, 4, 2, 1 forms a butterfly reduction.  
After the first step, each lane holds the combined value from 2 lanes; after the second, from 4;  
after all 5 steps (log₂ 32), every lane holds the result of all 32 lanes, so no separate broadcast is needed.  
This is an **efficient $O(\log N)$ scheme** — the step count grows logarithmically, not linearly.  

## So, Why is this faster than the naive softmax kernel

The naive kernel assigns one thread per row, so a single thread walks all N elements sequentially — three times (max, sum, normalize).  
The warp-level version improves on this in three ways:  

### 1. Parallelism within a row
32 lanes split the row, so each lane touches only N/32 elements, and the partial results are merged in just 5 shuffle steps.  
The per-row critical path drops from N sequential steps to roughly N/32 + 5.

### 2. Coalesced memory access
In the naive kernel, adjacent threads read different rows — addresses N elements apart — so most of each memory transaction is wasted.  
Here, adjacent lanes read adjacent elements of the same row, so every load is fully coalesced.

### 3. No shared memory, no barriers
The obvious way to parallelize a row is a block-level reduction: stage partial results in shared memory and synchronize with `__syncthreads()`.  
However, our kernel takes a different route: warp shuffles.  
They skip both — values move register to register, the fastest storage on the GPU, with no bank conflicts and no block-wide barriers.

## Code

### `warp_softmax_kernel()`

#### indexing

```cuda
    const int warps_per_block = blockDim.x / 32;
    const int warp = blockIdx.x * warps_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
```

Each warp owns one row.  
`warp` is the global warp index — which doubles as the row index — and lane (0–31) is the thread's position within the warp.  

#### max reduction

```cuda
    float max_val = -CUDART_INF_F;
    for (int j = lane; j < N; j += 32) {
        max_val = fmaxf(max_val, row[j]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
```

Each lane scans a strided slice of the row (`lane, lane+32, …`) for a local max, then 5 butterfly shuffles merge the 32 partials.  
After this, every lane holds the row max.

#### exp + sum reduction

```cuda
    float sum = 0.0f;
    for (int j = lane; j < N; j += 32) {
        float e = __expf(row[j] - max_val);
        row[j] = e;
        sum += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
```

Each lane computes `exp(x − max)` for its slice, writes it back in place for the final pass to reuse, and accumulates a local sum.  
The same butterfly pattern then reduces the partial sums.

## Measurements
