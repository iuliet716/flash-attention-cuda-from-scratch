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

TEST

```cuda
__global__ void warp_softmax_kernel(
    float* __restrict__ S,
    int N,
    int total_rows)
{
    // one warp for softmax by row
    // consecutive lanes read consecutive elements at each iteration,
    // so accesses are coalesced (each lane strides by 32 across iterations)
    const int warps_per_block = blockDim.x / 32;
    const int warp = blockIdx.x * warps_per_block + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= total_rows) return;

    float* row = S + (size_t)warp * N;

    // max by row: per-lane partial max, then warp reduction
    float max_val = -CUDART_INF_F;
    for (int j = lane; j < N; j += 32) {
        max_val = fmaxf(max_val, row[j]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }

    // exp(x - max) written in place, per-lane partial sum, then warp reduction
    float sum = 0.0f;
    for (int j = lane; j < N; j += 32) {
        float e = __expf(row[j] - max_val);
        row[j] = e;
        sum += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // normalization
    const float inv_sum = 1.0f / sum;
    for (int j = lane; j < N; j += 32) {
        row[j] *= inv_sum;
    }
}
```

## Measurements
