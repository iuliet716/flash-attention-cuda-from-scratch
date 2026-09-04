# Step 2. Warp-reduction Softmax

## What this step implements

Replace the single-thread-per-row softmax with a **one-warp-per-row** implementation.

## Warp-level reduction

A warp consists of **32 threads** that are scheduled and executed as a group.

CUDA warp shuffle instructions allow lanes to exchange register values directly without using shared memory.

This implementation uses `__shfl_xor_sync()` to perform the reductions.  
```cuda
T __shfl_xor_sync(unsigned mask, T var, int laneMask);

// mask — which lanes participate; 0xFFFFFFFF means all 32 lanes.
// var — the register value this lane contributes.
// laneMask — each lane receives var from lane laneId ^ laneMask.
```

```cuda
for (int offset = 16; offset > 0; offset >>= 1) {
    max_val = fmaxf(
        max_val,
        __shfl_xor_sync(0xffffffff, max_val, offset));
}
```

## Design
Each warp owns one score row, and each lane processes a strided subset:

```cuda
for (int j = lane; j < N; j += 32) {
    max_val = fmaxf(max_val, row[j]);
}
```

Compared with the previous one-thread-per-row implementation, this provides:

- **Row-level parallelism** — 32 lanes divide the work within each row.
- **Coalesced memory access** — adjacent lanes access adjacent elements.
- **Register-level reduction** — partial results are combined with warp shuffles without shared memory or block-wide synchronization.

The same pattern is used for the exponential sum:

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

## Nsight Compute summary

For `B=8, H=16, N=4096, d=64`:

| Kernel       | Main observation                                         |
| ------------ | -------------------------------------------------------- |
| cuBLAS `QKᵀ` | 24.1% SM throughput with similar behavior to Step 01     |
| Warp softmax | 4.2% SM throughput and ~0.06 eligible warps/scheduler    |
| cuBLAS `PV`  | 25.2% SM throughput with similar optimized GEMM behavior |

Warp-level parallelism improves SM utilization and memory access efficiency.

However, low eligible warps and LG-throttle stalls indicate that the **global memory instruction path still limits performance.**

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 02](ncu/02_warp_softmax.md)

## Conclusion

Warp-level reduction improves softmax by distributing each row across 32 lanes and using warp shuffles for reduction.

However, the full $N \times N$ score matrix is still materialized and repeatedly accessed in global memory.

Step 03 introduces online softmax, preparing for later fused steps that eliminate this intermediate matrix.
