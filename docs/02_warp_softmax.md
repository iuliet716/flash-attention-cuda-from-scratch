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

The warp-level implementation removes the severe thread-level serialization and inefficient memory accesses of the previous softmax kernel.

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 02](ncu/02_warp_softmax.md)

## Conclusion

Warp-level reduction improves softmax by distributing each row across 32 lanes and using warp shuffles for reduction.

However, the full $N \times N$ score matrix is still materialized and repeatedly accessed in global memory.

Step 03 introduces online softmax, while later fused steps address the larger intermediate-memory cost.
