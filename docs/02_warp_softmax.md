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

## Measurements

### Nsight Compute profile

A full Nsight Compute profile was collected on RTX 5090 for the representative FP32 shape `B=8, H=16, N=4096, d=64`.

| Kernel       | SM Throughput | L1/TEX |    L2 |  DRAM | Main signal                                                        |
| ------------ | ------------: | -----: | ----: | ----: | ------------------------------------------------------------------ |
| cuBLAS `QKᵀ` |         24.1% |  29.0% | 45.2% | 22.5% | same optimized GEMM behavior as Step 01                            |
| Warp softmax |          4.2% |   8.9% | 20.5% | 35.1% | high occupancy, but global-memory instruction throttling dominates |
| cuBLAS `PV`  |         25.2% |  29.0% | 45.7% | 22.4% | similar optimized GEMM behavior                                    |

#### Warp softmax

The warp-level implementation removes the severe thread-level serialization of the previous softmax kernel.

Compared with Step 01, softmax SM throughput increases from about **0.9% to 4.2%**, while DRAM throughput increases from roughly **13.6% to 35.1%**.  
Nsight Compute also reports **zero excessive global sectors**, confirming that the lane mapping produces coalesced global-memory accesses.

Occupancy is no longer the limiting factor.  
The kernel reaches about **96.1% achieved occupancy** against **100% theoretical occupancy**, with approximately **11.64 active warps per scheduler**.

However, only about **0.06 warps per scheduler are eligible to issue each cycle**, and each scheduler issues roughly one instruction every **75 cycles**.

Nsight Compute identifies the dominant stall as **LG throttle**:  
each warp spends about **630 cycles per issued instruction** waiting for the local/global-memory instruction queue,  
accounting for roughly **72%** of the cycles between issued instructions.

This is consistent with the kernel structure.
Even though accesses are now coalesced, conventional softmax still scans the full `(N × N)` score matrix multiple times:

1. read once to compute the row maximum,
2. read and write once to compute exponentials and the row sum,
3. read and write once more for normalization.

For the profiled shape, the FP32 score matrix alone is **8 GiB**, so these three phases generate about **40 GiB of logical score-matrix traffic** inside softmax.

Therefore, the kernel is **memory-path limited rather than compute-limited**.  
This does not mean that peak DRAM bandwidth is saturated—the measured DRAM throughput is only about 35% of peak.  
Instead, frequent global load/store instructions leave very few warps ready to issue arithmetic instructions, which keeps SM throughput low despite high occupancy.

#### Overall

Warp-level reduction solves the most obvious parallelization and coalescing problems of the naive softmax implementation.

The remaining profile exposes the next limitation more clearly: the algorithm still repeatedly streams the materialized `N × N` score matrix through global memory.

Thus Step 02 improves how softmax accesses the matrix, but it does not remove the **O(N²) intermediate-memory traffic**.
This motivates the online-softmax formulation in Step 03 and,  
ultimately, fusing attention so that the score matrix no longer needs to be materialized and repeatedly transferred through global memory.

