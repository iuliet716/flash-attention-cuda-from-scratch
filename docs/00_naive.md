# Step 0. Naive Standard Attention (Baseline)

## What this step implements

A straightforward CUDA implementation of standard attention:

$$
S = \mathrm{scale} \cdot QK^\top,\qquad
P = \mathrm{softmax}(S),\qquad
O = PV
$$

The computation is split into three separate kernels, with the full $(N \times N)$ attention matrix materialized in device memory.

## Baseline design

### QKᵀ

Each thread computes one element of the score matrix:

```cuda
float acc = 0.0f;
for (int k = 0; k < d; ++k) {
    acc += Qb[row * d + k] * Kb[col * d + k];
}
Sb[row * N + col] = acc * scale;
```

No shared-memory tiling or data reuse is applied.

### Softmax

Each thread processes one complete row sequentially:

```cuda
for (int j = 0; j < N; ++j)
    max_val = fmaxf(max_val, row[j]);

for (int j = 0; j < N; ++j)
    sum += __expf(row[j] - max_val);

for (int j = 0; j < N; ++j)
    row[j] = __expf(row[j] - max_val) / sum;
```

This requires three passes over $N$ elements for max, exponential sum, and normalization.

### PV

Each thread computes one output element:

```cuda
float acc = 0.0f;
for (int k = 0; k < N; ++k) {
    acc += Pb[row * N + k] * Vb[k * d + col];
}
```

Again, no tiling or reuse of `P` and `V` is applied.

## Measurements

### Nsight Compute profile

A full Nsight Compute profile was collected on RTX 5090 for the representative FP32 shape `B=8, H=16, N=4096, d=64`.

| Kernel  | SM Throughput | L1/TEX |    L2 |  DRAM | Main signal                                                       |
| ------- | ------------: | -----: | ----: | ----: | ----------------------------------------------------------------- |
| `QKᵀ`   |         22.5% |  99.6% | 10.4% |  2.6% | L1/TEX saturation and inefficient global-memory accesses          |
| Softmax |          1.0% |  18.4% | 37.2% | 12.8% | very low eligible-warps rate and highly serialized execution      |
| `PV`    |         90.0% |  90.4% | 15.2% |  9.1% | high utilization, but significant memory-dependency stalls remain |

#### QKᵀ

`QKᵀ` reaches almost **100% L1/TEX throughput while DRAM throughput is only 2.6%**.

This does not indicate a DRAM-bandwidth bottleneck.  
Instead, the naive thread mapping repeatedly accesses the same `Q` elements and poorly coalesced `K` elements without explicit shared-memory reuse.

As a result, the **L1/TEX path becomes saturated well before DRAM bandwidth is fully utilized**.

Nsight Compute explicitly flags the global-load access pattern as inefficient:

> "Only 4 of 32 bytes per sector are utilized," with 66% excessive sectors.

This motivates explicit tiling and data reuse rather than relying on the hardware cache hierarchy.

#### Softmax

Softmax shows only **1.0% SM throughput**, despite relatively higher L1/L2 activity.

The main issue is the one-thread-per-row mapping.  
Each thread scans the same row three times for max, exponential sum, and normalization, while neighboring threads in a warp access different rows with a large stride.

Nsight Compute shows that very few warps are eligible to issue instructions, indicating that execution is dominated by latency and serialization rather than raw memory bandwidth.

> Only about 0.02 warps per scheduler are eligible to issue, with roughly 234 cycles between issued instructions.

This motivates parallelizing each row across a warp and using warp-level reductions.

#### PV

`PV` reaches about **90% SM and L1/TEX throughput**, making it the most efficiently utilized of the three naive kernels.

Its `(32, 8)` thread-block layout gives `V` a more favorable access pattern across each warp,  
but the kernel still performs a long scalar dot-product loop without shared-memory tiling or explicit data reuse.

Nsight Compute shows that memory dependencies remain a major source of stalls.

> Long-scoreboard stalls account for about 58% of the cycles between issued instructions.

This motivates replacing the naive scalar matmul with a tiled GEMM implementation that improves reuse and hides memory latency.

#### Overall

The three naive kernels expose different bottlenecks:

* `QKᵀ`: L1/TEX saturation from redundant and uncoalesced global-memory accesses.
* Softmax: low instruction issue efficiency from sequential per-thread row processing.
* `PV`: high utilization, but still limited by memory-dependency stalls and lack of explicit data reuse.

Together, these results show that the baseline is limited not by a single hardware resource, but by inefficient memory access patterns, insufficient parallelism, and lack of locality.

Nsight Compute also confirms that the kernel is memory-bound.  
L1/TEX throughput reaches 99.6% while SM throughput remains at 22.5%, with LG-throttle stalls dominating execution.

### $O(N^2)$ Memory Access

<img width="800" height="298" alt="Device-memory traffic" src="assets/2_hbm.png" />

For each attention operation:

1. `QKᵀ` writes the $(N \times N)$ score matrix.
2. Softmax reads the score matrix repeatedly for max, sum, and normalization.
3. `PV` reads the normalized $(N \times N)$ matrix again.

This produces $O(N^2)$ device-memory traffic in addition to the attention computation itself.

> Following FlashAttention terminology, HBM traffic refers to off-chip device-memory traffic; on RTX 5090, this is GDDR7.
