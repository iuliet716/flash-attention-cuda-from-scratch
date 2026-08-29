# Step 1. cuBLAS GEMM

## What this step implements

In this step, we replace the kernels for $QK^\top$ and $PV$ matrix multiplication with cuBLAS library calls.  

### Why cuBLAS

cuBLAS provides highly optimized GPU implementations of linear algebra operations such as GEMM.

Compared with the naive kernels in Step 0, cuBLAS GEMMs improve data reuse and memory access through optimized tiling and GPU-specific kernel tuning.

### Why `cublasSgemmStridedBatched()` is used

Self-attention performs the same GEMM independently for each batch and head.

Because these matrices have the **same shape** and a **regular memory stride**,  
`cublasSgemmStridedBatched()` can process all batch/head matrices with a single cuBLAS call.

More flexible interfaces such as `cublasGemmStridedBatchedEx()` and `cublasLtMatmul()` are available,  
but this step uses the simpler FP32 SGEMM API as an optimized library baseline.

## Row-major mapping

cuBLAS assumes column-major storage, while the attention tensors are stored in row-major layout.

For $QK^\top$, the desired row-major result is

$$
S_{\text{row}} = \mathrm{scale} \cdot Q_{\text{row}}K_{\text{row}}^\top.
$$

The same buffers are interpreted by cuBLAS in column-major form, so the call computes the transposed result:

$$
S_{\text{row}}^\top = K_{\text{row}}Q_{\text{row}}^\top.
$$

This avoids explicit tensor transposes or additional memory copies.

```cuda
return cublasSgemmStridedBatched(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    N, N, d,
    &scale,
    dK, d, (long long)N * d,
    dQ, d, (long long)N * d,
    &beta,
    dS, N, (long long)N * N,
    batch_count);
```

The same layout mapping is used for $PV$:

$$
O_{\text{row}} = P_{\text{row}}V_{\text{row}}
\qquad\Longleftrightarrow\qquad
O_{\text{row}}^\top = V_{\text{row}}^\top P_{\text{row}}^\top.
$$

```cuda
return cublasSgemmStridedBatched(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    d, N, N,
    &alpha,
    dV, d, (long long)N * d,
    dP, N, (long long)N * N,
    &beta,
    dO, d, (long long)N * d,
    batch_count);
```

## Measurements

### Nsight Compute profile

A full Nsight Compute profile was collected on RTX 5090 for the representative FP32 shape `B=8, H=16, N=4096, d=64`.

| Kernel       | SM Throughput | L1/TEX |    L2 |  DRAM | Main signal                                                             |
| ------------ | ------------: | -----: | ----: | ----: | ----------------------------------------------------------------------- |
| cuBLAS `QKᵀ` |         24.9% |  29.1% | 45.5% | 22.3% | naive L1/TEX saturation removed; remaining latency and occupancy limits |
| Softmax      |          0.9% |  16.4% | 44.7% | 13.6% | extremely low eligible-warps rate and serialized memory accesses        |
| cuBLAS `PV`  |         25.2% |  29.0% | 45.5% | 22.3% | similar optimized GEMM behavior                                         |

#### cuBLAS GEMM

Replacing the naive matrix-multiplication kernels with cuBLAS substantially changes the memory behavior.

In Step 0, `QKᵀ` reached almost 100% L1/TEX throughput while DRAM throughput was only 2.6%.  
With cuBLAS, L1/TEX throughput drops to about **29%**, while L2 and DRAM throughput increase to roughly **46% and 22%**.

This does not mean that lower cache utilization is itself the goal.  
Rather, it shows that the pathological L1/TEX saturation of the naive implementation has disappeared.  
The tiled cuBLAS kernel reuses data more effectively instead of repeatedly issuing inefficient scalar accesses.

The GEMM kernels are still not saturating the GPU's peak compute throughput.  
Nsight Compute reports approximately **0.56 eligible warps per scheduler**, with each scheduler issuing roughly one instruction every **4 cycles**.

It also reports a theoretical occupancy of about **58%**, limited by high register usage, while memory latency remains the main source of stalls.

Therefore, the optimized GEMM is no longer dominated by the severe access-pattern problem seen in the naive implementation,  
but this FP32 workload still exhibits latency and resource limitations.

#### Softmax

The softmax kernel is unchanged from Step 0 and retains the same bottleneck.

After optimizing the GEMMs with cuBLAS, softmax becomes the next clear optimization target.

#### Overall

cuBLAS removes the severe memory-access inefficiency of the naive matrix-multiplication kernels and reduces the end-to-end latency.

After this optimization, the remaining bottleneck becomes much clearer:  
 the naive softmax kernel has very low instruction-issue efficiency and continues to perform serialized memory accesses.

The fundamental standard-attention dataflow also remains unchanged:

1. `QKᵀ` materializes the full `(N × N)` score matrix.
2. Softmax repeatedly accesses that matrix.
3. `PV` reads the normalized `(N × N)` matrix again.

Therefore, cuBLAS improves the efficiency of the GEMM operations,  
but it does not eliminate the **O(N²) intermediate-memory traffic** that ultimately motivates fused, IO-aware attention.
