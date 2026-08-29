# Step 1. cuBLAS GEMM

## What this step implements

This step replaces the naive $QK^\top$ and $PV$ kernels from Step 00 with cuBLAS GEMM.

The attention pipeline remains unchanged:

$$
S = \mathrm{scale} \cdot QK^\top,\qquad
P = \mathrm{softmax}(S),\qquad
O = PV
$$

Only the two matrix multiplications are replaced.

The naive softmax kernel is kept unchanged.

This removes the inefficient scalar GEMM implementations from Step 00 while preserving the standard attention dataflow.

## Strided batched GEMM

Attention performs the same GEMM independently for every batch and head.

Because the matrices have identical shapes and fixed strides,  
this step uses `cublasSgemmStridedBatched()` to process all batch/head matrices in a single call.

The implementation stays in FP32 to provide an optimized GEMM baseline before later attention-specific optimizations.

## Row-major mapping

cuBLAS uses column-major matrix semantics, while the tensors in this project are stored in row-major layout.

Instead of explicitly transposing the tensors, the operands are arranged so that cuBLAS computes the equivalent transposed operation.

### QKᵀ

The desired row-major operation is:

$$
S_{\text{row}} =
\mathrm{scale}\cdot
Q_{\text{row}}K_{\text{row}}^\top.
$$

Its transpose is:

$$
S_{\text{row}}^\top =
\mathrm{scale}\cdot
K_{\text{row}}Q_{\text{row}}^\top.
$$

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

No explicit transpose or additional memory copy is required.

### PV

The same layout mapping is used for $PV$:

$$
O_{\text{row}} =
P_{\text{row}}V_{\text{row}}
$$

which is equivalent to:

$$
O_{\text{row}}^\top =
V_{\text{row}}^\top P_{\text{row}}^\top.
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

## Nsight Compute summary

For `B=8, H=16, N=4096, d=64`:

| Kernel       | Main observation                                             |
| ------------ | ------------------------------------------------------------ |
| cuBLAS `QKᵀ` | naive L1/TEX saturation is removed                           |
| Softmax      | very low eligible-warp rate and serialized execution remain  |
| cuBLAS `PV`  | scalar GEMM is replaced by an optimized tiled implementation |

The GEMM access behavior improves substantially compared with Step 00.

Softmax is unchanged and becomes the next clear kernel-level bottleneck.

The full $(N \times N)$ attention matrix is still materialized between operations, so the standard attention dataflow remains unchanged.

Detailed profiler metrics are documented separately:

→ [Nsight Compute Analysis — Step 01](ncu/01_cublas.md)

## Conclusion

Replacing the naive matrix multiplications with cuBLAS removes their major implementation inefficiencies.

The next step focuses on parallelizing softmax.

The larger $O(N^2)$ intermediate-memory cost remains and is addressed later through fused attention.
