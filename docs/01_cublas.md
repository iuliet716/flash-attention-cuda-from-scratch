# Step 1. cuBLAS GEMM

## What this step implements

In this step, we replace the kernels for $QK^\top$ and $PV$ matrix multiplication with cuBLAS library calls.  

### What is cuBLAS

cuBLAS is NVIDIA's GPU-accelerated Basic Linear Algebra Subprograms library.  
It provides **highly optimized** implementations of common **linear algebra operations**. 

By using cuBLAS, we can easily leverage highly optimized matrix multiplication kernels without implementing them manually.  

### Why cuBLAS is fast

These kernels are carefully **tuned for NVIDIA GPUs** and improve data reuse through tiling, shared memory, and registers.  
They also optimize memory access through coalesced access patterns and leverage Tensor Cores when applicable.  

### Why `cublasSgemmStridedBatched()` is used in this code

For each of the $QK^\top$ and $PV$ computations, self-attention repeats the same GEMM shape independently for each batch and head.  

Since all batch/head matrices have the **same shape** and are stored with a **regular memory stride**,  
we can use strided batched GEMM to compute all of them with a single cuBLAS call.

### Are there better cuBLAS alternatives?

`cublasGemmStridedBatchedEx()` provides more control over input/output data types, compute precision, and GEMM algorithms.  
`cublasLtMatmul()` provides a more flexible and tunable GEMM interface, but it requires more setup through descriptors and heuristics.  

This step focuses on replacing the naive CUDA matmul kernels with a simple optimized library baseline.  
**More advanced APIs can be explored in later steps** when we introduce mixed precision, Tensor Cores, or more aggressive GEMM tuning.  

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
