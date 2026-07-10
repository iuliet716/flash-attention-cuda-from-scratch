# Step 1. cuBLAS GEMM

## What this step implements

In this step, we replace the kernels for $QK^\top$ and $PV$ matrixs multiplication with cuBLAS library calls.  

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

## Code

### `gemm_qk`

[NVIDIA Docs](https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmstridedbatched) describe `cublasSgemmStridedBatched()`.  

cuBLAS assumes column-major layout by default, while we store tensors in row-major layout.  
Therefore, the operands are passed in reversed order with appropriate transpose flags.  

**Conceptually**, we want to compute the following **row-major** attention score matrix:  

$$S_{\text{row}} = \text{scale} \cdot Q_{\text{row}} K_{\text{row}}^\top$$

**However**, a row-major matrix is interpreted by cuBLAS as a column-major matrix.  
Therefore, instead of directly calling GEMM as $QK^\top$, **we pass $K$ first and $Q$ second, and use transpose flags**:  

$$\text{scale} \cdot K_{\text{row}} Q_{\text{row}}^\top$$

This is not a problem because the cuBLAS output buffer is interpreted as a row-major matrix in our implementation.  
The column-major result written by cuBLAS corresponds to the transpose of the row-major output:

$$S_{\text{row}} = C_{\text{col}}^\top = \left(\text{scale} \cdot K_{\text{row}} Q_{\text{row}}^\top\right)^\top = \text{scale} \cdot Q_{\text{row}} K_{\text{row}}^\top$$

```CUDA
cublasStatus_t gemm_qk(
    cublasHandle_t handle,
    const float* dQ, const float* dK, float* dS,
    int N, int d, float scale, int batch_count)
{
    const float beta = 0.0f;
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
}
```

### `gemm_pv`

This kernel computes the final attention output:

$$O_{\text{row}} = P_{\text{row}} V_{\text{row}}$$

Because the row-major output is interpreted as the transpose of the cuBLAS column-major result, the cuBLAS call computes:

$$C_{\text{col}} = V_{\text{row}}^\top P_{\text{row}}^\top = \left(P_{\text{row}} V_{\text{row}}\right)^\top$$

which corresponds to the desired row-major output $O_{\text{row}}$.  

```CUDA
cublasStatus_t gemm_pv(
    cublasHandle_t handle,
    const float* dP, const float* dV, float* dO,
    int N, int d, int batch_count)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
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
}
```

## Measurements
