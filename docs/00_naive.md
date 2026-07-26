# Step 0. Naive Standard Attention (Baseline)

## What this step implements

Three separate kernels, with the intermediate `S = scale·QKᵀ` and `P = softmax(S)` matrices stored in HBM.

## Code

### `naive_qk_kernel`

Calculate $QK^\top$.  

We assign each thread for an element of score matrix $S$.

$S[i][j] = \sum_{k}Q[i][k] \cdot K[j][j]$  

```cuda
float acc = 0.0f;
for (int k = 0; k < d; ++k) {
    acc += Qb[row * d + k] * Kb[col * d + k];
}
```

The $(N \times N)$ score matrix is computed by **(16×16) thread blocks**.  
It provides a **simple 2D mapping** between threads and score elements.

```cuda
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x,
          (N + block.y - 1) / block.y,
          batch_count);
naive_qk_kernel<<<grid, block, 0, stream>>>(dQ, dK, dS, N, d, scale);
```

### `naive_softmax_kernel`

Calculate Softmax of score matrix $S$.  

A **256-thread linear block** provides **8 warps** that cooperatively perform the **row-wise reduction**.

```cuda
const int total_rows = batch_count * N;
const int threads = 256;
const int blocks = (total_rows + threads - 1) / threads;
naive_softmax_kernel<<<blocks, threads, 0, stream>>>(dS, N, total_rows);
```

### `naive_pv_kernel`

Calculate $PV$.  

A **(32×8) thread block** maps each warp to **32 contiguous output dimensions** of one row, enabling **coalesced memory access**.  

>For $(d \neq 32)$, **warp underutilization or extra block overhead** may reduce efficiency.  
>Therefore, the block configuration can be further tuned for different head dimensions.  

```cuda
dim3 block(32, 8);
dim3 grid((d + block.x - 1) / block.x,
          (N + block.y - 1) / block.y,
          batch_count);
naive_pv_kernel<<<grid, block, 0, stream>>>(dP, dV, dO, N, d);
```

## Measurements

For a more detailed analysis, we use NVIDIA Nsight Compute (NCU).

### Kernel breakdown

<img width="800" height="350" alt="image" src="assets/1_kernel_breakdown.png" />

### HBM traffic — why this is the bottleneck

<img width="800" height="298" alt="image" src="assets/2_hbm.png" />

The kernels communicate through HBM:

1. `S` (N×N) is written by `naive_qk_kernel`
2. `S` (N×N) is re-read ×3 by `softmax_kernel` (max, sum, normalize)
3. `S` (N×N) is read again by `naive_pv_nernel`

Naive attention materializes N×N score/probability matrices, causing $O(N²)$ off-chip memory traffic.  
The gray line shows the ideal $O(N)$ I/O lower bound, not the compute complexity of attention.

<br>

> [!NOTE]
*In FlashAttention literature, “HBM traffic” refers to traffic to off-chip device memory.  
On RTX 5090, the corresponding off-chip device memory is GDDR7, so we measure it as GDDR7-based device-memory traffic.
