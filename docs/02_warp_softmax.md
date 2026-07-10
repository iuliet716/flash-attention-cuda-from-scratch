# Step 0. Warp-reduction Softmax

## What this step implements



### `warp_softmax_kernel()`

```CUDA
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
