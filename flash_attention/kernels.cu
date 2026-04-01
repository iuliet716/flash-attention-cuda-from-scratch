#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>

#define kBlockM 32         // row tile size (Q)
#define kBlockN 32         // column tile size (K, V)
#define kNThreads 128      // threads per block

static_assert(kNThreads % kBlockM == 0, "kNThreads must be divisible by kBlockM");

namespace {
    constexpr int kThreadsPerRow = kNThreads / kBlockM;
    constexpr int kMaxHeadDim = 128;                                                       // max head dimension
    constexpr int kMaxDPerThread = (kMaxHeadDim + kThreadsPerRow - 1) / kThreadsPerRow;    // max num of head dimension elements handled by a single thread
}

// Split a warp into kThreadsPerRow-thread groups and compute the sum within each group,
// then broadcast that sum to the other threads in the same group
__device__ __forceinline__ float subgroup_sum_kThreadsPerRow(float v)
{
    #pragma unroll
    for (int offset = kThreadsPerRow / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset, kThreadsPerRow);
    }
    return __shfl_sync(0xffffffffu, v, 0, kThreadsPerRow);
}

__global__ void flash_attention_forward_kernel(
    const half* __restrict__ dQ,
    const half* __restrict__ dK,
    const half* __restrict__ dV,
    half* __restrict__ dO,
    int N,
    int d,
    float scale)
{

    const int m_block  = blockIdx.x;          // row tile index
    const int bh_index = blockIdx.y;
    const int tid      = threadIdx.x;

    // Example) kBlockM = 32, kNThreads = 128, kThreadsPerRow = 4
    // Row_00 = [Thread_000, Thread_001, Thread_002, Thread_003]
    // Row_01 = [Thread_004, Thread_005, Thread_006, Thread_007]
    // ...                                                         
    // Row_31 = [Thread_124, Thread_125, Thread_126, Thread_127]
    const int row_in_tile   = tid / kThreadsPerRow;
    const int thread_in_row = tid % kThreadsPerRow;

    const int row_start  = m_block * kBlockM;
    int       valid_rows = kBlockM;
    if (row_start + kBlockM > N) {           // last block
        valid_rows = N - row_start;
    }
    const bool active_row = (row_in_tile < valid_rows);

    const size_t stride_bh = static_cast<size_t>(N) * static_cast<size_t>(d);
    const size_t base_bh = static_cast<size_t>(bh_index) * stride_bh;

    // shared memory layout
    // | Q_tile (kBlockM x d) | K_tile (kBlockN x d) | V_tile (kBlockN x d) |
    extern __shared__ half smem[];
    half* sQ = smem;
    half* sK = sQ + static_cast<size_t>(kBlockM) * d;
    half* sV = sK + static_cast<size_t>(kBlockN) * d;

    // ----------------------------------------------------------------------
    // 1. Load the Q tile to shared memory
    // ----------------------------------------------------------------------
    // source in global memory: dQ[row_start:row_start + valid_rows][:d]
    // the Q tile is flattened into 1D array:
    // idx = 0, 1, ..., valid_rows * d - 1

    // each thread copies multiple elements of the tile, with a stride of kNThreads
    // Example) kBlockM (valid_rows) = 32, kNThreads = 128, d = 64
    // Thread_000 copies: [0, 128, 256, ..., 1920]
    // Thread_001 copies: [1, 129, 257, ..., 1921]
    // ...
    // Thread_127 copies: [127, 255, 383, ..., 2047]

    for (int idx = tid; idx < valid_rows * d; idx += kNThreads) {
        const int row_i = idx / d;              // row index within the tile
        const int d_i   = idx % d;              // head dimension index

        // flattened tile index: idx = row_i * d + d_i
        sQ[idx] = dQ[base_bh + static_cast<size_t>(row_start + row_i) * d + d_i];
    }
    __syncthreads();

    // ----------------------------------------------------------------------
    // 2. Initialize the running state for online softmax
    // ----------------------------------------------------------------------
    // each thread keeps its own online softmax state:
    // m and l track the assigned row
    // acc tracks the assigned output fragment
    float m = -CUDART_INF_F;         // running max
    float l = 0.0f;                  // running exponential sum (denominator; normalization term)
    float acc[kMaxDPerThread];       // running partial sum of the output fragment (numerator)

    #pragma unroll
    for (int i = 0; i < kMaxDPerThread; ++i) {
        acc[i] = 0.0f;
    }

    // ----------------------------------------------------------------------
    // 3. Iterate over K/V column tiles
    // ----------------------------------------------------------------------
    for (int col_start = 0; col_start < N; col_start += kBlockN) {
        int valid_cols = kBlockN;
        if (col_start + kBlockN > N) {       // last block
            valid_cols = N - col_start;
        }
        // Load the K/V tile to shared memory
        // source in global memory:
        // dK[col_start:col_start + valid_cols][:d]
        // dV[col_start:col_start + valid_cols][:d]
        // the K/V tile are flattened into 1D arrays:
        // idx = 0, 1, ..., valid_cols * d - 1

        // each thread copies multiple elements of the tile, with a stride of kNThreads

        for (int idx = tid; idx < valid_cols * d; idx += kNThreads) {
            const int col_i = idx / d;          // column index within the tile
            const int   d_i = idx % d;          // head dimension index

            sK[idx] = dK[base_bh + static_cast<size_t>(col_start + col_i) * d + d_i];
            sV[idx] = dV[base_bh + static_cast<size_t>(col_start + col_i) * d + d_i];
        }
        __syncthreads();

        if (active_row) {
            const int q_row_offset = row_in_tile * d;
            // ----------------------------------------------------------------------
            // 3-1. Compute running max for this row
            // ----------------------------------------------------------------------
            float tile_m = -CUDART_INF_F;

            for (int c = 0; c < valid_cols; ++c) {
                float partial_dot = 0.0f;

                // calculate QK^T tile
                // by iterating over all head dimension indices (di)
                for (int di = thread_in_row; di < d; di += kThreadsPerRow) {
                    partial_dot += __half2float(sQ[q_row_offset + di]) * __half2float(sK[static_cast<size_t>(c) * d + di]);
                }
                const float dot = subgroup_sum_kThreadsPerRow(partial_dot);       // broadcast the final QK^T dot product within the row group
                const float score = dot * scale;
                tile_m = fmaxf(tile_m, score);
            }
            // ----------------------------------------------------------------------
            // 3-2. Compute running exponential sum and partial sum
            // ----------------------------------------------------------------------
            float tile_l = 0.0f;
            float tile_acc[kMaxDPerThread];
            #pragma unroll
            for (int i = 0; i < kMaxDPerThread; ++i) {
                tile_acc[i] = 0.0f;
            }

            for (int c = 0; c < valid_cols; ++c) {
                float partial_dot = 0.0f;

                // calculate QK^T tile
                // by iterating over all head dimension indices (di)
                for (int di = thread_in_row; di < d; di += kThreadsPerRow) {
                    partial_dot += __half2float(sQ[q_row_offset + di]) * __half2float(sK[static_cast<size_t>(c) * d + di]);
                }

                const float dot = subgroup_sum_kThreadsPerRow(partial_dot);       // broadcast the final QK^T dot product within the row group
                const float score = dot * scale;
                const float p = __expf(score - tile_m);

                // only the first thread in the row group updates tile_l for this query row
                if (thread_in_row == 0) {
                    tile_l += p;
                }

                // each thread accumulates p * V for its assigned head-dimension indices
                int acc_idx = 0;
                for (int di = thread_in_row; di < d; di += kThreadsPerRow, ++acc_idx) {
                    tile_acc[acc_idx] += p * __half2float(sV[static_cast<size_t>(c) * d + di]);
                }
            }
    
            tile_l = subgroup_sum_kThreadsPerRow(tile_l);                         // broadcast the updated tile_l within the row group
            // --------------------------------------------------------------
            // 3-3. Merge online softmax
            //   merge the current tile's softmax statistics with the running state.
            //   rescale the running state and the current tile's local softmax statistics to the new row-wise maximum, 
            //   then update the normalizer and accumulator.
            //
            //   new running max = MAX(old running max, current tile row max)
            //
            //   new running exponential sum = exp(old running max  - new running max) * old running exponential sum 
            //                               + exp(current tile row max - new running max) * current tile row exponential sum
            //
            //   new accumulator = exp(old running max  - new running max) * old accumulator
            //                   + exp(current tile row max - new running max) * current tile accumulator
            // --------------------------------------------------------------
            const float new_m = fmaxf(m, tile_m);
            const float alpha = __expf(m - new_m);
            const float beta = __expf(tile_m - new_m);

            l = alpha * l + beta * tile_l;
            m = new_m;

            int acc_idx = 0;
            for (int di = thread_in_row; di < d; di += kThreadsPerRow, ++acc_idx) {
                acc[acc_idx] = alpha * acc[acc_idx] + beta * tile_acc[acc_idx];
            }
        }

        __syncthreads();
    }

    // ----------------------------------------------------------------------
    // 4) Normalize the accumulator with the updated normalizer, and write output.
    // output = accumulator / exponential sum
    // ----------------------------------------------------------------------
    if (active_row) {
        const int q_row = row_start + row_in_tile;
        half* o_ptr = dO + base_bh + static_cast<size_t>(q_row) * d;

        int acc_idx = 0;
        for (int di = thread_in_row; di < d; di += kThreadsPerRow, ++acc_idx) {
            o_ptr[di] = __float2half_rn(acc[acc_idx] / l);
        }
    }
}

void launch_flash_attention_forward(
    const half* dQ,
    const half* dK,
    const half* dV,
    half* dO,
    int N,
    int d,
    float scale,
    int batch_count,
    cudaStream_t stream)
{
    const int num_m_block = (N + kBlockM - 1) / kBlockM;                                    // one block processes one row tile
    dim3 grid(num_m_block, batch_count);
    const size_t shared_mem_bytes = ((kBlockM * d) + 2 * (kBlockN * d)) * sizeof(half);     // Q_tile + (K_tile + V_tile)
                                                                                            // one block handles a fixed Q row block and iterates over multiple K/V column blocks
    flash_attention_forward_kernel<<<grid, kNThreads, shared_mem_bytes, stream>>>(
        dQ,
        dK,
        dV,
        dO,
        N,
        d,
        scale);
}