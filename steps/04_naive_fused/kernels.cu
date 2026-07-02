#include "kernels.cuh"

#include <float.h>
#include <math.h>

#include <cuda_runtime.h>

// Fused attention: O = softmax(scale * QK^T) V in a single kernel.
//
// The N x N score matrix never touches HBM. K and V are streamed through
// shared memory (SRAM) in tiles of Bc rows, and each query row folds one
// tile at a time into its running softmax state (m, l) and output
// accumulator using the online-softmax merge rule from step 03:
//   m_new = max(m, m_tile)
//   l     = l * exp(m - m_new) + l_tile
//   acc   = acc * exp(m - m_new) + P_tile * V_tile
//
// Work split:
// - one thread block per Br query rows of one (batch, head)
// - one warp per query row; its (m, l) state and output accumulator
//   (d floats, strided across the 32 lanes) live in registers
// - within a tile, lane c owns key column c: it computes the full
//   dot product q . k_c and the unnormalized probability p_c

constexpr int BR = 8;                  // Q rows per block (= warps per block)
constexpr int BC = 32;                 // K, V rows per tile (= warp size)
constexpr int ACC = FUSED_D_MAX / 32;  // output accumulator floats per lane

__global__ void fused_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int d,
    float scale)
{
    // shared tiles:
    // Q (Br x d) loaded once
    // K, V (Bc x d) reloaded per tile
    extern __shared__ float smem[];
    float* Qtile = smem;               // BR * d
    float* Ktile = Qtile + BR * d;     // BC * d
    float* Vtile = Ktile + BC * d;     // BC * d

    const int batch = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * BR + warp;      // query row of this warp
    const bool row_valid = row < N;              // warp-uniform

    const float* Qb = Q + (size_t)batch * N * d;
    const float* Kb = K + (size_t)batch * N * d;
    const float* Vb = V + (size_t)batch * N * d;
    float* Ob = O + (size_t)batch * N * d;

    // cooperative Q tile load (rows past N are never read back, skip them)
    const int q_base = blockIdx.x * BR;
    for (int idx = threadIdx.x; idx < BR * d; idx += blockDim.x) {
        const int r = q_base + idx / d;
        if (r < N) Qtile[idx] = Qb[(size_t)r * d + idx % d];
    }

    // per-row online softmax state and output accumulator.
    // lane holds output elements k = lane, lane + 32, ... (ACC of them);
    // -FLT_MAX instead of -inf for the same NaN reason as step 03.
    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[ACC];
    for (int i = 0; i < ACC; ++i) acc[i] = 0.0f;

    for (int tile = 0; tile < N; tile += BC) {
        // cooperative K, V tile load; zero-pad rows past N so the
        // PV accumulation below can run without bounds checks
        for (int idx = threadIdx.x; idx < BC * d; idx += blockDim.x) {
            const int r = tile + idx / d;
            const int c = idx % d;
            const bool in = r < N;
            Ktile[idx] = in ? Kb[(size_t)r * d + c] : 0.0f;
            Vtile[idx] = in ? Vb[(size_t)r * d + c] : 0.0f;
        }
        __syncthreads();

        if (row_valid) {
            // lane's score vs. its key column. 
            // Q tile reads: broadcast (whole warp reads the same address)
            // K tile reads: 32-way bank conflicts (stride d) -> fixed in step 06
            const int key = tile + lane;
            float s = -FLT_MAX;
            if (key < N) {
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot += Qtile[warp * d + k] * Ktile[lane * d + k];
                }
                s = dot * scale;
            }

            // tile max, then merge into the running state
            float m_tile = s;
            for (int offset = 16; offset > 0; offset >>= 1) {
                m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, offset));
            }
            const float m_new = fmaxf(m, m_tile);

            // unnormalized probability of lane's key column
            const float p = (key < N) ? __expf(s - m_new) : 0.0f;
            float l_tile = p;
            for (int offset = 16; offset > 0; offset >>= 1) {
                l_tile += __shfl_xor_sync(0xffffffff, l_tile, offset);
            }

            // rescale old state to the new max, then fold the tile in
            const float alpha = __expf(m - m_new);
            l = l * alpha + l_tile;
            m = m_new;
            for (int i = 0; i < ACC; ++i) acc[i] *= alpha;

            // acc += P_tile * V_tile: broadcast p_c from lane c
            // every lane accumulates its strided slice of the output row
            // consecutive lanes read consecutive Vtile addresses -> no conflicts
            for (int c = 0; c < BC; ++c) {
                const float pc = __shfl_sync(0xffffffff, p, c);
                for (int i = 0; i < ACC; ++i) {
                    const int k = lane + 32 * i;
                    if (k < d) acc[i] += pc * Vtile[c * d + k];
                }
            }
        }
        __syncthreads();  // compute done before the next tile overwrites K, V
    }

    // normalization + write the output (first and only HBM write)
    if (row_valid) {
        const float inv_l = 1.0f / l;
        for (int i = 0; i < ACC; ++i) {
            const int k = lane + 32 * i;
            if (k < d) Ob[(size_t)row * d + k] = acc[i] * inv_l;
        }
    }
}

void launch_fused_attention(
    const float* dQ, const float* dK, const float* dV, float* dO,
    int N, int d, float scale, int batch_count, cudaStream_t stream)
{
    const int threads = BR * 32;
    const dim3 grid((N + BR - 1) / BR, batch_count);
    const size_t smem_bytes = (size_t)(BR + 2 * BC) * d * sizeof(float);
    fused_attention_kernel<<<grid, threads, smem_bytes, stream>>>(
        dQ, dK, dV, dO, N, d, scale);
}
