#include "kernels.cuh"

#include <float.h>
#include <math.h>

#include <cuda_runtime.h>

constexpr int BR = 8;                  // Q rows per block (= warps per block)
constexpr int BC = 32;                 // K, V rows per tile (= warp size)
constexpr int ACC = FUSED_D_MAX / 32;  // output dimensions per lane

__global__ void fused_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int d,
    float scale)
{
    // Tiling: Q_tile x (K_tile)^T -> QK^T computed in SRAM
    //
    //   +-------+        +------+-------+------------------+
    //   |       |        |      :       :                  |
    //   |   Q   |        |      : K tile:        K         |
    //   +.......+        |      :       :                  |
    //   : Q tile:        +------+.......+------------------+
    //   +.......+  \            |
    //   |       |   \           v
    //   |       |    \   +===============================+
    //   |       |     \  |   +......+                    |
    //   |       |      ->|   : QK^T :                    |
    //   +-------+        |   +......+                    |
    //                    |  SRAM                         |
    //                    +===============================+
    //
    //
    // SRAM layout:
    // Q_0, Q_1, ..., Q_7 | K_0, K_1, ... K_31 | V_0, V_1, ... V_31 |
    //       Q tile       |       K tile       |       V tile       |
    extern __shared__ float smem[];
    float* Qtile = smem;               
    float* Ktile = Qtile + BR * d;     
    float* Vtile = Ktile + BC * d;     

    const int batch = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * BR + warp;

    const float* Qb = Q + (size_t)batch * N * d;
    const float* Kb = K + (size_t)batch * N * d;
    const float* Vb = V + (size_t)batch * N * d;
    float* Ob = O + (size_t)batch * N * d;

    // cooperative Q tile load
    const int q_base = blockIdx.x * BR;
    for (int idx = threadIdx.x; idx < BR * d; idx += blockDim.x) {
        const int r = q_base + idx / d;
        if (r < N) Qtile[idx] = Qb[(size_t)r * d + idx % d];
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[ACC];
    for (int i = 0; i < ACC; ++i) acc[i] = 0.0f;

    for (int tile = 0; tile < N; tile += BC) {
        // cooperative K, V tile load
        for (int idx = threadIdx.x; idx < BC * d; idx += blockDim.x) {
            const int r = tile + idx / d;
            const int c = idx % d;
            const bool in = r < N;
            Ktile[idx] = in ? Kb[(size_t)r * d + c] : 0.0f;
            Vtile[idx] = in ? Vb[(size_t)r * d + c] : 0.0f;
        }
        __syncthreads();

            // tile QK^T
            if (row < N) {
                const int key = tile + lane;      // global row index of K
                float s = -FLT_MAX;
                if (key < N) {
                    float dot = 0.0f;
                    for (int k = 0; k < d; ++k) {
                        dot += Qtile[warp * d + k] * Ktile[lane * d + k];
                    }
                    s = dot * scale;
                }

            // tile max
            float m_tile = s;
            for (int offset = 16; offset > 0; offset >>= 1) {
                m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, offset));
            }
            const float m_new = fmaxf(m, m_tile);

            // tile exponential
            const float p = (key < N) ? __expf(s - m_new) : 0.0f;
            float l_tile = p;
            for (int offset = 16; offset > 0; offset >>= 1) {
                l_tile += __shfl_xor_sync(0xffffffff, l_tile, offset);
            }

            // rescale and merge
            const float alpha = __expf(m - m_new);
            l = l * alpha + l_tile;
            m = m_new;
            for (int i = 0; i < ACC; ++i) acc[i] *= alpha;

            // tile PV
            for (int c = 0; c < BC; ++c) {
                // broadcast p[c] across the warp
                const float pc = __shfl_sync(0xffffffff, p, c);
                for (int i = 0; i < ACC; ++i) {
                    const int k = lane + 32 * i;
                    if (k < d) acc[i] += pc * Vtile[c * d + k];
                }
            }
        }
        __syncthreads();
    }

    // normalization + write the output (first and only HBM write)
    if (row < N) {
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
