#include "kernels.cuh"

#include <float.h>
#include <math.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BR = 64;      // Q rows per block
constexpr int BC = 64;      // K, V rows per tile
constexpr int WARPS = 4;    // one warp per 16-row Q slice
constexpr int STAGES = 1;   // one synchronous K/V stage
constexpr int WMMA_M = 16;  // wmma tile: 16 x 16 x 16
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int ROWS_PER_WARP = BR / WARPS;

static_assert(ROWS_PER_WARP == WMMA_M, "each warp owns one WMMA row tile");

// 32-byte row padding for WMMA alignment and bank-conflict reduction
constexpr int SKEW = 16;

__global__ void fused_attention_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N,
    int d,
    float scale)
{
    // Split-Q attention: each warp owns 16 Q/O rows for the whole kernel
    //
    // shared memory:
    //   FP32: O, S, m, l, alpha
    //   FP16: Q, P, K/V staging
    //
    // K and V share one buffer because QK^T finishes before PV starts,
    // which keeps d = 128 under the opt-in shared memory limit
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* Osm = reinterpret_cast<float*>(smem_raw);
    float* Ssm = Osm + BR * d;
    float* m_sm = Ssm + BR * BC;
    float* l_sm = m_sm + BR;
    float* a_sm = l_sm + BR;
    __half* Qs = reinterpret_cast<__half*>(a_sm + BR);

    const int ldh = d + SKEW;      // row stride of the half tiles
    __half* KVsm = Qs + BR * ldh;  // K during QK^T, then V during PV
    __half* Ps = KVsm + STAGES * BC * ldh;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int batch = blockIdx.y;
    const int q_base = blockIdx.x * BR;
    const int r0 = warp * ROWS_PER_WARP;

    const int d8 = d / 8;      // row length in 8-half (16-byte) chunks
    const int ldh8 = ldh / 8;  // row stride in 8-half (16-byte) chunks

    const __half* Qb = Q + (size_t)batch * N * d;
    const __half* Kb = K + (size_t)batch * N * d;
    const __half* Vb = V + (size_t)batch * N * d;
    __half* Ob = O + (size_t)batch * N * d;

    const float4* Qb4 = reinterpret_cast<const float4*>(Qb);
    const float4* Kb4 = reinterpret_cast<const float4*>(Kb);
    const float4* Vb4 = reinterpret_cast<const float4*>(Vb);
    float4* Qs4 = reinterpret_cast<float4*>(Qs);
    float4* KVsm4 = reinterpret_cast<float4*>(KVsm);
    const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};

    // cooperative Q tile load; zero-fill out-of-range rows
    for (int idx = tid; idx < BR * d8; idx += blockDim.x) {
        const int r = idx / d8;
        const int c = idx % d8;
        const int gr = q_base + r;
        Qs4[r * ldh8 + c] = (gr < N) ? Qb4[(size_t)gr * d8 + c] : zero4;
    }

    // init output accumulator and softmax state
    for (int idx = tid; idx < BR * d; idx += blockDim.x) Osm[idx] = 0.0f;
    if (tid < BR) {
        m_sm[tid] = -FLT_MAX;
        l_sm[tid] = 0.0f;
    }
    __syncthreads();

    const int n_tiles = (N + BC - 1) / BC;
    for (int t = 0; t < n_tiles; ++t) {
        const int tile = t * BC;

        // cooperative K tile load; zero-fill out-of-range rows
        for (int idx = tid; idx < BC * d8; idx += blockDim.x) {
            const int r = idx / d8;
            const int c = idx % d8;
            const int gr = tile + r;
            KVsm4[r * ldh8 + c] =
                (gr < N) ? Kb4[(size_t)gr * d8 + c] : zero4;
        }
        __syncthreads();

        // S = QK^T
        // each warp computes every column tile of the 16 rows it owns
        for (int j = 0; j < BC; j += WMMA_N) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);
            for (int k = 0; k < d; k += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::row_major> q_frag;
                // interpret row-major K as col-major K^T
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::col_major> k_frag;
                wmma::load_matrix_sync(q_frag, Qs + r0 * ldh + k, ldh);
                wmma::load_matrix_sync(k_frag, KVsm + j * ldh + k, ldh);
                wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }
            wmma::store_matrix_sync(
                Ssm + r0 * BC + j, s_frag, BC, wmma::mem_row_major);
        }
        // only the owner warp reads this S slice; K stays live until the
        // block-wide barrier below
        __syncwarp();

        // online softmax for the current score tile.
        // the owner warp assigns one lane to each of its query rows.
        if (lane < ROWS_PER_WARP) {
            const int r = r0 + lane;
            float m_tile = -FLT_MAX;
            for (int c = 0; c < BC; ++c) {
                if (tile + c < N) {
                    m_tile = fmaxf(m_tile, Ssm[r * BC + c] * scale);
                }
            }
            const float m_new = fmaxf(m_sm[r], m_tile);

            float l_tile = 0.0f;
            for (int c = 0; c < BC; ++c) {
                const float p = (tile + c < N)
                    ? __expf(Ssm[r * BC + c] * scale - m_new)
                    : 0.0f;
                Ps[r * BC + c] = __float2half(p);
                l_tile += p;
            }

            const float alpha = __expf(m_sm[r] - m_new);
            l_sm[r] = l_sm[r] * alpha + l_tile;
            m_sm[r] = m_new;
            a_sm[r] = alpha;
        }
        __syncthreads();

        // K is done, so the same buffer now stages V
        for (int idx = tid; idx < BC * d8; idx += blockDim.x) {
            const int r = idx / d8;
            const int c = idx % d8;
            const int gr = tile + r;
            KVsm4[r * ldh8 + c] =
                (gr < N) ? Vb4[(size_t)gr * d8 + c] : zero4;
        }
        __syncthreads();

        // rescale the accumulated output of the owned rows
        for (int idx = lane; idx < ROWS_PER_WARP * d; idx += 32) {
            const int r = r0 + idx / d;
            const int c = idx % d;
            Osm[r * d + c] *= a_sm[r];
        }
        __syncwarp();

        // add the current PV contribution
        for (int j = 0; j < d; j += WMMA_N) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(
                o_frag, Osm + r0 * d + j, d, wmma::mem_row_major);
            for (int k = 0; k < BC; k += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::row_major> v_frag;
                wmma::load_matrix_sync(p_frag, Ps + r0 * BC + k, BC);
                wmma::load_matrix_sync(v_frag, KVsm + k * ldh + j, ldh);
                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }
            wmma::store_matrix_sync(
                Osm + r0 * d + j, o_frag, d, wmma::mem_row_major);
        }
        __syncthreads();  // protect V and shared state before the next tile
    }

    // normalization + write the output (first and only HBM write)
    for (int idx = tid; idx < BR * d; idx += blockDim.x) {
        const int r = idx / d;
        const int c = idx % d;
        const int gr = q_base + r;
        if (gr < N) {
            Ob[(size_t)gr * d + c] = __float2half(Osm[r * d + c] / l_sm[r]);
        }
    }
}

void launch_fused_attention(
    const __half* dQ, const __half* dK, const __half* dV, __half* dO,
    int N, int d, float scale, int batch_count, cudaStream_t stream)
{
    const int threads = WARPS * 32;
    const dim3 grid((N + BR - 1) / BR, batch_count);

    // K and V share one staging buffer: 94,976 bytes at d = 128,
    // against 113,408 bytes for two separate buffers
    const size_t smem_bytes =
        (size_t)(BR * d + BR * BC + 3 * BR) * sizeof(float) +
        (size_t)((BR + STAGES * BC) * (d + SKEW) + BR * BC) * sizeof(__half);
    if (smem_bytes > 48 * 1024) {
        // request a larger dynamic shared memory limit when needed
        cudaFuncSetAttribute(fused_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
    }
    fused_attention_kernel<<<grid, threads, smem_bytes, stream>>>(
        dQ, dK, dV, dO, N, d, scale);
}
