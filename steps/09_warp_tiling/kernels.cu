#include "kernels.cuh"

#include <float.h>
#include <math.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BR = 64;      // Q rows per block
constexpr int BC = 32;      // K, V rows per tile
constexpr int WARPS = 8;    // warps per block
constexpr int WMMA_M = 16;  // wmma tile: 16 x 16 x 16
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// 4 x 2 WMMA tiles, one per warp
constexpr int RGROUPS = BR / WMMA_M;       // row groups
constexpr int CGROUPS = BC / WMMA_N;       // column groups
constexpr int ROWS_PER_WARP = BR / WARPS;  // softmax rows per warp
static_assert(RGROUPS * CGROUPS == WARPS, "one QK^T tile per warp");
static_assert(BR % WARPS == 0, "softmax rows split evenly across warps");

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
    // Tensor-core attention: QK^T -> softmax -> PV
    //
    // shared memory:
    //   FP32: O, S, m, l, alpha
    //   FP16: Q, K, V, P
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* Osm = reinterpret_cast<float*>(smem_raw);
    float* Ssm = Osm + BR * d;
    float* m_sm = Ssm + BR * BC;
    float* l_sm = m_sm + BR;
    float* a_sm = l_sm + BR;
    __half* Qs = reinterpret_cast<__half*>(a_sm + BR);
    __half* Ks = Qs + BR * (d + SKEW);
    __half* Vs = Ks + BC * (d + SKEW);
    __half* Ps = Vs + BC * (d + SKEW);

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int batch = blockIdx.y;
    const int q_base = blockIdx.x * BR;

    // 2D warp mapping:
    //
    //            K columns
    //          0:16    16:32
    // Q  0:16   W0      W1
    //   16:32   W2      W3
    //   32:48   W4      W5
    //   48:64   W6      W7
    const int warp_r = warp / CGROUPS;
    const int warp_c = warp % CGROUPS;

    const int ldh = d + SKEW;  // row stride of the half tiles
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
    float4* Ks4 = reinterpret_cast<float4*>(Ks);
    float4* Vs4 = reinterpret_cast<float4*>(Vs);
    const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};

    // cooperative Q tile load
    // Q tile is reused across all K, V tiles processed by this block
    for (int idx = tid; idx < BR * d8; idx += blockDim.x) {
        const int r = idx / d8;
        const int c = idx % d8;
        const int gr = q_base + r;
        // zero-fill out-of-range rows
        Qs4[r * ldh8 + c] = (gr < N) ? Qb4[(size_t)gr * d8 + c] : zero4;
    }

    // init output accumulator and softmax state
    for (int idx = tid; idx < BR * d; idx += blockDim.x) Osm[idx] = 0.0f;
    if (tid < BR) {
        m_sm[tid] = -FLT_MAX;
        l_sm[tid] = 0.0f;
    }

    // Each warp owns:
    //
    //   one 16-row Q group
    //   one output-column group
    //
    // d=64:
    //
    //   warp_c = 0 -> O columns  0:32
    //   warp_c = 1 -> O columns 32:64
    const int dw = d / CGROUPS;
    const int r0 = warp_r * WMMA_M;
    const int c0 = warp_c * dw;

    // Iterate over K, V tiles:
    //
    //   load tile
    //      ↓
    //   compute tile
    //      ↓
    //   load next tile
    for (int tile = 0; tile < N; tile += BC) {
        // cooperative K, V tile load
        for (int idx = tid; idx < BC * d8; idx += blockDim.x) {
            const int r = idx / d8;
            const int c = idx % d8;
            const int gr = tile + r;
            const bool in = gr < N;
            Ks4[r * ldh8 + c] = in ? Kb4[(size_t)gr * d8 + c] : zero4;
            Vs4[r * ldh8 + c] = in ? Vb4[(size_t)gr * d8 + c] : zero4;
        }
        __syncthreads();

        // -------------------------------------------------------------
        // QK^T
        // -------------------------------------------------------------
        //
        // BR x BC = 64 x 32
        //
        // decomposed into eight 16 x 16 WMMA tiles:
        //
        //            K
        //        0      16     32
        //     +-------+-------+
        //  0  |  W0   |  W1   |
        // 16  |  W2   |  W3   |
        // 32  |  W4   |  W5   |
        // 48  |  W6   |  W7   |
        // 64  +-------+-------+
        {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);
            for (int k = 0; k < d; k += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::row_major> a_frag;
                // interpret row-major K as col-major K^T
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Qs + r0 * ldh + k, ldh);
                wmma::load_matrix_sync(b_frag, Ks + (warp_c * WMMA_N) * ldh + k, ldh);
                wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
            }
            wmma::store_matrix_sync(Ssm + r0 * BC + warp_c * WMMA_N, s_frag, BC, wmma::mem_row_major);
        }
        __syncthreads();

        // -------------------------------------------------------------
        // Online softmax
        // -------------------------------------------------------------
        //
        // Step 08:
        //   warp 0 handled all 16 query rows
        //
        // Step 09:
        //   64 rows / 8 warps = 8 rows per warp
        //
        //   W0 -> rows  0:8
        //   W1 -> rows  8:16
        //   ...
        //   W7 -> rows 56:64
        if (lane < ROWS_PER_WARP) {
            const int r = warp * ROWS_PER_WARP + lane;
            float m_tile = -FLT_MAX;
            for (int c = 0; c < BC; ++c) {
                if (tile + c < N) m_tile = fmaxf(m_tile, Ssm[r * BC + c] * scale);
            }
            const float m_new = fmaxf(m_sm[r], m_tile);

            float l_tile = 0.0f;
            for (int c = 0; c < BC; ++c) {
                const float p =
                    (tile + c < N) ? __expf(Ssm[r * BC + c] * scale - m_new) : 0.0f;
                Ps[r * BC + c] = __float2half(p);
                l_tile += p;
            }

            const float alpha = __expf(m_sm[r] - m_new);
            l_sm[r] = l_sm[r] * alpha + l_tile;
            m_sm[r] = m_new;
            a_sm[r] = alpha;  // rescale factor for O
        }
        __syncthreads();

        // -------------------------------------------------------------
        // Rescale previous O
        // -------------------------------------------------------------
        //
        // Each warp rescales only the O sub-tile it owns
        for (int idx = lane; idx < WMMA_M * dw; idx += 32) {
            const int r = r0 + idx / dw;
            const int c = c0 + idx % dw;
            Osm[r * d + c] *= a_sm[r];
        }
        __syncwarp();

        // -------------------------------------------------------------
        // O += PV
        // -------------------------------------------------------------
        //
        // Each warp owns:
        //
        //   rows    r0 : r0 + 16
        //   columns c0 : c0 + dw
        //
        // and computes its O sub-tile with WMMA
        for (int j = 0; j < dw; j += WMMA_N) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag, Osm + r0 * d + c0 + j, d, wmma::mem_row_major);
            for (int k = 0; k < BC; k += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::row_major> v_frag;
                wmma::load_matrix_sync(p_frag, Ps + r0 * BC + k, BC);
                wmma::load_matrix_sync(v_frag, Vs + k * ldh + c0 + j, ldh);
                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }
            wmma::store_matrix_sync(Osm + r0 * d + c0 + j, o_frag, d, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // normalization + write the output (first and only HBM write)
    for (int idx = tid; idx < BR * d; idx += blockDim.x) {
        const int r = idx / d;
        const int c = idx % d;
        const int gr = q_base + r;
        if (gr < N) Ob[(size_t)gr * d + c] = __float2half(Osm[r * d + c] / l_sm[r]);
    }
}

void launch_fused_attention(
    const __half* dQ, const __half* dK, const __half* dV, __half* dO,
    int N, int d, float scale, int batch_count, cudaStream_t stream)
{
    const int threads = WARPS * 32;
    const dim3 grid((N + BR - 1) / BR, batch_count);
    const size_t smem_bytes =
        (size_t)(BR * d + BR * BC + 3 * BR) * sizeof(float) +
        (size_t)((BR + 2 * BC) * (d + SKEW) + BR * BC) * sizeof(__half);
    if (smem_bytes > 48 * 1024) {
        // request a larger dynamic shared memory limit when needed
        cudaFuncSetAttribute(fused_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
    }
    fused_attention_kernel<<<grid, threads, smem_bytes, stream>>>(
        dQ, dK, dV, dO, N, d, scale);
}