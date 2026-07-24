#include "kernels.cuh"

#include <float.h>
#include <math.h>

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BR = 16;     // Q rows per block (one wmma tile of rows)
constexpr int BC = 64;     // K, V rows per tile (one wmma tile per warp)
constexpr int WARPS = 4;   // warps per block
constexpr int STAGES = 2;  // double buffering
constexpr int WMMA_M = 16;  // wmma tile: 16 x 16 x 16
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Extra halves appended to each Q/K/V tile row. 16 halves = 32 bytes:
// keeps every wmma fragment pointer 32-byte aligned while shifting
// consecutive rows to different banks (the wmma equivalent of step 06).
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
    // Step 08 alternates load and compute phases: while the K/V tile is
    // being fetched from HBM, the tensor cores sit idle.
    //
    // This step overlaps the two with double buffering. K/V get two SRAM
    // buffers, and cp.async (__pipeline_memcpy_async) copies HBM -> SRAM
    // directly, without staging through registers, so the copy proceeds in
    // the background while the same threads run wmma on the other buffer:
    //
    //   prefetch tile 0 into buffer 0
    //   loop t:  issue async load of tile t+1 into buffer (t+1) % 2
    //            wait for tile t, compute on buffer t % 2
    //
    // SRAM layout (FP32 first, then FP16):
    //   float Osm[BR][d]       running output accumulator
    //   float Ssm[BR][BC]      raw scores of the current tile
    //   float m[BR], l[BR], alpha[BR]
    //   half  Qs[BR][d+SKEW]
    //   half  Ks[STAGES][BC][d+SKEW]
    //   half  Vs[STAGES][BC][d+SKEW]
    //   half  Ps[BR][BC]       tile probabilities, rounded to half for wmma
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* Osm = reinterpret_cast<float*>(smem_raw);
    float* Ssm = Osm + BR * d;
    float* m_sm = Ssm + BR * BC;
    float* l_sm = m_sm + BR;
    float* a_sm = l_sm + BR;
    __half* Qs = reinterpret_cast<__half*>(a_sm + BR);
    __half* Ks = Qs + BR * (d + SKEW);
    __half* Vs = Ks + STAGES * BC * (d + SKEW);
    __half* Ps = Vs + STAGES * BC * (d + SKEW);

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int batch = blockIdx.y;
    const int q_base = blockIdx.x * BR;

    const int ldh = d + SKEW;  // row stride of the half tiles
    const int d8 = d / 8;      // row length in 8-half (16-byte) chunks
    const int ldh8 = ldh / 8;

    const __half* Qb = Q + (size_t)batch * N * d;
    const __half* Kb = K + (size_t)batch * N * d;
    const __half* Vb = V + (size_t)batch * N * d;
    __half* Ob = O + (size_t)batch * N * d;

    const float4* Qb4 = reinterpret_cast<const float4*>(Qb);
    const float4* Kb4 = reinterpret_cast<const float4*>(Kb);
    const float4* Vb4 = reinterpret_cast<const float4*>(Vb);
    float4* Qs4 = reinterpret_cast<float4*>(Qs);
    const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};

    // async cooperative K, V tile load into one of the two buffers.
    // cp.async moves 16-byte chunks HBM -> SRAM without a register
    // round-trip; rows past N are zero-filled with plain stores.
    auto load_kv_async = [&](int stage, int tile) {
        float4* Ks4 = reinterpret_cast<float4*>(Ks + stage * BC * ldh);
        float4* Vs4 = reinterpret_cast<float4*>(Vs + stage * BC * ldh);
        for (int idx = tid; idx < BC * d8; idx += blockDim.x) {
            const int r = idx / d8;
            const int c = idx % d8;
            const int gr = tile + r;
            if (gr < N) {
                __pipeline_memcpy_async(
                    &Ks4[r * ldh8 + c], &Kb4[(size_t)gr * d8 + c], sizeof(float4));
                __pipeline_memcpy_async(
                    &Vs4[r * ldh8 + c], &Vb4[(size_t)gr * d8 + c], sizeof(float4));
            } else {
                Ks4[r * ldh8 + c] = zero4;
                Vs4[r * ldh8 + c] = zero4;
            }
        }
    };

    // cooperative Q tile load
    // rows past N are zero-filled: wmma always computes full 16 x 16
    // tiles, and zeros keep the padding scores finite (no NaN in softmax)
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

    const int dw = d / WARPS;   // O columns owned by each warp
    const int c0 = warp * dw;   // first O column of this warp

    // prologue: prefetch the first tile
    load_kv_async(0, 0);
    __pipeline_commit();

    const int n_tiles = (N + BC - 1) / BC;
    for (int t = 0; t < n_tiles; ++t) {
        const int stage = t % STAGES;
        const int tile = t * BC;

        // issue the next tile's copy, then wait for the current tile only.
        // the buffer being filled was last read in iteration t-1, which
        // ended with __syncthreads, so the overwrite is safe.
        if (t + 1 < n_tiles) {
            load_kv_async((t + 1) % STAGES, tile + BC);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        const __half* Kst = Ks + stage * BC * ldh;
        const __half* Vst = Vs + stage * BC * ldh;

        // S tile = Q_tile * K_tile^T on tensor cores.
        // Each warp accumulates its 16 x 16 block over d in steps of 16.
        {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);
            for (int k = 0; k < d; k += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::row_major> a_frag;
                // reading row-major K as col-major yields K^T: B(k, n) = K[n][k]
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Qs + k, ldh);
                wmma::load_matrix_sync(b_frag, Kst + (warp * WMMA_N) * ldh + k, ldh);
                wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
            }
            wmma::store_matrix_sync(Ssm + warp * WMMA_N, s_frag, BC, wmma::mem_row_major);
        }
        __syncthreads();

        // online softmax over the BR x BC score tile: warp 0, lane r owns row r
        if (warp == 0 && lane < BR) {
            const int r = lane;
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
            a_sm[r] = alpha;  // published for the O rescale below
        }
        __syncthreads();

        // rescale this warp's O columns by alpha (per-row scalar code) ...
        for (int idx = lane; idx < BR * dw; idx += 32) {
            const int r = idx / dw;
            const int c = c0 + idx % dw;
            Osm[r * d + c] *= a_sm[r];
        }
        __syncwarp();

        // ... then accumulate P * V_tile into it on tensor cores
        for (int j = 0; j < dw; j += WMMA_N) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag, Osm + c0 + j, d, wmma::mem_row_major);
            for (int k = 0; k < BC; k += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                               wmma::row_major> v_frag;
                wmma::load_matrix_sync(p_frag, Ps + k, BC);
                wmma::load_matrix_sync(v_frag, Vst + k * ldh + c0 + j, ldh);
                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }
            wmma::store_matrix_sync(Osm + c0 + j, o_frag, d, wmma::mem_row_major);
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
        (size_t)((BR + STAGES * 2 * BC) * (d + SKEW) + BR * BC) * sizeof(__half);
    if (smem_bytes > 48 * 1024) {
        // beyond the 48 KB default the limit must be raised explicitly
        cudaFuncSetAttribute(fused_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
    }
    fused_attention_kernel<<<grid, threads, smem_bytes, stream>>>(
        dQ, dK, dV, dO, N, d, scale);
}
