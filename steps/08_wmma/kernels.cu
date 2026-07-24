#include "kernels.cuh"

#include <float.h>
#include <math.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BR = 16;     // Q rows per block (one wmma tile of rows)
constexpr int BC = 64;     // K, V rows per tile (one wmma tile per warp)
constexpr int WARPS = 4;   // warps per block
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
    // The scalar dot products of step 07 become tensor-core matrix
    // products (wmma = warp matrix multiply-accumulate). Both matmuls of
    // one KV tile run on tensor cores with FP16 inputs and FP32
    // accumulation:
    //
    //   S (BR x BC) = Q_tile (BR x d) * K_tile^T (d x BC)
    //   O (BR x d) += P_tile (BR x BC) * V_tile (BC x d)
    //
    // Warp work split (4 warps, one 16 x 16 wmma tile at a time):
    //   QK^T   : warp w computes S columns [16w, 16w + 16)   (BC = 64)
    //   softmax: warp 0, one lane per row of S               (BR = 16)
    //   PV     : warp w computes O columns [w*d/4, (w+1)*d/4)
    //
    // wmma accumulator fragments have an opaque, per-architecture register
    // layout, so a thread cannot tell which row of O its registers hold --
    // but the online-softmax rescale is per-row. The output accumulator
    // therefore lives in SRAM: rescale it with scalar code, then let wmma
    // load / mma / store it.
    //
    // SRAM layout (FP32 first, then FP16):
    //   float Osm[BR][d]       running output accumulator
    //   float Ssm[BR][BC]      raw scores of the current tile
    //   float m[BR], l[BR], alpha[BR]
    //   half  Qs[BR][d+SKEW]
    //   half  Ks[BC][d+SKEW]
    //   half  Vs[BC][d+SKEW]
    //   half  Ps[BR][BC]       tile probabilities, rounded to half for wmma
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
    float4* Ks4 = reinterpret_cast<float4*>(Ks);
    float4* Vs4 = reinterpret_cast<float4*>(Vs);
    const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};

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

    for (int tile = 0; tile < N; tile += BC) {
        // cooperative K, V tile load (zero-filled past N)
        for (int idx = tid; idx < BC * d8; idx += blockDim.x) {
            const int r = idx / d8;
            const int c = idx % d8;
            const int gr = tile + r;
            const bool in = gr < N;
            Ks4[r * ldh8 + c] = in ? Kb4[(size_t)gr * d8 + c] : zero4;
            Vs4[r * ldh8 + c] = in ? Vb4[(size_t)gr * d8 + c] : zero4;
        }
        __syncthreads();

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
                wmma::load_matrix_sync(b_frag, Ks + (warp * WMMA_N) * ldh + k, ldh);
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
                wmma::load_matrix_sync(v_frag, Vs + k * ldh + c0 + j, ldh);
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
        (size_t)((BR + 2 * BC) * (d + SKEW) + BR * BC) * sizeof(__half);
    if (smem_bytes > 48 * 1024) {
        // beyond the 48 KB default the limit must be raised explicitly
        cudaFuncSetAttribute(fused_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
    }
    fused_attention_kernel<<<grid, threads, smem_bytes, stream>>>(
        dQ, dK, dV, dO, N, d, scale);
}
