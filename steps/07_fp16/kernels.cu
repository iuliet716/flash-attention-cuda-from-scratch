#include "kernels.cuh"

#include <float.h>
#include <math.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int BR = 8;                  // Q rows per block (= warps per block)
constexpr int BC = 32;                 // K, V rows per tile (= warp size)
constexpr int ACC = FUSED_D_MAX / 32;  // output dimensions per lane

// Same layout trick as step 06, now at 8-half (16-byte) chunk granularity:
//   physical chunk = logical chunk ^ (row % 8)
// Stays in bounds as long as d/8 is a multiple of 8 (d % 64 == 0).
__device__ __forceinline__ int swizzle(int r, int c8) {
    return c8 ^ (r % 8);
}

__global__ void fused_attention_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N,
    int d,
    float scale)
{
    // Q, K, V, O are stored and moved in FP16: HBM traffic and the SRAM
    // footprint are halved, and one float4 chunk now carries 8 elements
    // instead of 4.
    //
    // All arithmetic stays in FP32: values are converted right after each
    // load, and the softmax state (m, l) and PV accumulators never leave
    // FP32 registers. Only storage is half precision, so the accumulation
    // error does not grow with N.
    extern __shared__ __align__(16) __half smem[];
    __half* Qtile = smem;
    __half* Ktile = Qtile + BR * d;
    __half* Vtile = Ktile + BC * d;

    const int d8 = d / 8;  // head dimension in 8-half (16-byte) chunks
    const int batch = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * BR + warp;

    const __half* Qb = Q + (size_t)batch * N * d;
    const __half* Kb = K + (size_t)batch * N * d;
    const __half* Vb = V + (size_t)batch * N * d;
    __half* Ob = O + (size_t)batch * N * d;

    const float4* Qb4 = reinterpret_cast<const float4*>(Qb);
    const float4* Kb4 = reinterpret_cast<const float4*>(Kb);
    const float4* Vb4 = reinterpret_cast<const float4*>(Vb);
    float4* Qtile4 = reinterpret_cast<float4*>(Qtile);
    float4* Ktile4 = reinterpret_cast<float4*>(Ktile);
    float4* Vtile4 = reinterpret_cast<float4*>(Vtile);
    const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};

    // cooperative Q tile load (8 halves per chunk)
    const int q_base = blockIdx.x * BR;
    for (int idx = threadIdx.x; idx < BR * d8; idx += blockDim.x) {
        const int r = q_base + idx / d8;
        if (r < N) Qtile4[idx] = Qb4[(size_t)r * d8 + idx % d8];
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[ACC];
    for (int i = 0; i < ACC; ++i) acc[i] = 0.0f;

    for (int tile = 0; tile < N; tile += BC) {
        // cooperative K, V tile load (K stored swizzled)
        for (int idx = threadIdx.x; idx < BC * d8; idx += blockDim.x) {
            const int rl = idx / d8;  // row inside the tile
            const int c8 = idx % d8;
            const int r = tile + rl;
            const bool in = r < N;
            Ktile4[rl * d8 + swizzle(rl, c8)] = in ? Kb4[(size_t)r * d8 + c8] : zero4;
            Vtile4[idx] = in ? Vb4[(size_t)r * d8 + c8] : zero4;
        }
        __syncthreads();

        // tile QK^T
        if (row < N) {
            const int key = tile + lane;  // global row index of K
            float s = -FLT_MAX;
            if (key < N) {
                const float4* q8 = reinterpret_cast<const float4*>(Qtile + warp * d);
                const float4* k8 = reinterpret_cast<const float4*>(Ktile) + lane * d8;
                float dot = 0.0f;
                for (int c = 0; c < d8; ++c) {
                    const float4 a = q8[c];
                    const float4 b = k8[swizzle(lane, c)];  // conflict-free read
                    // unpack 8 halves and accumulate in FP32
                    const __half2* ah = reinterpret_cast<const __half2*>(&a);
                    const __half2* bh = reinterpret_cast<const __half2*>(&b);
                    for (int j = 0; j < 4; ++j) {
                        const float2 qf = __half22float2(ah[j]);
                        const float2 kf = __half22float2(bh[j]);
                        dot += qf.x * kf.x + qf.y * kf.y;
                    }
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
                    if (k < d) acc[i] += pc * __half2float(Vtile[c * d + k]);
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
            if (k < d) Ob[(size_t)row * d + k] = __float2half(acc[i] * inv_l);
        }
    }
}

void launch_fused_attention(
    const __half* dQ, const __half* dK, const __half* dV, __half* dO,
    int N, int d, float scale, int batch_count, cudaStream_t stream)
{
    const int threads = BR * 32;
    const dim3 grid((N + BR - 1) / BR, batch_count);
    const size_t smem_bytes = (size_t)(BR + 2 * BC) * d * sizeof(__half);
    fused_attention_kernel<<<grid, threads, smem_bytes, stream>>>(
        dQ, dK, dV, dO, N, d, scale);
}
