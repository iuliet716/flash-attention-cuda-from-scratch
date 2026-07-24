#include "kernels.cuh"

#include <float.h>
#include <math.h>

#include <cuda_runtime.h>

constexpr int BR = 8;                  // Q rows per block (= warps per block)
constexpr int BC = 32;                 // K, V rows per tile (= warp size)
constexpr int ACC = FUSED_D_MAX / 32;  // output dimensions per lane

__device__ __forceinline__ int swizzle(int r, int c4) {
    return c4 ^ (r % 8);
}

__global__ void fused_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int d,
    float scale)
{
    extern __shared__ __align__(16) float smem[];
    float* Qtile = smem;
    float* Ktile = Qtile + BR * d;
    float* Vtile = Ktile + BC * d;

    const int d4 = d / 4;  // head dimension in float4 chunks
    const int batch = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * BR + warp;

    const float* Qb = Q + (size_t)batch * N * d;
    const float* Kb = K + (size_t)batch * N * d;
    const float* Vb = V + (size_t)batch * N * d;
    float* Ob = O + (size_t)batch * N * d;

    const float4* Qb4 = reinterpret_cast<const float4*>(Qb);
    const float4* Kb4 = reinterpret_cast<const float4*>(Kb);
    const float4* Vb4 = reinterpret_cast<const float4*>(Vb);
    float4* Qtile4 = reinterpret_cast<float4*>(Qtile);
    float4* Ktile4 = reinterpret_cast<float4*>(Ktile);
    float4* Vtile4 = reinterpret_cast<float4*>(Vtile);
    const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};

    // cooperative Q tile load (float4-vectorized)
    const int q_base = blockIdx.x * BR;
    for (int idx = threadIdx.x; idx < BR * d4; idx += blockDim.x) {
        const int r = q_base + idx / d4;
        if (r < N) Qtile4[idx] = Qb4[(size_t)r * d4 + idx % d4];
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[ACC];
    for (int i = 0; i < ACC; ++i) acc[i] = 0.0f;

    for (int tile = 0; tile < N; tile += BC) {
        // cooperative K, V tile load (K stored swizzled)
        for (int idx = threadIdx.x; idx < BC * d4; idx += blockDim.x) {
            const int rl = idx / d4;  // row inside the tile
            const int c4 = idx % d4;
            const int r = tile + rl;
            const bool in = r < N;
            Ktile4[rl * d4 + swizzle(rl, c4)] = in ? Kb4[(size_t)r * d4 + c4] : zero4;
            Vtile4[idx] = in ? Vb4[(size_t)r * d4 + c4] : zero4;
        }
        __syncthreads();

        // tile QK^T
        if (row < N) {
            const int key = tile + lane;  // global row index of K
            float s = -FLT_MAX;
            if (key < N) {
                const float4* q4 = reinterpret_cast<const float4*>(Qtile + warp * d);
                const float4* k4 = reinterpret_cast<const float4*>(Ktile) + lane * d4;
                float dot = 0.0f;
                for (int k = 0; k < d4; ++k) {
                    const float4 a = q4[k];
                    const float4 b = k4[swizzle(lane, k)];  // conflict-free read
                    dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
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