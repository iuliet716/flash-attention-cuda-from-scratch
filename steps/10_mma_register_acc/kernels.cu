#include "kernels.cuh"

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

constexpr int WARPS = 4;        // warps per block
constexpr int BR = WARPS * 16;  // Q rows per block
constexpr int BC = 64;          // K, V rows per tile
constexpr int STAGES = 2;       // double buffering

// 32-byte row padding for alignment and bank-conflict reduction
constexpr int SKEW = 16;

// m16n8k16 tensor-core MMA with explicit register layout.
// each thread holds values from two rows, allowing softmax and O to stay in registers.
__device__ __forceinline__ void mma_16816(
    float* acc, const uint32_t* a, const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t ld_half2(const __half* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ uint32_t pack_half2(__half lo, __half hi) {
    __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_float2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

template <int D>
__global__ void fused_attention_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N,
    float scale)
{
    // register-resident attention:
    //   registers: Q, S, P, O, m, l
    //   shared memory: double-buffered K, V
    //
    // D is fixed at compile time to keep register accumulator sizes static.
    constexpr int LDH = D + SKEW;  // K, V row stride in halves
    constexpr int D8 = D / 8;      // row length in 8-half (16-byte) chunks
    constexpr int LDH8 = LDH / 8;  // row stride in 8-half (16-byte) chunks
    constexpr int SB = BC / 8;     // S blocks
    constexpr int OB = D / 8;      // O blocks
    constexpr int KK = D / 16;     // QK^T steps
    constexpr int PK = BC / 16;    // PV steps

    extern __shared__ __align__(16) __half smem[];
    __half* Ks = smem;  
    __half* Vs = Ks + STAGES * BC * LDH;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g = lane / 4;    // row group
    const int tig = lane % 4;  // column pair
    const int batch = blockIdx.y;
    const int q_base = blockIdx.x * BR;
    const int r_lo = q_base + warp * 16 + g;
    const int r_hi = r_lo + 8;

    const __half* Qb = Q + (size_t)batch * N * D;
    const __half* Kb = K + (size_t)batch * N * D;
    const __half* Vb = V + (size_t)batch * N * D;
    __half* Ob = O + (size_t)batch * N * D;

    const float4* Kb4 = reinterpret_cast<const float4*>(Kb);
    const float4* Vb4 = reinterpret_cast<const float4*>(Vb);
    const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};

    // load Q fragments into registers; zero-fill out-of-range rows
    uint32_t qa[KK][4];
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
        const int c = 16 * kk + 2 * tig;
        qa[kk][0] = (r_lo < N) ? ld_half2(Qb + (size_t)r_lo * D + c) : 0u;
        qa[kk][1] = (r_hi < N) ? ld_half2(Qb + (size_t)r_hi * D + c) : 0u;
        qa[kk][2] = (r_lo < N) ? ld_half2(Qb + (size_t)r_lo * D + c + 8) : 0u;
        qa[kk][3] = (r_hi < N) ? ld_half2(Qb + (size_t)r_hi * D + c + 8) : 0u;
    }

    // init output accumulator and softmax state in registers
    // index 0: r_lo, index 1: r_hi
    float m[2] = {-FLT_MAX, -FLT_MAX};
    float l[2] = {0.0f, 0.0f};
    float o[OB][4];
#pragma unroll
    for (int jo = 0; jo < OB; ++jo)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[jo][e] = 0.0f;

    // async K, V tile load; zero-fill out-of-range rows
    auto load_kv_async = [&](int stg, int tile0) {
        float4* Ks4 = reinterpret_cast<float4*>(Ks + stg * BC * LDH);
        float4* Vs4 = reinterpret_cast<float4*>(Vs + stg * BC * LDH);
        for (int idx = tid; idx < BC * D8; idx += blockDim.x) {
            const int r = idx / D8;
            const int c = idx % D8;
            const int gr = tile0 + r;
            if (gr < N) {
                __pipeline_memcpy_async(
                    &Ks4[r * LDH8 + c], &Kb4[(size_t)gr * D8 + c], sizeof(float4));
                __pipeline_memcpy_async(
                    &Vs4[r * LDH8 + c], &Vb4[(size_t)gr * D8 + c], sizeof(float4));
            } else {
                Ks4[r * LDH8 + c] = zero4;
                Vs4[r * LDH8 + c] = zero4;
            }
        }
    };

    // prefetch the first K, V tile
    load_kv_async(0, 0);
    __pipeline_commit();

    const int n_tiles = (N + BC - 1) / BC;
    for (int t = 0; t < n_tiles; ++t) {
        const int stage = t % STAGES;
        const int tile = t * BC;

        // prefetch the next tile while processing the current tile
        if (t + 1 < n_tiles) {
            load_kv_async((t + 1) % STAGES, tile + BC);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        const __half* Kst = Ks + stage * BC * LDH;
        const __half* Vst = Vs + stage * BC * LDH;

        // S = QK^T
        float s[SB][4];
#pragma unroll
        for (int j = 0; j < SB; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) s[j][e] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < KK; ++kk) {
                const __half* kp = Kst + (8 * j + g) * LDH + 16 * kk + 2 * tig;
                uint32_t b[2] = {ld_half2(kp), ld_half2(kp + 8)};
                mma_16816(s[j], qa[kk], b);
            }
        }

        // online softmax on S registers
        float mt[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
        for (int j = 0; j < SB; ++j) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                if (tile + 8 * j + 2 * tig + e < N) {
                    mt[0] = fmaxf(mt[0], s[j][e]);
                    mt[1] = fmaxf(mt[1], s[j][2 + e]);
                }
            }
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            // reduce row max across each 4-lane group
            mt[0] = fmaxf(mt[0], __shfl_xor_sync(0xffffffff, mt[0], off));
            mt[1] = fmaxf(mt[1], __shfl_xor_sync(0xffffffff, mt[1], off));
        }

        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float m_new = fmaxf(m[r], mt[r]);
            // keep m in raw-score space; apply scale inside exp
            alpha[r] = __expf(scale * (m[r] - m_new));
            m[r] = m_new;
        }

        float lt[2] = {0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < SB; ++j) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const bool in = tile + 8 * j + 2 * tig + e < N;
                const float p0 = in ? __expf(scale * (s[j][e] - m[0])) : 0.0f;
                const float p1 = in ? __expf(scale * (s[j][2 + e] - m[1])) : 0.0f;
                s[j][e] = p0;
                s[j][2 + e] = p1;
                lt[0] += p0;
                lt[1] += p1;
            }
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            lt[0] += __shfl_xor_sync(0xffffffff, lt[0], off);
            lt[1] += __shfl_xor_sync(0xffffffff, lt[1], off);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) l[r] = l[r] * alpha[r] + lt[r];

        // rescale the accumulated output
#pragma unroll
        for (int jo = 0; jo < OB; ++jo) {
            o[jo][0] *= alpha[0];
            o[jo][1] *= alpha[0];
            o[jo][2] *= alpha[1];
            o[jo][3] *= alpha[1];
        }

        // O += PV using P from registers
#pragma unroll
        for (int kk = 0; kk < PK; ++kk) {
            // pack two 16x8 P blocks into one 16x16 MMA operand
            uint32_t pa[4];
            pa[0] = pack_float2(s[2 * kk][0], s[2 * kk][1]);
            pa[1] = pack_float2(s[2 * kk][2], s[2 * kk][3]);
            pa[2] = pack_float2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[3] = pack_float2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int jo = 0; jo < OB; ++jo) {
                const __half* vp = Vst + (16 * kk + 2 * tig) * LDH + 8 * jo + g;
                uint32_t b[2] = {pack_half2(vp[0], vp[LDH]),
                                 pack_half2(vp[8 * LDH], vp[9 * LDH])};
                mma_16816(o[jo], pa, b);
            }
        }
        __syncthreads();  // protect K/V before buffer reuse
    }

    // normalization + write the output (first and only HBM write)
    const float inv[2] = {1.0f / l[0], 1.0f / l[1]};
#pragma unroll
    for (int jo = 0; jo < OB; ++jo) {
        const int c = 8 * jo + 2 * tig;
        if (r_lo < N) {
            *reinterpret_cast<uint32_t*>(Ob + (size_t)r_lo * D + c) =
                pack_float2(o[jo][0] * inv[0], o[jo][1] * inv[0]);
        }
        if (r_hi < N) {
            *reinterpret_cast<uint32_t*>(Ob + (size_t)r_hi * D + c) =
                pack_float2(o[jo][2] * inv[1], o[jo][3] * inv[1]);
        }
    }
}

template <int D>
static void launch_impl(
    const __half* dQ, const __half* dK, const __half* dV, __half* dO,
    int N, float scale, int batch_count, cudaStream_t stream)
{
    const int threads = WARPS * 32;
    const dim3 grid((N + BR - 1) / BR, batch_count);
    // K, V double buffer only
    const size_t smem_bytes = (size_t)STAGES * 2 * BC * (D + SKEW) * sizeof(__half);
    if (smem_bytes > 48 * 1024) {
        // request a larger dynamic shared memory limit when needed
        cudaFuncSetAttribute(fused_attention_kernel<D>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
    }
    fused_attention_kernel<D><<<grid, threads, smem_bytes, stream>>>(
        dQ, dK, dV, dO, N, scale);
}

void launch_fused_attention(
    const __half* dQ, const __half* dK, const __half* dV, __half* dO,
    int N, int d, float scale, int batch_count, cudaStream_t stream)
{
    if (d == 64) {
        launch_impl<64>(dQ, dK, dV, dO, N, scale, batch_count, stream);
    } else if (d == 128) {
        launch_impl<128>(dQ, dK, dV, dO, N, scale, batch_count, stream);
    }
}