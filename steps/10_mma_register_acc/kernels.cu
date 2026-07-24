#include "kernels.cuh"

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

constexpr int WARPS = 4;        // warps per block, one 16-row group each
constexpr int BR = WARPS * 16;  // Q rows per block (64)
constexpr int BC = 64;          // K, V rows per tile
constexpr int STAGES = 2;       // double buffering
constexpr int SKEW = 16;        // halves of padding per K/V smem row

// ---- the m16n8k16 tensor-core primitive ---------------------------------
//
// wmma's fragments have an OPAQUE register layout, which is why steps
// 08/09 had to keep the output accumulator in SRAM: the per-row
// online-softmax rescale cannot be applied to registers whose row is
// unknown. The PTX mma.sync instruction exposes the same tensor cores
// with a DOCUMENTED layout (PTX ISA, "Matrix Fragments for mma.m16n8k16"):
//
//   groupID = lane / 4         tig = lane % 4
//   C/D (16x8 f32, 4 regs) : c0,c1 -> row groupID,   cols 2*tig, 2*tig+1
//                            c2,c3 -> row groupID+8, cols 2*tig, 2*tig+1
//   A (16x16 f16, 4x b32)  : R0 -> (groupID,   k = 2*tig..+1)  R2 -> k+8
//                            R1 -> (groupID+8, k = 2*tig..+1)  R3 -> k+8
//   B (16x8  f16, 2x b32)  : R0 -> (k = 2*tig..+1, n = groupID) R1 -> k+8
//
// Two consequences drive this whole step:
//   1. every thread knows which 2 rows its accumulator registers belong
//      to, so softmax, the online rescale and the final normalization all
//      run on registers -- S, P and O never touch SRAM;
//   2. the C layout doubles as the A layout: the probabilities produced
//      by softmax feed the PV mma directly from registers.
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
    // Register-resident FlashAttention step:
    //   - Q fragments load from HBM once and live in registers
    //   - S, P, O and the softmax state (m, l) live in registers
    //   - SRAM holds only the double-buffered K/V tiles (cp.async)
    //
    // Removing Osm/Ssm/Ps/Qs shrinks SRAM from 59-99 KB to 20-37 KB, so
    // several blocks fit per SM again -- occupancy AND intra-block
    // pipelining, instead of one at the expense of the other (step 09).
    //
    // The kernel is templated on the head dim: register arrays need
    // compile-time bounds, otherwise they spill to local memory.
    //
    // Work split: warp w owns Q rows [16w, 16w+16) and all d columns of
    // O. Within a warp, thread (g = lane/4, tig = lane%4) owns rows
    // r_lo = g and r_hi = g + 8 of every 16x8 accumulator tile.
    constexpr int LDH = D + SKEW;  // K/V smem row stride (halves)
    constexpr int D8 = D / 8;      // row length in 8-half (16-byte) chunks
    constexpr int LDH8 = LDH / 8;
    constexpr int SB = BC / 8;     // S accumulator blocks (8 cols each)
    constexpr int OB = D / 8;      // O accumulator blocks (8 cols each)
    constexpr int KK = D / 16;     // k-steps of the QK^T mma
    constexpr int PK = BC / 16;    // k-steps of the PV mma

    extern __shared__ __align__(16) __half smem[];
    __half* Ks = smem;                    // [STAGES][BC][LDH]
    __half* Vs = Ks + STAGES * BC * LDH;  // [STAGES][BC][LDH]

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g = lane / 4;    // this thread's row inside a 16x8 tile
    const int tig = lane % 4;  // this thread's column pair
    const int batch = blockIdx.y;
    const int q_base = blockIdx.x * BR;
    const int r_lo = q_base + warp * 16 + g;  // global Q row of c0, c1
    const int r_hi = r_lo + 8;                // global Q row of c2, c3

    const __half* Qb = Q + (size_t)batch * N * D;
    const __half* Kb = K + (size_t)batch * N * D;
    const __half* Vb = V + (size_t)batch * N * D;
    __half* Ob = O + (size_t)batch * N * D;

    const float4* Kb4 = reinterpret_cast<const float4*>(Kb);
    const float4* Vb4 = reinterpret_cast<const float4*>(Vb);
    const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};

    // Q fragments: loaded straight from HBM once, kept for the whole
    // kernel (no Q tile in SRAM at all). Rows past N load zeros so the
    // padding scores stay finite.
    uint32_t qa[KK][4];
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
        const int c = 16 * kk + 2 * tig;
        qa[kk][0] = (r_lo < N) ? ld_half2(Qb + (size_t)r_lo * D + c) : 0u;
        qa[kk][1] = (r_hi < N) ? ld_half2(Qb + (size_t)r_hi * D + c) : 0u;
        qa[kk][2] = (r_lo < N) ? ld_half2(Qb + (size_t)r_lo * D + c + 8) : 0u;
        qa[kk][3] = (r_hi < N) ? ld_half2(Qb + (size_t)r_hi * D + c + 8) : 0u;
    }

    // online-softmax state and output accumulator, all in registers.
    // index 0 belongs to row r_lo, index 1 to row r_hi.
    float m[2] = {-FLT_MAX, -FLT_MAX};
    float l[2] = {0.0f, 0.0f};
    float o[OB][4];
#pragma unroll
    for (int jo = 0; jo < OB; ++jo)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[jo][e] = 0.0f;

    // async cooperative K, V tile load (16-byte cp.async chunks,
    // zero-filled past N), same as step 09
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

    // prologue: prefetch the first tile
    load_kv_async(0, 0);
    __pipeline_commit();

    const int n_tiles = (N + BC - 1) / BC;
    for (int t = 0; t < n_tiles; ++t) {
        const int stage = t % STAGES;
        const int tile = t * BC;

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

        // ---- S = Q K^T: s[j] holds S columns [8j, 8j+8) ----------------
        // B operand (K^T) reads two contiguous halves per register, so
        // plain 4-byte loads suffice (ldmatrix comes in step 11)
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

        // ---- online softmax directly on the s registers ----------------
        // a row of a 16x8 tile is spread over the 4 threads of a quad
        // (same lane/4), so row statistics need only two XOR shuffles
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
            mt[0] = fmaxf(mt[0], __shfl_xor_sync(0xffffffff, mt[0], off));
            mt[1] = fmaxf(mt[1], __shfl_xor_sync(0xffffffff, mt[1], off));
        }

        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float m_new = fmaxf(m[r], mt[r]);
            // exp(scale*s - scale*m) == exp(scale*(s - m)): the max can be
            // tracked on raw scores and the scale applied inside the exp
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

        // rescale O in registers (an SRAM round-trip in steps 08/09)
#pragma unroll
        for (int jo = 0; jo < OB; ++jo) {
            o[jo][0] *= alpha[0];
            o[jo][1] *= alpha[0];
            o[jo][2] *= alpha[1];
            o[jo][3] *= alpha[1];
        }

        // ---- O += P V: P comes straight from the s registers -----------
        // the C layout doubles as the A layout: two adjacent 16x8 S
        // blocks repack into one 16x16 A operand, no SRAM involved
#pragma unroll
        for (int kk = 0; kk < PK; ++kk) {
            uint32_t pa[4];
            pa[0] = pack_float2(s[2 * kk][0], s[2 * kk][1]);
            pa[1] = pack_float2(s[2 * kk][2], s[2 * kk][3]);
            pa[2] = pack_float2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[3] = pack_float2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int jo = 0; jo < OB; ++jo) {
                // B operand of PV walks a COLUMN of row-major V: four
                // scalar half loads per fragment (fixed by ldmatrix.trans
                // in step 11)
                const __half* vp = Vst + (16 * kk + 2 * tig) * LDH + 8 * jo + g;
                uint32_t b[2] = {pack_half2(vp[0], vp[LDH]),
                                 pack_half2(vp[8 * LDH], vp[9 * LDH])};
                mma_16816(o[jo], pa, b);
            }
        }
        __syncthreads();  // all warps done with this stage's K/V before
                          // the next prefetch may overwrite it
    }

    // ---- epilogue: normalize in registers, single HBM write ------------
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
    // SRAM is the K/V double buffer only: 40 KB (d=64) / 72 KB (d=128)
    const size_t smem_bytes = (size_t)STAGES * 2 * BC * (D + SKEW) * sizeof(__half);
    if (smem_bytes > 48 * 1024) {
        // beyond the 48 KB default the limit must be raised explicitly
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
    // other head dims are rejected host-side in attention_forward
}
