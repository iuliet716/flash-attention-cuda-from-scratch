# Step 10. Warp-Owned Register Dataflow

## What this step implements

Step 09 establishes persistent **split-Q ownership**.

Each warp owns a separate 16-row Q/O slice for the entire attention computation:

```text
warp 0 : Q/O rows  0:16
warp 1 : Q/O rows 16:32
warp 2 : Q/O rows 32:48
warp 3 : Q/O rows 48:64
```

The same warp owns those rows through:

```text
QK^T
 ↓
softmax
 ↓
PV
```

However, most of the state belonging to those rows is still materialized in shared memory:

```text
Q       → shared memory
S       → shared memory
P       → shared memory
O       → shared memory
m, l, α → shared memory
```

This means that the ownership is warp-local, but the dataflow is not.

Step 10 changes the storage model to match the ownership:

```text
registers:
    Q
    S / P
    O
    m, l, α

shared memory:
    K
    V
```

The owner warp now keeps its attention state local from the beginning of the tile computation until the final output write.

Conceptually:

```text
Step 09

Q
↓
shared Q
↓
Tensor Core
↓
shared S
↓
softmax
↓
shared P
↓
Tensor Core
↓
shared O


Step 10

Q registers
     ↓
Tensor Core
     ↓
S/P registers
     ↓
Tensor Core
     ↓
O registers
     ↓
final HBM write
```

K and V remain in shared memory because the same K/V tile is reused by all four Q-owning warps.

The numerical format remains unchanged:

```text
Q, K, V, O     FP16
Tensor Core    FP16 × FP16 → FP32
softmax        FP32
O accumulator  FP32
```

## Explicit MMA register layout

Step 09 uses the high-level WMMA interface:

```cuda
wmma::fragment<...> frag;

wmma::load_matrix_sync(...);
wmma::mma_sync(...);
wmma::store_matrix_sync(...);
```

WMMA fragments provide a convenient matrix abstraction, but their internal per-thread element layout is intentionally opaque.

That becomes inconvenient once the Tensor Core result should be consumed directly by a warp-local softmax.

Step 09 therefore stores the score fragment into shared memory:

```text
WMMA accumulator
      ↓
shared S
      ↓
softmax
```

Step 10 instead uses an explicit Tensor Core instruction:

```cuda
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, "
    "{%4,%5,%6,%7}, "
    "{%8,%9}, "
    "{%0,%1,%2,%3};\n"
    ...
);
```

The MMA shape is:

```text
m16 n8 k16
```

Each thread explicitly supplies its operand registers and receives four FP32 accumulator values.

The accumulator layout is therefore directly accessible to the surrounding CUDA code:

```text
mma.sync
    ↓
thread registers
    ↓
softmax
```

No intermediate shared score matrix is required.

This explicit register layout is what allows the Tensor Core result, softmax, and PV input to share the same warp-local representation.

## Per-thread ownership

Each warp still owns 16 query rows.

Step 10 further divides the warp into eight four-lane groups:

```cuda
const int g = lane / 4;
const int tig = lane % 4;
```

Each group owns two rows:

```cuda
const int r_lo = q_base + warp * 16 + g;
const int r_hi = r_lo + 8;
```

Within one warp:

```text
lanes  0:3  → rows  0 and  8
lanes  4:7  → rows  1 and  9
lanes  8:11 → rows  2 and 10
...
lanes 28:31 → rows  7 and 15
```

Inside each four-lane group:

```text
tig = 0
tig = 1
tig = 2
tig = 3
```

identify the four column-pair positions required by the MMA register layout.

The same mapping is preserved through:

```text
Q fragments
    ↓
QK^T accumulators
    ↓
softmax
    ↓
P fragments
    ↓
PV accumulators
    ↓
output write
```

The result is a persistent register-level ownership model rather than only a warp-level scheduling model.

## Q stays in registers

Step 09 first copies the Q tile into shared memory:

```text
HBM Q
  ↓
shared Q
  ↓
WMMA
```

Step 10 removes this shared-memory stage.

Q fragments are loaded directly into registers:

```cuda
uint32_t qa[KK][4];
```

Each `uint32_t` contains two FP16 values:

```cuda
qa[kk][0] = ld_half2(...);
qa[kk][1] = ld_half2(...);
qa[kk][2] = ld_half2(...);
qa[kk][3] = ld_half2(...);
```

The dataflow becomes:

```text
HBM Q
  ↓
Q registers
  ↓
mma.sync
```

Q is invariant across all K/V tiles processed by the block.

Keeping it in registers avoids routing a warp-private operand through shared memory during every Tensor Core operation.

## Scores stay in registers

The score accumulators are:

```cuda
float s[SB][4];
```

with:

```text
SB = BC / 8
```

For:

```text
BC = 64
```

each warp computes eight `16 × 8` score tiles.

Each thread holds four FP32 accumulator values for every tile:

```text
QK^T
  ↓
s[0][0:4]
s[1][0:4]
...
s[7][0:4]
```

Step 09 instead follows:

```text
Tensor Core accumulator
        ↓
shared S
        ↓
softmax
```

Step 10 follows:

```text
Tensor Core accumulator
        ↓
S registers
        ↓
softmax
```

The complete `BR × BC` shared score matrix disappears.

## Softmax follows the MMA register layout

Step 09 performs the softmax with one lane per query row.

For each 64-column score tile:

```text
one lane
    ↓
64 score values
    ↓
max
    ↓
exp + sum
```

This leaves a long scalar loop inside each participating lane.

Step 10 instead distributes each row across a four-lane group.

Each lane already owns part of the row through the MMA accumulator layout.

The local maxima are first computed in registers:

```cuda
float mt[2] = {-FLT_MAX, -FLT_MAX};
```

and then reduced across the group:

```cuda
for (int off = 1; off <= 2; off <<= 1) {
    mt[0] = fmaxf(
        mt[0],
        __shfl_xor_sync(0xffffffff, mt[0], off)
    );

    mt[1] = fmaxf(
        mt[1],
        __shfl_xor_sync(0xffffffff, mt[1], off)
    );
}
```

The XOR offsets are:

```text
1
2
```

so the reduction remains inside each four-lane group.

The exponential sum uses the same reduction structure.

Conceptually:

```text
Step 09

1 lane
  ↓
64 scores / row


Step 10

4 lanes
  ↓
16 scores / lane
  ↓
shuffle reduction
```

The sequential score-processing depth per lane is therefore substantially shorter.

This is an important part of Step 10.

The optimization is not only moving the same softmax state from shared memory into registers.

The softmax mapping itself is redesigned around the explicit Tensor Core register layout.

## Online-softmax state stays in registers

The persistent online-softmax state is now:

```cuda
float m[2] = {-FLT_MAX, -FLT_MAX};
float l[2] = {0.0f, 0.0f};
```

Each thread keeps the state corresponding to its two owned rows.

For each K/V tile:

```cuda
const float m_new = fmaxf(m[r], mt[r]);

alpha[r] = __expf(scale * (m[r] - m_new));
m[r] = m_new;
```

and:

```cuda
l[r] = l[r] * alpha[r] + lt[r];
```

There are no longer shared arrays for:

```text
m
l
alpha
```

The entire online-softmax recurrence is maintained inside the owner warp.

## S and P reuse the same registers

The array:

```cuda
float s[SB][4];
```

initially contains raw QK^T scores.

After the tile maximum is known, those values are replaced in place by the unnormalized probabilities:

```cuda
const float p0 =
    in ? __expf(scale * (s[j][e] - m[0])) : 0.0f;

const float p1 =
    in ? __expf(scale * (s[j][2 + e] - m[1])) : 0.0f;

s[j][e] = p0;
s[j][2 + e] = p1;
```

The register lifetime therefore becomes:

```text
s registers
    │
    ├── raw QK^T scores
    │
    ▼
online softmax
    │
    ▼
unnormalized P
```

No separate P array is required.

Before PV, the FP32 probabilities are packed into FP16 MMA operands:

```cuda
pa[0] = pack_float2(...);
pa[1] = pack_float2(...);
pa[2] = pack_float2(...);
pa[3] = pack_float2(...);
```

The complete path is:

```text
S FP32 registers
      ↓
exp
      ↓
P FP32 registers
      ↓
pack to FP16
      ↓
PV Tensor Core
```

Step 09 instead uses:

```text
shared S FP32
      ↓
shared P FP16
      ↓
PV
```

Step 10 removes both shared-memory materializations.

## O stays live across all K/V tiles

The output accumulator is:

```cuda
float o[OB][4];
```

and is initialized once before the K/V tile loop.

When the online-softmax maximum changes, the previously accumulated output is rescaled:

```cuda
o[jo][0] *= alpha[0];
o[jo][1] *= alpha[0];
o[jo][2] *= alpha[1];
o[jo][3] *= alpha[1];
```

The current PV contribution is then accumulated directly into the same registers:

```cuda
mma_16816(o[jo], pa, b);
```

Conceptually:

```text
PV tile 0
    ↓
O registers
    ↓ × α
PV tile 1
    ↓
O registers
    ↓ × α
PV tile 2
    ↓
...
```

The partial output is never written to shared memory between tiles.

Only after all K/V tiles have been processed is the final normalization applied:

```cuda
const float inv[2] = {
    1.0f / l[0],
    1.0f / l[1]
};
```

The normalized FP16 result is then written directly to HBM.

The output therefore has one continuous register lifetime:

```text
initialize O
    ↓
PV accumulation
    ↓
online rescaling
    ↓
PV accumulation
    ↓
...
    ↓
final normalization
    ↓
HBM
```

## Shared memory now stages only K and V

Removing Q, S, P, O, and the softmax state substantially changes the role of shared memory.

Step 09 uses shared memory for:

```text
Q
K / V
S
P
O
m
l
alpha
```

Step 10 uses it only for:

```text
K
V
```

The buffers are allocated as:

```cuda
__half* Ks = smem;
__half* Vs = Ks + STAGES * BC * LDH;
```

K and V are loaded cooperatively:

```cuda
load_kv_sync(stage, tile);
__syncthreads();
```

Unlike Step 09, K and V can now remain in separate shared buffers.

There is enough capacity because the much larger Q/S/P/O shared-memory state has been removed.

For the profiled `d = 64` configuration, Nsight Compute reports:

```text
Dynamic shared memory / block
62,208 B → 20,480 B
```

The K/V path is still synchronous.

The kernel remains:

```cuda
constexpr int STAGES = 1;
```

so this step does not yet introduce an asynchronous or multi-stage copy pipeline.

## Compile-time head dimension

The register array sizes depend on the head dimension:

```cuda
constexpr int KK = D / 16;
constexpr int OB = D / 8;
```

These determine the size of the Q fragments and O accumulators.

Step 10 therefore specializes the kernel at compile time:

```cuda
fused_attention_kernel<64>
fused_attention_kernel<128>
```

rather than keeping `d` as a runtime dimension inside the register arrays.

This gives the compiler statically sized accumulators and allows the relevant loops to be fully unrolled.

## Why the improvement is so large

The transition from Step 09 to Step 10 changes several tightly coupled properties at the same time.

Nsight Compute reports:

```text
                                      Step 09    Step 10

Dynamic shared memory / block         62,208 B   20,480 B
Registers / thread                          72        124

Theoretical occupancy                     8.3%      33.3%
Achieved occupancy                        8.3%      32.5%

Active warps / scheduler                  1.00       3.89
Eligible warps / scheduler                0.11       0.51
Issue-active cycles                      10.9%      33.2%

SM Throughput                            12.0%      72.0%
Tensor-pipe utilization                  3.58%     40.87%

Short-scoreboard stall            4.29 cycles 0.42 cycles
```

The shared-memory wavefront count changes even more clearly:

```text
Total shared wavefronts

Step 09    ~9.41 billion
Step 10    ~0.87 billion
```

approximately a **10.8x reduction**.

The excessive portion falls from:

```text
~7.99 billion
```

to:

```text
~0.40 billion
```

approximately a **19.8x reduction**.

Shared-store requests also fall from:

```text
292.7 million
```

to:

```text
16.8 million
```

because the kernel no longer writes the large Q/S/P/O and softmax-state intermediates to shared memory.

The dominant Step 09 dependency changes correspondingly:

```text
Short scoreboard

4.29 → 0.42 cycles / issued instruction
```

This is approximately a tenfold reduction.

The lower shared-memory footprint also allows substantially more resident warps:

```text
Step 09

4 warps / SM
    ↓
8.3% theoretical occupancy


Step 10

16 warps / SM
    ↓
33.3% theoretical occupancy
```

The scheduler consequently sees:

```text
Active warps / scheduler
1.00 → 3.89

Eligible warps / scheduler
0.11 → 0.51

Issue-active cycles
10.9% → 33.2%
```

At the same time, the softmax itself changes from:

```text
1 lane × 64 scores
```

to:

```text
4 lanes × 16 scores / lane
```

with shuffle reductions.

The large improvement should therefore not be attributed to one isolated optimization.

It is the combined effect of:

```text
warp-owned data kept in registers
        +
far fewer shared-memory accesses
        +
far fewer shared-memory dependencies
        +
higher warp residency
        +
more scheduler latency-hiding opportunity
        +
shorter per-lane softmax dependency chains
```

These profiler ratios are not independent speedups and should not be multiplied together.

They are different measurements of the same underlying transition from shared-memory-heavy execution to a register-local dataflow.

## The improvement is primarily on chip

The global-load count remains close to Step 09:

```text
Global-load requests

Step 09    ~16.91 million
Step 10    ~17.30 million
```

The large execution improvement therefore does not come from another major reduction in K/V HBM loads.

Step 09 already gains most of that reuse from:

```text
BR = 64
BC = 64
```

Step 10 preserves the same tile sizes.

The important change is what happens after the data reaches the SM:

```text
similar global tile traffic
        ↓
much less shared intermediate traffic
        ↓
more register-local computation
```

This is why the step should be interpreted primarily as an **on-chip dataflow optimization**.

## Higher register use does not spill

The main cost of the redesign is increased register pressure:

```text
Registers / thread

72 → 124
```

Nsight Compute reports an allocated count of:

```text
128 registers / thread
```

However, the profiled `d = 64` kernel reports:

```text
Local-memory loads     0
Local-memory stores    0
```

The additional register state therefore does not spill into local memory for this profile.

This is important because otherwise the removed shared-memory traffic could have been replaced by a slower spill path.

Instead, the Q/S/P/O and softmax state remain genuinely register resident.

## Relation to FlashAttention-2

Step 09 adopts the split-Q ownership direction associated with FlashAttention-2:

```text
K, V shared by the block

warp 0 → independent Q rows
warp 1 → independent Q rows
warp 2 → independent Q rows
warp 3 → independent Q rows
```

Step 10 makes that ownership local at the data level:

```text
                  shared K / V
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼

    warp 0          warp 1          warp ...
 Q registers      Q registers
 S/P registers    S/P registers
 O registers      O registers
 m/l registers    m/l registers
```

This follows the general FlashAttention-2 direction of:

* partitioning Q across warps
* reducing communication between warps
* keeping warp-owned intermediate state local
* spending less execution time moving intermediate attention state

However, this kernel is still not a complete reproduction of the official FlashAttention-2 implementation.

In particular, K/V loading remains:

```text
synchronous
single-stage
STAGES = 1
```

and there is no copy/compute overlap.

Step 10 should therefore be understood as the **warp-local register-dataflow stage** built on top of the split-Q ownership introduced in Step 09.

## Remaining bottlenecks

Step 10 removes the dominant shared S/P/O dataflow, but several limits remain.

### Register pressure

The kernel now uses:

```text
124 registers / thread
```

with:

```text
128 allocated registers / thread
```

Nsight Compute reports a register residency limit of:

```text
4 blocks / SM
```

Registers are therefore now an explicit occupancy constraint.

Further register expansion cannot be treated as free.

### K/V shared-memory conflicts

K and V remain shared operands.

The profile still reports approximately:

```text
Shared-load average conflict     2.0-way
Shared-store average conflict    4.7-way
```

The conflict problem is substantially smaller than Step 09, but it has not disappeared.

### Synchronous K/V loading

The tile loop still follows:

```text
load K/V
    ↓
__syncthreads()
    ↓
compute
    ↓
__syncthreads()
```

There is no overlap between loading the next tile and computing the current tile.

### Math-pipe throttle

After most of the short-scoreboard dependency is removed, the largest warp-stall category becomes:

```text
Math-pipe throttle
≈ 4.75 cycles / issued instruction
```

The execution limit has therefore shifted away from the previous shared-memory dependency chain.

This does not mean that the Tensor pipeline itself is saturated.

Tensor-pipe utilization is approximately:

```text
40.9%
```

The kernel now contains a denser mixture of:

```text
Tensor Core MMA
softmax exponentials
max / sum operations
FP16 packing and conversion
register arithmetic
```

The next bottleneck is therefore more compute-oriented than in Step 09.

## Conclusion

Step 09 establishes warp ownership:

```text
one warp
   ↓
16 Q rows
   ↓
QK^T → softmax → PV
```

but keeps much of the corresponding state in shared memory.

Step 10 changes the dataflow to follow that ownership:

```text
Q
│
▼
Q registers
│
▼
QK^T
│
▼
S registers
│
▼
online softmax
│
▼
P in the same registers
│
▼
PV
│
▼
O registers
│
▼
final normalization
│
▼
HBM
```

Only K and V remain block-shared.

Nsight Compute shows the resulting transition clearly:

```text
shared-memory footprint    ↓
shared wavefronts          ↓
bank-conflict overhead     ↓
short-scoreboard stalls    ↓

warp residency             ↑
eligible warps             ↑
issue activity             ↑
Tensor utilization         ↑
```

The improvement is not a pure register-caching effect.

Step 10 combines a register-resident MMA dataflow, a four-lane softmax mapping, a much smaller shared-memory footprint, and substantially higher warp residency.

The result is a transition from a shared-memory-heavy implementation to a substantially more local **warp-owned register dataflow**.
