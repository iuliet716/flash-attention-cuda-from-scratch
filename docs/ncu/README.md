# Nsight Compute Profiling

This directory contains Nsight Compute analyses for the optimization steps in this repository.

The profiles use the same representative workload as the main benchmark:

```text
GPU: NVIDIA GeForce RTX 5090
B = 8
H = 16
N = 4096
d = 64
```

Profiling is performed with:

```text
warmup = 0
iterations = 1
Nsight Compute --set full
```

The benchmark itself uses separate warm-up and repeated timing runs.  
Nsight Compute is used only for kernel-level profiling and should not be interpreted as the source of the latency values reported in the main README.

## General command

```bash
ncu \
  --set full \
  --kernel-name-base function \
  --kernel-name regex:"<kernel-name>" \
  --launch-skip <N> \
  --launch-count 3 \
  -o <output-name> \
  -f \
  $(pyenv which python) benchmark.py \
  --steps <step> \
  --warmup 0 \
  --iters 1
```

`--kernel-name` limits profiling to the kernel relevant to the selected step.

`--launch-skip` skips earlier launches of the same kernel that occur during correctness checks or setup.

`--launch-count 3` profiles three representative launches rather than relying on a single launch.

The generated `.ncu-rep` files are intentionally excluded from Git because they are binary profiler artifacts.  
The Markdown files in this directory record the relevant metrics and interpretation.

## Fused-kernel steps

For Steps 04–10, the primary kernel is:

```text
fused_attention_kernel
```

For example, Step 10 can be profiled with:

```bash
ncu \
  --set full \
  --kernel-name-base function \
  --kernel-name regex:"fused_attention_kernel" \
  --launch-skip 6 \
  --launch-count 3 \
  -o step10_register_dataflow \
  -f \
  $(pyenv which python) benchmark.py \
  --steps 10 \
  --warmup 0 \
  --iters 1
```

The exact `--launch-skip` value can depend on how many validation launches occur before the timed kernel.  
Verify the Nsight Compute console output when reproducing a profile.

## Interpreting the results

The profiler documents focus on metrics such as:

```text
SM throughput
Tensor-pipe utilization
L1 / L2 / DRAM throughput

registers per thread
dynamic shared memory per block
theoretical / achieved occupancy

active and eligible warps
warp-stall reasons

global-memory requests and sectors
shared-memory requests, wavefronts, and bank conflicts
```

Profiler ratios are used to explain changes in execution behavior.

They should not be treated as independent speedup factors or multiplied together to predict end-to-end latency.
