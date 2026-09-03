# Documentation

Each step has a design note describing the implementation change.  
Selected steps also include an [Nsight Compute](#nsight-compute-profiling) analysis that explains the corresponding GPU execution behavior.

| Step | Optimization                        | Design                           | Nsight Compute                           |
| ---- | ----------------------------------- | -------------------------------- | ---------------------------------------- |
| 00   | Naive Standard Attention            | [doc](./00_naive.md)             | [profile](./ncu/00_naive.md)             |
| 01   | cuBLAS GEMM                         | [doc](./01_cublas.md)            | [profile](./ncu/01_cublas.md)            |
| 02   | Warp-reduction Softmax              | [doc](./02_warp_softmax.md)      | [profile](./ncu/02_warp_softmax.md)      |
| 03   | Online Softmax                      | [doc](./03_online_softmax.md)    | —                                        |
| 04   | Naive Fused Attention (SRAM Tiling) | [doc](./04_naive_fused.md)       | [profile](./ncu/04_naive_fused.md)       |
| 05   | Coalescing + Vectorized Load        | [doc](./05_vectorized_load.md)   | [profile](./ncu/05_vectorized_load.md)   |
| 06   | Shared-Memory Swizzling             | [doc](./06_swizzling.md)         | [profile](./ncu/06_swizzling.md)         |
| 07   | Half-Precision (FP16)               | [doc](./07_fp16.md)              | [profile](./ncu/07_fp16.md)              |
| 08   | WMMA Tensor Cores                   | [doc](./08_wmma.md)              | [profile](./ncu/08_wmma.md)              |
| 09   | Split-Q Warp Partitioning           | [doc](./09_split_q.md)           | [profile](./ncu/09_split_q.md)           |
| 10   | Warp-Owned Register Dataflow        | [doc](./10_register_dataflow.md) | [profile](./ncu/10_register_dataflow.md) |

## Nsight Compute Profiling

Profiler analyses use the same representative workload as the main benchmark:

```text
GPU: NVIDIA GeForce RTX 5090
B = 8
H = 16
N = 4096
d = 64
```

Nsight Compute is used for kernel-level analysis only.  
Latency values in the main README come from separate warm-up and repeated benchmark runs.

Profiles are collected with:

```text
warmup = 0
iterations = 1
Nsight Compute --set full
```

General command:

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

`--kernel-name` selects the kernel being analyzed, while `--launch-skip` skips validation or setup launches that occur before it.  
The required skip count may vary by step.

The generated `.ncu-rep` files are excluded from Git.
