# Flash-Attention-CUDA-from-scratch
CUDA implementation of FlashAttention for learning

[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) provides the official implementation.

## Get Started
```bash
$ source env.sh         # Run this command if you need the environment for RTX 5090
$ python benchmark.py
  # python benchmark.py --preset llm
  # python benchmark.py -B 8 -H 16 -N 4096 -d 64
```

## What is FlashAttention
Fast and Memory-efficient Attention

### How it works
FlashAttention’s advantage comes from GPU hardware characteristics.  
The latest GPUs have enormous computing power (TFLOPs), but the memory bandwidth is relatively limited.  

**Standard Attention needs to read and write $N \times N$ matrices in HBM several times**.  
This results in $O(N^2)$ memory accesses and makes Self-Attention be considered a **memory-bound algorithm**.

**FlashAttention views the main bottleneck of Self-Attention as memory traffic rather than FLOPs**

<img width="350" height="350" alt="image" src="https://github.com/user-attachments/assets/0f290693-10c8-47b4-a553-33e363fa3b93" />

**FlashAttention computes Self-Attention in on-chip tiles (SRAM), without storing the full $N \times N$ matrices in HBM**.  

The implementation uses **tiling, online softmax, kernel fusion, and other techniques described below**.

## Implementation

We implement these techniques step by step and evaluate how each step affects performance.  
**Detailed implementation notes and benchmarks for each step are provided in `/docs` directory.**

## Benchmark

**Currently achieves ~70% of PyTorch SDPA FlashAttention speed.**  
Further kernel optimizations are planned.

### Environment

NVIDIA RTX 5090 32GB  
`PyTorch 2.7.0`, `CUDA 12.8`  

B=8, H=16, N=4096, d=64 (10 warm-ups, median value from 50 iterations)  

### References

| Reference | dtype | Latency | TFLOPS |
|---|---|---|---|
| PyTorch matmul + softmax | FP32 | 38.025 ms | 14.5 |
| PyTorch SDPA FlashAttention | FP16 | 2.482 ms | 221.5 |

### Track A: Unfused kernel

| Step | Technique | dtype | Latency | vs. prev. | TFLOPS |
|---|---|---|---|---|---|
| 00 | Naive Standard Attention | FP32 | 253.330 ms | - | 2.2 |
| 01 | CuBLAS GEMM | FP32 | 64.719 ms | 3.91x | 8.5 |
| 02 | Warp-reduction Softmax | FP32 | 28.816 ms | 2.25x | 19.1 |
| 03 | Online Softmax | FP32 | 30.135 ms | 0.96x | 18.2 |

### Track B: Fused FlashAttention Kernel

| Step | Technique | dtype | Latency | vs. prev. | TFLOPS | % SDPA |
|---|---|---|---|---|---|---|
| 04 | Naive Fused Attention (SRAM Tiling) | FP32 | 327.255 ms | 0.09x | 1.7 | 0.8 % |
| 05 | Coalescing + Vectorized Load | FP32 | 119.058 ms | 2.75x | 4.6 | 2.1 % |
| 06 | Bank Conflict Avoidance (Swizzling) | FP32 | 63.985 ms | 1.86x | 8.6 | 3.9 % |
| 07 | Half-Precision (FP16) | FP16 | 57.849 ms | 1.11x | 9.5 | 4.3 % |
| 08 | WMMA TensorCore | FP16 | 38.430 ms | 1.51x | 14.3 | 6.5 % |
| 09 | Double Buffering | FP16 | 23.896 ms | 1.61x | 23.0 | 10.4 % |
| 10 | Register-Resident Accumulators | FP16 | **3.566 ms** | 6.70x | **154.2** | **69.6 %** |
