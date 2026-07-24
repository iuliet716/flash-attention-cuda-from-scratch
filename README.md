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

### Benchmark
B=8, H=16, N=4096, d=64 (10 warm-ups, median value from 50 iterations)  
NVIDIA RTX 5090 32GB
| Step | Technique | Latency | Speedup vs. prev. | Speedup vs. Baseline | TFLOPS* | Speed vs. PyTorch matmul + softmax (%) | Speed vs. PyTorch SDPA FlashAttention* (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| 00 |[Naive Standard Attention (Baseline)](docs/00_naive.md) | 253.330 ms | N/A | N/A | 2.2 | 15.0 % | 1.0 % |
| 01 | cuBLAS GEMM | 64.719 ms | 3.91x | 3.91x | 8.5 | 58.8 % | 3.8 % |
| 02 | Warp-reduction Softmax | 28.816 ms | 2.25x | 8.79x | 19.1 | 132.0 % | 8.6 % |
| 03 | Online Softmax | 30.135 ms | 0.96x | 8.41x | 18.2 | 126.2 % | 8.2 % |
| 04 | Naive Fused Attention (SRAM Tiling) | 327.255 ms | 0.09x | 0.77x | 1.7 | 11.6 % | 0.8 % |
| 05 | Coalescing + Vectorized Load | 119.058 ms | 2.75x | 2.13x | 4.6 | 31.9 % | 2.1 % |
| 06 | Bank Conflict Avoidance (Swizzling) | 63.985 ms | 1.86x | 3.96x | 8.6 | 59.4 % | 3.9 % |
| 07 | Half-Precision (FP16) | 57.849 ms | 1.11x | 4.38x | 9.5 | 65.7 % | 4.3 % |
| 08 | WMMA TensorCore | 38.430 ms | 1.51x | 6.59x | 14.3 | 98.9 % | 6.5 % |
| 09 | Double Buffering | 23.896 ms | 1.61x | 10.60x | 23.0 | 159.1 % | 10.4 % |
| 10 | Register-Resident Accumulators | 3.566 ms | 6.70x | 71.04x | 154.2 | 1066.3 % | 69.6 % |
| -- | PyTorch matmul + softmax | 38.025 ms | N/A | 6.66x | 14.5 | 100.0 % | 6.5 % |
| -- | PyTorch SDPA FlashAttention | 2.482 ms | N/A | 102.08x | 221.5 | 1532.2 % | 100.0 % |

#### Note
- The last two columns show how each step progressively approaches the two PyTorch references.
- *Steps 00–06 run in FP32, so part of the gap vs. SDPA (FP16) is inherent to precision, not kernel quality.
- *Likewise, TFLOPS reflects each dtype's hardware peak — FP32 steps have a much lower ceiling than 07+ steps.