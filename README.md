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
| 00 | Naive Standard Attention (Baseline) | 287.232 ms | N/A | N/A | 1.9 | 14.6 % | 0.9 % |
| 01 | cuBLAS GEMM | 73.695 ms | 3.90x | 3.90x | 7.5 | 56.8 % | 3.3 % |
| 02 | Warp-reduction Softmax | 31.958 ms | 2.31x | 8.99x | 17.2 | 131.1 % | 7.7 % |
| 03 | Online Softmax | 32.768 ms | 0.98x | 8.77x | 16.8 | 127.8 % | 7.5 % |
| 04 | Naive Fused Attention (SRAM Tiling) | 0.0 ms |0.0x | 0.0x | 0.0 | 0.0 % | 0.0 % |
| 05 | Coalescing + Vectorized Load | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % | 0.0 % |
| 06 | Bank Conflict Avoidance (Swizzling) | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % | 0.0 % |
| 07 | Half-Precision (FP16) | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % | 0.0 % |
| 08 | WMMA TensorCore | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % | 0.0 % |
| 09 | Double Buffering | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % | 0.0 % |
| -- | PyTorch matmul + softmax | 41.887 ms | N/A | 6.86x | 13.1 | 100.0 % | 5.9 % |
| -- | PyTorch SDPA FlashAttention | 2.458 ms | N/A | 116.83x | 223.6 | 1704.1 % | 100.0 % |

#### Note
The last two columns show how each step progressively approaches the two PyTorch references (FP32 matmul + softmax, and FP16 SDPA FlashAttention).  
*Steps 00–06 run in FP32, so part of the gap vs. SDPA (FP16) is inherent to precision, not kernel quality.  
*Likewise, TFLOPS reflects each dtype's hardware peak — FP32 steps have a much lower ceiling than 07+ steps.


