# Flash-Attention-CUDA-from-scratch
CUDA implementation of FlashAttention for learning

[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) provides the official implementation.

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
Detailed implementation notes for each step are provided in `/docs` directory.

### Benchmark
B = 00, H = 00, N = 00, d = 00
| Step | Technique | Latency | Speedup vs. prev. | Speedup vs. Baseline | TFLOPS | Speed vs. PyTorch SDPA FlashAttention (%) |
|---|---|---:|---:|---:|---:|---:|
| 00 | Naive Standard Attention (Baseline) | 0.0 ms | N/A | N/A | 0.0 | 0.0 % |
| 01 | cuBLAS GEMM | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % |
| 02 | Warp-reduction Softmax | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % |
| 03 | Online Softmax | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % |
| 04 | Naive Fused Attention (SRAM Tiling) | 0.0 ms |0.0x | 0.0x | 0.0 | 0.0 % |
| 05 | Coalescing + Vectorized Load | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % |
| 06 | Bank Conflict Avoidance (Swizzling) | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % |
| 07 | Half-Precision (FP16) | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % |
| 08 | WMMA TensorCore | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % |
| 09 | Double Buffering | 0.0 ms | 0.0x | 0.0x | 0.0 | 0.0 % |

#### Note
The last two columns (TFLOPS, Speed vs. SDPA) show how each step progressively approaches PyTorch SDPA FlashAttention.  
Steps 00–06 run in FP32, so part of the gap vs. SDPA (FP16) is inherent to precision, not kernel quality.  
Likewise, TFLOPS reflects each dtype's hardware peak — FP32 steps have a much lower ceiling than 07+ steps.


