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

The implementation uses **tiling, online softmax, kernel fusion, recomputation, and other techniques described below**.

## Implementation

We implement these techniques step by step and evaluate how each step affects performance.  
Detailed implementation notes for each step are provided in `/docs` directory.

### Benchmark
B = 00, H = 00, N = 00, d = 00
| Step | Technique | Time | Speedup vs. prev. | Speedup vs. Baseline | Performance vs. PyTorch SDPA FlashAttention (%) |
|---|---|---:|---:|---:|---:|
| 00 | Naive Standard Attention (Baseline) | 0.0 ms | None | 0.0x | 0.0 % |
| 01 | cuBLAS + Warp-reduction Softmax | 0.0 ms | 0.0x | 0.0x | 0.0 % |
| 02 | Online Softmax | 0.0 ms | 0.0x | 0.0x | 0.0 % |
| 03 | Naive Fused Attention (SRAM Tiling) | 0.0 ms | 0.0x | 0.0x | 0.0 % |
| 04 | Memory Access Optimization | 0.0 ms | 0.0x | 0.0x | 0.0 % |
| 05 | Data Type Optimization | 0.0 ms | 0.0x | 0.0x | 0.0 % |
| 06 | WMMA TensorCore | 0.0 ms | 0.0x | 0.0x | 0.0 % |


