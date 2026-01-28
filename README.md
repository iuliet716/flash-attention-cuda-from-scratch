# Flash-Attention-CUDA-from-scratch
CUDA implementation of FlashAttention for learning

[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) provides the official implementation.

## Get Started

```bash
$ nvcc flash_attn.cu -o build/flash_attn

$ ./build/flash_attn
```

## What is FlashAttention

Fast and Memory-efficient Attention

## Key Idea: IO-Awareness

The bottleneck lies not in FLOPS but in memory traffic.

In standard attention,  
the intermediate matrices $S = QK^T$ and $P = \text{softmax}(S)$ are read from and written to HBM, resulting in $O(N²)$ memory accesses.

In FlashAttention, these operations are performed entirely on-chip in SRAM, **without materializing $S$ or $P$ in HBM.**

## Performance

### HBM accesses
- Standard Attention : $\Theta(Nd + N^2)$
- FlashAttention : $\Theta(\displaystyle\frac{N^2d^2}{M})$

$N$ : sequence length  
$d$ : head dimension  
$M$ : size of SRAM with $d \le M \le Nd$

For typical values of $d$ (64-128) and $M$ (around 100KB), $d^2 << M$ 

## How it works

### Tiling

It does not compute the entire attention at once, but processes it in **small blocks within SRAM**.

1. Splits the $Q, K, V$ matrices into smaller blocks.
2. Loads these blocks from the slow HBM into the fast on-chip SRAM.
3. Fuses all operations (MatMul, Softmax, and the final MatMul with $V$) into a **single kernel**.
4. Executes these **fused operations** **on the small blocks** entierly within SRAM.

<img width="1790" height="690" alt="image" src="https://github.com/user-attachments/assets/ef365239-e55b-4aa6-ac59-22dfe63a6e91" />

### Recomputation

Standard Attention’s backward pass needs intermediate $N \times N$ matrices ($S, P$) calculated during the forward pass, resulting in $O(N^2)$ HBM aceesses.

Instead of storing the $N \times N$ matrices, FlashAttention **recomputes $S, P$ for each block on-the-fly.**  
This does increase FLOPs slightly but it’s acceptable because the dominant bottleneck is HBM I/O, not compute.

### Online Softmax

The Softmax computation requires the maximum and the sum of exponentials for each row.  
Since we divide each row into blocks, **block-wise rescaling** is needed to **ensure the exact output**.
