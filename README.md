# Flash-Attention-CUDA-from-scratch
CUDA implementation of FlashAttention for learning and Benchmark against Standard Attention baselines

[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) provides the official implementation.

## What is FlashAttention
Fast and Memory-efficient Attention

### How it works
FlashAttention’s advantage comes from GPU hardware characteristics.  
The latest GPUs have enormous computing power (TFLOPs), but the memory bandwidth is relatively limited.  

**Standard Attention needs to read and write $N \times N$ matrices in HBM several times**, while calculating Self-Attention.  
This results in $O(N^2)$ memory accesses and makes Self-Attention be considered a **memory-bound algorithm**.

For these reasons, **The bottleneck lies not in FLOPS but in memory traffic**.

<img width="350" height="350" alt="image" src="https://github.com/user-attachments/assets/0f290693-10c8-47b4-a553-33e363fa3b93" />

To reduce memory traffic, **FlashAttention computes Self-Attention in on-chip tiles (SRAM), without storing the full $N \times N$ matrices in HBM**.

## Implementation Details

### Must-have  
- [ ] Tiling
- [ ] Recomputation
- [ ] Online Softmax

### To-do
- [ ] Warp-level optimization
- [ ] MMA (Matrix Multiply and Accumulate)
- [ ] Memory access optimization
- [ ] Dropout
- [ ] Multi-head optimization
- [ ] Backward
- [ ] Mixed Precision

## Benchmark

### Baseline: Standard Attention

To ensure parity, well-optimized Standard Attentions are required.  
First, we need **reliable Standard Attention baselines for benchmark**.
- [Pytorch Attention ('Math' Backend)](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Nvidia Megatron-LM ('local', 'fused' Backend)](https://github.com/NVIDIA/Megatron-LM)

Second, we should implement **Standard Attention in CUDA to highlight its memory bottleneck**.

Implementation References
> [CUTLASS Reference Implementation](https://github.com/NVIDIA/cutlass/blob/main/examples/41_fused_multi_head_attention/fused_multihead_attention_fixed_seqlen.cu)  
> [PyTorch C++ Native 'Math' Backend](https://github.com/pytorch/pytorch/blob/f50e264a8688b966989efab4fbbe547d5eaf3c5b/aten/src/ATen/native/transformers/attention.cpp#L850)  
> [Megatron-LM CoreAttention](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer)


