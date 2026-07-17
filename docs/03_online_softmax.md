# Step 3. Online Softmax

## What this step implements

We replace the row-wise softmax kernel with an online softmax algorithm that performs **incremental tile-wise normalization**.

## What is this for?

Online softmax computes the exact result incrementally **without storing the entire row in on-chip memory**.  

The implementation is slower than the previous warp-reduction softmax kernel.  
However, it introduces the tile-wise normalization mechanism required for the next steps.  

It will later be fused with the attention computation to reduce memory access and kernel-launch overhead.

## How Online Softmax Works

Softmax requires the **maximum value and the sum of exponentials for each row**.  
When a row is processed tile by tile, **these statistics must be updated and combined across tiles**.  
Online softmax achieves this by maintaining a running maximum and normalization factor.

### Tile-wise Statistics Update

For each tile, the algorithm first computes the local maximum and exponential sum:

$$
\tilde{m} = \max_{x \in \text{tile}} x
$$

$$
\tilde{l} = \sum_{x \in \text{tile}} e^{x - \tilde{m}}
$$

It then updates the running maximum:

$$
m^{\text{new}} = \max\left(m^{\text{old}}, \tilde{m}\right)
$$

The previous and current exponential sums are rescaled to the new maximum before being combined:

$$
l^{\text{new}} =
e^{m^{\text{old}} - m^{\text{new}}} l^{\text{old}}
+
e^{\tilde{m} - m^{\text{new}}} \tilde{l}
$$

This rescaling places statistics computed with different local maxima on the same numerical scale.  

After all tiles are processed, $m$ and $l$ are equivalent to the maximum and normalization factor computed over the full row.








