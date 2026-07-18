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

## Code

### `online_softmax_kernel()`

The kernel assigns **one warp to each row** of the score matrix. Each lane processes a strided subset of the row:

```cuda
for (int j = lane; j < N; j += 32)
```

Lane 0 processes elements (0, 32, 64, $\ldots$), lane 1 processes elements (1, 33, 65, $\ldots$), and so on.  
Because adjacent lanes access adjacent elements during each iteration, global-memory accesses are coalesced.

Each lane independently maintains an online softmax state:

* $m$ : the maximum value seen by the lane
* $s$ : the exponential sum normalized with respect to $m$

The state represented by $(m, s)$ is

$$
s = \sum_{x \in \mathcal{X}} e^{x-m},
$$

where $\mathcal{X}$ is the set of elements processed by that lane.

#### Per-element Online Update

```cuda
const float x = row[j];
```

For each new value $x$, the lane updates its running state.

If $x$ does not exceed the current maximum, the normalization scale remains unchanged:

$$
m^{\text{new}} = m^{\text{old}},
$$

$$
s^{\text{new}} =
s^{\text{old}}
+
e^{x-m^{\text{old}}}.
$$

The following code implements this

```cuda
if (x <= m) {
    s += __expf(x - m);
```

If $x$ becomes the new maximum, the previously accumulated sum must be rescaled to the new maximum:

$$
m^{\text{new}} = x,
$$

$$
s^{\text{new}} = e^{m^{\text{old}}-x}s^{\text{old}} + 1.
$$

The $1$ is the contribution of the new element because

$$
e^{x-x}=1.
$$

This update is implemented as

```cuda
else {
    s = s * __expf(m - x) + 1.0f;
    m = x;
}
```

Therefore, the maximum and exponential sum are computed together in a single pass over each lane's elements.  
Unlike the previous softmax implementation, the kernel does not require one full pass for the maximum followed by another full pass for the exponential sum.

The initial maximum is set to `-FLT_MAX` rather than negative infinity:

```cuda
float m = -FLT_MAX;
float s = 0.0f;
```

Using a finite initial value prevents undefined expressions such as

$$
-\infty - (-\infty),
$$

which could otherwise produce `NaN` when empty states are merged.

#### Warp-level State Merge

After each lane has processed its subset of the row, the warp contains 32 independent softmax states:

$$
(m_0,s_0), (m_1,s_1), \ldots, (m_{31},s_{31}).
$$

These states must be combined to obtain the statistics for the complete row.

For two states $(m_a,s_a)$ and $(m_b,s_b)$, the merged maximum is

$$
m = \max(m_a,m_b).
$$

Their exponential sums are rescaled to this common maximum and then added:

$$
s =
e^{m_a-m}s_a
+
e^{m_b-m}s_b.
$$

The kernel performs this merge using warp shuffle operations:

```cuda
for (int offset = 16; offset > 0; offset >>= 1) {
    const float m_other = __shfl_xor_sync(0xffffffff, m, offset);
    const float s_other = __shfl_xor_sync(0xffffffff, s, offset);
    const float m_new = fmaxf(m, m_other);
    s = s * __expf(m - m_new) + s_other * __expf(m_other - m_new);
    m = m_new;
}
```

At each reduction step, lanes exchange their $(m,s)$ states and merge them. The offsets (16, 8, 4, 2,) and $1$ progressively combine all 32 lane-local states.

**After the reduction, every lane holds the same row-level statistics**:

$$
m = \max_j S_j,
$$

$$
s = \sum_j e^{S_j-m}.
$$

**No shared memory or block-level synchronization is required because all communication occurs within a single warp.**

#### Final Normalization

Once the row-level maximum and exponential sum have been obtained, the kernel makes a second pass over the row to write the normalized softmax values:

$$
P_j =
\frac{e^{S_j-m}}{s}.
$$

Each lane then normalizes the same elements that it processed during the first pass:

```cuda
const float inv_sum = 1.0f / s;
for (int j = lane; j < N; j += 32) {
    row[j] = __expf(row[j] - m) * inv_sum;
```

The score matrix is updated in place, so the original score values in `S` are replaced by the normalized probability matrix $P$.

Although the maximum and exponential sum are computed together online, a second memory pass is still required to write the final normalized values.  
The kernel therefore avoids a separate maximum pass, **but it does not yet fuse softmax with the subsequent $PV$ multiplication**.  
**This fusion will be introduced in the next step**.

In this step, **the row is not yet divided into explicit contiguous tiles**.  
Instead, each lane computes a partial softmax state for its strided subset of the row, and these states are merged at the warp level.  
**The next step extends this mechanism to explicit tile-wise processing**.






