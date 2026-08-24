# Step 3. Online Softmax

## What this step implements

Replace the conventional max-then-sum softmax with an online softmax formulation.

Instead of computing the maximum and exponential sum in separate passes,  
each lane maintains a running softmax state that can later be extended to tiled attention.

## Online softmax state

For a set of processed values, maintain two statistics:

* $m$: running maximum
* $s$: exponential sum normalized with respect to $m$

$$
s = \sum_x e^{x-m}
$$

When a new value $x$ becomes the maximum, the previously accumulated sum must be rescaled:

$$
s_{\text{new}}
= s_{\text{old}}e^{m_{\text{old}}-x}+1,
\qquad
m_{\text{new}}=x
$$

Otherwise, its contribution can be added directly:

$$
s_{\text{new}}
= s_{\text{old}}+e^{x-m}.
$$

This allows the maximum and exponential sum to be updated together in a single pass.

```cuda
float m = -FLT_MAX;
float s = 0.0f;

for (int j = lane; j < N; j += 32) {
    const float x = row[j];

    if (x <= m) {
        s += __expf(x - m);
    } else {
        s = s * __expf(m - x) + 1.0f;
        m = x;
    }
}
```

## Merging partial states

Each lane independently computes a partial $(m, s)$ state.

Two states $(m_a, s_a)$ and $(m_b, s_b)$ can be combined by rescaling both exponential sums to a common maximum:

$$
m = \max(m_a, m_b)
$$

$$
s =
e^{m_a-m}s_a
+
e^{m_b-m}s_b
$$

The kernel uses the same warp-shuffle reduction pattern introduced in Step 02:

```cuda
for (int offset = 16; offset > 0; offset >>= 1) {
    const float m_other = __shfl_xor_sync(0xffffffff, m, offset);
    const float s_other = __shfl_xor_sync(0xffffffff, s, offset);

    const float m_new = fmaxf(m, m_other);

    s = s * __expf(m - m_new)
      + s_other * __expf(m_other - m_new);

    m = m_new;
}
```
