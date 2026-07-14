---
title: "Exercise: Scaled Dot-Product Attention"
description: Implement the core attention equation in NumPy and verify its key properties.
type: exercise
domain: 06-transformers
tags: [transformers, attention, numpy]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 06-transformers/lessons/attention-fundamentals
---

# Exercise: Scaled Dot-Product Attention

> Practice for **[Attention Fundamentals](../lessons/attention-fundamentals.md)**.

---

## Problem

Implement

$$
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

in NumPy and verify three properties: (1) each row of the attention-weight
matrix sums to 1, (2) the output shape is `(n_q, d_v)`, and (3) removing the
$\sqrt{d_k}$ scaling makes the softmax sharper (more peaked) at high dimensions.

## Requirements

- [ ] `attention(Q, K, V)` returning `(output, weights)`.
- [ ] Numerically stable softmax (subtract the row max).
- [ ] Asserts for properties (1) and (2).
- [ ] A short demonstration of property (3).

---

## Hints

<details>
<summary>Hint 1</summary>

Stable softmax: `e = np.exp(s - s.max(axis=-1, keepdims=True))`, then divide by
`e.sum(axis=-1, keepdims=True)`.

</details>

<details>
<summary>Hint 2</summary>

For (3): draw random Q,K with `d_k = 512`, compare `weights.max(axis=-1)` with
and without the scaling — unscaled rows concentrate near 1.0.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))   # stable: shift by row max
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                 # (n_q, n_k)
    weights = softmax(scores)
    return weights @ V, weights                     # (n_q, d_v), (n_q, n_k)


rng = np.random.default_rng(0)
Q, K, V = rng.standard_normal((3, 8)), rng.standard_normal((5, 8)), rng.standard_normal((5, 4))
out, w = attention(Q, K, V)
assert np.allclose(w.sum(axis=-1), 1.0)             # (1) rows are distributions
assert out.shape == (3, 4)                          # (2) (n_q, d_v)

# (3) why the scaling matters: at d_k=512, unscaled softmax saturates
Qb, Kb = rng.standard_normal((3, 512)), rng.standard_normal((5, 512))
w_scaled = softmax(Qb @ Kb.T / np.sqrt(512))
w_unscaled = softmax(Qb @ Kb.T)
print(w_scaled.max(axis=-1))    # moderate peaks, e.g. ~0.3-0.6
print(w_unscaled.max(axis=-1))  # ≈1.0 — one-hot, gradients vanish
```

**Explanation:** Dot products of $d_k$-dimensional random vectors have variance
proportional to $d_k$, so at large dimension the logits are huge and softmax
collapses to near one-hot — killing gradients. Dividing by $\sqrt{d_k}$
normalizes the variance back to ~1, keeping the distribution (and training)
healthy. This tiny constant is what makes deep attention stacks trainable.

</details>

---

## Navigation

- ⬆️ [Module 6 Exercises](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
