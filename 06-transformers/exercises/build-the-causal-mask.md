---
title: "Exercise: Build the Causal Mask"
description: Construct and apply the upper-triangular mask that stops a decoder from seeing the future.
type: exercise
domain: 06-transformers
tags: [transformers, causal-mask, attention]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 06-transformers/lessons/masked-and-cross-attention
---

# Exercise: Build the Causal Mask

> Practice for **[Masked and Cross-Attention](../lessons/masked-and-cross-attention.md)**.

---

## Problem

Implement `causal_attention(Q, K, V)` for a single sequence so that position $i$
can attend only to positions $\le i$. The mask must be added to the scores
**before** softmax (as $-\infty$), not applied after.

For a 4-token sequence, print the attention-weight matrix and confirm it is
lower-triangular (zeros above the diagonal).

## Requirements

- [ ] Build the mask with `np.triu` (or `torch.triu`).
- [ ] Add `-inf` to masked positions before softmax.
- [ ] Weight matrix is lower-triangular; each row still sums to 1.

---

## Hints

<details>
<summary>Hint 1</summary>

`np.triu(np.ones((n, n)), k=1)` is 1 strictly above the diagonal — those are the
positions to forbid.

</details>

<details>
<summary>Hint 2</summary>

Set masked scores to `-np.inf`; after softmax those become exactly 0, and the
remaining (allowed) entries renormalize to sum to 1.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    n, d_k = Q.shape
    scores = Q @ K.T / np.sqrt(d_k)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # True = forbidden (future)
    scores = np.where(mask, -np.inf, scores)          # add -inf BEFORE softmax
    weights = softmax(scores)
    return weights @ V, weights


rng = np.random.default_rng(0)
Q = K = rng.standard_normal((4, 8))
V = rng.standard_normal((4, 8))
_, w = causal_attention(Q, K, V)
print(np.round(w, 2))
# Row 0 attends only to token 0; row 3 attends to 0..3. Upper triangle is 0.
assert np.allclose(np.triu(w, k=1), 0.0)
assert np.allclose(w.sum(axis=-1), 1.0)
```

**Explanation:** During training all positions are scored at once for speed, but
a language model must predict token $i{+}1$ from tokens $\le i$ only. Masking
with $-\infty$ *before* softmax removes future tokens from the normalization, so
each row remains a valid probability distribution over just the allowed
positions. Zeroing *after* softmax would break that (rows wouldn't sum to 1).

</details>

---

## Navigation

- ⬆️ [Module 6 Exercises](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
