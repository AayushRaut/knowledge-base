---
title: "Exercise: Sinusoidal Positional Encoding"
description: Implement the original transformer's positional encoding and inspect its structure.
type: exercise
domain: 06-transformers
tags: [transformers, positional-encoding, numpy]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 06-transformers/lessons/positional-encoding
---

# Exercise: Sinusoidal Positional Encoding

> Practice for **[Positional Encoding](../lessons/positional-encoding.md)**.

---

## Problem

Implement the sinusoidal positional encoding from *Attention Is All You Need*:

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

where $pos$ is the position, $i$ indexes the dimension pair, and $d$ is the model
dimension. Build the `(seq_len, d)` matrix and confirm two properties: values are
bounded in $[-1, 1]$, and nearby positions have more similar encodings than
distant ones.

## Requirements

- [ ] `positional_encoding(seq_len, d)` returns a `(seq_len, d)` array.
- [ ] Even dimensions use sin, odd use cos.
- [ ] Show that dot-product similarity between rows decays with distance.

---

## Hints

<details>
<summary>Hint 1</summary>

Compute the division term once:
`div = 10000 ** (np.arange(0, d, 2) / d)`, then `angles = pos / div`.

</details>

<details>
<summary>Hint 2</summary>

Fill `PE[:, 0::2] = sin(angles)` and `PE[:, 1::2] = cos(angles)`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def positional_encoding(seq_len: int, d: int) -> np.ndarray:
    pos = np.arange(seq_len)[:, None]                 # (seq_len, 1)
    div = 10000.0 ** (np.arange(0, d, 2) / d)         # (d/2,)
    angles = pos / div                                # (seq_len, d/2)
    pe = np.zeros((seq_len, d))
    pe[:, 0::2] = np.sin(angles)                      # even dims
    pe[:, 1::2] = np.cos(angles)                      # odd dims
    return pe


pe = positional_encoding(50, 128)
assert pe.shape == (50, 128)
assert pe.min() >= -1.0 and pe.max() <= 1.0

# Similarity decays with distance: row 0 vs its neighbours
sims = pe @ pe[0]
print(sims[0], sims[1], sims[10], sims[40])  # highest at 0, decreasing away
```

**Explanation:** Each dimension pair oscillates at a different frequency (fast
for small $i$, slow for large $i$) — like a smooth, continuous binary clock, so
every position gets a unique fingerprint. Because it's a fixed function of
position (not learned), it extrapolates to longer sequences in principle, and
relative offsets can be expressed as linear combinations — the property later
schemes like RoPE exploit more directly. You **add** this matrix to the token
embeddings (same shape), injecting order into an otherwise permutation-blind model.

</details>

---

## Navigation

- ⬆️ [Module 6 Exercises](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
