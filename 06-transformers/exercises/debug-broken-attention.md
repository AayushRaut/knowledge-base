---
title: "Debugging: Fix the Broken Attention"
description: Three subtle bugs in an attention implementation — wrong softmax axis, missing scaling, and a transpose error.
type: exercise
domain: 06-transformers
tags: [transformers, debugging, attention, softmax]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 06-transformers/lessons/attention-fundamentals
  - 06-transformers/lessons/self-attention
---

# Debugging: Fix the Broken Attention

> Practice for **[Attention Fundamentals](../lessons/attention-fundamentals.md)**
> and **[Self-Attention](../lessons/self-attention.md)**.

---

## Problem

This attention function runs without error and returns the right *shape*, but the
outputs are wrong. It has **three** bugs. Find them all.

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    # Q, K, V: (batch, seq, d_k)
    scores = Q @ K                       # BUG 1
    weights = F.softmax(scores, dim=0)   # BUG 2
    out = weights @ V
    return out                           # BUG 3: where did scaling go?
```

## Requirements

- [ ] Identify all three bugs and the symptom of each.
- [ ] Produce a correct version.
- [ ] Note why "right shape, wrong values" bugs are dangerous.

---

## Hints

<details>
<summary>Hint 1</summary>

To multiply `Q (…, s, d)` by `K (…, s, d)` you must transpose K's last two dims.

</details>

<details>
<summary>Hint 2</summary>

Softmax must run over the **key** axis (the last one), so each query's weights
sum to 1. And scores need the $1/\sqrt{d_k}$ scaling.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

**The three bugs:**

1. **`Q @ K`** — shape-incompatible in general and semantically wrong; it must be
   `Q @ K.transpose(-2, -1)` to get the `(seq_q, seq_k)` score matrix. (With
   equal seq lengths it may not error, but it computes nonsense.)
2. **`softmax(..., dim=0)`** — normalizes over the batch dimension, so weights
   don't sum to 1 per query and information leaks across the batch. It must be
   `dim=-1` (over keys).
3. **Missing `/ sqrt(d_k)`** — without scaling, large-dimension dot products
   saturate the softmax and gradients vanish.

```python
import torch
import torch.nn.functional as F


def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)   # fixes 1 and 3
    weights = F.softmax(scores, dim=-1)               # fixes 2 (over keys)
    return weights @ V
```

**Why these are dangerous:** all three preserve the output *shape*, so the code
runs and tensors flow — the model just learns poorly or not at all, with no
error to point you at the cause. Shape-correct-but-value-wrong bugs are the
hardest class in deep learning; guard against them with unit tests that check
properties (weights sum to 1, attention matches a hand-computed tiny example),
not just shapes.

</details>

---

## Navigation

- ⬆️ [Module 6 Exercises](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
