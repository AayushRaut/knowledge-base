---
title: "Exercise: Count a Transformer Block's Parameters"
description: Derive the parameter count of one transformer block by hand, then verify with PyTorch.
type: exercise
domain: 06-transformers
tags: [transformers, parameters, architecture]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 30 min
reinforces:
  - 06-transformers/lessons/transformer-architecture
---

# Exercise: Count a Transformer Block's Parameters

> Practice for **[The Transformer Architecture](../lessons/transformer-architecture.md)**.

---

## Problem

For one encoder block with $d_{model}=512$, $h=8$ heads, and FFN inner dimension
$d_{ff}=2048$, compute the number of parameters by hand, broken down by
component, then verify with PyTorch. Ignore LayerNorm's tiny parameter counts at
first, then add them.

## Requirements

- [ ] Hand-derive params for: multi-head attention ($W^Q, W^K, W^V, W^O$) and
      the FFN (two linear layers).
- [ ] Add the two LayerNorms.
- [ ] Verify the total against a PyTorch module with `sum(p.numel() ...)`.
- [ ] State which component dominates.

---

## Hints

<details>
<summary>Hint 1</summary>

A `Linear(in, out)` has `in*out + out` parameters (weights + bias). The four
attention projections are each `d_model × d_model` (+ bias).

</details>

<details>
<summary>Hint 2</summary>

FFN: `Linear(d_model, d_ff)` then `Linear(d_ff, d_model)`. LayerNorm has
`2 * d_model` params (scale + shift).

</details>

---

## Solution

<details>
<summary>Show solution</summary>

**By hand ($d=512$, $d_{ff}=2048$), with biases:**

- Attention projections: $4 \times (512 \times 512 + 512) = 4 \times 262{,}656 = 1{,}050{,}624$
- FFN: $(512 \times 2048 + 2048) + (2048 \times 512 + 512) = 1{,}050{,}624 + 1{,}049{,}088 = 2{,}099{,}712$
- LayerNorms: $2 \times (2 \times 512) = 2{,}048$
- **Total ≈ 3,152,384** (~3.15M per block)

```python
import torch.nn as nn

d_model, h, d_ff = 512, 8, 2048


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, h, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)


total = sum(p.numel() for p in Block().parameters())
print(f"{total:,}")   # ≈ 3.15M (nn.MultiheadAttention packs QKV; count matches)
```

**Explanation:** The **FFN holds about two-thirds of the parameters** of a block
— the attention gets the headlines, but most of the weights (and much of the
compute) live in the position-wise feed-forward layers. Multiply ~3.15M by the
number of layers (e.g. 6 → ~19M, plus embeddings) to sanity-check any model's
size. This back-of-envelope skill is invaluable for estimating memory and cost.

</details>

---

## Navigation

- ⬆️ [Module 6 Exercises](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
