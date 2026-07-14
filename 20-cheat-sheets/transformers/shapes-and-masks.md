---
title: Transformer Shapes & Masks Cheat Sheet
description: Tensor shapes through a transformer and copy-paste mask recipes.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [transformers, shapes, masks, pytorch, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Transformer Shapes & Masks Cheat Sheet

> Fast reference. For depth, see
> [Multi-Head Attention](../../06-transformers/lessons/multi-head-attention.md) and
> [Masked & Cross-Attention](../../06-transformers/lessons/masked-and-cross-attention.md).

---

## Shapes through the model

Let `B`=batch, `S`=sequence, `d`=d_model, `h`=heads, `d_k=d/h`, `V`=vocab.

| Stage | Shape |
|-------|-------|
| token ids | `(B, S)` |
| embeddings + positional | `(B, S, d)` |
| Q/K/V projections | `(B, S, d)` |
| split into heads | `(B, h, S, d_k)` |
| attention scores | `(B, h, S, S)` |
| attention output (per head) | `(B, h, S, d_k)` |
| combined heads | `(B, S, d)` |
| after FFN / block | `(B, S, d)` |
| LM head logits | `(B, S, V)` |

## Mask recipes (PyTorch)

```python
import torch

# Causal mask: True = block (future). Shape (S, S).
causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)

# Key-padding mask from lengths: True = pad. Shape (B, S) -> broadcast to (B,1,1,S).
pad = (tokens == pad_id)                        # (B, S)
pad = pad[:, None, None, :]                     # (B, 1, 1, S)

# Apply (additive) before softmax:
scores = scores.masked_fill(causal, float("-inf"))
scores = scores.masked_fill(pad,    float("-inf"))
```

## Split / combine heads

```python
def split(x, h):     # (B,S,d) -> (B,h,S,d_k)
    B, S, d = x.shape
    return x.view(B, S, h, d // h).transpose(1, 2)

def combine(x):      # (B,h,S,d_k) -> (B,S,d)
    B, h, S, d_k = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, h * d_k)
```

---

## Gotchas

- ⚠️ `.contiguous()` before `view` after a `transpose`, or PyTorch errors.
- ⚠️ Causal mask is `(S, S)`; padding mask is `(B, S)` — broadcast, then combine.
- ⚠️ LM training shifts targets by one: predict token `t+1` from `≤ t`.
- ⚠️ Scores are `(B, h, S, S)` — the $O(S^2)$ memory that limits context length.

---

## Quick Links

- 🧩 [Multi-head shape gymnastics](../../06-transformers/exercises/multi-head-shapes.md) · [Build the causal mask](../../06-transformers/exercises/build-the-causal-mask.md)
- 📖 [Building a Transformer from Scratch](../../06-transformers/lessons/transformer-from-scratch.md)

---

## Navigation

- ⬆️ [Transformers Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
