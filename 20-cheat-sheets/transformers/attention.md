---
title: Attention Cheat Sheet
description: The attention equation and its variants at a glance.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [transformers, attention, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Attention Cheat Sheet

> Fast reference. For depth, see
> [Module 6 lessons](../../06-transformers/lessons/README.md).

---

## The core equation

$$
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- $Q$ queries `(n_q, d_k)`, $K$ keys `(n_k, d_k)`, $V$ values `(n_k, d_v)`.
- $QK^\top$ → `(n_q, n_k)` score matrix; softmax over the **key axis**.
- $\sqrt{d_k}$ keeps logits' variance ~1 so softmax doesn't saturate.

## Variants

| Variant | Q from | K,V from | Mask | Used in |
|---------|--------|----------|------|---------|
| Self-attention | input | same input | none | encoder |
| Causal self-attention | input | same input | upper-triangular −∞ | decoder / GPT |
| Cross-attention | decoder | encoder memory | (padding) | enc-dec decoder |
| Multi-head | h projections | h projections | any | everywhere |

## Multi-head

$$
\text{MultiHead}=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O,\quad d_k=d_{model}/h
$$

Split `(B,S,d_model)→(B,h,S,d_k)`, attend per head, concat, project with $W^O$.
Total compute ≈ constant (heads split the dimension).

```python
import torch, torch.nn.functional as F

def attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ V
```

---

## Gotchas

- ⚠️ Softmax over the **last** axis (keys), never the batch.
- ⚠️ Mask with −∞ **before** softmax, not by zeroing after.
- ⚠️ Don't forget the `/ sqrt(d_k)` scaling.
- ⚠️ `Q @ K` needs `K.transpose(-2,-1)`.
- ⚠️ Attention is $O(n^2)$ in sequence length — the context-window bottleneck.

---

## Quick Links

- 📖 [Attention Fundamentals](../../06-transformers/lessons/attention-fundamentals.md) · [Multi-Head](../../06-transformers/lessons/multi-head-attention.md) · [Masked & Cross](../../06-transformers/lessons/masked-and-cross-attention.md)
- 🔗 Attention Is All You Need (https://arxiv.org/abs/1706.03762)

---

## Navigation

- ⬆️ [Transformers Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
