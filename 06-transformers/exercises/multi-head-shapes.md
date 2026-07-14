---
title: "Exercise: Multi-Head Shape Gymnastics"
description: Master the reshape/transpose dance that splits and recombines attention heads.
type: exercise
domain: 06-transformers
tags: [transformers, multi-head-attention, tensor-shapes, pytorch]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 30 min
reinforces:
  - 06-transformers/lessons/multi-head-attention
---

# Exercise: Multi-Head Shape Gymnastics

> Practice for **[Multi-Head Attention](../lessons/multi-head-attention.md)**.

---

## Problem

The hardest part of multi-head attention is the tensor bookkeeping. Given an
input of shape `(batch=2, seq=5, d_model=8)` and `n_heads=2`, implement
`split_heads` and `combine_heads` so that:

- `split_heads`: `(B, S, d_model) → (B, h, S, d_k)` with `d_k = d_model // h`
- `combine_heads`: `(B, h, S, d_k) → (B, S, d_model)` (exact inverse)

Then confirm `combine_heads(split_heads(x)) == x`.

## Requirements

- [ ] Both functions use `view`/`reshape` + `transpose` (no loops).
- [ ] Round-trip is exact (`torch.allclose`).
- [ ] Explain why the `transpose` (not just `view`) is necessary.

---

## Hints

<details>
<summary>Hint 1</summary>

Split: `view(B, S, h, d_k)` then `transpose(1, 2)` → `(B, h, S, d_k)`.

</details>

<details>
<summary>Hint 2</summary>

Combine reverses it: `transpose(1, 2)` → `(B, S, h, d_k)`, then
`.contiguous().view(B, S, d_model)`. The `.contiguous()` is required after a
transpose before `view`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import torch


def split_heads(x: torch.Tensor, h: int) -> torch.Tensor:
    B, S, d_model = x.shape
    d_k = d_model // h
    # (B, S, d_model) -> (B, S, h, d_k) -> (B, h, S, d_k)
    return x.view(B, S, h, d_k).transpose(1, 2)


def combine_heads(x: torch.Tensor) -> torch.Tensor:
    B, h, S, d_k = x.shape
    # (B, h, S, d_k) -> (B, S, h, d_k) -> (B, S, d_model)
    return x.transpose(1, 2).contiguous().view(B, S, h * d_k)


x = torch.randn(2, 5, 8)
assert torch.allclose(combine_heads(split_heads(x, h=2)), x)
print("round-trip exact ✓")
```

**Why the transpose?** You want each head to own a contiguous `(S, d_k)` slice so
attention runs independently per head with a single batched matmul over the
`(B, h)` dimensions. A bare `view(B, h, S, d_k)` would **not** do this — it would
regroup elements incorrectly, mixing sequence positions across heads. Splitting
the last dim (`view` to `(B, S, h, d_k)`) then moving `h` forward (`transpose`)
keeps each head's tokens together. And because `transpose` only changes strides
(not memory layout), you must call `.contiguous()` before the final `view` or
PyTorch raises an error.

</details>

---

## Navigation

- ⬆️ [Module 6 Exercises](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
