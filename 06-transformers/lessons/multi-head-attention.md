---
title: Multi-Head Attention
description: Why transformers run several attention heads in parallel, how the projections and tensor reshapes work, and how to implement it in PyTorch.
type: lesson
domain: 06-transformers
tags: [transformers, multi-head-attention, projections]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 45 min
prerequisites:
  - 06-transformers/lessons/self-attention
---

# Multi-Head Attention

> **TL;DR:** One attention head produces a single weighted average per token, so it can only capture one kind of relationship at a time. Multi-head attention splits the model dimension into $h$ subspaces, runs scaled dot-product attention in each in parallel, then concatenates and projects the results — at roughly the same total compute cost as a single full-width head.

---

## Overview

You already know how a single self-attention head works: every token builds a query, compares it against all keys, and takes a softmax-weighted average of the values. This lesson explains why one head is not enough, how the multi-head mechanism splits the representation into parallel subspaces, and exactly how the tensors are reshaped along the way — the part where most implementations go wrong.

**By the end, you will be able to:**
- Explain why a single softmax-weighted average limits what one attention head can express
- Derive the shapes at every step of the `(batch, seq, d_model) → (batch, h, seq, d_k)` reshape and back
- Implement a `MultiHeadAttention` module in PyTorch with correct shape handling

---

## Intuition

A single attention head has one softmax per query token. A softmax produces one probability distribution, and the output is one weighted average of the value vectors. That is a strong constraint: if the token "it" needs to attend to its antecedent noun (coreference), to the verb it is the subject of (syntax), *and* to nearby tokens (position), a single distribution has to smear its probability mass across all three. The result is a blurry average that serves none of the relationships well.

The fix is to ask several smaller questions in parallel instead of one big one. Give the model $h$ independent attention heads, each with its own learned projections. Each head sees the same input tokens but through a different learned "lens": one head is free to specialize in syntactic dependencies, another in positional neighbors, another in something no human has named yet. Each head computes its own softmax and its own weighted average; the outputs are concatenated and mixed back together with a final linear layer.

The crucial design decision in "Attention Is All You Need" is that the heads **split** the model dimension rather than multiply it. With $d_{model} = 512$ and $h = 8$, each head works in a $64$-dimensional subspace. You get 8 different attention patterns for roughly the price of one full-width head.

---

## Details

### Mathematics

Let:

- $X \in \mathbb{R}^{n \times d_{model}}$ — the input sequence of $n$ token embeddings, each of dimension $d_{model}$
- $h$ — the number of attention heads
- $d_k = d_v = d_{model} / h$ — the per-head dimension for keys/queries and values ($d_{model}$ must be divisible by $h$)
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$ and $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$ — learned projection matrices for head $i$
- $W^O \in \mathbb{R}^{h d_v \times d_{model}}$ — the learned output projection

Each head runs the scaled dot-product attention you already know, but in its own projected subspace:

$$
\text{head}_i = \text{Attention}(X W_i^Q,\; X W_i^K,\; X W_i^V) = \text{softmax}\!\left(\frac{(X W_i^Q)(X W_i^K)^\top}{\sqrt{d_k}}\right) X W_i^V
$$

The heads are concatenated along the feature dimension and projected back to $d_{model}$:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

**Why compute stays roughly constant.** A single head at full width would multiply $n \times d_{model}$ matrices against $d_{model} \times d_{model}$ projections. With $h$ heads, each projection is $d_{model} \times d_k$, and $h \cdot d_k = d_{model}$ — so the total projection work is the same. The attention score matrices are $n \times n$ per head, computed over $d_k$-dimensional vectors, so $h$ heads over $d_k$ dims cost about the same dot-product work as one head over $d_{model}$ dims. You are re-slicing the computation, not adding to it (the only extra cost is the $W^O$ projection and some reshape overhead).

**The tensor-shape walkthrough.** In practice all $h$ heads are computed in one batched operation. This is the part to internalize. Start with a batch of inputs:

1. `X` has shape `(batch, seq, d_model)`.
2. Apply one big linear layer `W_q` of size `d_model × d_model` (it packs all $h$ per-head $W_i^Q$ matrices side by side). Output: `(batch, seq, d_model)`.
3. **Split** the last dimension into heads with `view`: `(batch, seq, d_model)` → `(batch, seq, h, d_k)`. Nothing moves in memory; you are just reinterpreting the last axis as two axes.
4. **Transpose** dims 1 and 2: `(batch, seq, h, d_k)` → `(batch, h, seq, d_k)`. Now each head is a separate "matrix" of shape `(seq, d_k)`, and batched matrix multiply treats `(batch, h)` as leading batch dimensions.
5. Compute scores: `Q @ K.transpose(-2, -1)` gives `(batch, h, seq, seq)` — one $n \times n$ attention map **per head**.
6. Softmax over the last dimension, then multiply by `V`: `(batch, h, seq, seq) @ (batch, h, seq, d_k)` → `(batch, h, seq, d_k)`.
7. **Undo** the split: transpose back to `(batch, seq, h, d_k)`, call `.contiguous()` (the transpose made memory non-contiguous), then `view` to `(batch, seq, d_model)`. This is the concatenation — the heads simply sit side by side in the last dimension again.
8. Apply `W_o`: `(batch, seq, d_model)` → `(batch, seq, d_model)`.

If you remember one thing: **`view` splits/merges the feature axis; `transpose` moves the head axis next to the batch axis so matmul batches over heads.**

### Python implementation

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # One big projection per role; packs all heads' W_i side by side.
        self.w_q = nn.Linear(d_model, d_model)   # (d_model) -> (d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)   # the W^O output projection
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape                            # (batch, seq, d_model)
        x = x.view(batch, seq, self.num_heads, self.d_k)   # (batch, seq, h, d_k)
        return x.transpose(1, 2)                           # (batch, h, seq, d_k)

    def forward(
        self,
        query: torch.Tensor,   # (batch, seq_q, d_model)
        key: torch.Tensor,     # (batch, seq_k, d_model)
        value: torch.Tensor,   # (batch, seq_k, d_model)
        mask: torch.Tensor | None = None,  # broadcastable to (batch, h, seq_q, seq_k)
    ) -> torch.Tensor:
        batch, seq_q, _ = query.shape

        q = self._split_heads(self.w_q(query))   # (batch, h, seq_q, d_k)
        k = self._split_heads(self.w_k(key))     # (batch, h, seq_k, d_k)
        v = self._split_heads(self.w_v(value))   # (batch, h, seq_k, d_k)

        # Attention scores per head.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # (batch, h, seq_q, seq_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)          # (batch, h, seq_q, seq_k)
        attn = self.dropout(attn)

        out = attn @ v                            # (batch, h, seq_q, d_k)
        out = out.transpose(1, 2).contiguous()    # (batch, seq_q, h, d_k)
        out = out.view(batch, seq_q, self.d_model)  # (batch, seq_q, d_model) — concat
        return self.w_o(out)                      # (batch, seq_q, d_model)


if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)                  # (batch=2, seq=10, d_model=512)
    y = mha(x, x, x)                             # self-attention: Q = K = V = x
    print(y.shape)                               # torch.Size([2, 10, 512])
```

This mirrors what `torch.nn.MultiheadAttention` does internally (that module also packs Q/K/V projections into a single weight and defaults to `(seq, batch, d_model)` layout unless you pass `batch_first=True`). Use the built-in in production; write your own once so the shapes stop being magic.

## Worked Example

Take $d_{model} = 512$, $h = 8$, so $d_k = 64$. Batch of 2 sentences, 10 tokens each.

1. Input `X`: `(2, 10, 512)`.
2. `w_q(X)`: still `(2, 10, 512)` — one linear layer computed all 8 heads' queries at once.
3. `view(2, 10, 8, 64)`: the 512 features are reinterpreted as 8 chunks of 64. Head 0's query for token 3 of sentence 1 is `q[1, 3, 0, :]` — features 0–63 of the projected vector.
4. `transpose(1, 2)` → `(2, 8, 10, 64)`: now index order is (sentence, head, token, feature).
5. `q @ k.transpose(-2, -1)` → `(2, 8, 10, 10)`: for each of the 2 sentences and each of the 8 heads, a full 10×10 map of how much every token attends to every other token. Sixteen independent attention maps in one matmul.
6. Softmax rows sum to 1 within each head. Multiply by `v` (`(2, 8, 10, 64)`) → `(2, 8, 10, 64)`: each token, in each head, is now a 64-dim weighted average of that head's values.
7. Transpose back and `view(2, 10, 512)`: token 3's output is head 0's 64 dims, then head 1's, ..., then head 7's — literal concatenation.
8. `w_o` mixes information **across** heads (until now the heads never interacted) and returns `(2, 10, 512)`.

Total scored dot products: $8 \times 10 \times 10$ pairs at 64 dims each — the same multiply count as $1 \times 10 \times 10$ pairs at 512 dims. The compute was split, not multiplied.

## Best Practices

- ✅ Assert `d_model % num_heads == 0` in the constructor — a silent mismatch produces garbage shapes far from the source.
- ✅ Comment the shape on every line of the forward pass while learning; it turns shape bugs from mysteries into diffs.
- ✅ Use `view` + `transpose` (or `einops.rearrange`) rather than manual index gymnastics; call `.contiguous()` after transposing before the final `view`.
- ✅ In production, prefer `nn.MultiheadAttention` or `F.scaled_dot_product_attention`, which dispatch to fused/optimized kernels.

## Common Mistakes

- ⚠️ **Transposing before the `view` split** (`transpose` then `view(batch, seq, h, d_k)`) silently interleaves features across heads. Fix: always `view` first to split the feature axis, then `transpose` to move the head axis.
- ⚠️ **Forgetting `.contiguous()` before the final `view`** — PyTorch raises an error after `transpose` because the memory layout no longer matches. Fix: `out.transpose(1, 2).contiguous().view(...)` (or use `.reshape(...)`).
- ⚠️ **Scaling by $\sqrt{d_{model}}$ instead of $\sqrt{d_k}$.** Each head's dot products live in $d_k$ dimensions, so the scale factor is $\sqrt{d_k}$. Wrong scaling changes softmax sharpness and hurts training.
- ⚠️ **Assuming more heads is strictly better.** With $d_{model}$ fixed, more heads means smaller $d_k$ per head; extremely small heads can underperform. The head count is a tunable trade-off, not a free win.

## Industry Tips

- 💡 Heads empirically learn diverse patterns — some track adjacent positions, some rare tokens, some syntactic-looking relations — but reading attention maps as explanations is contested; treat head interpretation as an active research area, not settled fact.
- 💡 A well-replicated finding: after training, a substantial fraction of heads can be pruned with only modest quality loss, which suggests heads are partly redundant. This motivates head-pruning as an inference-time optimization.
- 💡 Modern large models often use multi-query or grouped-query attention (sharing K/V across heads) to shrink the inference-time KV cache — the multi-head mechanics here are the foundation for understanding those variants.

## Real-World Use Cases

- Every transformer encoder and decoder layer — BERT-style encoders, GPT-style decoders, translation models — uses multi-head attention as its core mixing operation.
- Vision Transformers apply the identical mechanism to image patch embeddings instead of word tokens.
- Speech and multimodal models reuse the same module unchanged; only the tokenization of the input differs.

---

## Summary

- One head means one softmax and one weighted average per token; $h$ heads let the model attend to $h$ different relationship types simultaneously.
- Heads split $d_{model}$ into $h$ subspaces of size $d_k = d_{model}/h$, so total compute stays roughly constant versus a single full-width head.
- The implementation is a `view` (split features into heads), a `transpose` (move heads next to batch), batched attention, then the reverse dance plus a final $W^O$ projection.

## Practice

- [ ] Exercises: [Module 6 Exercises](../exercises/README.md)
- [ ] Self-check: Given `d_model=768` and `h=12`, what is `d_k`, and what is the shape of the attention score tensor for a batch of 4 sequences of length 128?

## Further Reading

- 📑 Attention Is All You Need — Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762)
- 🌐 The Illustrated Transformer — Jay Alammar (https://jalammar.github.io/illustrated-transformer/)
- 🌐 The Annotated Transformer (https://nlp.seas.harvard.edu/annotated-transformer/)
- 📘 Dive into Deep Learning (https://d2l.ai/)
- 🌐 PyTorch `nn.MultiheadAttention` docs (https://pytorch.org/docs/stable/)

## Related

- [Self-Attention](self-attention.md)
- [Masked and Cross-Attention](masked-and-cross-attention.md)
- [Word Embeddings](../../05-nlp/lessons/word-embeddings.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
