---
title: Self-Attention
description: How a sequence attends to itself through learned Q/K/V projections to build contextual token representations, with a full worked matrix example.
type: lesson
domain: 06-transformers
tags: [transformers, self-attention, qkv]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 50 min
prerequisites:
  - 06-transformers/lessons/attention-fundamentals
---

# Self-Attention

> **TL;DR:** In self-attention, queries, keys, and values all come from the *same* sequence via learned linear projections, so every token rebuilds its representation from every other token — in one parallel matrix multiplication.

---

## Overview

Self-attention is the core computation of the transformer: it is what turns static token embeddings into context-aware representations, and it is why transformers train orders of magnitude faster than RNNs. Every architecture decision downstream — multi-head attention, positional encoding, context-window limits — exists because of how self-attention works and what it lacks. Understand this lesson deeply and the rest of the transformer is bookkeeping.

**By the end, you will be able to:**
- Explain how $Q$, $K$, $V$ are derived from one input via learned projections, with exact shapes
- Walk through a complete self-attention computation by hand on small matrices
- Implement single-head self-attention as a PyTorch `nn.Module` and state its $O(n^2)$ cost and its two structural blind spots

---

## Intuition

In [Attention Fundamentals](attention-fundamentals.md), a decoder attended to a *separate* encoder sequence. Self-attention asks a simpler, more radical question: what if a sequence attends to **itself**?

Take the sentence *"The animal didn't cross the street because it was too tired."* What does "it" mean? Every fluent reader instantly links "it" to "animal". Self-attention gives each token that ability: the token "it" issues a query ("I'm a pronoun — who's my referent?"), every token in the sentence — including "it" itself — offers a key, and "animal" matches best. The output vector for "it" then becomes mostly a blend of "animal"'s value with its own.

Three ideas make this powerful:

1. **One input, three roles.** Each token's embedding is projected three ways: as a query (what am I looking for?), a key (what do I offer for matching?), and a value (what content do I pass along?). The projections are *learned*, so the model discovers which relationships matter.
2. **Contextual embeddings.** After self-attention, "bank" in *"river bank"* and "bank" in *"bank deposit"* have *different* vectors, because each blended in different neighbors. Static embeddings like Word2Vec (see [Word Embeddings](../../05-nlp/lessons/word-embeddings.md)) give one frozen vector per word regardless of context — self-attention is what removed that limitation.
3. **Everything at once.** There is no recurrence. Token 500 does not wait for token 499; all positions are computed in a single batched matrix multiply. This parallelism is the training-speed revolution that made modern LLMs feasible.

---

## Details

### Mathematics

Let the input be a matrix of token embeddings:

- $X \in \mathbb{R}^{n \times d_{model}}$ — $n$ tokens, each a row vector of dimension $d_{model}$ (the model's embedding width).

Self-attention introduces three learned weight matrices:

- $W^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W^K \in \mathbb{R}^{d_{model} \times d_k}$, $W^V \in \mathbb{R}^{d_{model} \times d_v}$

where $d_k$ is the query/key dimension and $d_v$ the value dimension. Queries, keys, and values are projections of the *same* $X$:

$$
Q = XW^Q \in \mathbb{R}^{n \times d_k}, \qquad
K = XW^K \in \mathbb{R}^{n \times d_k}, \qquad
V = XW^V \in \mathbb{R}^{n \times d_v}
$$

Then apply scaled dot-product attention from the previous lesson:

$$
\text{SelfAttention}(X) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \in \mathbb{R}^{n \times d_v}
$$

Row $i$ of the output is token $i$'s new, *contextual* representation: a weighted average of all tokens' values, weighted by how well token $i$'s query matched each token's key.

**Complexity.** The score matrix $QK^\top$ is $n \times n$: computing it costs $O(n^2 d_k)$ time and $O(n^2)$ memory. Doubling the sequence length quadruples the attention cost. This quadratic scaling is precisely why LLM context windows are a headline spec and why long-context inference is expensive.

**Parallelism.** An RNN needs $n$ *sequential* steps because state $h_t$ depends on $h_{t-1}$. Self-attention has no such dependency — the whole computation is three matmuls and a softmax, which GPUs execute in parallel across all positions. Any token can influence any other in a single layer (path length $O(1)$ vs an RNN's $O(n)$), which also helps gradients flow between distant positions.

**A note on masking.** In decoder (GPT-style) models, position $i$ must not see future tokens. This is enforced by adding $-\infty$ to score entries $(i, j)$ with $j > i$ before the softmax, which zeroes those attention weights:

$$
\text{score}_{ij} \leftarrow
\begin{cases}
q_i \cdot k_j / \sqrt{d_k} & j \le i \\
-\infty & j > i
\end{cases}
$$

The unmasked (bidirectional) form in this lesson is what BERT-style encoders use.

**Permutation-invariance.** Nothing in the formula knows *where* a token sits. Permute the rows of $X$ and the output rows permute identically — "dog bites man" and "man bites dog" produce the same set of vectors. Word order is real information, so transformers must inject it explicitly; that is the job of [Positional Encoding](positional-encoding.md).

### Python implementation

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Single-head self-attention."""

    def __init__(self, d_model: int, d_k: int, d_v: int) -> None:
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, n, d_model) -> (output (batch, n, d_v), weights (batch, n, n))."""
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (batch, n, n)
        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights

attn = SelfAttention(d_model=8, d_k=4, d_v=4)
x = torch.randn(2, 5, 8)          # batch of 2 sequences, 5 tokens each
out, w = attn(x)
print(out.shape, w.shape)          # torch.Size([2, 5, 4]) torch.Size([2, 5, 5])
print(w.sum(dim=-1))               # every row sums to 1
```

In production code, replace the explicit softmax path with `F.scaled_dot_product_attention`, which dispatches to fused, memory-efficient kernels (see the PyTorch docs: https://pytorch.org/docs/stable/).

## Worked Example

Three tokens, $d_{model} = 4$, $d_k = d_v = 2$. Input and learned projections (hand-picked for clean arithmetic):

$$
X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix},\quad
W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix},\quad
W^K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix},\quad
W^V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}
$$

**Step 1 — project.**

$$
Q = XW^Q = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix},\quad
K = XW^K = \begin{bmatrix} 0 & 2 \\ 2 & 0 \\ 1 & 1 \end{bmatrix},\quad
V = XW^V = \begin{bmatrix} 2 & 1 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
$$

**Step 2 — scores.** $QK^\top$, entry $(i,j) = q_i \cdot k_j$:

$$
QK^\top = \begin{bmatrix} 0 & 4 & 2 \\ 4 & 0 & 2 \\ 2 & 2 & 2 \end{bmatrix}
$$

**Step 3 — scale.** Divide by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:

$$
\frac{QK^\top}{\sqrt{2}} = \begin{bmatrix} 0 & 2.828 & 1.414 \\ 2.828 & 0 & 1.414 \\ 1.414 & 1.414 & 1.414 \end{bmatrix}
$$

**Step 4 — softmax (row-wise).** For row 1: $e^0 = 1$, $e^{2.828} \approx 16.92$, $e^{1.414} \approx 4.11$, sum $\approx 22.03$:

$$
A = \begin{bmatrix} 0.0454 & 0.7679 & 0.1867 \\ 0.7679 & 0.0454 & 0.1867 \\ 0.3333 & 0.3333 & 0.3333 \end{bmatrix}
$$

Token 1's query points at token 2's key ($q_1 \cdot k_2 = 4$ is the biggest score), so token 1 attends 77% to token 2. Token 3 matches everyone equally and attends uniformly.

**Step 5 — output.** $AV$, e.g. row 1 $= 0.0454\,[2,1] + 0.7679\,[0,1] + 0.1867\,[1,1]$:

$$
AV = \begin{bmatrix} 0.2775 & 1.0 \\ 1.7225 & 1.0 \\ 1.0 & 1.0 \end{bmatrix}
$$

Each token's new vector is a context-dependent mixture — token 1's output is dominated by token 2's value. That is a contextual embedding, computed with nothing but matrix algebra.

## Best Practices

- ✅ Track tensor shapes at every step ($X$: $n \times d_{model}$ → $Q,K$: $n \times d_k$ → scores: $n \times n$ → output: $n \times d_v$); most attention bugs are shape bugs.
- ✅ Keep $W^Q$, $W^K$, $W^V$ as separate `nn.Linear` layers (or one fused projection you split) — never reuse one matrix for all three roles.
- ✅ Test permutation-invariance in unit tests: shuffling input rows should shuffle output rows identically. It verifies your implementation *and* cements why positional encoding is needed.

## Common Mistakes

- ⚠️ **Thinking Q, K, V are separate inputs.** In self-attention they are three learned *views* of the same $X$. Distinct inputs for K/V is *cross*-attention, a different (encoder–decoder) configuration.
- ⚠️ **Forgetting that a token attends to itself.** Position $i$'s query is scored against its own key too; there is no self-exclusion in the standard formulation.
- ⚠️ **Ignoring the $O(n^2)$ memory of the score matrix.** A 32k-token sequence yields a billion-entry attention matrix per head. If long inputs OOM, attention memory is the first suspect — fix with fused kernels or shorter contexts, not bigger batch hacks.

## Industry Tips

- 💡 Context-window pricing and limits in LLM APIs trace directly back to this $O(n^2)$ cost — you are paying for the quadratic attention bill.
- 💡 Memory-aware exact-attention kernels (e.g. the FlashAttention line of work, exposed via `F.scaled_dot_product_attention`) avoid materializing the full $n \times n$ matrix and are the default in serious training and serving stacks.
- 💡 "Walk me through self-attention with shapes" is one of the most common ML-engineer interview questions; the worked example above is the level of detail interviewers expect.

## Real-World Use Cases

- **Contextual encoders** — BERT-style models built from stacked self-attention power search ranking, classification, and reranking in production.
- **LLM decoding** — GPT-style models use (causally masked) self-attention over the growing context at every generated token.
- **Vision Transformers** — image patches are treated as tokens; self-attention relates distant regions of an image in one layer.

---

## Summary

- Self-attention derives $Q = XW^Q$, $K = XW^K$, $V = XW^V$ from the *same* input, so every token rebuilds itself as a learned, weighted mixture of all tokens — that mixture *is* the contextual embedding.
- The computation is fully parallel across positions (no recurrence), which is why transformers train dramatically faster than RNNs, but the $n \times n$ score matrix makes cost quadratic in sequence length.
- The formula is permutation-invariant: word order must be injected separately via positional encoding.

## Practice

- [ ] Exercises: [Module 6 Exercises](../exercises/README.md)
- [ ] Self-check: Given $X \in \mathbb{R}^{n \times d_{model}}$, write the shapes of $Q$, $K$, $V$, the score matrix, and the output — then explain why doubling $n$ quadruples attention memory.

## Further Reading

- 📑 Attention Is All You Need — Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762)
- 🌐 The Illustrated Transformer — Jay Alammar (https://jalammar.github.io/illustrated-transformer/)
- 📘 Dive into Deep Learning (https://d2l.ai/)
- 🌐 The Annotated Transformer — Harvard NLP (https://nlp.seas.harvard.edu/annotated-transformer/)

## Related

- [Attention Fundamentals](attention-fundamentals.md) — the scaled dot-product mechanism this lesson builds on
- [Multi-Head Attention](multi-head-attention.md) — running several self-attention heads in parallel
- [Positional Encoding](positional-encoding.md) — fixing permutation-invariance
- [Word Embeddings](../../05-nlp/lessons/word-embeddings.md) — static embeddings, contrasted with the contextual ones built here

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
