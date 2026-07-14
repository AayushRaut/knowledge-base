---
title: "Transformers — Interview Questions"
description: A bank of 20 transformer interview questions with answers.
type: interview
domain: 19-interview-preparation
tags: [transformers, interview, attention, architecture]
category: ml-concepts
difficulty: medium
status: complete
created: 2026-07-02
updated: 2026-07-02
companies: []
---

# Transformers — Interview Questions

> **Category:** Transformers · **Difficulty:** mixed (easy → hard)
>
> A question bank tied to **[Module 6 — Transformers](../../06-transformers/README.md)**.
> Try to answer before reading the response.

---

## Contents

1. [What problem did transformers solve?](#1-what-problem-did-transformers-solve)
2. [Query, key, value](#2-query-key-value)
3. [Why divide by √dₖ?](#3-why-divide-by-dₖ)
4. [Self-attention vs cross-attention](#4-self-attention-vs-cross-attention)
5. [Why multi-head?](#5-why-multi-head)
6. [Why positional encoding?](#6-why-positional-encoding)
7. [Sinusoidal vs learned vs RoPE](#7-sinusoidal-vs-learned-vs-rope)
8. [The causal mask](#8-the-causal-mask)
9. [Why mask before softmax?](#9-why-mask-before-softmax)
10. [Residuals and LayerNorm](#10-residuals-and-layernorm)
11. [What the FFN does](#11-what-the-ffn-does)
12. [Attention's complexity](#12-attentions-complexity)
13. [Encoder-only vs decoder-only vs enc-dec](#13-encoder-only-vs-decoder-only-vs-enc-dec)
14. [Why did decoder-only win for LLMs?](#14-why-did-decoder-only-win-for-llms)
15. [Training vs inference cost](#15-training-vs-inference-cost)
16. [Transformers vs RNNs](#16-transformers-vs-rnns)
17. [Pre-LN vs post-LN](#17-pre-ln-vs-post-ln)
18. [How Vision Transformers work](#18-how-vision-transformers-work)
19. [What attention weights do and don't tell you](#19-what-attention-weights-do-and-dont-tell-you)
20. [Context-length limits](#20-context-length-limits)

---

### 1. What problem did transformers solve?
**A:** RNNs process sequences step by step — slow (no parallelism) and prone to
losing long-range dependencies through the recurrent bottleneck. Transformers
replace recurrence with attention: every position interacts with every other in
one parallel operation, capturing long-range relationships and training far
faster on modern hardware.

### 2. Query, key, value
**A:** A soft dictionary lookup. Each token emits a **query**; it's compared
against every token's **key** to get relevance weights (softmax of scaled dot
products); the output is the weighted sum of **values**. In self-attention Q, K,
V are all linear projections of the same input.

### 3. Why divide by √dₖ?
**A:** Dot products of $d_k$-dimensional vectors have variance ∝ $d_k$, so at
high dimension the logits are large, softmax saturates to near one-hot, and
gradients vanish. Dividing by $\sqrt{d_k}$ normalizes the variance to ~1, keeping
the softmax and gradients well-behaved.

### 4. Self-attention vs cross-attention
**A:** In **self-attention** Q, K, V come from the same sequence (a token attends
to its own sequence). In **cross-attention** the queries come from one sequence
(the decoder) and keys/values from another (the encoder's output) — how a
translation decoder consults the source sentence.

### 5. Why multi-head?
**A:** A single attention head produces one weighted average — it can't attend to
several kinds of relationship at once. Multiple heads project into different
subspaces and attend in parallel (e.g. one to syntax, one to nearby tokens),
then concatenate. Because each head uses $d_k = d_{model}/h$, total compute stays
roughly constant.

### 6. Why positional encoding?
**A:** Self-attention is permutation-equivariant: with no position information,
"dog bites man" and "man bites dog" produce the same set of representations. We
inject order by adding positional encodings to the token embeddings.

### 7. Sinusoidal vs learned vs RoPE
**A:** **Sinusoidal** — fixed sin/cos of position at many frequencies; no
parameters, extrapolates in principle. **Learned** — a trainable position lookup
(BERT/GPT-2); simple but capped at max trained length. **RoPE** — rotates Q/K by
position-dependent angles so attention depends on *relative* position; used by
Llama-family and many modern LLMs.

### 8. The causal mask
**A:** In a decoder, position $i$ must not attend to positions $>i$ (the future),
or it would cheat during training (the answer is the next token). The mask sets
those score entries to $-\infty$ (an upper-triangular pattern) before softmax.

### 9. Why mask before softmax?
**A:** Adding $-\infty$ before softmax makes those entries exactly 0 **and**
renormalizes the remaining (allowed) weights to sum to 1. Zeroing *after* softmax
would leave rows that don't sum to 1 — an invalid attention distribution whose
normalization still depended on the future.

### 10. Residuals and LayerNorm
**A:** Residual connections ($x + \text{sublayer}(x)$) give gradients a direct
path, enabling very deep stacks (same idea as ResNet). LayerNorm normalizes each
token's features to stabilize training. Together they're what make deep
transformers trainable.

### 11. What the FFN does
**A:** After attention mixes information *across* tokens, the position-wise
feed-forward network ($\text{Linear}\to\text{activation}\to\text{Linear}$, ~4×
expansion) processes each token independently, adding nonlinear capacity. It
holds roughly two-thirds of a block's parameters.

### 12. Attention's complexity
**A:** The score matrix is $n \times n$ for sequence length $n$, so self-attention
is $O(n^2)$ in time and memory. This quadratic cost is why context windows are
limited and why efficient-attention variants (sparse, linear, FlashAttention as a
systems optimization) exist.

### 13. Encoder-only vs decoder-only vs enc-dec
**A:** **Encoder-only** (BERT): bidirectional, masked-LM pretraining, great for
understanding/classification, can't generate. **Decoder-only** (GPT): causal,
next-token pretraining, generation-native. **Encoder-decoder** (T5/BART): encoder
reads input, decoder generates output with cross-attention — natural for
translation/summarization.

### 14. Why did decoder-only win for LLMs?
**A:** A reasoned observation (not a hard law): decoder-only models offer a
single, simple next-token objective and a unified generate-everything interface,
scale cleanly, and get in-context learning "for free" at scale. That simplicity
and generality made them the default for general-purpose LLMs.

### 15. Training vs inference cost
**A:** Training is parallel — teacher forcing scores all positions at once.
Inference is **autoregressive** — one token at a time, each conditioned on all
previous — so generation is the latency bottleneck. KV-caching avoids recomputing
past keys/values but the sequential dependency remains.

### 16. Transformers vs RNNs
**A:** Transformers parallelize over sequence positions and model long-range
dependencies directly via attention; RNNs are sequential and forget over long
spans. The trade-off: attention is $O(n^2)$ in length, while RNNs are $O(n)$ —
so RNNs still appear in streaming/on-device niches.

### 17. Pre-LN vs post-LN
**A:** The original transformer applied LayerNorm *after* the sublayer + residual
(post-LN), which can need learning-rate warmup to stay stable. Most modern
implementations use **pre-LN** (normalize before the sublayer), which is more
stable to train and is standard in current LLMs.

### 18. How Vision Transformers work
**A:** ViT splits an image into fixed patches (e.g. 16×16), linearly projects
each to a token, prepends a learnable [CLS] token, adds positional embeddings,
and feeds a standard transformer encoder; classification reads the CLS output.
Lacking CNNs' built-in locality bias, ViTs need large-scale pretraining to
surpass CNNs.

### 19. What attention weights do and don't tell you
**A:** They show which positions a head weights when forming each output — often
suggestive (previous-token heads, delimiter heads). But high attention ≠
causal importance, weights are one of many factors, and interpretation is an
active research debate. Treat them as hints, not explanations.

### 20. Context-length limits
**A:** The $O(n^2)$ attention cost in compute and memory is the core limit;
positional schemes also constrain extrapolation beyond trained lengths. Mitigations
include efficient-attention variants, relative/rotary positions (RoPE/ALiBi) that
extend better, and systems tricks (FlashAttention, KV-caching).

---

## Common Mistakes

- ⚠️ Saying the $\sqrt{d_k}$ scaling is "just normalization" without the variance/gradient reason.
- ⚠️ Claiming attention weights explain the model's decision.
- ⚠️ Forgetting decoder-only models have no cross-attention.
- ⚠️ Zeroing attention after softmax instead of masking before it.

## Key Takeaways

- Be able to write the attention equation and justify every term.
- Connect each architectural piece to the failure mode it addresses.

## Related

- [Module 6 — Transformers](../../06-transformers/README.md)
- [Attention Cheat Sheet](../../20-cheat-sheets/transformers/attention.md)

---

## Navigation

- ⬆️ [Transformers Interview Questions](README.md)
- 📚 [Interview Preparation](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
