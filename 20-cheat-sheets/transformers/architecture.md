---
title: Transformer Architecture Cheat Sheet
description: Block structure, positional encoding, and the three model families at a glance.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [transformers, architecture, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Transformer Architecture Cheat Sheet

> Fast reference. For depth, see
> [The Transformer Architecture](../../06-transformers/lessons/transformer-architecture.md).

---

## The block (pre-LN, modern default)

```text
x → LayerNorm → Multi-Head Attention → + x        (residual)
  → LayerNorm → FFN (Linear→GELU→Linear) → + x     (residual)
```

- **FFN** expands 4×: `Linear(d, 4d) → activation → Linear(4d, d)`. Holds ~⅔ of a block's params.
- **Residuals** = gradient highways (as in ResNet); **LayerNorm** normalizes per token over features.
- Post-LN (original) vs pre-LN (most modern): pre-LN trains more stably.

## Positional encoding

| Type | Idea | Used by |
|------|------|---------|
| Sinusoidal | fixed sin/cos of position × frequency | original transformer |
| Learned | trainable position lookup table | BERT, GPT-2 |
| RoPE | rotate Q/K by position angle (relative) | Llama family, many modern LLMs |

Needed because self-attention is permutation-equivariant (no built-in order).

## Model families

| Family | Attention | Pretraining | Good at | Examples |
|--------|-----------|-------------|---------|----------|
| Encoder-only | bidirectional | masked LM | understanding (classify, NER, QA) | BERT, RoBERTa |
| Decoder-only | causal | next-token | generation, in-context learning | GPT, Llama |
| Encoder-decoder | both + cross | denoising / span | input→output transforms | T5, BART |

Original hyperparameters (Vaswani et al. 2017): N=6, d_model=512, h=8, d_ff=2048.

---

## Gotchas

- ⚠️ Training is parallel (teacher-forced); generation is sequential (slow part).
- ⚠️ Add positional info or the model is order-blind.
- ⚠️ Decoder-only models have no cross-attention.
- ⚠️ ViT: image → patches → tokens; needs large-scale pretraining to beat CNNs.

---

## Quick Links

- 📖 [Positional Encoding](../../06-transformers/lessons/positional-encoding.md) · [Architecture](../../06-transformers/lessons/transformer-architecture.md) · [Variants](../../06-transformers/lessons/architecture-variants.md)
- 🔗 The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/)

---

## Navigation

- ⬆️ [Transformers Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
