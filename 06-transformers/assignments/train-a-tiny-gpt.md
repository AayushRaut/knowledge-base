---
title: "Assignment: Train a Tiny GPT"
description: Build and train a small decoder-only transformer that generates coherent text from a character-level corpus.
type: assignment
domain: 06-transformers
tags: [transformers, gpt, language-model, pytorch]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 6–8 hours
covers:
  - 06-transformers/lessons/transformer-architecture
  - 06-transformers/lessons/transformer-from-scratch
  - 06-transformers/lessons/positional-encoding
  - 06-transformers/lessons/masked-and-cross-attention
---

# Assignment: Train a Tiny GPT

> A larger, assessed task that integrates multiple concepts from
> **[Module 6](../README.md)**.

---

## Context

You've built the parts; now assemble and *train* them. You will implement a small
decoder-only transformer (GPT-style), train it on a character-level corpus, and
generate text — the capstone that ties every attention and architecture lesson
together. This is the transformer successor to the Module 4
[char-level RNN generator](../../04-deep-learning/projects/char-level-text-generator.md).

## Objectives

- Assemble a working decoder-only transformer from your own components.
- Train it with the shifted next-token objective and generate samples.
- Reason about model size, context length, and sampling.

---

## Tasks

1. **Data** — Character-level dataset from a plain-text corpus; build the
   vocab; produce `(x, y)` where `y` is `x` shifted by one. Context length is a
   hyperparameter.
2. **Model** — Token + positional embeddings → N pre-LN blocks (causal
   multi-head self-attention + FFN with residuals) → final LayerNorm → LM head.
   Build on [Building a Transformer from Scratch](../lessons/transformer-from-scratch.md).
3. **Train** — Cross-entropy over all positions (flatten batch × seq); AdamW;
   gradient clipping; track train/val loss. Keep it CPU-runnable (tiny config).
4. **Generate** — Autoregressive sampling with **temperature** and **top-k**;
   show samples at several checkpoints (untrained → trained).
5. **Analyze** — Vary one of {n_layers, n_heads, context length} and report the
   effect on validation loss and sample quality.

## Constraints

- Pure PyTorch; your own model (may reference nanoGPT for structure, but write it).
- Correct shift-by-one objective (state where the off-by-one bug would be).
- Fixed seeds; report parameter count and the config.

---

## Deliverables

- [ ] `model.py` (the transformer) + `train.py` + `generate.py`.
- [ ] Loss curves and generated samples at ≥3 training stages.
- [ ] The ablation table + short analysis.
- [ ] `README.md` with config, param count, and results.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Correctness | 40% | Model + causal masking + objective are right; it learns. |
| Generation | 20% | Working temperature/top-k sampling; quality improves visibly. |
| Analysis | 25% | Thoughtful ablation and interpretation. |
| Code quality | 15% | Clean, reproducible, documented. |

---

## Submission

Push to a branch `add/module-6-tiny-gpt` and open a pull request.

## Further Reading

- Andrej Karpathy — nanoGPT (https://github.com/karpathy/nanoGPT)
- The Annotated Transformer (https://nlp.seas.harvard.edu/annotated-transformer/)

---

## Navigation

- ⬆️ [Module 6 Assignments](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
