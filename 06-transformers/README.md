# Module 6 — Transformers

> The architecture behind modern AI, **explained from scratch** — attention,
> multi-head and masked attention, positional encoding, the full encoder-decoder,
> a from-scratch build, and the BERT/GPT/T5/ViT variants. Taught
> **intuition → mathematics → PyTorch**. This is Module 6 of the
> [Master Curriculum](../.claude/MASTER_CURRICULUM.md).

**Status:** 🟢 Complete · **Difficulty:** intermediate → advanced
**Estimated time:** ~40–55 hours · **Last reviewed:** 2026-07-02

---

## 🎯 Learning outcomes

By the end of this module you will be able to:

- Derive scaled dot-product attention and explain the $\sqrt{d_k}$ scaling.
- Implement self-, multi-head, masked, and cross-attention correctly.
- Explain positional encoding (sinusoidal, learned, RoPE) and why it's needed.
- Assemble the full transformer and build a working mini-GPT from scratch.
- Distinguish encoder-only, decoder-only, and encoder-decoder models — and ViT.

## 📋 Prerequisites

[Module 4 — Deep Learning](../04-deep-learning/README.md) (backprop, PyTorch,
RNNs) and [Module 5 — NLP](../05-nlp/README.md) (tokenization, embeddings,
seq2seq). Linear algebra from [Module 2](../02-mathematics-foundations/README.md).

---

## 📚 Lessons

The 9 lessons, grouped into three arcs. Full index: **[lessons/README.md](lessons/README.md)**.

| Arc | Lessons |
|-----|---------|
| 🎯 Attention | [Attention Fundamentals](lessons/attention-fundamentals.md) · [Self-Attention](lessons/self-attention.md) · [Multi-Head Attention](lessons/multi-head-attention.md) · [Masked & Cross-Attention](lessons/masked-and-cross-attention.md) |
| 🏗️ Architecture | [Positional Encoding](lessons/positional-encoding.md) · [The Transformer Architecture](lessons/transformer-architecture.md) · [Building a Transformer from Scratch](lessons/transformer-from-scratch.md) |
| 🧬 Variants | [Architecture Variants (BERT/GPT/T5)](lessons/architecture-variants.md) · [Vision Transformers](lessons/vision-transformers.md) |

## 🧩 Practice & assessment

| Resource | What's inside |
|----------|---------------|
| [Exercises](exercises/README.md) | 5 coding + 2 debugging exercises with hints & solutions |
| [Assignments](assignments/README.md) | 2 rubric-graded assignments (MHA from scratch; train a tiny GPT) |
| [Mini Projects](projects/README.md) | 3 builds (attention visualizer, annotated block, fine-tuning) |
| [Portfolio Project](../18-projects/advanced/transformer-from-scratch.md) | A trained, documented mini-GPT built entirely from scratch |

## ⚡ Quick reference & prep

| Resource | Location |
|----------|----------|
| Cheat sheets | [Attention](../20-cheat-sheets/transformers/attention.md) · [Architecture](../20-cheat-sheets/transformers/architecture.md) · [Shapes & Masks](../20-cheat-sheets/transformers/shapes-and-masks.md) |
| Interview questions | [Transformers (20 Q&A)](../19-interview-preparation/transformers/transformers.md) |
| References | [Papers, explainers, code](references.md) |

---

## 🗺️ Suggested path

1. Work the **lessons** arc by arc (attention → architecture → variants).
2. After each arc, do the matching **[exercises](exercises/README.md)**.
3. Consolidate with the two **[assignments](assignments/README.md)** and the **[mini projects](projects/README.md)**.
4. Finish with the **[portfolio project](../18-projects/advanced/transformer-from-scratch.md)** and review with the **[interview bank](../19-interview-preparation/transformers/transformers.md)**.

## ➡️ What's next

Module 7 — [Large Language Models](../07-large-language-models/README.md) (not yet written; see the [Roadmap](../.claude/ROADMAP.md)).

---

## 🧭 Navigation

- 🏠 [Knowledge Base Home](../README.md)
- ⬅️ [Module 5 — Natural Language Processing](../05-nlp/README.md)
- 📖 [Master Curriculum](../.claude/MASTER_CURRICULUM.md) · 🗺️ [Roadmap](../.claude/ROADMAP.md)

---
*Module 6 is content-complete. Contributions and corrections welcome per
[CONTRIBUTING.md](../CONTRIBUTING.md).*
