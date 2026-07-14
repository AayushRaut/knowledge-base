---
title: "Assignment: Implement Multi-Head Attention from Scratch"
description: Build a fully-tested multi-head attention module in PyTorch and verify it against the reference implementation.
type: assignment
domain: 06-transformers
tags: [transformers, multi-head-attention, pytorch, testing]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 4–6 hours
covers:
  - 06-transformers/lessons/self-attention
  - 06-transformers/lessons/multi-head-attention
  - 06-transformers/lessons/masked-and-cross-attention
---

# Assignment: Implement Multi-Head Attention from Scratch

> A larger, assessed task that integrates multiple concepts from
> **[Module 6](../README.md)**.

---

## Context

Multi-head attention is *the* transformer primitive. You will implement it
end-to-end in PyTorch — supporting self-attention, causal masking, and
cross-attention — and prove it correct against PyTorch's built-in
`nn.MultiheadAttention`.

## Objectives

- Implement scaled dot-product attention and the multi-head wrapper yourself.
- Support causal and padding masks and cross-attention.
- Validate rigorously with tests, not vibes.

---

## Tasks

1. **Scaled dot-product attention** — a function taking `Q, K, V` and an optional
   additive mask, returning output and weights; numerically stable, correctly
   scaled. See [Attention Fundamentals](../lessons/attention-fundamentals.md).
2. **Multi-head wrapper** — an `nn.Module` with `W^Q, W^K, W^V, W^O`, the
   split/combine-heads logic, and clean shape handling. See
   [Multi-Head Attention](../lessons/multi-head-attention.md).
3. **Masking & cross-attention** — accept a causal mask and a key-padding mask;
   support distinct query vs key/value inputs (cross-attention). See
   [Masked and Cross-Attention](../lessons/masked-and-cross-attention.md).
4. **Validate** — tests asserting: attention weights sum to 1; causal masking
   zeroes the upper triangle; your module's output matches
   `nn.MultiheadAttention` on the same weights (or agrees on shapes and key
   properties); a hand-computed 2-token example matches.

## Constraints

- Pure PyTorch tensor ops (no `nn.MultiheadAttention` inside your module — only
  use it as a test oracle).
- Every public method type-annotated with shape docstrings.
- Tests run with `pytest`.

---

## Deliverables

- [ ] `attention.py` with the function + `MultiHeadAttention` module.
- [ ] `test_attention.py` covering properties, masking, and the oracle check.
- [ ] `README.md` documenting the shapes and design.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Correctness | 40% | Matches the reference; masks and cross-attention work. |
| Testing | 25% | Property + oracle + hand-example tests, all passing. |
| Code clarity | 20% | Clean shapes, type hints, docstrings. |
| Understanding | 15% | README explains the shape logic and scaling. |

---

## Submission

Push to a branch `add/module-6-mha-from-scratch` and open a pull request.

## Further Reading

- Attention Is All You Need — Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762)
- The Annotated Transformer (https://nlp.seas.harvard.edu/annotated-transformer/)

---

## Navigation

- ⬆️ [Module 6 Assignments](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
