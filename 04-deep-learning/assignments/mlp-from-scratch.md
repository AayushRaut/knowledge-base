---
title: "Assignment: MLP from Scratch in NumPy"
description: Implement a multi-layer perceptron — forward pass, backprop, and training — with no deep-learning framework.
type: assignment
domain: 04-deep-learning
tags: [deep-learning, numpy, backpropagation, mlp]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 5–7 hours
covers:
  - 04-deep-learning/lessons/neural-networks
  - 04-deep-learning/lessons/activation-functions
  - 04-deep-learning/lessons/loss-functions
  - 04-deep-learning/lessons/backpropagation
  - 04-deep-learning/lessons/optimizers
---

# Assignment: MLP from Scratch in NumPy

> A larger, assessed task that integrates multiple concepts from
> **[Module 4](../README.md)**.

---

## Context

Nothing cements understanding of deep learning like building it yourself. You
will implement a small multi-layer perceptron in pure NumPy — forward pass,
cross-entropy loss, manual backpropagation, and mini-batch SGD — and prove it
learns a real dataset.

## Objectives

- Translate the forward-pass and backprop math into working array code.
- Verify hand-derived gradients numerically.
- Train to a sensible accuracy with mini-batch SGD.

---

## Tasks

1. **Architecture** — Implement a 2-layer MLP (input → hidden ReLU → softmax
   output) as a class holding `W1, b1, W2, b2`, with He-style random
   initialization. See [Neural Networks](../lessons/neural-networks.md).
2. **Forward + loss** — Implement the forward pass and a numerically stable
   softmax + cross-entropy (subtract the row max before exponentiating). See
   [Loss Functions](../lessons/loss-functions.md).
3. **Backward** — Derive and implement the gradients for all four parameters
   (use the $\hat{p} - y$ shortcut for softmax+CE at the output). See
   [Backpropagation](../lessons/backpropagation.md).
4. **Gradient check** — Verify every gradient against central finite differences
   on a tiny batch (agreement to ~1e-6 relative error).
5. **Train** — Mini-batch SGD on a small image/tabular dataset (e.g.
   scikit-learn's `load_digits`); plot the loss curve and report train/test
   accuracy. Add momentum as a stretch.

## Constraints

- NumPy only (Matplotlib for plots; scikit-learn only to load data/split).
- No autograd — all gradients derived and coded by hand.
- Include the gradient derivations in the README (math, not just code).

---

## Deliverables

- [ ] `mlp.py` with forward, backward, and training code.
- [ ] A passing numerical gradient check.
- [ ] Loss curve + final train/test accuracy on `load_digits` (or similar).
- [ ] `README.md` with the derivations and results.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Correctness | 40% | Gradient check passes; model trains and clearly learns. |
| Math clarity | 25% | Derivations are correct and readable. |
| Code quality | 20% | Clean, typed, vectorized NumPy (no per-sample Python loops). |
| Analysis | 15% | Loss curve discussed; effect of learning rate/hidden size noted. |

---

## Submission

Push to a branch `add/module-4-mlp-from-scratch` and open a pull request.

## Further Reading

- Deep Learning — Goodfellow, Bengio & Courville (https://www.deeplearningbook.org/)
- Dive into Deep Learning — Zhang, Lipton, Li & Smola (https://d2l.ai/)

---

## Navigation

- ⬆️ [Module 4 Assignments](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
