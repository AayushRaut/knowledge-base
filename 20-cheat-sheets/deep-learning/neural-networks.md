---
title: Neural Networks Cheat Sheet
description: Activations, losses, initialization, and layer choices at a glance.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [deep-learning, neural-networks, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Neural Networks Cheat Sheet

> Fast reference. For depth, see
> [Module 4 lessons](../../04-deep-learning/lessons/README.md).

---

## Activations

| Activation | Formula / behavior | Use |
|-----------|--------------------|-----|
| ReLU | $\max(0, z)$ | Default hidden activation. |
| LeakyReLU | small slope for $z<0$ | If dying ReLU is a problem. |
| GELU | smooth ReLU-like | Transformers. |
| Sigmoid | $1/(1+e^{-z})$ | Binary **output** only (saturates). |
| Tanh | $[-1, 1]$ | RNN hidden states; zero-centered. |
| Softmax | normalizes logits to probs | Multiclass **output** (with CE loss). |

## Losses

| Task | Loss | PyTorch | Input |
|------|------|---------|-------|
| Regression | MSE / MAE / Huber | `MSELoss` / `L1Loss` / `SmoothL1Loss` | predictions |
| Binary classification | BCE | `BCEWithLogitsLoss` | **logits** |
| Multiclass | Cross-entropy | `CrossEntropyLoss` | **logits** (no softmax layer!) |

## Initialization & layers

| Item | Rule of thumb |
|------|---------------|
| Init | He for ReLU, Xavier for tanh/sigmoid; never all-zeros. |
| Batch norm | After linear/conv, before/after activation (be consistent). |
| Dropout | Hidden layers, 0.1–0.5; **off at eval** (`model.eval()`). |
| Output layer | No activation for regression; logits for classification. |

---

## Key formulas

$$
\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b}) \qquad
o = \left\lfloor \tfrac{n + 2p - k}{s} \right\rfloor + 1 \;\;\text{(conv output size)}
$$

---

## Gotchas

- ⚠️ Softmax + `CrossEntropyLoss` double-applies — pass raw logits.
- ⚠️ Stacked linear layers without nonlinearity collapse to one linear layer.
- ⚠️ Sigmoid/tanh saturate → vanishing gradients in deep stacks.
- ⚠️ Dropout/batch-norm behave differently in train vs eval mode.

---

## Quick Links

- 📖 [Activation Functions](../../04-deep-learning/lessons/activation-functions.md) · [Loss Functions](../../04-deep-learning/lessons/loss-functions.md) · [CNNs](../../04-deep-learning/lessons/cnn.md)
- 🔗 [PyTorch nn docs](https://pytorch.org/docs/stable/nn.html)

---

## Navigation

- ⬆️ [Deep Learning Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
