---
title: "Exercise: Overfit, Then Regularize"
description: Deliberately overfit a small network, then bring the gap down with dropout and weight decay.
type: exercise
domain: 04-deep-learning
tags: [deep-learning, regularization, dropout, overfitting]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 30 min
reinforces:
  - 04-deep-learning/lessons/regularization-and-training
---

# Exercise: Overfit, Then Regularize

> Practice for **[Regularization and Training Techniques](../lessons/regularization-and-training.md)**.

---

## Problem

The fastest way to understand regularization is to watch it work. Take a small
subset of a dataset (e.g. 500 samples), train an oversized MLP until train
accuracy ≈ 100% while validation lags, then add **dropout** and **weight decay**
and compare the train/validation gap.

## Requirements

- [ ] Baseline: train an MLP on a deliberately small training set; record
      train/val accuracy per epoch and show a large gap.
- [ ] Regularized: same architecture + `nn.Dropout` + `weight_decay` in the
      optimizer; show the gap shrink.
- [ ] Report both curves (or final numbers) side by side.

---

## Hints

<details>
<summary>Hint 1</summary>

Overfitting is easiest to induce with a big model, a tiny dataset, and many
epochs. Deliberately overfitting first is also a real debugging technique — it
proves your pipeline can learn.

</details>

<details>
<summary>Hint 2</summary>

`nn.Dropout(0.5)` between hidden layers; `torch.optim.AdamW(params, weight_decay=1e-2)`.
Remember dropout is only active in `model.train()` mode.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import torch
from torch import nn


def make_mlp(p_drop: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512), nn.ReLU(), nn.Dropout(p_drop),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p_drop),
        nn.Linear(512, 10),
    )

# Baseline: big net, tiny data, no regularization
baseline = make_mlp(p_drop=0.0)
opt_base = torch.optim.AdamW(baseline.parameters(), lr=1e-3, weight_decay=0.0)

# Regularized: same net + dropout + weight decay
regular = make_mlp(p_drop=0.5)
opt_reg = torch.optim.AdamW(regular.parameters(), lr=1e-3, weight_decay=1e-2)

# Train both with the SAME loop on the same 500-sample subset
# (reuse train_one_epoch/evaluate from the training-loop exercise) and
# log train vs validation accuracy each epoch.
```

**Expected outcome:** the baseline reaches ~100% train accuracy with validation
stuck well below it (a wide generalization gap). The regularized run trains a
little slower, tops out lower on train accuracy, but validation is closer to
train — the gap shrinks, which is the whole point.

**Explanation:** Dropout randomly zeroes activations during training so the
network can't rely on any single co-adapted path; weight decay penalizes large
weights, biasing toward simpler functions. Both reduce variance at a small cost
in bias — the deep-learning version of the
[bias–variance tradeoff](../../03-machine-learning/lessons/bias-variance-overfitting.md).

</details>

---

## Navigation

- ⬆️ [Module 4 Exercises](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
