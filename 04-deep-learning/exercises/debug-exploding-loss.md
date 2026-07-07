---
title: "Debugging: Fix the Exploding Loss"
description: Diagnose a loss that goes to NaN — learning rate, unscaled inputs, and a double log-softmax.
type: exercise
domain: 04-deep-learning
tags: [deep-learning, debugging, numerical-stability, learning-rate]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 04-deep-learning/lessons/optimizers
  - 04-deep-learning/lessons/loss-functions
---

# Debugging: Fix the Exploding Loss

> Practice for **[Optimizers](../lessons/optimizers.md)** and
> **[Loss Functions](../lessons/loss-functions.md)**.

---

## Problem

Training starts, the loss climbs, then becomes `nan` within a few epochs. The
script has **three** contributing bugs. Find and fix them.

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256), nn.ReLU(),
    nn.Linear(256, 10),
    nn.Softmax(dim=1),                       # BUG?
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=10.0)   # BUG?

for x, y in train_loader:                    # x are raw pixel values 0..255
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)              # BUG?
    loss.backward()
    optimizer.step()
```

## Requirements

- [ ] Identify all three problems and why each destabilizes training.
- [ ] Produce a corrected version.

---

## Hints

<details>
<summary>Hint 1</summary>

What does `nn.CrossEntropyLoss` expect as input — probabilities or logits?

</details>

<details>
<summary>Hint 2</summary>

Look at the learning rate's magnitude, and at the scale of the raw inputs
(0–255) hitting the first linear layer.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

**The three bugs:**

1. **`Softmax` before `CrossEntropyLoss`** — `CrossEntropyLoss` applies
   log-softmax internally and expects **raw logits**. Feeding it probabilities
   squashes gradients and is numerically unstable. Remove the `Softmax` layer.
2. **Learning rate 10.0** — absurdly large for SGD here; each step overshoots,
   the weights grow, activations saturate, and the loss diverges to `inf`/`nan`.
   Start around `1e-2` for SGD (or `1e-3` for Adam) and tune from there.
3. **Unscaled inputs (0–255)** — huge input magnitudes produce huge activations
   and gradients, compounding the instability. Normalize to roughly unit scale
   (e.g. divide by 255, or standardize).

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256), nn.ReLU(),
    nn.Linear(256, 10),                      # raw logits out — no Softmax
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

for x, y in train_loader:
    x = x.float() / 255.0                    # scale inputs to [0, 1]
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
```

**Explanation:** NaN losses are almost always numerics: an unstable loss
pairing, a too-hot learning rate, or unnormalized data (often all three, as
here). The debugging order that works in practice: check the loss/output
contract first, then shrink the learning rate, then inspect input scale —
adding gradient clipping if recurrence-driven spikes remain.

</details>

---

## Navigation

- ⬆️ [Module 4 Exercises](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
