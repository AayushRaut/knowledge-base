---
title: "Exercise: Write the Canonical Training Loop"
description: Write a complete, correct PyTorch train/validate loop from memory — the skill you'll use daily.
type: exercise
domain: 04-deep-learning
tags: [deep-learning, pytorch, training-loop]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 30 min
reinforces:
  - 04-deep-learning/lessons/pytorch
  - 04-deep-learning/lessons/regularization-and-training
---

# Exercise: Write the Canonical Training Loop

> Practice for **[PyTorch Essentials](../lessons/pytorch.md)**.

---

## Problem

Without copying from the lesson, write `train_one_epoch(model, loader, loss_fn,
optimizer, device)` and `evaluate(model, loader, loss_fn, device)` with every
correctness detail in place.

## Requirements

- [ ] Train: `model.train()`, move batches to device, `zero_grad` → forward →
      loss → `backward` → `step`.
- [ ] Eval: `model.eval()` **and** `torch.no_grad()`; no optimizer calls.
- [ ] Both return average loss; eval also returns accuracy.
- [ ] Device-agnostic (works on CPU or CUDA).

---

## Hints

<details>
<summary>Hint 1</summary>

The five training steps per batch, in order: `optimizer.zero_grad()`,
`out = model(x)`, `loss = loss_fn(out, y)`, `loss.backward()`, `optimizer.step()`.

</details>

<details>
<summary>Hint 2</summary>

`model.eval()` switches dropout/batch-norm behavior; `torch.no_grad()` disables
gradient tracking. You need **both** in evaluation.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn,
                    optimizer, device: torch.device) -> float:
    model.train()                                  # enable dropout/BN updates
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()                      # clear stale gradients
        loss = loss_fn(model(x), y)
        loss.backward()                            # accumulate fresh gradients
        optimizer.step()                           # update parameters
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn,
             device: torch.device) -> tuple[float, float]:
    model.eval()                                   # freeze dropout/BN stats
    total, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total += loss_fn(out, y).item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total / n, correct / n
```

**Explanation:** Every line guards against a real bug: forgetting `zero_grad`
silently accumulates gradients across batches; forgetting `model.eval()` leaves
dropout on during validation; forgetting `no_grad` wastes memory and can leak
graph state; weighting `loss.item()` by batch size makes the epoch average
correct when the last batch is smaller.

</details>

---

## Navigation

- ⬆️ [Module 4 Exercises](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
