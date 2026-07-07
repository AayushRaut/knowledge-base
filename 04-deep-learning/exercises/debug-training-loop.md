---
title: "Debugging: Fix the Broken Training Loop"
description: Find three classic PyTorch bugs — missing zero_grad, missing eval mode, and missing no_grad.
type: exercise
domain: 04-deep-learning
tags: [deep-learning, debugging, pytorch, training-loop]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 04-deep-learning/lessons/pytorch
  - 04-deep-learning/lessons/regularization-and-training
---

# Debugging: Fix the Broken Training Loop

> Practice for **[PyTorch Essentials](../lessons/pytorch.md)**.

---

## Problem

This training script "runs" but the loss is erratic and validation numbers are
wrong. It contains **three** classic bugs. Find and fix all of them.

```python
import torch

def train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)                    # BUG territory below…
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        correct = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
        print(f"epoch {epoch}: val acc {correct / len(val_loader.dataset):.3f}")
```

## Requirements

- [ ] Identify all three bugs and the symptom each causes.
- [ ] Produce a corrected version.

---

## Hints

<details>
<summary>Hint 1</summary>

What happens to `.grad` buffers between batches if you never clear them?

</details>

<details>
<summary>Hint 2</summary>

The model contains dropout/batch-norm. What mode is it in during validation?
And is autograd recording the validation forward passes?

</details>

---

## Solution

<details>
<summary>Show solution</summary>

**The three bugs:**

1. **No `optimizer.zero_grad()`** — PyTorch *accumulates* gradients, so each
   step applies the sum of all previous batches' gradients: erratic, exploding
   updates.
2. **No `model.train()` / `model.eval()` switching** — dropout stays active and
   batch-norm keeps updating running stats during validation, corrupting both
   training behavior and validation numbers.
3. **No `torch.no_grad()` in validation** — autograd builds graphs for every
   validation batch, wasting memory (and risking OOM) for gradients you never use.

```python
import torch

def train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()                              # fix 2a
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()                  # fix 1
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()                               # fix 2b
        correct = 0
        with torch.no_grad():                      # fix 3
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        print(f"epoch {epoch}: val acc {correct / len(val_loader.dataset):.3f}")
```

**Explanation:** These three omissions are probably the most common real-world
PyTorch bugs. None of them raises an error — the code runs and produces numbers,
just wrong ones. That's why knowing the canonical loop cold matters.

</details>

---

## Navigation

- ⬆️ [Module 4 Exercises](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
