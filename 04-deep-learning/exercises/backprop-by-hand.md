---
title: "Exercise: Backprop by Hand (and Check It)"
description: Derive the gradients of a tiny network manually, implement them, and verify against PyTorch autograd.
type: exercise
domain: 04-deep-learning
tags: [deep-learning, backpropagation, autograd, gradients]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 35 min
reinforces:
  - 04-deep-learning/lessons/backpropagation
---

# Exercise: Backprop by Hand (and Check It)

> Practice for **[Backpropagation](../lessons/backpropagation.md)**.

---

## Problem

For the tiny network $\hat{y} = w_2 \cdot \tanh(w_1 x)$ with loss
$L = (\hat{y} - y)^2$ (all scalars), derive $\partial L/\partial w_1$ and
$\partial L/\partial w_2$ by the chain rule, implement them in NumPy, and verify
both against PyTorch autograd.

## Requirements

- [ ] Write out the two gradients symbolically.
- [ ] Implement forward + manual backward in NumPy.
- [ ] Match `torch.autograd` gradients to ~1e-6.

---

## Hints

<details>
<summary>Hint 1</summary>

Chain rule pieces: $\partial L/\partial \hat{y} = 2(\hat{y}-y)$;
$\partial \hat{y}/\partial w_2 = \tanh(w_1 x)$;
$\partial \hat{y}/\partial h = w_2$ with $h = \tanh(w_1 x)$ and
$\tanh'(z) = 1 - \tanh^2(z)$.

</details>

<details>
<summary>Hint 2</summary>

In PyTorch, make `w1`, `w2` tensors with `requires_grad=True`, compute the same
loss, call `loss.backward()`, and compare `.grad`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np
import torch

x, y = 0.7, 1.5
w1, w2 = 0.4, -0.9

# --- manual (NumPy) ---
z = w1 * x
h = np.tanh(z)
y_hat = w2 * h
dL_dyhat = 2 * (y_hat - y)
grad_w2 = dL_dyhat * h                                # dL/dw2
grad_w1 = dL_dyhat * w2 * (1 - np.tanh(z) ** 2) * x   # dL/dw1 via chain rule

# --- autograd check ---
tw1 = torch.tensor(w1, requires_grad=True)
tw2 = torch.tensor(w2, requires_grad=True)
loss = (tw2 * torch.tanh(tw1 * torch.tensor(x)) - y) ** 2
loss.backward()

assert abs(tw1.grad.item() - grad_w1) < 1e-6
assert abs(tw2.grad.item() - grad_w2) < 1e-6
print("manual gradients match autograd ✓")
```

**Explanation:** Backprop is nothing more than the chain rule evaluated backward
through the computation graph, reusing the forward pass's cached values
($h$, $z$). Autograd automates exactly this bookkeeping. Verifying a manual
gradient against autograd (or a finite-difference check) is the standard way to
trust custom layers and losses.

</details>

---

## Navigation

- ⬆️ [Module 4 Exercises](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
