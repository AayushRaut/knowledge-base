---
title: Calculus & Optimization Cheat Sheet
description: Fast reference for derivatives, gradients, gradient descent, and loss functions.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [mathematics, calculus, optimization, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Calculus & Optimization Cheat Sheet

> Fast reference. For depth, see
> [Module 2 lessons](../../02-mathematics-foundations/lessons/README.md).

---

## Core Concepts

| Term | Definition |
|------|------------|
| Derivative | Instantaneous rate of change / slope. |
| Gradient $\nabla f$ | Vector of partial derivatives; points uphill. |
| Chain rule | Derivative of composed functions; the basis of backprop. |
| Convex | Bowl-shaped; any local min is the global min. |
| Learning rate $\eta$ | Step size in gradient descent. |

---

## Key Formulas

$$
\nabla f = \left[\tfrac{\partial f}{\partial x_1},\dots,\tfrac{\partial f}{\partial x_n}\right]
\qquad
\theta \leftarrow \theta - \eta\,\nabla_\theta L(\theta)
$$

- **MSE:** $L=\frac{1}{n}\sum (\hat{y}_i-y_i)^2$.
- **Cross-entropy:** $L=-\sum_i y_i \log \hat{y}_i$.
- **Chain rule:** $\frac{d}{dx}f(g(x))=f'(g(x))\,g'(x)$.

---

## Common Commands / API

```python
import numpy as np
from scipy.optimize import minimize

# numerical gradient check (central difference)
def num_grad(f, x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g

minimize(f, x0, jac=grad, method="BFGS")   # off-the-shelf optimizer
```

---

## Decision Guide

| If you need… | Use… | Notes |
|--------------|------|-------|
| Train on huge data | mini-batch SGD | Balances speed and stability. |
| A robust default optimizer | Adam | Adaptive per-parameter steps. |
| Verify a hand-coded gradient | numerical gradient check | Central difference. |
| Discourage large weights | $L_2$ regularization | Adds $\lambda\|\theta\|_2^2$. |

---

## Gotchas

- ⚠️ Learning rate too high → diverges; too low → crawls.
- ⚠️ Always scale/standardize features before gradient descent.
- ⚠️ Non-convex loss → result depends on initialization.
- ⚠️ `log(0)` → clip probabilities or compute loss from logits.

---

## Quick Links

- 📖 [Calculus](../../02-mathematics-foundations/lessons/calculus.md) · [Gradient Descent](../../02-mathematics-foundations/lessons/gradient-descent.md) · [Optimization](../../02-mathematics-foundations/lessons/optimization.md)
- 🔗 [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)

---

## Navigation

- ⬆️ [Mathematics Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
