---
title: "Exercise: Gradient Descent in 1D"
description: Implement gradient descent to minimize a one-dimensional function and observe learning-rate behavior.
type: exercise
domain: 02-mathematics-foundations
tags: [mathematics, optimization, gradient-descent, calculus]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 02-mathematics-foundations/lessons/gradient-descent
---

# Exercise: Gradient Descent in 1D

> Practice for **[Gradient Descent](../lessons/gradient-descent.md)**.

---

## Problem

Minimize $f(x) = (x - 3)^2 + 2$ using gradient descent, starting from $x_0 = 0$.
The analytic minimum is at $x = 3$.

The gradient is $f'(x) = 2(x - 3)$, and the update rule is:

$$
x \leftarrow x - \eta\, f'(x)
$$

## Requirements

- [ ] Implement `gradient_descent(grad, x0, lr, steps)` returning the final `x` and the history.
- [ ] Confirm it converges to $x \approx 3$ with a good learning rate.
- [ ] Show what happens when `lr` is too large (e.g. `1.1`) — it diverges.

---

## Hints

<details>
<summary>Hint 1</summary>

Each step: `x = x - lr * grad(x)`. Record `x` each iteration to inspect convergence.

</details>

<details>
<summary>Hint 2</summary>

For $f'(x)=2(x-3)$, the update is $x \leftarrow x - \eta\,2(x-3)$. Divergence occurs
when the step overshoots so far that $|x-3|$ grows — here when $\eta > 1$.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
from collections.abc import Callable


def gradient_descent(
    grad: Callable[[float], float], x0: float, lr: float, steps: int
) -> tuple[float, list[float]]:
    x = x0
    history = [x]
    for _ in range(steps):
        x = x - lr * grad(x)
        history.append(x)
    return x, history


grad = lambda x: 2 * (x - 3)  # f'(x) = 2(x - 3)

x_final, hist = gradient_descent(grad, x0=0.0, lr=0.1, steps=50)
print(f"converged to {x_final:.4f}")          # ~3.0000

x_div, _ = gradient_descent(grad, x0=0.0, lr=1.1, steps=20)
print(f"diverged to {x_div:.1f}")             # blows up
```

**Explanation:** With $f'(x)=2(x-3)$, each step moves $x$ toward 3 by a fraction
set by $\eta$. A moderate `lr` (0.1) converges smoothly; `lr` above 1 overshoots
the minimum by more than it started, so the error grows each step and the iterate
diverges. This 1D case is the intuition behind tuning learning rates for real models.

</details>

---

## Navigation

- ⬆️ [Module 2 Exercises](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
