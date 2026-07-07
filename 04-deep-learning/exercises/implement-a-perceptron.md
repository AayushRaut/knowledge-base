---
title: "Exercise: Implement a Perceptron"
description: Build the classic perceptron and its learning rule in NumPy, and see why it can't learn XOR.
type: exercise
domain: 04-deep-learning
tags: [deep-learning, perceptron, numpy]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 04-deep-learning/lessons/neural-networks
---

# Exercise: Implement a Perceptron

> Practice for **[Neural Networks and the Perceptron](../lessons/neural-networks.md)**.

---

## Problem

Implement a perceptron with the classic learning rule
$w \leftarrow w + \eta\,(y - \hat{y})\,x$, train it on the linearly separable
**AND** problem, then show it fails on **XOR**.

## Requirements

- [ ] `Perceptron` class with `fit(X, y)` and `predict(X)`.
- [ ] Converges on AND (all 4 points correct).
- [ ] Demonstrably fails on XOR (never reaches 4/4 within many epochs).
- [ ] Explain *why* it fails on XOR.

---

## Hints

<details>
<summary>Hint 1</summary>

Prediction is `1 if w @ x + b > 0 else 0`. Update `w` and `b` only on
misclassified points.

</details>

<details>
<summary>Hint 2</summary>

XOR's positive points sit on one diagonal, negatives on the other — no single
line separates them.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


class Perceptron:
    def __init__(self, lr: float = 0.1, epochs: int = 100) -> None:
        self.lr, self.epochs = lr, epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = int(self.w @ xi + self.b > 0)
                if pred != yi:                      # update only on mistakes
                    self.w += self.lr * (yi - pred) * xi
                    self.b += self.lr * (yi - pred)
                    errors += 1
            if errors == 0:                         # converged
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.w + self.b > 0).astype(int)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_xor = np.array([0, 1, 1, 0])

print(Perceptron().fit(X, y_and).predict(X))  # [0 0 0 1] — converges
print(Perceptron().fit(X, y_xor).predict(X))  # never all correct
```

**Explanation:** The perceptron learns a **linear** decision boundary
($w\cdot x + b = 0$). AND is linearly separable, so the update rule converges
(the perceptron convergence theorem). XOR is not — its classes sit on opposite
diagonals — so no weight vector exists, and the updates cycle forever. Solving
XOR needs a hidden layer with a nonlinearity: the historical motivation for
multi-layer networks.

</details>

---

## Navigation

- ⬆️ [Module 4 Exercises](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
