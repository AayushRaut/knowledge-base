---
title: "Debugging: Fix the Unstable Cross-Entropy"
description: Fix NaNs from log(0) in a cross-entropy loss and make it numerically stable.
type: exercise
domain: 02-mathematics-foundations
tags: [mathematics, information-theory, debugging, numerical-stability]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 02-mathematics-foundations/lessons/information-theory
---

# Debugging: Fix the Unstable Cross-Entropy

> Practice for **[Information Theory](../lessons/information-theory.md)**.

---

## Problem

This cross-entropy loss returns `nan` on some batches. The cross-entropy between
true labels $y$ (one-hot) and predicted probabilities $\hat{y}$ is:

$$
H(y, \hat{y}) = -\sum_i y_i \log \hat{y}_i
$$

```python
import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # BUG: log(0) = -inf when a predicted probability is exactly 0,
    # and 0 * -inf = nan.
    return float(-np.sum(y_true * np.log(y_pred)))
```

## Requirements

- [ ] Explain when and why it produces `nan`.
- [ ] Fix it so it never returns `nan`/`inf`.
- [ ] Prefer the standard fix used in real ML libraries.

---

## Hints

<details>
<summary>Hint 1</summary>

If any `y_pred` entry is exactly `0` (or `1`), `np.log` overflows to `-inf`.

</details>

<details>
<summary>Hint 2</summary>

Clip predictions into `[eps, 1 - eps]` before taking the log. The best practice
is to compute the loss directly from logits with a log-sum-exp trick, but
clipping is the minimal fix here.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    # Clip away exact 0 and 1 so log is always finite.
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.sum(y_true * np.log(y_pred)))
```

**Explanation:** A model can output a probability of exactly 0 for the true class
(e.g. after rounding or a saturated softmax); $\log 0 = -\infty$, and
$0 \cdot (-\infty)$ is `nan`. Clipping into $[\varepsilon, 1-\varepsilon]$ keeps
the log finite. In production, frameworks avoid the problem entirely by computing
the loss from **logits** with a fused `log_softmax` + negative-log-likelihood
(the log-sum-exp trick), which is more accurate than clipping after a softmax.

</details>

---

## Navigation

- ⬆️ [Module 2 Exercises](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
