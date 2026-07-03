---
title: "Debugging: Fix the Sample-Variance Bug"
description: Diagnose a biased variance estimate caused by the wrong degrees-of-freedom setting.
type: exercise
domain: 02-mathematics-foundations
tags: [mathematics, statistics, debugging, variance]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 15 min
reinforces:
  - 02-mathematics-foundations/lessons/statistics
---

# Debugging: Fix the Sample-Variance Bug

> Practice for **[Statistics](../lessons/statistics.md)**.

---

## Problem

This function is meant to estimate the variance of a population **from a sample**,
but its estimate is consistently too small. Find and fix the bug.

```python
import numpy as np

def sample_variance(x: np.ndarray) -> float:
    # BUG: this computes the *population* variance (divides by n),
    # which is a biased estimator when x is only a sample.
    return np.var(x)
```

## Requirements

- [ ] Explain why the estimate is biased low.
- [ ] Fix it so it is an unbiased estimator of the population variance.
- [ ] Confirm the fix with a simulation.

---

## Hints

<details>
<summary>Hint 1</summary>

The unbiased sample variance divides by $n-1$, not $n$ (Bessel's correction).

</details>

<details>
<summary>Hint 2</summary>

`np.var` takes a `ddof` (delta degrees of freedom) argument.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def sample_variance(x: np.ndarray) -> float:
    # ddof=1 divides by (n - 1): the unbiased sample variance.
    return float(np.var(x, ddof=1))


# Simulation: average sample variance should match the true variance (=1).
rng = np.random.default_rng(0)
true_var = 1.0
biased = np.mean([np.var(rng.standard_normal(5)) for _ in range(200_000)])
unbiased = np.mean([np.var(rng.standard_normal(5), ddof=1) for _ in range(200_000)])
print(f"ddof=0 -> {biased:.3f} (too low)   ddof=1 -> {unbiased:.3f} (~{true_var})")
```

**Explanation:** The sample mean is itself estimated from the data, so deviations
around it underestimate deviations around the true mean. Dividing by $n-1$
(Bessel's correction) corrects this bias. The effect is largest for small samples
— exactly where it matters most.

</details>

---

## Navigation

- ⬆️ [Module 2 Exercises](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
