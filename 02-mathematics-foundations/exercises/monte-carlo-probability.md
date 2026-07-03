---
title: "Exercise: Estimate a Probability by Simulation"
description: Use Monte Carlo sampling to estimate a probability and compare to the analytic answer.
type: exercise
domain: 02-mathematics-foundations
tags: [mathematics, probability, monte-carlo, simulation]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 02-mathematics-foundations/lessons/probability
---

# Exercise: Estimate a Probability by Simulation

> Practice for **[Probability](../lessons/probability.md)**.

---

## Problem

The **birthday problem**: in a room of $n$ people, what is the probability that at
least two share a birthday (365 equally likely days, ignoring leap years)?

Estimate it with a Monte Carlo simulation for $n = 23$, then compare to the exact
value:

$$
P(\text{match}) = 1 - \frac{365!}{(365-n)!\,365^{n}}
$$

## Requirements

- [ ] Simulate many rooms with `np.random.default_rng`.
- [ ] Estimate the probability as the fraction of rooms with a collision.
- [ ] Compute the exact value and check the estimate is close.

---

## Hints

<details>
<summary>Hint 1</summary>

For one room, sample `n` birthdays with `rng.integers(0, 365, size=n)`; a
collision exists when `len(np.unique(room)) < n`.

</details>

<details>
<summary>Hint 2</summary>

The exact probability is `1 - prod((365 - i) / 365 for i in range(n))`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def birthday_mc(n: int, trials: int = 100_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    rooms = rng.integers(0, 365, size=(trials, n))
    # A collision means fewer unique days than people.
    unique_counts = np.array([len(np.unique(r)) for r in rooms])
    return float(np.mean(unique_counts < n))


def birthday_exact(n: int) -> float:
    p_no_match = np.prod([(365 - i) / 365 for i in range(n)])
    return 1.0 - float(p_no_match)


est, exact = birthday_mc(23), birthday_exact(23)
print(f"MC estimate: {est:.3f}   exact: {exact:.3f}")  # both ~0.507
assert abs(est - exact) < 0.02
```

**Explanation:** Monte Carlo turns a hard counting problem into an easy averaging
problem: simulate the random process many times and take the sample mean of the
event indicator. By the law of large numbers the estimate converges to the true
probability — the same principle behind simulation-based evaluation in ML.

</details>

---

## Navigation

- ⬆️ [Module 2 Exercises](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
