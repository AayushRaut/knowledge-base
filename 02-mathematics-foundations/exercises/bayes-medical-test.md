---
title: "Exercise: The Medical-Test Problem"
description: Apply Bayes' rule to a base-rate problem and see why a positive test can still be probably wrong.
type: exercise
domain: 02-mathematics-foundations
tags: [mathematics, probability, bayes, base-rate]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 02-mathematics-foundations/lessons/bayes-rule
---

# Exercise: The Medical-Test Problem

> Practice for **[Bayes' Rule](../lessons/bayes-rule.md)**.

---

## Problem

A disease affects **1 in 1000** people. A test has:

- **Sensitivity** (true positive rate) $P(+\mid D) = 0.99$
- **Specificity** $P(-\mid \neg D) = 0.95$ (so the false-positive rate is $0.05$)

A randomly chosen person tests **positive**. What is $P(D \mid +)$? Compute it
with Bayes' rule and verify by simulation.

## Requirements

- [ ] Compute $P(D \mid +)$ analytically.
- [ ] Confirm with a Monte Carlo simulation.
- [ ] Explain why the answer is so much lower than 99%.

---

## Hints

<details>
<summary>Hint 1</summary>

$$P(D\mid +) = \dfrac{P(+\mid D)\,P(D)}{P(+\mid D)P(D) + P(+\mid \neg D)P(\neg D)}$$

</details>

<details>
<summary>Hint 2</summary>

The denominator is dominated by false positives because the disease is rare
(the base rate $P(D)$ is tiny).

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def posterior(prevalence: float, sensitivity: float, specificity: float) -> float:
    p_d = prevalence
    p_pos_given_d = sensitivity
    p_pos_given_notd = 1 - specificity
    p_pos = p_pos_given_d * p_d + p_pos_given_notd * (1 - p_d)
    return p_pos_given_d * p_d / p_pos


analytic = posterior(0.001, 0.99, 0.95)
print(f"P(D | +) = {analytic:.4f}")   # ~0.0194

# Simulation check
rng = np.random.default_rng(0)
n = 2_000_000
has_disease = rng.random(n) < 0.001
test_pos = np.where(
    has_disease,
    rng.random(n) < 0.99,      # sensitivity
    rng.random(n) < 0.05,      # false-positive rate
)
sim = has_disease[test_pos].mean()
print(f"simulated  = {sim:.4f}")
assert abs(sim - analytic) < 0.005
```

**Explanation:** Even a very accurate test produces mostly false positives when
the condition is rare, because 5% of the huge healthy population dwarfs the 99%
of the tiny sick population. This **base-rate** effect is why priors matter — the
same reasoning underlies spam filters and anomaly detection.

</details>

---

## Navigation

- ⬆️ [Module 2 Exercises](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
