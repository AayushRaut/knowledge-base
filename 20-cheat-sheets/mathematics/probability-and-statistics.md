---
title: Probability & Statistics Cheat Sheet
description: Fast reference for distributions, expectation, Bayes' rule, estimation, and hypothesis testing.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [mathematics, probability, statistics, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Probability & Statistics Cheat Sheet

> Fast reference. For depth, see
> [Module 2 lessons](../../02-mathematics-foundations/lessons/README.md).

---

## Core Concepts

| Term | Definition |
|------|------------|
| Random variable | A quantity whose value depends on a random outcome. |
| Expectation | Long-run average, $\mathbb{E}[X]=\sum x\,p(x)$. |
| Variance | Spread, $\mathrm{Var}(X)=\mathbb{E}[(X-\mu)^2]$. |
| Prior / Posterior | Belief before / after seeing data. |
| p-value | P(data at least this extreme \| null true) — **not** P(null true). |

---

## Key Formulas

$$
\mathbb{E}[X]=\sum_x x\,p(x)
\qquad
P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)}
\qquad
\text{SE}=\frac{\sigma}{\sqrt{n}}
$$

- **CLT:** sample means → Normal as $n$ grows, spread $\sigma/\sqrt{n}$.
- **Sample variance:** divide by $n-1$ (Bessel's correction).

---

## Distributions

| Distribution | Use | Mean |
|--------------|-----|------|
| Bernoulli($p$) | single yes/no trial | $p$ |
| Binomial($n,p$) | # successes in $n$ trials | $np$ |
| Normal($\mu,\sigma^2$) | continuous, CLT limit | $\mu$ |
| Poisson($\lambda$) | counts per interval | $\lambda$ |

---

## Common Commands / API

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
rng.normal(0, 1, size=1000)          # sample
np.mean(x); np.var(x, ddof=1)        # unbiased sample variance
stats.norm.pdf(x, loc=0, scale=1)    # density
stats.ttest_ind(a, b)                # two-sample t-test
```

---

## Decision Guide

| If you need… | Use… | Notes |
|--------------|------|-------|
| Update beliefs with evidence | Bayes' rule | Watch the base rate. |
| Compare two group means | t-test | Report effect size too. |
| Estimate uncertainty of a mean | confidence interval / SE | Shrinks like $1/\sqrt{n}$. |
| Model rare event counts | Poisson | — |

---

## Gotchas

- ⚠️ A p-value is not the probability the hypothesis is true.
- ⚠️ Rare conditions → most positives are false (base-rate fallacy).
- ⚠️ `np.var` defaults to `ddof=0` (population); use `ddof=1` for samples.
- ⚠️ Correlation ≠ causation.

---

## Quick Links

- 📖 [Probability](../../02-mathematics-foundations/lessons/probability.md) · [Statistics](../../02-mathematics-foundations/lessons/statistics.md) · [Bayes' Rule](../../02-mathematics-foundations/lessons/bayes-rule.md)
- 🔗 [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)

---

## Navigation

- ⬆️ [Mathematics Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
