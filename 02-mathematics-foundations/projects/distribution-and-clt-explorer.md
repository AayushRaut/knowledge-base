---
title: "Mini Project: Distribution & CLT Explorer"
description: Simulate distributions and demonstrate the Central Limit Theorem interactively.
type: project
domain: 02-mathematics-foundations
tags: [mathematics, probability, statistics, clt, simulation]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 3–4 hours
prerequisites:
  - 02-mathematics-foundations/lessons/probability
  - 02-mathematics-foundations/lessons/statistics
tech_stack: [python, numpy, matplotlib]
---

# Mini Project: Distribution & CLT Explorer

> **What you'll build:** A small tool that samples from various distributions and
> shows the **Central Limit Theorem** in action — sample means becoming Gaussian
> regardless of the source distribution.

---

## Objective

The CLT is the reason the normal distribution is everywhere in statistics. You'll
build a simulator that makes it visible: draw samples from skewed, uniform, or
discrete distributions and watch the distribution of their means converge to a
bell curve.

## Learning Goals

- Sample from and visualize common distributions.
- See sampling distributions and the CLT empirically.
- Relate sample size to the spread of the sample mean.

---

## Prerequisites

- [Probability](../lessons/probability.md), [Statistics](../lessons/statistics.md)
- NumPy + Matplotlib.

## Architecture

```mermaid
flowchart LR
  A[pick source distribution] --> B[draw many samples of size n]
  B --> C[compute each sample mean]
  C --> D[histogram of means]
  D --> E[overlay Normal fit, vary n]
```

---

## Steps

### 1. Distribution zoo
Implement sampling for several distributions (uniform, exponential, Bernoulli,
Poisson) using `np.random.default_rng`, and plot each one's histogram.

### 2. Sampling distribution of the mean
For a chosen distribution, draw many samples of size $n$, compute each sample
mean, and histogram those means.

### 3. Demonstrate the CLT
Repeat for increasing $n$ and overlay a normal curve with mean $\mu$ and standard
deviation $\sigma/\sqrt{n}$. Show the means become bell-shaped as $n$ grows.

### 4. Write up
Explain why the standard error shrinks like $1/\sqrt{n}$ and what that implies for
sample-size decisions.

---

## Deliverables

- [ ] `clt_explorer.py` with sampling + plotting.
- [ ] Figures showing the CLT for at least two source distributions and three sample sizes.
- [ ] `README.md` explaining the standard-error scaling.

## Success Criteria

The histograms of sample means visibly approach a normal shape as $n$ increases,
and the fitted normal's spread matches $\sigma/\sqrt{n}$.

---

## Extensions (Optional)

- 🚀 Add a bootstrap confidence-interval demo.
- 🚀 Show a distribution where convergence is slow (heavy tails).

## Further Reading

- Think Stats — Allen B. Downey (https://greenteapress.com/wp/think-stats-2e/)
- [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)

---

## Navigation

- ⬆️ [Module 2 Mini Projects](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
