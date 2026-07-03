---
title: Statistics
description: Statistical inference for AI — descriptive statistics, sampling, the Central Limit Theorem, confidence intervals, and hypothesis testing.
type: lesson
domain: 02-mathematics-foundations
tags: [mathematics, statistics, inference]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 45 min
prerequisites:
  - 02-mathematics-foundations/lessons/probability
---

# Statistics

> **TL;DR:** Statistics turns data into decisions. Learn how sample summaries estimate population truths, why the Central Limit Theorem makes averages Gaussian, and how confidence intervals and hypothesis tests quantify what you actually know.

---

## Overview
Probability starts from a known model and predicts data; statistics runs it backwards — you *have* the data and want to infer the model. This is the daily work of AI engineering: estimating a model's true accuracy from a test set, deciding whether variant B really beats variant A, and reporting uncertainty honestly. Sloppy statistics is how teams ship regressions they mistook for improvements.

**By the end, you will be able to:**
- Compute and interpret descriptive statistics and explain the population-vs-sample distinction.
- State the Central Limit Theorem and build a confidence interval for a mean.
- Run and correctly interpret a t-test, including what a p-value does and does not mean.

---

## Intuition
Imagine you want the average height of everyone in a country (the **population**) but can only measure a few thousand people (the **sample**). Your sample average is a good guess, but a *different* sample would give a slightly different number. Statistics is the study of that wiggle: how much your estimate would bounce around if you could redo the sampling.

The magic result is the **Central Limit Theorem**: no matter how weirdly shaped the population is, the *average* of a large enough sample behaves like a smooth bell curve. That single fact is why we can attach error bars — confidence intervals — to almost any average, and why p-values from t-tests are trustworthy at reasonable sample sizes.

---

## Details

### Mathematics

**Descriptive statistics.** For observations $x_1, \dots, x_n$, the sample **mean** and **median** measure center; the sample **variance** and **standard deviation** measure spread:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad
s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2, \qquad
s = \sqrt{s^2}.
$$

Here $n$ is the sample size and $\bar{x}$ the sample mean.

**Population vs sample and Bessel's correction.** The *population* is the full set of interest with true parameters mean $\mu$ and variance $\sigma^2$. A *sample* is a finite draw from it. Dividing the sample variance by $n-1$ instead of $n$ — **Bessel's correction** — makes it an *unbiased* estimator of $\sigma^2$. The reason: the deviations are measured from $\bar{x}$, which is itself fit to the data, so it hugs the points too closely; dividing by $n$ would systematically *underestimate* $\sigma^2$. One degree of freedom is "used up" estimating the mean, leaving $n-1$.

**Estimators and bias.** An *estimator* $\hat{\theta}$ is a function of the data used to guess a parameter $\theta$. Its *bias* is $\mathbb{E}[\hat{\theta}] - \theta$; an estimator is **unbiased** when this is 0. The sample mean is unbiased for $\mu$; the $n-1$ variance is unbiased for $\sigma^2$.

**Sampling distribution and the Central Limit Theorem.** The *sampling distribution* is the distribution of a statistic across repeated samples. For the sample mean of $n$ i.i.d. draws with mean $\mu$ and finite variance $\sigma^2$, the CLT states:

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \;\xrightarrow{\;d\;}\; \mathcal{N}(0, 1) \quad \text{as } n \to \infty,
$$

meaning $\bar{X}_n$ is approximately $\mathcal{N}\!\big(\mu, \sigma^2/n\big)$. The quantity $\sigma/\sqrt{n}$ is the **standard error** of the mean — the typical size of the sampling wiggle.

**Confidence intervals.** A $95\%$ confidence interval for $\mu$ (using the sample estimate $s$) is

$$
\bar{x} \pm t_{0.975,\,n-1}\,\frac{s}{\sqrt{n}},
$$

where $t_{0.975,\,n-1}$ is the $97.5$th percentile of the $t$-distribution with $n-1$ degrees of freedom. Interpretation: across many repeated samples, $95\%$ of such intervals contain the true $\mu$.

**Hypothesis testing.** State a *null hypothesis* $H_0$ (e.g. two group means are equal, $\mu_A = \mu_B$) and an *alternative* $H_1$ (they differ). Compute a test statistic and its **p-value** — the probability of a result at least as extreme as observed *if $H_0$ were true*. A two-sample t-test statistic is

$$
t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\dfrac{s_A^2}{n_A} + \dfrac{s_B^2}{n_B}}}.
$$

If $p < \alpha$ (commonly $\alpha = 0.05$) you reject $H_0$. Caveat: the p-value is **not** the probability that $H_0$ is true, and failing to reject is **not** proof that $H_0$ holds.

### Python implementation

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=7)

# Descriptive statistics with the correct (unbiased) sample variance.
data = rng.normal(loc=50.0, scale=8.0, size=200)
print(f"mean   = {data.mean():.2f}")
print(f"median = {np.median(data):.2f}")
print(f"std    = {data.std(ddof=1):.2f}")   # ddof=1 => divide by n-1 (Bessel)

# 95% confidence interval for the mean via the t-distribution.
n = data.size
se = data.std(ddof=1) / np.sqrt(n)          # standard error
t_crit = stats.t.ppf(0.975, df=n - 1)
lo, hi = data.mean() - t_crit * se, data.mean() + t_crit * se
print(f"95% CI = [{lo:.2f}, {hi:.2f}]")

# Two-sample t-test: does group B differ from group A?
group_a = rng.normal(loc=0.30, scale=0.10, size=120)  # e.g. conversion rates
group_b = rng.normal(loc=0.34, scale=0.10, size=120)
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"t = {t_stat:.3f}, p = {p_value:.4f}")
```

## Worked Example
You measure model accuracy on 30 held-out batches: $\bar{x} = 0.82$ with sample standard deviation $s = 0.05$. What is a 95% confidence interval for the true mean accuracy?

The standard error is $s/\sqrt{n} = 0.05/\sqrt{30} \approx 0.00913$. With $n-1 = 29$ degrees of freedom, $t_{0.975,29} \approx 2.045$. So the margin is $2.045 \times 0.00913 \approx 0.0187$, giving

$$
0.82 \pm 0.019 \;=\; [0.801,\; 0.839].
$$

You would report accuracy as roughly $0.82$ with a 95% CI of about $[0.80, 0.84]$ — narrow enough to be meaningful, and honest about the remaining uncertainty.

## Best Practices
- ✅ Use `ddof=1` in NumPy for sample standard deviation; the default `ddof=0` divides by $n$ and is biased for the population variance.
- ✅ Always report an interval or standard error alongside a point estimate.
- ✅ Fix the sample size and significance level *before* looking at results to avoid p-hacking.

## Common Mistakes
- ⚠️ Interpreting a p-value as "the probability the null is true." Fix: it is the probability of the data (or more extreme) *given* the null.
- ⚠️ Using the default `np.std` (population, $n$) when you mean the sample estimate. Fix: pass `ddof=1`.
- ⚠️ Peeking at A/B results repeatedly and stopping at the first significant moment inflates false positives. Fix: pre-register the sample size or use sequential-testing corrections.

## Industry Tips
- 💡 Report confidence intervals on evaluation metrics, not bare numbers — reviewers and stakeholders trust error bars.
- 💡 The CLT is why bootstrap resampling and mini-batch gradient averages behave stably at scale.

## Real-World Use Cases
- A/B testing product changes and model variants with two-sample t-tests.
- Estimating a model's true accuracy or latency with confidence intervals.
- Detecting data drift by testing whether a new sample's mean differs from the training baseline.

---

## Summary
- Descriptive statistics summarize data; inference generalizes from sample to population.
- Bessel's $n-1$ correction removes bias in the sample variance.
- The CLT makes sample means approximately Gaussian, enabling confidence intervals.
- Hypothesis tests quantify evidence against a null; interpret p-values carefully.

## Practice
- [ ] Exercises: [Module 2 Exercises](../exercises/README.md)
- [ ] Self-check: Why does a larger $n$ produce a narrower confidence interval?

## Further Reading
- 📘 Mathematics for Machine Learning — Deisenroth, Faisal & Ong (https://mml-book.github.io/)
- 📘 Think Stats — Allen B. Downey (https://greenteapress.com/wp/think-stats-2e/)
- 📄 [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- ▶️ StatQuest (https://www.youtube.com/@statquest)

## Related
- [Probability](probability.md)
- [Bayes' Rule](bayes-rule.md)
- Cross-domain (AI evaluation / A-B testing): [Machine Learning](../../03-machine-learning/README.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
