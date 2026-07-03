---
title: "Assignment: Statistical Analysis + Naive Bayes"
description: Run a proper statistical analysis of a dataset and build a Naive Bayes classifier from scratch.
type: assignment
domain: 02-mathematics-foundations
tags: [mathematics, statistics, probability, bayes, classification]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 3–4 hours
covers:
  - 02-mathematics-foundations/lessons/probability
  - 02-mathematics-foundations/lessons/statistics
  - 02-mathematics-foundations/lessons/bayes-rule
---

# Assignment: Statistical Analysis + Naive Bayes

> A larger, assessed task that integrates multiple concepts from
> **[Module 2](../README.md)**.

---

## Context

Probability and statistics turn raw data into defensible claims. You will (a)
analyze a dataset rigorously and (b) build a **Naive Bayes** classifier from
scratch — applying Bayes' rule end to end. No `sklearn` for the classifier.

## Objectives

- Summarize and test hypotheses about real data with correct interpretation.
- Implement a probabilistic classifier from first principles.
- Communicate uncertainty honestly (intervals, caveats).

---

## Tasks

1. **Explore & summarize** — Pick a small labeled dataset (e.g. a public CSV).
   Report descriptive statistics and at least one visualization. See
   [Statistics](../lessons/statistics.md).
2. **Hypothesis test** — Form a question ("does group A differ from group B on
   feature X?"), run an appropriate test, and interpret the p-value **and effect
   size** — no p-hacking, state assumptions. See
   [Probability](../lessons/probability.md).
3. **Confidence intervals** — Report a confidence interval for a key estimate and
   explain what it does and does not mean.
4. **Naive Bayes from scratch** — Implement a Gaussian or multinomial Naive Bayes
   classifier using Bayes' rule with log-probabilities. Evaluate accuracy on a
   held-out split. See [Bayes' Rule](../lessons/bayes-rule.md).

## Constraints

- NumPy / SciPy / pandas allowed; **no `sklearn` for the classifier itself**.
- Work in log-space to avoid numerical underflow.
- Split train/test before fitting anything (no leakage).

---

## Deliverables

- [ ] A notebook or script with the analysis and plots.
- [ ] `naive_bayes.py` implementing `fit` / `predict` with log-probabilities.
- [ ] A short report: findings, test interpretation, and classifier accuracy.
- [ ] Tests for the classifier on a tiny synthetic dataset.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Statistical correctness | 35% | Right test, correct interpretation, honest caveats. |
| Classifier | 35% | Correct Bayes derivation and working log-space implementation. |
| Communication | 20% | Clear report; uncertainty stated properly. |
| Code quality | 10% | Clean, typed, tested. |

---

## Submission

Push to a branch `add/module-2-stats-naive-bayes` and open a pull request.

## Further Reading

- Think Stats — Allen B. Downey (https://greenteapress.com/wp/think-stats-2e/)

---

## Navigation

- ⬆️ [Module 2 Assignments](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
