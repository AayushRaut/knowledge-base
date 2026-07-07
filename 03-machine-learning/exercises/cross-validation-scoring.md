---
title: "Exercise: Cross-Validate a Model Correctly"
description: Use stratified k-fold cross-validation with a pipeline to get an honest performance estimate.
type: exercise
domain: 03-machine-learning
tags: [machine-learning, cross-validation, model-selection]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 03-machine-learning/lessons/cross-validation
---

# Exercise: Cross-Validate a Model Correctly

> Practice for **[Cross-Validation](../lessons/cross-validation.md)**.

---

## Problem

A single train/test split gives a noisy score. Estimate a classifier's
performance with **stratified 5-fold** cross-validation, keeping preprocessing
inside the pipeline so nothing leaks across folds.

## Requirements

- [ ] Use a `Pipeline` (scaling + model).
- [ ] Use `StratifiedKFold` with 5 folds.
- [ ] Report the mean and standard deviation of the fold scores.

---

## Hints

<details>
<summary>Hint 1</summary>

Pass the `Pipeline` (not a pre-fitted model) to `cross_val_score` so each fold
re-fits preprocessing on its own training portion.

</details>

<details>
<summary>Hint 2</summary>

`cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cv_estimate(X, y) -> tuple[float, float]:
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())
```

**Explanation:** Passing the whole `Pipeline` to `cross_val_score` means the
scaler is re-fit on each fold's training data only — never on the held-out fold —
so the estimate is unbiased. Stratification keeps class proportions stable in
each fold, which matters for imbalanced data. Reporting the **std** communicates
how much the estimate would wobble on different splits.

</details>

---

## Navigation

- ⬆️ [Module 3 Exercises](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
