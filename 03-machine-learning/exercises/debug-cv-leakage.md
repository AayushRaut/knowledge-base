---
title: "Debugging: Fix the Cross-Validation Leakage"
description: Find why cross-validation scores are too optimistic — preprocessing fit outside the CV loop.
type: exercise
domain: 03-machine-learning
tags: [machine-learning, debugging, cross-validation, data-leakage]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 03-machine-learning/lessons/cross-validation
  - 03-machine-learning/lessons/scikit-learn-workflow
---

# Debugging: Fix the Cross-Validation Leakage

> Practice for **[Cross-Validation](../lessons/cross-validation.md)** and
> **[The scikit-learn Workflow](../lessons/scikit-learn-workflow.md)**.

---

## Problem

This code cross-validates a model but the scores are suspiciously high and don't
hold up on new data. There is **leakage**: preprocessing (scaling and feature
selection) is fit on the whole dataset *before* cross-validation. Fix it.

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def evaluate(X, y):
    # BUG: scaler and feature selector see the ENTIRE dataset, including the
    # rows that will later be used as validation folds -> leakage.
    X_scaled = StandardScaler().fit_transform(X)
    X_sel = SelectKBest(f_classif, k=10).fit_transform(X_scaled, y)
    return cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=5).mean()
```

## Requirements

- [ ] Explain exactly what leaks and why it inflates the score.
- [ ] Fix it so every transform is fit only on each fold's training data.

---

## Hints

<details>
<summary>Hint 1</summary>

`fit_transform(X)` before CV lets the scaler's mean/std and the feature selector's
choices depend on the validation rows.

</details>

<details>
<summary>Hint 2</summary>

Put every step in a `Pipeline` and pass that to `cross_val_score` — each fold then
re-fits preprocessing on its own training portion.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(X, y) -> float:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=10)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    # Now cross_val_score refits the whole pipeline per fold — no leakage.
    return float(cross_val_score(pipe, X, y, cv=5).mean())
```

**Explanation:** `SelectKBest` and `StandardScaler` are *learned* from data. Fitting
them on the full dataset lets information from the validation folds influence
scaling and which features are kept, so CV scores are optimistically biased. Inside
a `Pipeline`, `cross_val_score` re-fits every step on each fold's training data
only — the validation fold stays truly unseen, giving an honest estimate.

</details>

---

## Navigation

- ⬆️ [Module 3 Exercises](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
