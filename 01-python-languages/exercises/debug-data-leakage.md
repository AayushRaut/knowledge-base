---
title: "Debugging: Fix the Data-Leakage Bug"
description: Find and fix a subtle train/test leakage bug in a preprocessing pipeline.
type: exercise
domain: 01-python-languages
tags: [python, debugging, data-leakage, scikit-learn, feature-engineering]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 01-python-languages/lessons/feature-engineering
---

# Debugging: Fix the Data-Leakage Bug

> Practice for **[Feature Engineering](../lessons/feature-engineering.md)**.

---

## Problem

The code below scales features and evaluates a model. It reports a suspiciously
high score. There is a **data-leakage** bug — the model has seen information from
the test set during preprocessing. Find it and fix it.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_and_score(X: np.ndarray, y: np.ndarray) -> float:
    # BUG: scaling is fit on the full dataset before splitting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=0
    )
    model = LogisticRegression().fit(X_train, y_train)
    return model.score(X_test, y_test)
```

## Requirements

- [ ] Explain *why* this leaks.
- [ ] Fix it so no test statistics influence training.
- [ ] Prefer a solution that also works inside cross-validation.

---

## Hints

<details>
<summary>Hint 1</summary>

`StandardScaler.fit_transform` computes the mean and standard deviation. What data
does it see here, and does the test set contribute to those statistics?

</details>

<details>
<summary>Hint 2</summary>

Fit transforms on **training data only**, then `transform` the test set. A
`Pipeline` makes this automatic and CV-safe.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_and_score(X: np.ndarray, y: np.ndarray) -> float:
    # Split FIRST, so the test set never influences preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    # The pipeline fits the scaler on train only, then applies it to test.
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
```

**Why it leaked:** `fit_transform(X)` computed the mean/std over *all* rows,
including the test rows. Those statistics encode information about the test
distribution, so the reported score is optimistically biased. Fitting the scaler
inside a `Pipeline` after the split guarantees the test set is only ever
`transform`-ed with training-derived statistics — and the same pipeline plugs
directly into `cross_val_score` without leaking across folds.

</details>

---

## Navigation

- ⬆️ [Module 1 Exercises](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
