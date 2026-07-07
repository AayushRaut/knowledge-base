---
title: "Exercise: Build a Classification Pipeline"
description: Assemble a leak-free scikit-learn pipeline that preprocesses features and trains a classifier.
type: exercise
domain: 03-machine-learning
tags: [machine-learning, scikit-learn, pipeline, classification]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 03-machine-learning/lessons/scikit-learn-workflow
  - 03-machine-learning/lessons/classification
---

# Exercise: Build a Classification Pipeline

> Practice for **[The scikit-learn Workflow](../lessons/scikit-learn-workflow.md)**
> and **[Classification](../lessons/classification.md)**.

---

## Problem

Given a tabular dataset with numeric and categorical columns, build a single
scikit-learn `Pipeline` that scales numerics, one-hot encodes categoricals, and
trains a `LogisticRegression` — then evaluate it on a held-out test set.

## Requirements

- [ ] Use `ColumnTransformer` for per-type preprocessing.
- [ ] Wrap preprocessing + model in one `Pipeline`.
- [ ] Fit on train only; report test accuracy.
- [ ] No preprocessing fit on the test set (no leakage).

---

## Hints

<details>
<summary>Hint 1</summary>

`ColumnTransformer` takes a list of `(name, transformer, columns)` tuples — one
branch for numeric columns, one for categorical.

</details>

<details>
<summary>Hint 2</summary>

Put the `ColumnTransformer` and the classifier into a single `Pipeline` so
`fit`/`predict` handle everything and nothing leaks from test to train.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_and_score(X, y, numeric, categorical):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model.fit(X_train, y_train)          # preprocessing fit on TRAIN only
    return model.score(X_test, y_test)
```

**Explanation:** The `Pipeline` guarantees the scaler and encoder are fit only on
training folds and merely applied to the test set — the single most common way to
avoid data leakage. `handle_unknown="ignore"` keeps unseen categories at inference
from crashing. `stratify=y` preserves class balance across the split.

</details>

---

## Navigation

- ⬆️ [Module 3 Exercises](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
