---
title: scikit-learn API Cheat Sheet
description: The estimator API, pipelines, model selection, and persistence at a glance.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [machine-learning, scikit-learn, api, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# scikit-learn API Cheat Sheet

> Fast reference. For depth, see
> [The scikit-learn Workflow](../../03-machine-learning/lessons/scikit-learn-workflow.md).

---

## The estimator API

| Method | On | Does |
|--------|----|----|
| `fit(X, y)` | estimators | Learn parameters from training data. |
| `predict(X)` | models | Produce predictions. |
| `predict_proba(X)` | classifiers | Class probabilities. |
| `transform(X)` | transformers | Apply a learned transformation. |
| `fit_transform(X)` | transformers | Fit then transform (train only!). |
| `score(X, y)` | models | Default metric (accuracy / $R^2$). |

---

## Core workflow

```python
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                           stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=0))])

grid = GridSearchCV(pipe, {"clf__n_estimators": [100, 300]},
                    cv=StratifiedKFold(5, shuffle=True, random_state=0))
grid.fit(X_tr, y_tr)
print(grid.best_params_, grid.score(X_te, y_te))
```

---

## Persistence

```python
import joblib
joblib.dump(grid.best_estimator_, "model.joblib")
model = joblib.load("model.joblib")
```

---

## Decision Guide

| If you need… | Use… |
|--------------|------|
| Per-column preprocessing | `ColumnTransformer` |
| Leak-free preprocessing + model | `Pipeline` |
| Hyperparameter search | `GridSearchCV` / `RandomizedSearchCV` |
| Honest CV estimate | `cross_val_score` on the **pipeline** |
| Imbalanced classes | `class_weight="balanced"`, stratified CV |

---

## Gotchas

- ⚠️ Never `fit`/`fit_transform` on the test set — only `transform`.
- ⚠️ Address params inside a pipeline with `step__param` (double underscore).
- ⚠️ Pass the **pipeline** (not fitted data) to CV so folds refit preprocessing.
- ⚠️ Set `random_state` for reproducibility.

---

## Quick Links

- 📖 [The scikit-learn Workflow](../../03-machine-learning/lessons/scikit-learn-workflow.md) · [Cross-Validation](../../03-machine-learning/lessons/cross-validation.md)
- 🔗 [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)

---

## Navigation

- ⬆️ [Machine Learning Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
