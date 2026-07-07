---
title: "Exercise: ROC, AUC & Threshold Tuning"
description: Move beyond the default 0.5 threshold — plot an ROC curve and pick a threshold from the cost of errors.
type: exercise
domain: 03-machine-learning
tags: [machine-learning, metrics, roc, auc, threshold]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 03-machine-learning/lessons/model-evaluation-metrics
---

# Exercise: ROC, AUC & Threshold Tuning

> Practice for **[Model Evaluation Metrics](../lessons/model-evaluation-metrics.md)**.

---

## Problem

A classifier outputs probabilities, but the default 0.5 cutoff is rarely optimal.
Compute the ROC curve and AUC, then choose a decision threshold that reflects the
real cost of false negatives vs false positives.

## Requirements

- [ ] Get predicted **probabilities** (not just labels).
- [ ] Compute ROC points and AUC.
- [ ] Select a threshold that maximizes recall subject to precision ≥ 0.8 (or a similar stated rule) and report the resulting confusion matrix.

---

## Hints

<details>
<summary>Hint 1</summary>

Use `model.predict_proba(X)[:, 1]` for the positive-class score, then
`roc_curve` and `roc_auc_score`.

</details>

<details>
<summary>Hint 2</summary>

`precision_recall_curve` gives precision/recall at every threshold — scan it to
apply your rule.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)


def tune_threshold(y_true, proba, min_precision: float = 0.8) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    # prec/rec have length len(thr)+1; align by dropping the last point.
    ok = prec[:-1] >= min_precision
    if not ok.any():
        return 0.5
    # Among thresholds meeting the precision floor, maximize recall.
    best = np.argmax(np.where(ok, rec[:-1], -1))
    return float(thr[best])


# Usage
# proba = model.predict_proba(X_test)[:, 1]
# auc = roc_auc_score(y_test, proba)
# fpr, tpr, _ = roc_curve(y_test, proba)   # for plotting
# t = tune_threshold(y_test, proba, 0.8)
# preds = (proba >= t).astype(int)
# cm = confusion_matrix(y_test, preds)
```

**Explanation:** AUC summarizes ranking quality across all thresholds and is
threshold-independent, but deployment needs a single cutoff. The right threshold
depends on the **asymmetric cost** of the two error types (missing fraud vs
flagging a good customer). Tuning it on a validation set — never the test set —
often matters more than swapping model families.

</details>

---

## Navigation

- ⬆️ [Module 3 Exercises](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
