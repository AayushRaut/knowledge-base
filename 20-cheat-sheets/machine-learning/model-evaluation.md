---
title: Model Evaluation Cheat Sheet
description: Classification and regression metrics — formulas and when to use each.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [machine-learning, metrics, evaluation, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Model Evaluation Cheat Sheet

> Fast reference. For depth, see
> [Model Evaluation Metrics](../../03-machine-learning/lessons/model-evaluation-metrics.md).

---

## Classification

| Metric | Formula / meaning | Use when |
|--------|-------------------|----------|
| Accuracy | $(TP+TN)/\text{total}$ | Balanced classes only. |
| Precision | $TP/(TP+FP)$ | False positives are costly. |
| Recall (TPR) | $TP/(TP+FN)$ | Missing positives is costly. |
| F1 | harmonic mean of P & R | Imbalanced; want balance. |
| ROC-AUC | ranking quality over thresholds | Threshold-independent comparison. |
| PR-AUC | precision vs recall area | Highly imbalanced positives. |

**Confusion matrix:**

|  | Pred + | Pred − |
|--|--------|--------|
| **Actual +** | TP | FN |
| **Actual −** | FP | TN |

---

## Regression

| Metric | Meaning | Notes |
|--------|---------|-------|
| MAE | mean absolute error | Robust to outliers; same units as target. |
| MSE | mean squared error | Penalizes large errors. |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as target; outlier-sensitive. |
| $R^2$ | variance explained vs mean | Negative = worse than predicting the mean. |

---

## Decision Guide

| If… | Then | Notes |
|-----|------|-------|
| Classes imbalanced | precision/recall/F1, PR-AUC | **Not** accuracy. |
| Comparing rankers | ROC-AUC | Independent of threshold. |
| Errors have asymmetric cost | tune the threshold | Optimize on validation, not test. |
| Reporting to stakeholders | MAE / RMSE in real units | Interpretable. |

---

## Gotchas

- ⚠️ **Accuracy paradox:** 98% accuracy can mean 0% recall on the rare class.
- ⚠️ Pick the metric **before** modeling, from the business cost.
- ⚠️ Tune thresholds on validation data — never on the test set.
- ⚠️ Report variability (CV std), not a single split's score.

---

## Quick Links

- 📖 [Model Evaluation Metrics](../../03-machine-learning/lessons/model-evaluation-metrics.md) · [Cross-Validation](../../03-machine-learning/lessons/cross-validation.md)
- 🔗 [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## Navigation

- ⬆️ [Machine Learning Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
