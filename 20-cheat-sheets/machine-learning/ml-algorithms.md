---
title: ML Algorithms Cheat Sheet
description: When to reach for which classical ML algorithm, and the knobs that matter.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [machine-learning, algorithms, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# ML Algorithms Cheat Sheet

> Fast reference. For depth, see
> [Module 3 lessons](../../03-machine-learning/lessons/README.md).

---

## Pick an algorithm

| Task | Start with | Then try | Notes |
|------|-----------|----------|-------|
| Regression | Linear / Ridge | Random Forest, Gradient Boosting | Scale for linear; trees are scale-free. |
| Classification | Logistic Regression | Random Forest, Gradient Boosting, SVM | Strong, interpretable baseline. |
| Many features, few samples | Linear + $L_1$ (Lasso) | Linear SVM | Regularize hard; Lasso selects features. |
| Nonlinear tabular | Gradient Boosting (XGBoost/LightGBM) | Random Forest | Usually the top tabular performers. |
| Clustering | k-means | DBSCAN, Agglomerative | Scale first; DBSCAN for arbitrary shapes/noise. |
| Dimensionality reduction | PCA | t-SNE / UMAP (viz only) | PCA for features; t-SNE/UMAP for plots. |

---

## Key hyperparameters

| Algorithm | Watch these |
|-----------|-------------|
| Ridge / Lasso | `alpha` (regularization strength) |
| kNN | `n_neighbors`, distance metric, **scaling** |
| Decision Tree | `max_depth`, `min_samples_leaf` (control overfitting) |
| Random Forest | `n_estimators`, `max_features`, `max_depth` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` (trade-off!) |
| SVM | `C`, `kernel`, `gamma` |
| k-means | `n_clusters`, `n_init` |

---

## Decision Guide

| If you need… | Use… | Notes |
|--------------|------|-------|
| Interpretability | Linear / trees | Coefficients or feature importance. |
| Best tabular accuracy | Gradient boosting | Tune `learning_rate` × `n_estimators`. |
| Probabilities | Logistic regression, calibrated models | Check calibration if used for decisions. |
| Robust to feature scale | Tree-based | No scaling needed. |

---

## Gotchas

- ⚠️ Distance/gradient models (kNN, SVM, linear, k-means) need **feature scaling**.
- ⚠️ More trees rarely overfit RF; more boosting rounds **can** — use early stopping.
- ⚠️ Trees extrapolate poorly outside the training range.
- ⚠️ Always fit preprocessing inside a `Pipeline` to avoid leakage.

---

## Quick Links

- 📖 [Regression](../../03-machine-learning/lessons/regression.md) · [Classification](../../03-machine-learning/lessons/classification.md) · [Ensembles](../../03-machine-learning/lessons/ensemble-methods.md)
- 🔗 [scikit-learn supervised learning](https://scikit-learn.org/stable/supervised_learning.html)

---

## Navigation

- ⬆️ [Machine Learning Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
