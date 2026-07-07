---
title: "Machine Learning — Interview Questions"
description: A bank of 20 classical ML interview questions with answers.
type: interview
domain: 19-interview-preparation
tags: [machine-learning, interview, ml-concepts]
category: ml-concepts
difficulty: medium
status: complete
created: 2026-07-02
updated: 2026-07-02
companies: []
---

# Machine Learning — Interview Questions

> **Category:** ml-concepts · **Difficulty:** mixed (easy → hard)
>
> A question bank tied to **[Module 3 — Machine Learning](../../03-machine-learning/README.md)**.
> Try to answer before reading the response.

---

## Contents

1. [Supervised vs unsupervised](#1-supervised-vs-unsupervised)
2. [Bias–variance tradeoff](#2-biasvariance-tradeoff)
3. [Overfitting & how to prevent it](#3-overfitting--how-to-prevent-it)
4. [Why split train/validation/test?](#4-why-split-trainvalidationtest)
5. [What is cross-validation?](#5-what-is-cross-validation)
6. [Data leakage](#6-data-leakage)
7. [Precision vs recall](#7-precision-vs-recall)
8. [The accuracy paradox](#8-the-accuracy-paradox)
9. [ROC-AUC](#9-roc-auc)
10. [L1 vs L2 regularization](#10-l1-vs-l2-regularization)
11. [Logistic regression](#11-logistic-regression)
12. [How decision trees split](#12-how-decision-trees-split)
13. [Bagging vs boosting](#13-bagging-vs-boosting)
14. [Why random forests work](#14-why-random-forests-work)
15. [k-means and choosing k](#15-k-means-and-choosing-k)
16. [When to scale features](#16-when-to-scale-features)
17. [PCA](#17-pca)
18. [Handling class imbalance](#18-handling-class-imbalance)
19. [Generative vs discriminative](#19-generative-vs-discriminative)
20. [Curse of dimensionality](#20-curse-of-dimensionality)

---

### 1. Supervised vs unsupervised
**Q:** What's the difference?

**A:** Supervised learning trains on **labeled** examples to predict a target
(regression or classification). Unsupervised learning finds structure in
**unlabeled** data (clustering, dimensionality reduction). Semi-supervised mixes a
few labels with many unlabeled points; reinforcement learning learns from reward
signals via interaction. See [Learning Paradigms](../../03-machine-learning/lessons/learning-paradigms.md).

### 2. Bias–variance tradeoff
**Q:** Explain it.

**A:** **Bias** is error from overly simple assumptions (underfitting); **variance**
is error from sensitivity to the training sample (overfitting). Expected error ≈
bias² + variance + irreducible noise. Increasing model complexity lowers bias but
raises variance; the goal is the sweet spot, found via validation.

### 3. Overfitting & how to prevent it
**Q:** What is it and how do you fight it?

**A:** Overfitting is fitting noise in the training data so test performance drops.
Remedies: more data, simpler models, **regularization** ($L_1$/$L_2$), early
stopping, dropout (NNs), cross-validation for model selection, and pruning/limiting
tree depth. See [Bias, Variance, Overfitting & Underfitting](../../03-machine-learning/lessons/bias-variance-overfitting.md).

### 4. Why split train/validation/test?
**A:** **Train** fits parameters, **validation** tunes hyperparameters / selects
models, **test** gives a final unbiased estimate. If you tune on the test set, its
score is no longer an honest estimate of generalization.

### 5. What is cross-validation?
**A:** Rotating k-fold evaluation: split data into $k$ folds, train on $k-1$ and
validate on the held-out fold, repeat, average. It uses data efficiently and gives
a less noisy estimate than one split. Use **stratified** folds for classification.

### 6. Data leakage
**Q:** What is it and give an example.

**A:** Leakage is when information unavailable at prediction time influences
training, inflating scores. Classic example: fitting a scaler or feature selector
on the full dataset before CV, so validation statistics leak into training. Fix:
put all preprocessing inside a `Pipeline` fit per fold. See [Cross-Validation](../../03-machine-learning/lessons/cross-validation.md).

### 7. Precision vs recall
**A:** Precision $=TP/(TP+FP)$ — of predicted positives, how many are right.
Recall $=TP/(TP+FN)$ — of actual positives, how many you caught. Precision matters
when false positives are costly; recall when misses are costly. F1 balances them.

### 8. The accuracy paradox
**A:** On imbalanced data, a model that always predicts the majority class can have
very high accuracy while being useless (0 recall on the minority class). Use
precision/recall/F1, confusion matrix, and PR-AUC instead.

### 9. ROC-AUC
**A:** The ROC curve plots TPR vs FPR across thresholds; AUC is the area under it,
i.e. the probability a random positive is ranked above a random negative. It's
threshold-independent. For very imbalanced data, PR-AUC is often more informative.

### 10. L1 vs L2 regularization
**A:** Both penalize large weights. **L2 (Ridge)** shrinks weights smoothly and
handles correlated features; **L1 (Lasso)** drives some weights to exactly zero,
giving sparse, feature-selecting models. Elastic Net combines both.

### 11. Logistic regression
**Q:** Is it regression? How does it work?

**A:** It's a **classification** model. It applies the sigmoid to a linear
combination of features to output a probability, and is trained by minimizing
cross-entropy (log loss). The decision boundary is linear in feature space.

### 12. How decision trees split
**A:** They greedily choose the feature/threshold that most reduces impurity —
**Gini** or **entropy** for classification, variance for regression. They keep
splitting until a stopping criterion; unconstrained trees overfit, so limit depth
or leaf size.

### 13. Bagging vs boosting
**A:** **Bagging** trains many models in parallel on bootstrap samples and averages
them to reduce **variance** (e.g. random forests). **Boosting** trains models
sequentially, each correcting the previous one's errors, reducing **bias** (e.g.
gradient boosting). See [Ensemble Methods](../../03-machine-learning/lessons/ensemble-methods.md).

### 14. Why random forests work
**A:** They average many de-correlated trees (bootstrap samples + random feature
subsets). Individual trees are high-variance; averaging many diverse trees cuts
variance without adding much bias, and they need little tuning.

### 15. k-means and choosing k
**A:** k-means alternates assigning points to the nearest centroid and recomputing
centroids to minimize within-cluster sum of squares. Choose $k$ with the **elbow**
of inertia and the **silhouette** score. Scale features first; k-means assumes
roughly spherical, similar-size clusters.

### 16. When to scale features
**A:** For distance- and gradient-based methods — kNN, SVM, k-means, PCA, linear/
logistic regression with regularization. **Tree-based models don't need scaling.**
Always fit the scaler on training data only.

### 17. PCA
**A:** Principal Component Analysis projects data onto the orthogonal directions of
greatest variance — the top eigenvectors of the covariance matrix. It reduces
dimensionality while retaining variance; inspect the explained-variance ratio to
choose the number of components. See [Dimensionality Reduction](../../03-machine-learning/lessons/dimensionality-reduction.md).

### 18. Handling class imbalance
**A:** Options: resampling (oversample minority / SMOTE, undersample majority),
class weights (`class_weight="balanced"`), threshold tuning, and evaluating with
precision/recall/PR-AUC rather than accuracy. Choose based on the cost of each
error type.

### 19. Generative vs discriminative
**A:** Discriminative models learn $P(y\mid x)$ directly (logistic regression, SVM,
most classifiers). Generative models learn $P(x\mid y)$ and $P(y)$ then apply
Bayes' rule (Naive Bayes, GDA). Generative models can generate data and handle
missing features; discriminative often classify better with enough data.

### 20. Curse of dimensionality
**A:** As dimensions grow, data becomes sparse, distances concentrate (everything
looks equidistant), and models need exponentially more data. Mitigate with feature
selection, dimensionality reduction (PCA), and regularization.

---

## Common Mistakes

- ⚠️ Reporting accuracy on imbalanced data.
- ⚠️ Tuning hyperparameters on the test set.
- ⚠️ Forgetting to scale for distance/gradient models.
- ⚠️ Leaking preprocessing outside the CV loop.

## Key Takeaways

- Know the *why* and the trade-offs — interviewers push on them.
- Tie every metric choice to the cost of errors.

## Related

- [Module 3 — Machine Learning](../../03-machine-learning/README.md)
- [Model Evaluation Cheat Sheet](../../20-cheat-sheets/machine-learning/model-evaluation.md)

---

## Navigation

- ⬆️ [ML Concepts Interview Questions](README.md)
- 📚 [Interview Preparation](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
