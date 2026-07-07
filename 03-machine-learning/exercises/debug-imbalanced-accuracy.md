---
title: "Debugging: Fix the Misleading Accuracy"
description: Diagnose the accuracy paradox on imbalanced data and choose an honest metric.
type: exercise
domain: 03-machine-learning
tags: [machine-learning, debugging, metrics, imbalanced-data]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 03-machine-learning/lessons/model-evaluation-metrics
---

# Debugging: Fix the Misleading Accuracy

> Practice for **[Model Evaluation Metrics](../lessons/model-evaluation-metrics.md)**.

---

## Problem

A fraud model reports **98% accuracy** and everyone is thrilled — but it never
catches fraud. Explain what's wrong and fix the *evaluation* (not necessarily the
model).

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# 98% of transactions are legitimate (class 0), 2% fraud (class 1).
# This "model" predicts "not fraud" for everything.
clf = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("accuracy:", accuracy_score(y_test, clf.predict(X_test)))  # ~0.98 (!)
```

## Requirements

- [ ] Explain why 98% accuracy is meaningless here.
- [ ] Report metrics that expose the failure.
- [ ] Suggest how to evaluate (and later fix) such a model.

---

## Hints

<details>
<summary>Hint 1</summary>

With 98% of one class, always predicting that class scores 98% accuracy while
catching zero positives.

</details>

<details>
<summary>Hint 2</summary>

Look at recall for the positive class, the confusion matrix, and PR-AUC.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
from sklearn.metrics import classification_report, confusion_matrix

preds = clf.predict(X_test)
print(confusion_matrix(y_test, preds))
# All fraud cases fall in the "predicted not-fraud" column: recall(fraud) = 0.
print(classification_report(y_test, preds, digits=3))
```

**Explanation:** On imbalanced data, **accuracy is dominated by the majority
class** — the "accuracy paradox." The always-negative classifier has recall 0 on
the class you actually care about. Evaluate with **precision, recall, F1 on the
positive class**, the **confusion matrix**, and **PR-AUC**; choose the metric from
the business cost of a missed fraud. To improve the model itself: resampling
(SMOTE/undersampling), class weights (`class_weight="balanced"`), or threshold
tuning — but first, measure the right thing.

</details>

---

## Navigation

- ⬆️ [Module 3 Exercises](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
