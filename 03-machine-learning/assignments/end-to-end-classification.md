---
title: "Assignment: End-to-End Classification Project"
description: Build, tune, and honestly evaluate a full supervised classification pipeline on a real tabular dataset.
type: assignment
domain: 03-machine-learning
tags: [machine-learning, classification, pipeline, cross-validation, evaluation]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 4–6 hours
covers:
  - 03-machine-learning/lessons/scikit-learn-workflow
  - 03-machine-learning/lessons/classification
  - 03-machine-learning/lessons/ensemble-methods
  - 03-machine-learning/lessons/cross-validation
  - 03-machine-learning/lessons/model-evaluation-metrics
  - 03-machine-learning/lessons/bias-variance-overfitting
---

# Assignment: End-to-End Classification Project

> A larger, assessed task that integrates multiple concepts from
> **[Module 3](../README.md)**.

---

## Context

This is the module's centrepiece: take a real tabular dataset from raw data to a
tuned, honestly evaluated classifier — the workflow you'll repeat in every applied
ML role.

## Objectives

- Assemble a leak-free preprocessing + modeling pipeline.
- Select and tune models with cross-validation.
- Evaluate with metrics that match the problem, and diagnose bias/variance.

---

## Tasks

1. **EDA** — Load a labeled tabular dataset; explore distributions, missing
   values, and class balance. See [Data Cleaning](../../01-python-languages/lessons/data-cleaning-and-wrangling.md).
2. **Pipeline** — Build a `Pipeline` + `ColumnTransformer` for numeric/categorical
   preprocessing. See [The scikit-learn Workflow](../lessons/scikit-learn-workflow.md).
3. **Model selection** — Compare at least three models (e.g. logistic regression,
   random forest, gradient boosting) with **stratified cross-validation**, and tune
   the best with `GridSearchCV`. See [Cross-Validation](../lessons/cross-validation.md)
   and [Ensemble Methods](../lessons/ensemble-methods.md).
4. **Evaluation** — Report precision/recall/F1, ROC-AUC, and a confusion matrix on
   a held-out test set; justify the metric you optimize. See
   [Model Evaluation Metrics](../lessons/model-evaluation-metrics.md).
5. **Diagnose** — Use learning/validation curves to argue whether the model
   under- or over-fits, and what you'd do next. See
   [Bias, Variance, Overfitting & Underfitting](../lessons/bias-variance-overfitting.md).

## Constraints

- All preprocessing lives **inside** the pipeline/CV (no leakage).
- Split a test set once, up front, and touch it only for the final report.
- State which metric matters for this problem and why.

---

## Deliverables

- [ ] A notebook or script: EDA → pipeline → CV/tuning → final evaluation.
- [ ] A short report with the confusion matrix, key metrics, and a bias/variance diagnosis.
- [ ] Reproducible code (fixed random seeds, pinned deps).

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Correctness & no leakage | 35% | Pipeline is leak-free; CV and test usage are sound. |
| Model selection | 25% | Sensible comparison and tuning with justified choices. |
| Evaluation | 25% | Right metrics, correct interpretation, honest test report. |
| Communication | 15% | Clear report and reproducible code. |

---

## Submission

Push to a branch `add/module-3-classification` and open a pull request.

## Further Reading

- Hands-On Machine Learning — Aurélien Géron
- [scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)

---

## Navigation

- ⬆️ [Module 3 Assignments](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
