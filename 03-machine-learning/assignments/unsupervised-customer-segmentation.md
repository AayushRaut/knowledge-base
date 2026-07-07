---
title: "Assignment: Unsupervised Customer Segmentation"
description: Segment customers with clustering and dimensionality reduction, then interpret and validate the segments.
type: assignment
domain: 03-machine-learning
tags: [machine-learning, clustering, dimensionality-reduction, unsupervised]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 3–5 hours
covers:
  - 03-machine-learning/lessons/clustering
  - 03-machine-learning/lessons/dimensionality-reduction
  - 03-machine-learning/lessons/learning-paradigms
---

# Assignment: Unsupervised Customer Segmentation

> A larger, assessed task that integrates multiple concepts from
> **[Module 3](../README.md)**.

---

## Context

Without labels, you must find structure yourself. You will segment a customer
dataset using clustering, use dimensionality reduction to visualize and validate
the segments, and translate clusters into a business narrative.

## Objectives

- Cluster unlabeled data and choose the number of clusters defensibly.
- Use PCA/UMAP to visualize high-dimensional structure.
- Interpret clusters and state the limits of unsupervised findings.

---

## Tasks

1. **Prepare** — Load a customer/behavioral dataset; clean it and **scale**
   features (distance-based methods require it).
2. **Cluster** — Run k-means and one density/hierarchical method; choose $k$ with
   the elbow and silhouette. See [Clustering](../lessons/clustering.md).
3. **Reduce & visualize** — Use PCA (report explained variance) and a 2-D
   projection (PCA or t-SNE/UMAP) colored by cluster. See
   [Dimensionality Reduction](../lessons/dimensionality-reduction.md).
4. **Interpret** — Profile each segment (feature means, sizes) and give each a
   plain-language description and a possible action.
5. **Critique** — Discuss stability (do clusters change with seed/method?) and
   what unsupervised results can and cannot tell you.

## Constraints

- Scale features before clustering; justify the distance/method choice.
- Report cluster-quality metrics (e.g. silhouette), not just pretty plots.
- Be honest about ambiguity — there is no ground-truth label.

---

## Deliverables

- [ ] A notebook/script: preprocess → cluster → reduce → visualize → profile.
- [ ] A 2-D visualization of the segments and a metrics summary.
- [ ] A short report describing each segment and its reliability.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Method correctness | 35% | Proper scaling, sensible $k$ selection, valid metrics. |
| Interpretation | 30% | Clear, plausible segment profiles and actions. |
| Visualization | 20% | Effective, honest dimensionality-reduced plots. |
| Critique | 15% | Thoughtful discussion of stability and limitations. |

---

## Submission

Push to a branch `add/module-3-segmentation` and open a pull request.

## Further Reading

- [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html)
- An Introduction to Statistical Learning — James, Witten, Hastie & Tibshirani (https://www.statlearning.com/)

---

## Navigation

- ⬆️ [Module 3 Assignments](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
