---
title: "Mini Project: Digit Clustering Explorer"
description: Cluster handwritten digits without labels and visualize the structure with PCA.
type: project
domain: 03-machine-learning
tags: [machine-learning, clustering, dimensionality-reduction, unsupervised]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 4–5 hours
prerequisites:
  - 03-machine-learning/lessons/clustering
  - 03-machine-learning/lessons/dimensionality-reduction
tech_stack: [python, scikit-learn, matplotlib]
---

# Mini Project: Digit Clustering Explorer

> **What you'll build:** An unsupervised pipeline that clusters the `digits`
> dataset and shows, via PCA, how well the discovered clusters line up with the
> true digits.

---

## Objective

Unsupervised learning finds structure without labels. Using the `digits` dataset
(which *has* labels you deliberately hide), you'll cluster the images and then use
the hidden labels only to *validate* how meaningful your clusters are.

## Learning Goals

- Apply k-means and evaluate clusters with silhouette and label-based scores.
- Use PCA to visualize high-dimensional data in 2-D.
- Understand what clustering can recover without supervision.

---

## Prerequisites

- [Clustering](../lessons/clustering.md), [Dimensionality Reduction](../lessons/dimensionality-reduction.md)
- `scikit-learn` (ships the `digits` dataset).

## Architecture

```mermaid
flowchart LR
  A[load digits] --> B[scale]
  B --> C[KMeans k=10]
  C --> D[silhouette + ARI vs true labels]
  B --> E[PCA to 2D]
  E --> F[scatter colored by cluster]
```

---

## Steps

### 1. Load & scale
Load `load_digits()`; scale the pixel features.

### 2. Cluster
Run k-means with `k=10`; also try DBSCAN and compare.

### 3. Validate
Hold out the true labels until now — compute silhouette (unsupervised) and
Adjusted Rand Index / homogeneity (vs true labels) to judge cluster quality.

### 4. Visualize
Project to 2-D with PCA and scatter, colored by predicted cluster and by true
digit side by side.

### 5. Write up
Discuss which digits get confused and why clustering can't perfectly recover labels.

---

## Deliverables

- [ ] Clustering script/notebook with silhouette + ARI scores.
- [ ] PCA 2-D visualizations (cluster vs true label).
- [ ] `README.md` interpreting the results and confusions.

## Success Criteria

Clusters achieve a clearly-better-than-random ARI, and your visualization and
write-up explain where and why clustering struggles.

---

## Extensions (Optional)

- 🚀 Replace PCA with t-SNE/UMAP for the visualization and compare.
- 🚀 Use cluster labels as features for a downstream classifier.

## Further Reading

- [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html)
- Cross-topic: [Eigenvalues and Eigenvectors](../../02-mathematics-foundations/lessons/eigenvalues-and-eigenvectors.md)

---

## Navigation

- ⬆️ [Module 3 Mini Projects](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
