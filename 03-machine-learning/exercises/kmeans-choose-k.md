---
title: "Exercise: Choose k for k-means"
description: Use the elbow method and silhouette score to select the number of clusters.
type: exercise
domain: 03-machine-learning
tags: [machine-learning, clustering, kmeans, model-selection]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 03-machine-learning/lessons/clustering
---

# Exercise: Choose k for k-means

> Practice for **[Clustering](../lessons/clustering.md)**.

---

## Problem

k-means needs you to pick the number of clusters $k$. On a dataset with unknown
structure, use two standard heuristics — the **elbow** (inertia vs $k$) and the
**silhouette score** — to choose $k$, and reconcile them.

## Requirements

- [ ] Standardize features before clustering.
- [ ] Compute inertia and silhouette for $k = 2 \ldots 10$.
- [ ] Pick a $k$ and justify it from both curves.

---

## Hints

<details>
<summary>Hint 1</summary>

`KMeans(n_clusters=k, n_init="auto").fit(X)` exposes `.inertia_`; silhouette comes
from `silhouette_score(X, labels)`.

</details>

<details>
<summary>Hint 2</summary>

Inertia always decreases with $k$ — look for the "elbow". Silhouette has a peak;
prefer the $k$ where both roughly agree.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def scan_k(X: np.ndarray, k_range=range(2, 11)) -> dict[int, tuple[float, float]]:
    X_scaled = StandardScaler().fit_transform(X)
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X_scaled)
        results[k] = (km.inertia_, silhouette_score(X_scaled, km.labels_))
    return results


# results[k] = (inertia, silhouette); pick the k where silhouette peaks
# and inertia's marginal gain flattens (the elbow).
```

**Explanation:** Inertia (within-cluster sum of squares) falls monotonically, so
it can't pick $k$ alone — you look for the elbow where extra clusters stop helping
much. The silhouette score (how well points sit in their cluster vs the next)
peaks at a good $k$. Standardizing first is essential: k-means uses Euclidean
distance, so unscaled features dominate the clustering.

</details>

---

## Navigation

- ⬆️ [Module 3 Exercises](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
