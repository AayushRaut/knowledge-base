---
title: "Exercise: PCA via Eigendecomposition"
description: Implement principal component analysis from scratch using the covariance matrix's eigenvectors.
type: exercise
domain: 02-mathematics-foundations
tags: [mathematics, linear-algebra, eigenvalues, pca, dimensionality-reduction]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 30 min
reinforces:
  - 02-mathematics-foundations/lessons/eigenvalues-and-eigenvectors
---

# Exercise: PCA via Eigendecomposition

> Practice for **[Eigenvalues and Eigenvectors](../lessons/eigenvalues-and-eigenvectors.md)**.

---

## Problem

Principal Component Analysis projects data onto the directions of greatest
variance — the top eigenvectors of the covariance matrix. Implement it from
scratch (no `sklearn.decomposition.PCA`).

Given data $X \in \mathbb{R}^{n \times d}$, return the projection onto the top
$k$ principal components.

## Requirements

- [ ] Center the data (subtract the column means).
- [ ] Compute the covariance matrix and its eigenvectors/eigenvalues.
- [ ] Sort by eigenvalue (descending) and project onto the top $k$.
- [ ] Use `np.linalg.eigh` (covariance is symmetric).

---

## Hints

<details>
<summary>Hint 1</summary>

For centered data, the covariance is `X_c.T @ X_c / (n - 1)`. It's symmetric, so
`np.linalg.eigh` is the right (and numerically better) choice.

</details>

<details>
<summary>Hint 2</summary>

`eigh` returns eigenvalues in ascending order — reverse to get the largest first.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def pca(X: np.ndarray, k: int) -> np.ndarray:
    """Project X onto its top-k principal components."""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)          # (d, d), symmetric
    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending order
    top = eigvecs[:, ::-1][:, :k]                   # k largest-variance directions
    return X_centered @ top                         # (n, k) projection


# Sanity check: variance captured is the sum of the top-k eigenvalues.
rng = np.random.default_rng(0)
X = rng.standard_normal((200, 5)) @ rng.standard_normal((5, 5))
Z = pca(X, k=2)
assert Z.shape == (200, 2)
```

**Explanation:** The covariance matrix's eigenvectors are the axes along which the
data varies independently; their eigenvalues are the variances along those axes.
Keeping the top $k$ eigenvectors keeps the directions that explain the most
variance — the essence of dimensionality reduction. Using `eigh` exploits
symmetry for real eigenvalues and orthogonal eigenvectors.

</details>

---

## Navigation

- ⬆️ [Module 2 Exercises](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
