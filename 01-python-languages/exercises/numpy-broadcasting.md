---
title: "Exercise: Vectorize a Distance Matrix"
description: Replace nested loops with broadcasted NumPy to compute pairwise distances.
type: exercise
domain: 01-python-languages
tags: [python, numpy, broadcasting, vectorization]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 01-python-languages/lessons/numpy
---

# Exercise: Vectorize a Distance Matrix

> Practice for **[NumPy for AI Engineering](../lessons/numpy.md)**.

---

## Problem

Given a matrix `X` of shape `(n, d)` (n vectors, e.g. embeddings, in d dimensions),
compute the `(n, n)` matrix of pairwise **squared Euclidean distances** without
Python loops. Start from this correct-but-slow reference:

```python
import numpy as np

def pairwise_sq_dists_loop(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            diff = X[i] - X[j]
            out[i, j] = diff @ diff
    return out
```

## Requirements

- [ ] Return an array equal to the loop version (within floating-point tolerance).
- [ ] Use broadcasting — no Python-level `for` loops.
- [ ] Explain the shapes involved.

---

## Hints

<details>
<summary>Hint 1</summary>

`X[:, None, :]` has shape `(n, 1, d)` and `X[None, :, :]` has shape `(1, n, d)`.
Subtracting them broadcasts to `(n, n, d)`.

</details>

<details>
<summary>Hint 2</summary>

Square the differences and sum over the last axis (`axis=-1`).

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every pair of rows in X."""
    # (n, 1, d) - (1, n, d) -> (n, n, d) via broadcasting
    diff = X[:, None, :] - X[None, :, :]
    # Sum of squares over the feature axis -> (n, n)
    return np.einsum("ijk,ijk->ij", diff, diff)


# Verify against the reference
X = np.random.default_rng(0).standard_normal((50, 8))
assert np.allclose(pairwise_sq_dists(X), pairwise_sq_dists_loop(X))
```

**Explanation:** Broadcasting expands the two reshaped arrays to a common
`(n, n, d)` grid of differences without copying data until the subtraction.
`np.einsum("ijk,ijk->ij", diff, diff)` squares and sums over `k` (the feature
axis) in one pass. This is the pattern behind similarity search over embeddings —
and it runs in optimized C instead of a Python double loop.

</details>

---

## Navigation

- ⬆️ [Module 1 Exercises](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
