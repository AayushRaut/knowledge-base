---
title: "Exercise: Cosine Similarity from Scratch"
description: Implement L2 normalization and cosine similarity with NumPy, then batch it.
type: exercise
domain: 02-mathematics-foundations
tags: [mathematics, linear-algebra, vectors, embeddings]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 02-mathematics-foundations/lessons/vectors
---

# Exercise: Cosine Similarity from Scratch

> Practice for **[Vectors](../lessons/vectors.md)**.

---

## Problem

Cosine similarity measures the angle between two vectors and is the workhorse of
embedding search. For vectors $\mathbf{a}, \mathbf{b}$:

$$
\text{cos\_sim}(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|_2\,\|\mathbf{b}\|_2}
$$

Implement it in NumPy without using a library's built-in cosine helper.

## Requirements

- [ ] `cosine_similarity(a, b)` returns a float in $[-1, 1]$.
- [ ] Handle the zero-vector case without dividing by zero (return `0.0`).
- [ ] Bonus: `cosine_similarity_matrix(X)` returns the `(n, n)` pairwise matrix for rows of `X`.

---

## Hints

<details>
<summary>Hint 1</summary>

The dot product is `a @ b`; the L2 norm is `np.linalg.norm(a)`.

</details>

<details>
<summary>Hint 2</summary>

For the matrix version, L2-normalize each row first, then a single `X_norm @ X_norm.T` gives all pairwise cosines.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:  # a or b is the zero vector — undefined angle
        return 0.0
    return float((a @ b) / denom)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of X, shape (n, n)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0            # avoid division by zero
    X_norm = X / norms
    return X_norm @ X_norm.T
```

**Explanation:** Normalizing each vector to unit length turns the dot product
into the cosine of the angle between them. Doing it once per row and multiplying
`X_norm @ X_norm.T` computes every pairwise similarity in one vectorized matrix
product — exactly how a brute-force vector search scores candidates.

</details>

---

## Navigation

- ⬆️ [Module 2 Exercises](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
