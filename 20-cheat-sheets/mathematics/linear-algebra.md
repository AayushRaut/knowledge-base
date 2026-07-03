---
title: Linear Algebra Cheat Sheet
description: Fast reference for vectors, matrices, norms, eigenvalues, and PCA.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [mathematics, linear-algebra, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Linear Algebra Cheat Sheet

> Fast reference. For depth, see
> [Module 2 lessons](../../02-mathematics-foundations/lessons/README.md).

---

## Core Concepts

| Term | Definition |
|------|------------|
| Vector | An ordered list of numbers; a point/direction, or a data/feature row. |
| Norm | Length of a vector; $L_2 = \sqrt{\sum x_i^2}$, $L_1 = \sum |x_i|$. |
| Dot product | $\mathbf{a}\cdot\mathbf{b}=\sum a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$. |
| Matrix | A 2D array; data table **or** a linear transformation. |
| Eigenvector | Direction unchanged (only scaled) by a matrix: $A\mathbf{v}=\lambda\mathbf{v}$. |

---

## Key Formulas

$$
\text{cosine} = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}
\qquad
A\mathbf{v}=\lambda\mathbf{v}
\qquad
\mathbf{w}^\star = (X^\top X)^{-1}X^\top \mathbf{y}
$$

- **Matmul shape rule:** $(m\times k)(k\times n) \to (m\times n)$.
- **Symmetric matrix:** real eigenvalues, orthogonal eigenvectors.

---

## Common Commands / API

```python
import numpy as np

a @ b                     # dot / matrix product (NOT *)
np.linalg.norm(a)         # L2 norm
A.T                       # transpose
np.linalg.solve(A, b)     # solve Ax = b  (prefer over inv)
vals, vecs = np.linalg.eigh(S)   # symmetric eigendecomposition
U, s, Vt = np.linalg.svd(A)      # SVD
```

---

## Decision Guide

| If you need… | Use… | Notes |
|--------------|------|-------|
| Similarity of two vectors | cosine similarity | Normalize, then dot. |
| Solve a linear system | `np.linalg.solve` | Don't invert explicitly. |
| Reduce dimensions | PCA (`eigh` of covariance) | Keep top-k eigenvectors. |
| Factorize any matrix | SVD | More stable/general than eig. |

---

## Gotchas

- ⚠️ `*` is elementwise; `@` is matrix multiply.
- ⚠️ Inverting a matrix is slow and unstable — solve the system instead.
- ⚠️ `eig` may give complex values; use `eigh` for symmetric matrices.
- ⚠️ Check shapes: most bugs are shape/axis mismatches.

---

## Quick Links

- 📖 [Vectors](../../02-mathematics-foundations/lessons/vectors.md) · [Matrices](../../02-mathematics-foundations/lessons/matrices.md) · [Eigenvalues](../../02-mathematics-foundations/lessons/eigenvalues-and-eigenvectors.md)
- 🔗 [NumPy linalg](https://numpy.org/doc/stable/reference/routines.linalg.html)

---

## Navigation

- ⬆️ [Mathematics Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
