---
title: Eigenvalues and Eigenvectors
description: The directions a matrix only scales, eigendecomposition, and the link to PCA and SVD.
type: lesson
domain: 02-mathematics-foundations
tags: [mathematics, linear-algebra, eigenvalues, pca]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 45 min
prerequisites:
  - 01-python-languages/lessons/numpy
  - 02-mathematics-foundations/lessons/matrices
---

# Eigenvalues and Eigenvectors

> **TL;DR:** An eigenvector is a direction a matrix leaves pointing the same way, only stretching it by a factor called the eigenvalue. These special directions reveal a transformation's "axes" and are the mathematical backbone of PCA and dimensionality reduction.

---

## Overview

Most vectors get rotated *and* stretched by a matrix. A precious few are only stretched — their direction is preserved. Those are **eigenvectors**, and the stretch factors are **eigenvalues**. Finding them decomposes a complicated transformation into simple independent scalings along fixed axes. This is exactly what Principal Component Analysis exploits to find the directions of greatest variance in data, and it connects to the Singular Value Decomposition that underlies compression, recommendation, and embeddings.

**By the end, you will be able to:**
- State the eigenvalue equation $A\mathbf{v} = \lambda\mathbf{v}$ and explain it geometrically.
- Perform eigendecomposition and know why symmetric matrices are special.
- Connect eigenvectors of a covariance matrix to PCA, and relate this to SVD.

---

## Intuition

Apply a matrix to every arrow in the plane and watch what happens. Most arrows swing to a new direction. But along a couple of special lines, the arrows just get longer or shorter (or flip) without turning — they stay on their own line. Those lines are the eigenvectors' directions; how much each one stretches is its eigenvalue.

Think of a transformation as having natural "grain," like wood. Push along the grain and it stretches cleanly; push across it and things twist. Eigenvectors are the grain. Once you know them, a messy transformation becomes "scale by $\lambda_1$ along axis 1, by $\lambda_2$ along axis 2," which is far easier to reason about — and it tells you where a dataset varies most.

---

## Details

### Mathematics

For a square matrix $A \in \mathbb{R}^{n\times n}$, a nonzero vector $\mathbf{v} \in \mathbb{R}^n$ is an **eigenvector** with **eigenvalue** $\lambda \in \mathbb{R}$ (or $\mathbb{C}$) if applying $A$ only scales it:

$$
A\mathbf{v} = \lambda\mathbf{v}, \qquad \mathbf{v} \neq \mathbf{0}.
$$

Rearranging gives $(A - \lambda I)\mathbf{v} = \mathbf{0}$. A nonzero solution $\mathbf{v}$ exists only when $A - \lambda I$ is singular, i.e. when

$$
\det(A - \lambda I) = 0.
$$

This is the **characteristic equation**; its roots are the eigenvalues. (You will let NumPy solve it — the point here is that eigenvalues are exactly the $\lambda$ that make the transformation collapse a direction to zero net effect.)

**Eigendecomposition.** If $A$ has $n$ linearly independent eigenvectors, collect them as columns of $V$ and the eigenvalues on the diagonal of $\Lambda$:

$$
A = V \Lambda V^{-1}, \qquad
\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n).
$$

This says: change to the eigenvector basis ($V^{-1}$), scale each axis ($\Lambda$), change back ($V$).

**Symmetric matrices are special.** If $A = A^\top$ (symmetric, real), then:
- all eigenvalues are **real**, and
- eigenvectors for distinct eigenvalues are **orthogonal**, so $V$ can be chosen orthogonal ($V^{-1} = V^\top$).

This is the **spectral theorem**, giving $A = Q\Lambda Q^\top$ with $Q$ orthogonal. Covariance matrices are symmetric, which is why PCA behaves so cleanly.

**Connection to PCA.** Given mean-centered data $X \in \mathbb{R}^{m\times n}$ ($m$ samples, $n$ features), the **covariance matrix** is

$$
C = \frac{1}{m-1} X^\top X \in \mathbb{R}^{n\times n}.
$$

$C$ is symmetric. Its eigenvectors are the **principal components** — orthogonal directions of variance — and each eigenvalue is the variance captured along that direction. Keeping the top-$k$ eigenvectors (largest eigenvalues) gives the best $k$-dimensional linear projection of the data.

**Relation to SVD.** The **Singular Value Decomposition** factors *any* matrix as $X = U\Sigma V^\top$. The right singular vectors $V$ are the eigenvectors of $X^\top X$, and the squared singular values $\sigma_i^2$ are its eigenvalues. So PCA can be computed via SVD directly on $X$ — which is more numerically stable than forming $C$ explicitly.

### Python implementation

```python
import numpy as np

A: np.ndarray = np.array([[2.0, 0.0],
                          [0.0, 3.0]])

# General (possibly non-symmetric) matrices: np.linalg.eig
values, vectors = np.linalg.eig(A)
print(values)    # [2. 3.]  eigenvalues
print(vectors)   # columns are eigenvectors (here the axes)

# Verify the defining equation A v = lambda v for the first pair
v0 = vectors[:, 0]
print(np.allclose(A @ v0, values[0] * v0))  # True

# Symmetric matrices: use eigh (faster, returns real, sorted ascending)
S = np.array([[2.0, 1.0],
              [1.0, 2.0]])
vals, vecs = np.linalg.eigh(S)
print(vals)      # [1. 3.]  real eigenvalues
print(np.allclose(vecs.T @ vecs, np.eye(2)))  # True: eigenvectors orthonormal
```

## Worked Example

A minimal PCA: reduce 2-D correlated data to its single most informative direction.

```python
import numpy as np

rng = np.random.default_rng(0)
# Data stretched mostly along the (1, 1) direction
base = rng.normal(size=(200, 1))
X = base @ np.array([[3.0, 1.0]]) + 0.3 * rng.normal(size=(200, 2))

# 1. Center the data
Xc = X - X.mean(axis=0)

# 2. Covariance matrix (symmetric) and its eigendecomposition
C = (Xc.T @ Xc) / (Xc.shape[0] - 1)
vals, vecs = np.linalg.eigh(C)   # ascending order

# 3. Top principal component = eigenvector with largest eigenvalue
order = np.argsort(vals)[::-1]
pc1 = vecs[:, order[0]]
print("Variance explained by PC1:",
      round(vals[order[0]] / vals.sum(), 3))  # ~0.98

# 4. Project data onto PC1 (1-D representation)
X_1d = Xc @ pc1
print(X_1d.shape)  # (200,)
```

The top eigenvector points along the direction of maximum spread, and its eigenvalue tells you it captures ~98% of the variance — so one number per sample preserves almost all the information.

## Best Practices
- ✅ Use `np.linalg.eigh` for symmetric matrices (covariance, Gram, kernel matrices): it is faster and returns sorted real values.
- ✅ Center your data before computing a covariance matrix, or PCA directions will be wrong.
- ✅ For PCA on real data, prefer SVD (`np.linalg.svd`) on the centered data matrix over forming $X^\top X$ — it is more numerically stable.

## Common Mistakes
- ⚠️ Assuming eigenvectors are unique. They are defined only up to scale (and sign); NumPy returns unit-length ones with an arbitrary sign.
- ⚠️ Expecting real eigenvalues from a non-symmetric matrix. `np.linalg.eig` can return complex values — that is correct, not a bug.
- ⚠️ Reading eigenvalue order. `eig` does not sort; `eigh` sorts ascending. Sort explicitly before picking "top" components.

## Industry Tips
- 💡 PCA is a standard preprocessing and visualization step: shrink hundreds of features to 2–3 for plotting, or to a compact code for downstream models.
- 💡 Truncated SVD (the same math) compresses embedding and recommendation matrices and speeds up retrieval.

## Real-World Use Cases
- Dimensionality reduction and denoising via PCA before training a model.
- Latent-factor recommendation systems factor a ratings matrix with SVD.
- Spectral clustering and graph analysis use eigenvectors of a graph's matrices.

---

## Summary
- An eigenvector is a direction a matrix only scales; the scale factor is its eigenvalue: $A\mathbf{v} = \lambda\mathbf{v}$.
- Eigendecomposition $A = V\Lambda V^{-1}$ turns a transformation into independent scalings; symmetric matrices give real eigenvalues and orthogonal eigenvectors.
- The eigenvectors of a covariance matrix are PCA's principal components, and SVD provides the same directions more stably.

## Practice
- [ ] Exercises: [Module 2 Exercises](../exercises/README.md)
- [ ] Self-check: Why must the covariance matrix be symmetric, and what does that guarantee about its principal components?

## Further Reading
- 📘 Mathematics for Machine Learning — Deisenroth, Faisal & Ong (https://mml-book.github.io/)
- 📖 Deep Learning — Goodfellow, Bengio & Courville (https://www.deeplearningbook.org/)
- 📄 [NumPy linear algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- ▶️ 3Blue1Brown — Essence of Linear Algebra (https://www.youtube.com/@3blue1brown)

## Related
- [Matrices](matrices.md)
- [Machine Learning](../../03-machine-learning/README.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
