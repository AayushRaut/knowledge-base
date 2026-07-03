---
title: "Mini Project: PCA Image Compression"
description: Use PCA (eigendecomposition) to compress grayscale images and visualize explained variance.
type: project
domain: 02-mathematics-foundations
tags: [mathematics, linear-algebra, eigenvalues, pca, compression]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 4–5 hours
prerequisites:
  - 02-mathematics-foundations/lessons/eigenvalues-and-eigenvectors
  - 02-mathematics-foundations/lessons/matrices
tech_stack: [python, numpy, matplotlib]
---

# Mini Project: PCA Image Compression

> **What you'll build:** An image compressor that keeps only the top principal
> components of an image and reconstructs it — making "explained variance" you can
> literally see.

---

## Objective

PCA is abstract until you watch an image reappear from a handful of components.
You'll implement PCA from scratch and use it to compress and reconstruct
grayscale images, plotting how reconstruction quality tracks explained variance.

## Learning Goals

- Apply eigendecomposition of a covariance matrix in practice.
- Understand explained variance and the rank–quality trade-off.
- Connect linear algebra to a tangible outcome.

---

## Prerequisites

- [Eigenvalues and Eigenvectors](../lessons/eigenvalues-and-eigenvectors.md), [Matrices](../lessons/matrices.md)
- NumPy + Matplotlib; a grayscale image to test with.

## Architecture

```mermaid
flowchart LR
  A[grayscale image matrix] --> B[center rows]
  B --> C[covariance + eigh]
  C --> D[keep top-k components]
  D --> E[reconstruct]
  E --> F[compare quality vs k]
```

---

## Steps

### 1. Load the image
Read a grayscale image into a 2D NumPy array (treat rows as samples, columns as features).

### 2. Implement PCA
Center the data, compute the covariance matrix, and get eigenvectors via
`np.linalg.eigh`. Sort by eigenvalue descending.

### 3. Compress & reconstruct
Project onto the top $k$ components and reconstruct. Do this for several $k$.

### 4. Explained variance
Plot cumulative explained variance vs $k$, and show reconstructed images side by
side to relate the curve to visual quality.

### 5. Write up
Report the compression ratio and where quality "saturates."

---

## Deliverables

- [ ] `pca_compress.py` implementing PCA and reconstruction.
- [ ] A grid of reconstructions for increasing $k$.
- [ ] An explained-variance plot.
- [ ] `README.md` with results and the quality/compression discussion.

## Success Criteria

Reconstructions visibly improve with more components, and your explained-variance
plot correctly matches the observed quality.

---

## Extensions (Optional)

- 🚀 Compare your eigendecomposition PCA against an SVD-based implementation.
- 🚀 Extend to color images (per-channel).

## Further Reading

- Mathematics for Machine Learning — Deisenroth, Faisal & Ong (https://mml-book.github.io/)
- Related domain: [Machine Learning](../../03-machine-learning/README.md)

---

## Navigation

- ⬆️ [Module 2 Mini Projects](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
