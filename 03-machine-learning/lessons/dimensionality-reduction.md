---
title: Dimensionality Reduction
description: Reduce high-dimensional data with PCA for features and t-SNE/UMAP for visualization.
type: lesson
domain: 03-machine-learning
tags: [machine-learning, dimensionality-reduction, pca, tsne]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 40 min
prerequisites:
  - 03-machine-learning/lessons/clustering
---

# Dimensionality Reduction

> **TL;DR:** High-dimensional data is sparse and hard to model — the "curse of dimensionality." PCA compresses features by keeping the directions of maximum variance; t-SNE and UMAP are nonlinear tools for *visualization*, not feature engineering.

---

## Overview

Real datasets often have dozens or thousands of features, many redundant or noisy. Dimensionality reduction maps data to fewer dimensions while preserving as much useful structure as possible. This speeds up models, reduces overfitting and storage, and — critically — lets you *see* structure in two or three dimensions.

**By the end, you will be able to:**
- Explain the curse of dimensionality and why reducing dimensions helps.
- Apply PCA as a variance-maximizing linear projection and read its explained-variance ratio.
- Choose between PCA (features) and t-SNE/UMAP (visualization) appropriately.

---

## Intuition

Picture a pancake floating in 3D space. Every point has three coordinates, but the pancake is essentially flat — two coordinates describe it almost perfectly. **PCA** finds those two natural axes (the ones along which the data spreads out most) and drops the thin third one, losing almost nothing.

Now picture data that curls like a Swiss roll. A flat projection would smear distant parts on top of each other. **t-SNE** and **UMAP** instead try to keep *neighbors* close: points that were near each other in high dimensions stay near each other on the 2D map, even if the global geometry gets distorted. That makes them superb for eyeballing clusters — but the distances and axes on the resulting plot are not meaningful features to feed a model.

---

## Details

### Theory

**Curse of dimensionality.** As dimension $d$ grows, the volume of the space explodes, so a fixed number of points becomes ever sparser. Distances between points become more uniform (nearest and farthest neighbors converge), which weakens distance-based methods, and models need exponentially more data to generalize. Reducing $d$ counteracts this.

**PCA (Principal Component Analysis).** PCA finds an orthogonal set of directions — **principal components** — ordered so the first captures the most variance in the data, the second the most of the remaining variance, and so on. Center the data so each feature has zero mean, giving matrix $X \in \mathbb{R}^{n \times d}$ for $n$ samples. The **covariance matrix** is

$$
\Sigma = \frac{1}{n-1} X^{\top} X, \qquad \Sigma \in \mathbb{R}^{d \times d}.
$$

The principal components are the **eigenvectors** of $\Sigma$, and each eigenvalue $\lambda_j$ equals the variance captured along its eigenvector. Sort eigenvectors by descending $\lambda_j$; projecting $X$ onto the top $m$ of them gives an $m$-dimensional representation that retains maximum variance. (In practice scikit-learn computes this via the SVD of $X$ for numerical stability, which is mathematically equivalent.)

The **explained-variance ratio** of component $j$ is

$$
r_j = \frac{\lambda_j}{\sum_{k=1}^{d} \lambda_k},
$$

the fraction of total variance it captures. Summing $r_j$ over the kept components tells you how much information you retained — e.g. "10 components explain 95% of the variance." Because PCA depends on variance, and variance depends on scale, **standardize features first**.

> PCA is a direct application of eigenvectors and eigenvalues of the covariance matrix — see [Eigenvalues and Eigenvectors](../../02-mathematics-foundations/lessons/eigenvalues-and-eigenvectors.md).

**t-SNE (t-distributed Stochastic Neighbor Embedding).** A nonlinear method that models pairwise similarities as probabilities and places points in 2D/3D so those similarities are preserved, using a heavy-tailed (Student-t) distribution in the low-dimensional space to avoid crowding. It excels at revealing local cluster structure but: distances *between* clusters and cluster sizes on the plot are not reliable, results depend on the `perplexity` setting and random seed, and it is computationally expensive. Use it to *look*, not to build features.

**UMAP (Uniform Manifold Approximation and Projection).** A newer nonlinear technique (in the separate `umap-learn` package, not scikit-learn) that is typically faster than t-SNE and tends to preserve more of the global structure. Like t-SNE, it is primarily a visualization tool; treat its output coordinates as a picture, not a feature set.

**Feature selection vs. extraction.** *Selection* keeps a subset of the original features (e.g. drop low-variance or highly correlated ones) — the survivors stay interpretable. *Extraction* (like PCA) builds new features as combinations of the originals — more compact but less interpretable. Both reduce dimensionality; pick based on whether you need to keep original meanings.

### Python implementation

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)          # 1797 samples, 64 features (8x8 pixels)
X_scaled: np.ndarray = StandardScaler().fit_transform(X)

# --- PCA for feature compression ---
pca = PCA(n_components=0.95, random_state=42)  # keep enough PCs for 95% variance
X_pca = pca.fit_transform(X_scaled)
print("Reduced from", X.shape[1], "to", X_pca.shape[1], "dimensions")
print("Total variance retained:", round(pca.explained_variance_ratio_.sum(), 3))

# PCA to 2D just for plotting
X_2d_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

# --- t-SNE for visualization only (do NOT use as model features) ---
X_2d_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)

# X_2d_pca / X_2d_tsne now hold (x, y) coords you can scatter-plot, colored by y.
```

## Worked Example

You train a classifier on the 64-pixel digits dataset and it is slow and overfits.

1. **Standardize** the 64 features, then fit `PCA(n_components=0.95)`. It keeps about 40 components — a ~35% reduction — while retaining 95% of the variance.
2. Retrain the classifier on `X_pca`. It trains faster, and validation accuracy holds or improves because you dropped noisy, low-variance directions.
3. To *understand* the data, project to 2D with t-SNE and scatter-plot colored by digit. You see ten well-separated blobs — evidence the classes are genuinely distinguishable.
4. You resist the temptation to feed the t-SNE coordinates into the model: those 2D positions are a visualization, not stable features, and would not generalize to new data.

## Best Practices
- ✅ Standardize features before PCA (and fit the scaler on training data only).
- ✅ Pick the number of PCA components by a variance target (e.g. `n_components=0.95`) or a scree/cumulative-variance plot.
- ✅ Fit PCA on the training set and `transform` the test set — never fit on test data.
- ✅ Use t-SNE/UMAP only to visualize; use PCA (or the raw/selected features) to train.

## Common Mistakes
- ⚠️ Running PCA on unscaled data → high-variance-by-scale features dominate. Fix: standardize first.
- ⚠️ Interpreting t-SNE cluster distances/sizes as meaningful → they are not. Fix: read only local grouping.
- ⚠️ Feeding t-SNE/UMAP coordinates into a downstream model → they are non-deterministic and non-generalizing. Fix: use PCA for features.
- ⚠️ Assuming PCA components are interpretable features → each is a mix of all originals. Fix: use feature selection if interpretability matters.

## Industry Tips
- 💡 PCA doubles as denoising: reconstructing from top components filters out low-variance noise.
- 💡 t-SNE is $O(n^2)$-ish and slow; run PCA down to ~50 dims first, then t-SNE — this is common and recommended.
- 💡 UMAP (via `umap-learn`) is usually the faster, more scalable choice for large embedding visualizations.

## Real-World Use Cases
- Compressing image or sensor features before modeling.
- Visualizing word or sentence embeddings to inspect semantic clusters.
- Denoising and speeding up pipelines with many correlated features.
- Exploratory data analysis to spot structure and separability.

---

## Summary
- The curse of dimensionality makes high-dimensional data sparse and hard to model; reducing dimensions helps speed, storage, and generalization.
- PCA is a linear, variance-maximizing projection onto eigenvectors of the covariance matrix; the explained-variance ratio tells you how much you kept.
- t-SNE and UMAP are nonlinear visualization tools — great for seeing clusters, wrong for producing model features.

## Practice
- [ ] Exercises: [Module 3 Exercises](../exercises/README.md)
- [ ] Self-check: Why must you standardize features before PCA, and why should you never feed t-SNE coordinates into a predictive model?

## Further Reading
- 📘 Hands-On Machine Learning — Aurélien Géron
- 📘 An Introduction to Statistical Learning — James, Witten, Hastie & Tibshirani (https://www.statlearning.com/)
- 📄 [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
- ▶️ StatQuest (https://www.youtube.com/@statquest)

## Related
- [Clustering](clustering.md)
- [Eigenvalues and Eigenvectors](../../02-mathematics-foundations/lessons/eigenvalues-and-eigenvectors.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
