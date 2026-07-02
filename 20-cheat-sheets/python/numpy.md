---
title: NumPy Cheat Sheet
description: Fast reference for NumPy arrays, dtypes, broadcasting, and vectorized operations.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [python, numpy, arrays, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# NumPy Cheat Sheet

> Fast reference for NumPy. For depth, see
> [NumPy for AI Engineering](../../01-python-languages/lessons/numpy.md).

---

## Core Concepts

| Term | Definition |
|------|------------|
| `ndarray` | N-dimensional, homogeneous, fixed-dtype array. |
| `dtype` | Element type (e.g. `float32`, `int64`); controls memory & precision. |
| Broadcasting | Rules for operating on arrays of different but compatible shapes. |
| View vs copy | Slices are views (share memory); fancy indexing copies. |
| Vectorization | Expressing loops as whole-array ops that run in C. |

---

## Create & inspect

```python
import numpy as np

a = np.array([1, 2, 3], dtype=np.float32)
z = np.zeros((2, 3)); o = np.ones(4); e = np.eye(3)
r = np.arange(0, 10, 2); l = np.linspace(0, 1, 5)
rng = np.random.default_rng(0); s = rng.standard_normal((3, 4))

a.shape, a.dtype, a.ndim, a.size      # metadata
```

## Reshape & index

```python
x = np.arange(12).reshape(3, 4)
x[0], x[:, 1], x[1:, ::2]             # rows, column, strided
x[x > 5]                              # boolean mask (copy)
x.T, x.reshape(-1), x.ravel()         # transpose / flatten
```

## Broadcasting & reductions

```python
x + 1                                 # scalar broadcasts
x - x.mean(axis=0)                    # (3,4) - (4,) -> center columns
x.sum(axis=1); x.max(axis=0)          # axis reductions
A @ B; np.linalg.inv(M); np.linalg.norm(v)   # linear algebra
```

---

## Decision Guide

| If you need… | Use… | Notes |
|--------------|------|-------|
| Elementwise math over data | vectorized ops | Avoid Python `for` loops. |
| Combine different shapes | broadcasting | Align trailing dims; size 1 stretches. |
| Reproducible randomness | `np.random.default_rng(seed)` | Preferred over legacy `np.random.*`. |
| Reduce memory | smaller `dtype` (`float32`) | Common for ML feature matrices. |
| Matrix product | `@` / `np.matmul` | Not `*`, which is elementwise. |

---

## Gotchas

- ⚠️ Slices are **views** — writing to a slice mutates the original; use `.copy()`.
- ⚠️ `*` is elementwise, `@` is matrix multiply.
- ⚠️ Integer arrays don't hold NaN — use a float dtype for missing values.
- ⚠️ Broadcasting can silently produce a huge array if shapes are wrong — check `.shape`.

---

## Quick Links

- 📖 [NumPy for AI Engineering](../../01-python-languages/lessons/numpy.md)
- 📖 [SciPy Essentials](../../01-python-languages/lessons/scipy.md)
- 🔗 [NumPy documentation](https://numpy.org/doc/stable/)

---

## Navigation

- ⬆️ [Python Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
