---
title: Vectors
description: Vectors as geometric arrows and as data rows, with dot products, norms, and cosine similarity for AI.
type: lesson
domain: 02-mathematics-foundations
tags: [mathematics, linear-algebra, vectors]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 35 min
prerequisites:
  - 01-python-languages/lessons/numpy
---

# Vectors

> **TL;DR:** A vector is an ordered list of numbers you can picture as an arrow. In AI it doubles as a row of features or an embedding, and the dot product powers everything from similarity search to neural network layers.

---

## Overview

Almost every object in modern AI — a data point, a word embedding, a set of model weights — is a vector. Understanding vectors geometrically (arrows with length and direction) and computationally (arrays of numbers) lets you reason about *distance*, *similarity*, and *direction*, which underpin retrieval, clustering, and learning. This lesson builds that dual intuition and grounds it in NumPy.

**By the end, you will be able to:**
- Read a vector both as a geometric arrow and as a feature/embedding row.
- Compute vector addition, scalar multiplication, dot products, and norms.
- Use cosine similarity to compare embeddings the way a retrieval system does.

---

## Intuition

Picture an arrow starting at the origin. Its **length** and **direction** are all that matter — where it points and how far. Adding two arrows means walking along the first, then the second (tip-to-tail). Scaling an arrow by a number stretches or flips it without changing the line it lies on.

Now switch lenses. That same arrow is just a list of coordinates: $[3, 4]$ is "3 right, 4 up." In AI you rarely stop at 2 or 3 numbers — a sentence embedding might be a 768-number arrow living in 768-dimensional space. You cannot draw it, but every geometric idea (length, angle, direction) still applies. When two embeddings point the *same way*, they mean similar things. That single insight drives semantic search.

---

## Details

### Mathematics

A **vector** $\mathbf{a} \in \mathbb{R}^n$ is an ordered tuple of $n$ real numbers, where $n$ is the **dimension** and each entry $a_i$ is a **component**:

$$
\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix}, \qquad a_i \in \mathbb{R}.
$$

**Addition** and **scalar multiplication** act component-wise, where $c \in \mathbb{R}$ is a scalar:

$$
\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ \vdots \\ a_n + b_n \end{bmatrix},
\qquad
c\,\mathbf{a} = \begin{bmatrix} c\,a_1 \\ \vdots \\ c\,a_n \end{bmatrix}.
$$

The **dot product** (inner product) of $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ is the sum of component products:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i.
$$

Geometrically it relates the two vectors' lengths and the angle $\theta$ between them:

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta.
$$

Here $\|\cdot\|$ is a **norm** (a length). The two you will use most are the **$L_2$ (Euclidean) norm** and the **$L_1$ (Manhattan) norm**:

$$
\|\mathbf{a}\|_2 = \sqrt{\sum_{i=1}^{n} a_i^2} = \sqrt{\mathbf{a}\cdot\mathbf{a}},
\qquad
\|\mathbf{a}\|_1 = \sum_{i=1}^{n} |a_i|.
$$

A **unit vector** has $L_2$ norm 1. **Normalizing** any nonzero $\mathbf{a}$ produces one that keeps its direction:

$$
\hat{\mathbf{a}} = \frac{\mathbf{a}}{\|\mathbf{a}\|_2}.
$$

**Cosine similarity** measures direction alone by dividing out both lengths; it lies in $[-1, 1]$:

$$
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2\,\|\mathbf{b}\|_2}.
$$

The **projection** of $\mathbf{a}$ onto $\mathbf{b}$ is the "shadow" of $\mathbf{a}$ along $\mathbf{b}$'s direction:

$$
\operatorname{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b}.
$$

When $\mathbf{a}\cdot\mathbf{b} = 0$ the vectors are **orthogonal** (perpendicular, $\theta = 90^\circ$).

### Python implementation

```python
import numpy as np

a: np.ndarray = np.array([3.0, 4.0])
b: np.ndarray = np.array([4.0, 3.0])

# Addition and scalar multiplication (component-wise)
print(a + b)          # [7. 7.]
print(2.0 * a)        # [6. 8.]

# Dot product (three equivalent spellings)
print(np.dot(a, b), a @ b, np.sum(a * b))  # 24.0 24.0 24.0

# Norms
print(np.linalg.norm(a))       # L2 = 5.0
print(np.linalg.norm(a, ord=1))  # L1 = 7.0

# Normalize to a unit vector (guard against divide-by-zero)
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

print(np.linalg.norm(normalize(a)))  # 1.0
```

## Worked Example

Suppose a tiny retrieval system stores document embeddings and you have a query embedding. You rank documents by cosine similarity — exactly what a vector database does under the hood.

```python
import numpy as np

def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom else 0.0

query: np.ndarray = np.array([0.9, 0.1, 0.0])
docs: dict[str, np.ndarray] = {
    "cats":    np.array([0.8, 0.2, 0.0]),
    "finance": np.array([0.0, 0.1, 0.99]),
    "kittens": np.array([0.85, 0.15, 0.05]),
}

ranked = sorted(docs, key=lambda k: cosine_similarity(query, docs[k]), reverse=True)
for name in ranked:
    print(f"{name:8s} {cosine_similarity(query, docs[name]):.3f}")
# kittens  ~0.998
# cats     ~0.992
# finance  ~0.109
```

Notice that cosine similarity ignores magnitude: a long "kittens" arrow and a short one rank the same as long as they point the same way. That is why embeddings are often normalized before storage — then a dot product *is* the cosine similarity.

## Best Practices
- ✅ Normalize embeddings once before storing them; then similarity reduces to a fast dot product.
- ✅ Use `np.linalg.norm` with an explicit `ord` argument rather than hand-rolling square-root sums.
- ✅ Keep dtypes as `float32`/`float64` for embeddings to avoid silent integer truncation.

## Common Mistakes
- ⚠️ Confusing the dot product (a scalar) with element-wise multiplication `a * b` (a vector). Use `@` or `np.dot` for the dot product.
- ⚠️ Comparing raw Euclidean distance when you meant *direction*; switch to cosine similarity for embeddings.
- ⚠️ Dividing by a zero norm when normalizing. Guard the denominator as shown above.

## Industry Tips
- 💡 Vector databases (FAISS, pgvector, Pinecone) index normalized vectors so a dot product recovers cosine similarity at scale.
- 💡 In neural networks, a single neuron's pre-activation is a dot product of inputs and weights — vectors are the atomic unit of the forward pass.

## Real-World Use Cases
- Semantic search and RAG retrieval rank passages by cosine similarity to a query embedding.
- Recommendation systems represent users and items as vectors and score matches with dot products.
- Word/sentence embeddings encode meaning as direction in high-dimensional space.

---

## Summary
- A vector is one object seen two ways: a geometric arrow (length + direction) and a numeric array (features/embedding).
- Addition and scaling are component-wise; the dot product ties algebra ($\sum a_i b_i$) to geometry ($\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$).
- Norms measure length; normalization keeps direction; cosine similarity compares direction and powers embedding retrieval.

## Practice
- [ ] Exercises: [Module 2 Exercises](../exercises/README.md)
- [ ] Self-check: Two embeddings have a large positive dot product but low cosine similarity. What does that tell you about their magnitudes versus their directions?

## Further Reading
- 📘 Mathematics for Machine Learning — Deisenroth, Faisal & Ong (https://mml-book.github.io/)
- 📄 [NumPy linear algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- ▶️ 3Blue1Brown — Essence of Linear Algebra (https://www.youtube.com/@3blue1brown)

## Related
- [Matrices](matrices.md)
- [NumPy for AI Engineering](../../01-python-languages/lessons/numpy.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
