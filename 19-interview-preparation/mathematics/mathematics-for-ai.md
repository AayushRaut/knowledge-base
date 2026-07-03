---
title: "Mathematics for AI — Interview Questions"
description: A bank of 20 math interview questions with answers, focused on the linear algebra, probability, calculus, and information theory used in AI.
type: interview
domain: 19-interview-preparation
tags: [mathematics, interview, linear-algebra, probability, calculus]
category: ml-concepts
difficulty: medium
status: complete
created: 2026-07-02
updated: 2026-07-02
companies: []
---

# Mathematics for AI — Interview Questions

> **Category:** ML concepts / math · **Difficulty:** mixed (easy → hard)
>
> A question bank tied to **[Module 2 — Mathematics for AI](../../02-mathematics-foundations/README.md)**.
> Try to answer before reading the response.

---

## Contents

1. [Dot product meaning](#1-dot-product-meaning)
2. [Why cosine similarity?](#2-why-cosine-similarity)
3. [What is matrix rank?](#3-what-is-matrix-rank)
4. [Eigenvalues & eigenvectors](#4-eigenvalues--eigenvectors)
5. [How does PCA work?](#5-how-does-pca-work)
6. [Solve vs invert](#6-solve-vs-invert)
7. [Expectation vs variance](#7-expectation-vs-variance)
8. [Why divide by n−1?](#8-why-divide-by-n1)
9. [Central Limit Theorem](#9-central-limit-theorem)
10. [What a p-value is (and isn't)](#10-what-a-p-value-is-and-isnt)
11. [Bayes' rule & base rates](#11-bayes-rule--base-rates)
12. [Prior vs likelihood vs posterior](#12-prior-vs-likelihood-vs-posterior)
13. [What is a gradient?](#13-what-is-a-gradient)
14. [The chain rule & backprop](#14-the-chain-rule--backprop)
15. [How gradient descent works](#15-how-gradient-descent-works)
16. [Learning rate effects](#16-learning-rate-effects)
17. [Convex vs non-convex](#17-convex-vs-non-convex)
18. [Entropy & cross-entropy](#18-entropy--cross-entropy)
19. [KL divergence](#19-kl-divergence)
20. [Why cross-entropy loss?](#20-why-cross-entropy-loss)

---

### 1. Dot product meaning
**Q:** What does the dot product tell you geometrically?

**A:** $\mathbf{a}\cdot\mathbf{b}=\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$, so it measures how aligned two vectors are. Zero means orthogonal; positive means a similar direction; negative means opposing. It underlies projections and similarity.

### 2. Why cosine similarity?
**Q:** Why use cosine similarity instead of Euclidean distance for embeddings?

**A:** Cosine compares **direction**, ignoring magnitude, so vectors of different lengths but the same orientation score as similar — useful when embedding norm isn't meaningful. It's a normalized dot product in $[-1,1]$. See [Vectors](../../02-mathematics-foundations/lessons/vectors.md).

### 3. What is matrix rank?
**Q:** Define rank and why it matters.

**A:** Rank is the number of linearly independent rows/columns — the dimension of the space the matrix maps onto. Low rank means redundancy (used in compression, LoRA). A square matrix is invertible iff it's full rank.

### 4. Eigenvalues & eigenvectors
**Q:** What are they intuitively?

**A:** An eigenvector of $A$ is a direction that $A$ only scales, not rotates: $A\mathbf{v}=\lambda\mathbf{v}$; the eigenvalue $\lambda$ is the scaling factor. They reveal a transformation's invariant axes. See [Eigenvalues](../../02-mathematics-foundations/lessons/eigenvalues-and-eigenvectors.md).

### 5. How does PCA work?
**Q:** Explain PCA in terms of linear algebra.

**A:** PCA finds the eigenvectors of the data's covariance matrix; those are the orthogonal directions of maximum variance, and their eigenvalues are the variances. Projecting onto the top-$k$ eigenvectors reduces dimensionality while keeping the most variance.

### 6. Solve vs invert
**Q:** Why prefer `np.linalg.solve(A, b)` over `inv(A) @ b`?

**A:** Solving is faster and numerically more stable; explicitly forming the inverse amplifies floating-point error and is wasteful. Only invert when you genuinely need the inverse matrix itself.

### 7. Expectation vs variance
**Q:** Define both.

**A:** Expectation $\mathbb{E}[X]$ is the probability-weighted average (the center); variance $\mathrm{Var}(X)=\mathbb{E}[(X-\mu)^2]$ measures spread around it. Standard deviation is $\sqrt{\mathrm{Var}}$, in the same units as $X$.

### 8. Why divide by n−1?
**Q:** Why does sample variance use $n-1$?

**A:** Bessel's correction. The sample mean is estimated from the data, so squared deviations around it underestimate the true variance; dividing by $n-1$ removes that bias. In NumPy: `np.var(x, ddof=1)`. See [Statistics](../../02-mathematics-foundations/lessons/statistics.md).

### 9. Central Limit Theorem
**Q:** State the CLT and why it matters.

**A:** The distribution of sample means approaches a Normal as sample size grows, regardless of the source distribution, with standard error $\sigma/\sqrt{n}$. It's why Normal-based confidence intervals and tests work so broadly.

### 10. What a p-value is (and isn't)
**Q:** Define a p-value.

**A:** It's $P(\text{data at least this extreme}\mid \text{null hypothesis true})$. It is **not** the probability the null is true, nor the probability results are due to chance. Pair it with effect size.

### 11. Bayes' rule & base rates
**Q:** A 99%-accurate test for a 1-in-1000 disease comes back positive. Likely sick?

**A:** No — usually not. With so few true cases, false positives dominate; $P(D\mid+)\approx 2\%$. This base-rate effect is exactly what Bayes' rule quantifies. See [Bayes' Rule](../../02-mathematics-foundations/lessons/bayes-rule.md).

### 12. Prior vs likelihood vs posterior
**Q:** Distinguish them.

**A:** Prior $P(H)$ = belief before data; likelihood $P(D\mid H)$ = how well a hypothesis explains the data; posterior $P(H\mid D)$ = updated belief after data, $\propto$ prior × likelihood.

### 13. What is a gradient?
**Q:** What does $\nabla f$ represent?

**A:** The vector of partial derivatives; it points in the direction of steepest **increase**, and its magnitude is the steepness. Optimizers step in $-\nabla f$ to descend. See [Calculus](../../02-mathematics-foundations/lessons/calculus.md).

### 14. The chain rule & backprop
**Q:** How does the chain rule relate to backpropagation?

**A:** Backprop is the chain rule applied through a network: the gradient of the loss w.r.t. early parameters is the product of local derivatives along the path. It lets you compute all gradients in one backward pass.

### 15. How gradient descent works
**Q:** State the update rule and the idea.

**A:** $\theta \leftarrow \theta - \eta\,\nabla_\theta L$. Repeatedly step downhill along the negative gradient to reduce the loss. Variants (batch/SGD/mini-batch) differ in how much data each gradient uses. See [Gradient Descent](../../02-mathematics-foundations/lessons/gradient-descent.md).

### 16. Learning rate effects
**Q:** What happens if the learning rate is too high or too low?

**A:** Too high → overshoots, oscillates, or diverges; too low → very slow convergence and can get stuck. Good practice: scale features, tune $\eta$, or use adaptive optimizers like Adam.

### 17. Convex vs non-convex
**Q:** Why does convexity matter?

**A:** A convex loss has a single global minimum, so gradient descent is guaranteed to reach it. Neural-network losses are non-convex, so results depend on initialization and you settle for good local minima.

### 18. Entropy & cross-entropy
**Q:** Define both.

**A:** Entropy $H(p)=-\sum p\log p$ is the average surprise / uncertainty of a distribution. Cross-entropy $H(p,q)=-\sum p\log q$ is the average surprise of using model $q$ when reality is $p$. See [Information Theory](../../02-mathematics-foundations/lessons/information-theory.md).

### 19. KL divergence
**Q:** What is $D_{KL}(p\parallel q)$ and its key properties?

**A:** $D_{KL}(p\parallel q)=\sum p\log\frac{p}{q}$ measures how far $q$ is from $p$. It's $\ge 0$ (zero iff equal) and **asymmetric** ($D_{KL}(p\parallel q)\neq D_{KL}(q\parallel p)$), so it's not a true distance. Cross-entropy $=H(p)+D_{KL}(p\parallel q)$.

### 20. Why cross-entropy loss?
**Q:** Why is cross-entropy the default classification loss?

**A:** Minimizing cross-entropy between the one-hot labels and predicted probabilities is equivalent to maximum-likelihood estimation, and (since $H(p)$ is fixed) to minimizing the KL divergence from predictions to truth. It also gives well-scaled gradients with softmax.

---

## Common Mistakes

- ⚠️ Saying a p-value is "the probability the hypothesis is true."
- ⚠️ Forgetting the base rate when reasoning about test accuracy.
- ⚠️ Confusing `*` (elementwise) with `@` (matrix multiply).

## Key Takeaways

- Interviewers want intuition **and** the formula — give both.
- Tie each concept to where it appears in ML (embeddings, PCA, training, loss).

## Related

- [Module 2 — Mathematics for AI](../../02-mathematics-foundations/README.md)
- [Linear Algebra Cheat Sheet](../../20-cheat-sheets/mathematics/linear-algebra.md)

---

## Navigation

- ⬆️ [Mathematics Interview Questions](README.md)
- 📚 [Interview Preparation](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
