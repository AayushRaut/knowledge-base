---
title: "Assignment: Linear Regression from Scratch"
description: Derive and implement linear regression two ways — the normal equation and gradient descent — using only NumPy.
type: assignment
domain: 02-mathematics-foundations
tags: [mathematics, linear-algebra, calculus, gradient-descent, regression]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 3–4 hours
covers:
  - 02-mathematics-foundations/lessons/vectors
  - 02-mathematics-foundations/lessons/matrices
  - 02-mathematics-foundations/lessons/calculus
  - 02-mathematics-foundations/lessons/gradient-descent
  - 02-mathematics-foundations/lessons/optimization
---

# Assignment: Linear Regression from Scratch

> A larger, assessed task that integrates multiple concepts from
> **[Module 2](../README.md)**.

---

## Context

Linear regression is where linear algebra, calculus, and optimization meet. You
will implement it **two independent ways** and show they agree: the closed-form
**normal equation** and iterative **gradient descent**. No `sklearn`.

## Objectives

- Connect the math (gradients of a loss) to working training code.
- Contrast a closed-form solution with an iterative optimizer.
- Reason about learning rate, scaling, and convergence.

---

## Tasks

1. **Set up the model** — For inputs $X \in \mathbb{R}^{n\times d}$ and weights
   $\mathbf{w}$, the prediction is $\hat{\mathbf{y}} = X\mathbf{w}$ (with a bias
   column). The loss is mean squared error
   $L(\mathbf{w}) = \frac{1}{n}\lVert X\mathbf{w} - \mathbf{y}\rVert_2^2$. See
   [Matrices](../lessons/matrices.md).
2. **Normal equation** — Derive and implement
   $\mathbf{w}^\star = (X^\top X)^{-1} X^\top \mathbf{y}$ (use `np.linalg.solve`,
   not an explicit inverse).
3. **Gradient descent** — Derive the gradient
   $\nabla_{\mathbf{w}} L = \frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y})$ and
   implement iterative training. See [Calculus](../lessons/calculus.md) and
   [Gradient Descent](../lessons/gradient-descent.md).
4. **Compare & analyze** — Show the two solutions agree on a synthetic dataset.
   Plot the loss curve; experiment with learning rate and feature scaling and
   write up what happens.

## Constraints

- NumPy only (Matplotlib allowed for plots).
- Include the gradient derivation in the README (math, not just code).
- Standardize features before gradient descent and explain why.

---

## Deliverables

- [ ] `linear_regression.py` with `fit_normal_equation` and `fit_gradient_descent`.
- [ ] A loss-vs-iteration plot and a short analysis of learning-rate behavior.
- [ ] Tests asserting the two methods agree within tolerance.
- [ ] `README.md` including the gradient derivation.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Correctness | 40% | Both methods fit correctly and agree; gradient is right. |
| Math clarity | 25% | Clear, correct derivation of loss gradient and normal equation. |
| Analysis | 20% | Thoughtful discussion of learning rate, scaling, convergence. |
| Code quality | 15% | Clean, typed, tested NumPy. |

---

## Submission

Push to a branch `add/module-2-linear-regression` and open a pull request.

## Further Reading

- Mathematics for Machine Learning — Deisenroth, Faisal & Ong (https://mml-book.github.io/)

---

## Navigation

- ⬆️ [Module 2 Assignments](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
