---
title: "Mini Project: Gradient Descent Visualizer"
description: Animate gradient descent on 2D loss surfaces to build intuition for learning rate and convergence.
type: project
domain: 02-mathematics-foundations
tags: [mathematics, gradient-descent, optimization, visualization]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 4–5 hours
prerequisites:
  - 02-mathematics-foundations/lessons/gradient-descent
  - 02-mathematics-foundations/lessons/calculus
tech_stack: [python, numpy, matplotlib]
---

# Mini Project: Gradient Descent Visualizer

> **What you'll build:** A tool that plots the path gradient descent takes across
> a 2D loss surface, so you can *see* how learning rate and starting point change
> the outcome.

---

## Objective

Gradient descent is easy to state and easy to misuse. Visualizing the optimizer's
trajectory on a contour plot makes learning rate, momentum, and non-convex traps
tangible — intuition you'll rely on when training real models.

## Learning Goals

- Compute gradients (analytic and numerical) and step downhill.
- See the effect of learning rate: convergence, oscillation, divergence.
- Contrast convex and non-convex surfaces.

---

## Prerequisites

- [Gradient Descent](../lessons/gradient-descent.md) and [Calculus](../lessons/calculus.md)
- NumPy + Matplotlib.

## Architecture

```mermaid
flowchart LR
  A[choose f(x,y) + grad] --> B[run GD, record path]
  B --> C[contour plot of f]
  C --> D[overlay descent path]
  D --> E[compare learning rates]
```

---

## Steps

### 1. Define surfaces
Implement a couple of loss surfaces with known gradients — a convex bowl
($f = x^2 + y^2$) and a non-convex one (e.g. Himmelblau or a saddle).

### 2. Implement the optimizer
Write `gradient_descent(f, grad, start, lr, steps)` that records every iterate.
Add a numerical-gradient check against your analytic gradient.

### 3. Visualize
Draw a contour plot of the surface and overlay the descent path as connected
points/arrows.

### 4. Experiment
Sweep learning rates (too small, good, too big) and starting points; capture the
qualitatively different behaviors. Optionally add momentum and compare.

### 5. Write up
Summarize what each learning-rate regime does and why non-convex surfaces make
initialization matter.

---

## Deliverables

- [ ] `gd_visualizer.py` with the optimizer and plotting.
- [ ] Figures for at least three learning rates on a convex surface.
- [ ] One non-convex example showing dependence on the starting point.
- [ ] `README.md` with the figures and a short analysis.

## Success Criteria

The visualizations clearly show convergence, oscillation, and divergence, and
your write-up correctly explains each in terms of the step size and gradient.

---

## Extensions (Optional)

- 🚀 Add momentum / Adam and compare trajectories.
- 🚀 Animate the descent as a GIF.

## Further Reading

- [Matplotlib contour docs](https://matplotlib.org/stable/)
- Deep Learning — Goodfellow, Bengio & Courville (https://www.deeplearningbook.org/)

---

## Navigation

- ⬆️ [Module 2 Mini Projects](README.md)
- 📚 [Module 2 — Mathematics for AI](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
