---
title: "Assignment: CNN Image Classifier in PyTorch"
description: Train a regularized CNN on an image dataset with a correct training loop, augmentation, and honest evaluation.
type: assignment
domain: 04-deep-learning
tags: [deep-learning, cnn, pytorch, image-classification]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 4–6 hours
covers:
  - 04-deep-learning/lessons/cnn
  - 04-deep-learning/lessons/pytorch
  - 04-deep-learning/lessons/regularization-and-training
  - 04-deep-learning/lessons/optimizers
---

# Assignment: CNN Image Classifier in PyTorch

> A larger, assessed task that integrates multiple concepts from
> **[Module 4](../README.md)**.

---

## Context

This is the applied core of the module: design, train, regularize, and evaluate a
convolutional network on a standard image dataset (FashionMNIST or CIFAR-10),
using the canonical PyTorch workflow end to end.

## Objectives

- Design a small CNN with correct shape arithmetic.
- Run a complete, correct training/validation loop with early stopping.
- Use regularization and augmentation to close the generalization gap.

---

## Tasks

1. **Data** — Load FashionMNIST or CIFAR-10 via `torchvision`; build train/val/test
   `DataLoader`s; apply normalization and (train-only!) augmentation
   (`RandomCrop`, `RandomHorizontalFlip`). See
   [Regularization and Training](../lessons/regularization-and-training.md).
2. **Model** — Build a CNN (2–3 conv blocks → pool → dense head) as an
   `nn.Module`; document the shape after every layer. See [CNNs](../lessons/cnn.md).
3. **Train** — AdamW + a learning-rate schedule; the canonical loop with
   train/eval modes; track train and validation loss/accuracy per epoch; early-stop
   on validation. See [PyTorch Essentials](../lessons/pytorch.md) and
   [Optimizers](../lessons/optimizers.md).
4. **Evaluate** — Final test accuracy, a confusion matrix, and a handful of
   misclassified examples with your interpretation.
5. **Ablate** — Re-train once without augmentation *or* without dropout and
   compare the generalization gap; report the effect.

## Constraints

- Test set is touched exactly once, at the end.
- Augmentation applies to training data only.
- Fixed seeds; report the device used (CPU is fine — keep the model small).

---

## Deliverables

- [ ] Model + training script (or notebook) with per-layer shape comments.
- [ ] Training curves, final test accuracy, confusion matrix, misclassified gallery.
- [ ] The ablation comparison with a short analysis.
- [ ] `README.md` documenting architecture and results.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Correctness | 35% | Loop, modes, shapes, and evaluation are right; no test leakage. |
| Training quality | 25% | Sensible optimizer/schedule/early-stopping choices. |
| Analysis | 25% | Confusion matrix, misclassifications, and ablation interpreted well. |
| Code quality | 15% | Clean, reproducible, documented. |

---

## Submission

Push to a branch `add/module-4-cnn-classifier` and open a pull request.

## Further Reading

- [PyTorch tutorials](https://pytorch.org/tutorials/)
- Stanford CS231n notes (https://cs231n.github.io/)

---

## Navigation

- ⬆️ [Module 4 Assignments](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
