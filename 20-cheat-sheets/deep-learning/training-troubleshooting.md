---
title: Training Troubleshooting Cheat Sheet
description: Symptom-to-fix table for deep-learning training problems.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [deep-learning, debugging, training, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Training Troubleshooting Cheat Sheet

> Fast reference. For depth, see
> [Regularization and Training Techniques](../../04-deep-learning/lessons/regularization-and-training.md).

---

## Symptom → likely cause → fix

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Loss is `nan`/`inf` | LR too high; unscaled inputs; log(0) in loss | Lower LR; normalize inputs; use `*WithLogitsLoss` |
| Loss never decreases | LR too low; broken labels; missing nonlinearity; logits/loss mismatch | Sanity-check on a tiny batch; raise LR; inspect data |
| Train ≫ validation accuracy | Overfitting | Dropout, weight decay, augmentation, early stopping, more data |
| Train ≈ validation, both poor | Underfitting | Bigger model, train longer, better features/LR |
| Loss oscillates wildly | LR too high; batch too small | Lower LR; larger batch; LR schedule |
| Erratic, compounding updates | Missing `zero_grad()` | Add `optimizer.zero_grad()` per step |
| Great in training, bad in eval | `model.eval()` missing (dropout/BN active) | Toggle modes correctly |
| RNN gradients explode | Long sequences | `clip_grad_norm_` |
| Can't even overfit 10 samples | Bug in pipeline/model | Fix before anything else — this is the first test |

---

## Sanity checklist (run in order)

1. **Overfit a tiny batch** (10 samples → ~100% train accuracy). If not: bug.
2. **Check the loss at init** — multiclass CE should start ≈ $\ln(C)$ for $C$ classes.
3. **Inspect one batch** — shapes, dtypes, label range, input scale.
4. **Verify the loop** — `zero_grad` → forward → loss → `backward` → `step`; modes toggled.
5. **Then** tune LR (the single highest-leverage knob), schedule, regularization.

---

## Reasonable starting points

| Knob | Default |
|------|---------|
| Optimizer | AdamW |
| LR | `1e-3` (Adam) / `1e-2` (SGD+momentum) |
| Weight decay | `1e-2` (AdamW) |
| Dropout | 0.1–0.5 |
| Batch size | 32–256 (fit memory) |
| Grad clipping | max-norm 1.0 (RNNs/transformers) |

---

## Gotchas

- ⚠️ A model that *runs* is not a model that's *right* — most training bugs are silent.
- ⚠️ Augment training data only, never validation/test.
- ⚠️ Compare runs with fixed seeds before crediting a change.

---

## Quick Links

- 📖 [Optimizers](../../04-deep-learning/lessons/optimizers.md) · [Regularization & Training](../../04-deep-learning/lessons/regularization-and-training.md)
- 🧩 Debug practice: [broken loop](../../04-deep-learning/exercises/debug-training-loop.md) · [exploding loss](../../04-deep-learning/exercises/debug-exploding-loss.md)

---

## Navigation

- ⬆️ [Deep Learning Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
