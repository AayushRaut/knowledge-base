---
title: Optimizers
description: From SGD to momentum, RMSProp, Adam, and AdamW — how gradient updates are shaped, scaled, and scheduled in modern training.
type: lesson
domain: 04-deep-learning
tags: [deep-learning, optimizers, sgd, adam]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 50 min
prerequisites:
  - 04-deep-learning/lessons/backpropagation
  - 02-mathematics-foundations/lessons/gradient-descent
---

# Optimizers

> **TL;DR:** Optimizers decide how to turn gradients into parameter updates. Momentum smooths the direction, RMSProp adapts the step size per parameter, Adam combines both with bias correction, and AdamW fixes weight decay — with learning-rate schedules controlling the step size over time.

---

## Overview

Backpropagation gives you gradients; an optimizer decides what to do with them. The choice of optimizer and learning-rate schedule often matters more for final performance than architecture tweaks. This lesson walks the standard lineage — SGD, momentum, RMSProp, Adam, AdamW — with the exact update rules and their PyTorch usage.

**By the end, you will be able to:**
- Write the update equations for SGD, momentum, RMSProp, Adam, and AdamW, and explain what each term buys you
- Choose a sensible optimizer and learning-rate schedule for a given task
- Wire any `torch.optim` optimizer and scheduler into a correct training loop

---

## Intuition

Picture the loss surface as a hilly landscape and the parameters as a ball you nudge downhill.

- **Vanilla SGD** takes a step directly along the (noisy, mini-batch) downhill direction. In a narrow ravine it zig-zags: steep walls dominate the gradient while progress along the valley floor is slow.
- **Momentum** gives the ball mass. Velocity accumulates in directions where gradients consistently agree (along the valley) and cancels where they oscillate (across it). The result: smoother, faster travel.
- **RMSProp** gives each parameter its own step size. Parameters with persistently large gradients get their steps shrunk; parameters with tiny gradients get relatively bigger steps. Useful when gradient scales vary wildly across layers.
- **Adam** combines both ideas: a momentum-like average of gradients *and* a per-parameter scale, plus a correction for the startup bias of both averages.
- **AdamW** additionally fixes how weight decay interacts with the adaptive scaling — decay is applied directly to weights, not mixed into the gradient.

A **learning-rate schedule** is the meta-control: start carefully (warmup), cruise, then decay so the ball settles into a minimum instead of rattling around it.

---

## Details

### Mathematics

Let $\theta_t$ denote the parameters at step $t$, $g_t = \nabla_\theta L(\theta_t)$ the mini-batch gradient of the loss $L$, and $\eta$ the learning rate.

**Mini-batch SGD.** Instead of the full dataset (slow) or a single sample (noisy), estimate the gradient on a batch of $B$ examples:

$$
\theta_{t+1} = \theta_t - \eta \, g_t
$$

The batch gradient is an unbiased estimate of the full gradient; batching trades noise for parallel hardware efficiency.

**Momentum.** Maintain a velocity $v_t$ that exponentially averages past gradients, with momentum coefficient $\beta \in [0, 1)$ (typically $0.9$):

$$
v_t = \beta \, v_{t-1} + g_t, \qquad
\theta_{t+1} = \theta_t - \eta \, v_t
$$

Consistent gradient directions compound (up to a factor $\tfrac{1}{1-\beta}$); oscillating components cancel.

**RMSProp.** Track an exponential moving average of *squared* gradients (elementwise), with decay $\rho$ (typically $0.99$) and a small $\epsilon$ (e.g. $10^{-8}$) for numerical stability:

$$
s_t = \rho \, s_{t-1} + (1 - \rho) \, g_t^2, \qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \, g_t
$$

Dividing by $\sqrt{s_t}$ normalizes each parameter's step by its typical gradient magnitude.

**Adam.** Combine a first-moment estimate $m_t$ (mean of gradients, coefficient $\beta_1$, default $0.9$) and a second-moment estimate $v_t$ (mean of squared gradients, coefficient $\beta_2$, default $0.999$):

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \, g_t, \qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2) \, g_t^2
$$

Both start at zero, so early estimates are biased toward zero. **Bias correction** divides by the missing mass:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad
\theta_{t+1} = \theta_t - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**AdamW — decoupled weight decay.** Classic "$L_2$ regularization" adds $\lambda \theta$ to the gradient, where $\lambda$ is the decay strength — but in Adam that term then gets rescaled by $\sqrt{\hat{v}_t}$, so effective decay varies per parameter. AdamW instead decays weights *directly*, outside the adaptive machinery:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \, \theta_t \right)
$$

This decoupling is why AdamW is the default in most transformer training recipes.

**Learning-rate schedules (conceptual).**

- *Step decay:* multiply $\eta$ by a factor (e.g. $0.1$) at fixed milestones.
- *Cosine annealing:* decay $\eta$ smoothly from its peak to near zero following a half cosine — no milestone tuning.
- *Warmup:* ramp $\eta$ linearly from ~0 over the first steps, giving Adam's moment estimates time to stabilize before large updates. Warmup + cosine decay is the common recipe for training transformers.

### Python implementation

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
loss_fn = nn.MSELoss()

# Pick one — identical interface:
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# Cosine schedule over 100 epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def train_epoch(loader) -> float:
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()          # clear accumulated grads
        loss = loss_fn(model(xb), yb)  # forward
        loss.backward()                # backward: fill .grad
        optimizer.step()               # apply the update rule
        total += loss.item() * len(xb)
    return total / len(loader.dataset)

for epoch in range(100):
    train_epoch(train_loader)
    scheduler.step()                   # once per epoch for this scheduler
```

The optimizer never touches your model's code — it only reads `.grad` and updates the parameters it was given. Warmup is available via `torch.optim.lr_scheduler.LinearLR` chained with `SequentialLR`, or in libraries built on top of PyTorch.

## Worked Example

Watch Adam's machinery on a single parameter. Suppose the first two gradients are $g_1 = 1.0$ and $g_2 = 1.0$, with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.001$, $\epsilon = 10^{-8}$.

**Step 1:**
- $m_1 = 0.9 \cdot 0 + 0.1 \cdot 1.0 = 0.1$; $v_1 = 0.999 \cdot 0 + 0.001 \cdot 1.0 = 0.001$
- Without correction the step would use $0.1 / \sqrt{0.001} \approx 3.16$ — distorted by zero-initialization.
- Corrected: $\hat{m}_1 = 0.1 / (1 - 0.9) = 1.0$; $\hat{v}_1 = 0.001 / (1 - 0.999) = 1.0$
- Update: $\Delta\theta = -\eta \cdot 1.0 / (\sqrt{1.0} + \epsilon) \approx -0.001$ — a clean, unit-scaled step.

**Step 2:** $m_2 = 0.9 \cdot 0.1 + 0.1 \cdot 1.0 = 0.19$, $\hat{m}_2 = 0.19 / (1 - 0.81) = 1.0$; similarly $\hat{v}_2 = 1.0$. The step is again $\approx -0.001$.

Takeaway: with steady gradients, Adam's step size is approximately $\eta$ regardless of the gradient's absolute scale — the second moment normalizes it. This is why Adam is far less sensitive to gradient scale than SGD, and why $\eta \approx 10^{-3}$ works across many problems.

```python
import torch

theta = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([theta], lr=1e-3)
for _ in range(2):
    opt.zero_grad()
    theta.grad = torch.ones(1)   # inject g_t = 1.0
    opt.step()
print(theta)   # ≈ -0.002 after two steps of ≈ -0.001
```

## Best Practices

- ✅ Default to AdamW (`lr≈3e-4`, `weight_decay≈0.01`) for transformers and most new problems; SGD + momentum remains competitive for CNNs on vision benchmarks when you can afford tuning.
- ✅ Tune the learning rate first — it dominates every other optimizer hyperparameter. Search on a log scale.
- ✅ Pair the optimizer with a schedule: warmup + cosine is a robust default for long runs.
- ✅ When resuming training, restore `optimizer.state_dict()` too — Adam's moments are part of the training state.

## Common Mistakes

- ⚠️ Using `weight_decay` in `torch.optim.Adam` and expecting AdamW behavior — Adam couples decay into the adaptive update. Fix: use `torch.optim.AdamW` when you want decoupled decay.
- ⚠️ Forgetting `optimizer.zero_grad()` — updates use stale, accumulated gradients. Fix: zero at the top of every step.
- ⚠️ Calling `scheduler.step()` at the wrong granularity (per batch vs per epoch differs by scheduler). Fix: check each scheduler's documented convention.
- ⚠️ Carrying over an SGD-tuned learning rate (e.g. $0.1$) to Adam — Adam typically wants $10^{-3}$ or smaller; too-large rates diverge quickly.
- ⚠️ Applying weight decay to bias and normalization parameters. Fix: use parameter groups to exempt them, a standard trick in transformer recipes.

## Industry Tips

- 💡 Learning-rate warmup matters most early in training with Adam-family optimizers, when second-moment estimates are still noisy; a few hundred to a few thousand steps is common.
- 💡 If loss curves oscillate, lower the learning rate before switching optimizers — most "optimizer problems" are learning-rate problems.
- 💡 Adam keeps two extra tensors per parameter, so optimizer state roughly triples parameter memory — a real constraint for large models and a motivation for memory-efficient variants.

## Real-World Use Cases

- LLM and transformer pretraining almost universally uses AdamW with warmup and cosine (or similar) decay.
- Classic vision training (e.g. ResNet-style CNNs) often uses SGD with momentum plus step or cosine schedules.
- Fine-tuning pretrained models typically uses AdamW with a small learning rate and short warmup.

---

## Summary

- All these optimizers are variations on $\theta \leftarrow \theta - \eta \cdot (\text{step})$: momentum smooths the step direction, RMSProp scales it per parameter, Adam does both with bias correction.
- AdamW decouples weight decay from the adaptive update and is the standard for transformer training.
- The learning rate and its schedule are the highest-impact knobs; warmup + decay is the workhorse recipe.

## Practice

- [ ] Exercises: [Module 4 Exercises](../exercises/README.md)
- [ ] Self-check: Why does Adam need bias correction, and what would the first update look like without it?

## Further Reading

- 📘 Deep Learning — Goodfellow, Bengio & Courville (https://www.deeplearningbook.org/)
- 📘 Dive into Deep Learning — Zhang, Lipton, Li & Smola (https://d2l.ai/)
- 📄 [PyTorch documentation](https://pytorch.org/docs/stable/)
- ▶️ Andrej Karpathy (https://www.youtube.com/@AndrejKarpathy)

## Related

- [Backpropagation](backpropagation.md)
- [Regularization and Training Techniques](regularization-and-training.md)
- [Optimization](../../02-mathematics-foundations/lessons/optimization.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
