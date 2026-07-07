---
title: PyTorch Cheat Sheet
description: Tensors, autograd, modules, data loading, and the canonical training loop.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [deep-learning, pytorch, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# PyTorch Cheat Sheet

> Fast reference. For depth, see
> [PyTorch Essentials](../../04-deep-learning/lessons/pytorch.md).

---

## Tensors

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])
x = torch.zeros(2, 3); torch.randn(2, 3)
x.shape, x.dtype, x.device
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)                      # move to device
a = x.numpy(); t = torch.from_numpy(a)  # NumPy interop (CPU, shared memory)
```

## Autograd

```python
w = torch.randn(3, requires_grad=True)
loss = (w ** 2).sum()
loss.backward()                       # gradients into w.grad
w.grad.zero_()                        # clear before next backward
with torch.no_grad():                 # inference / no tracking
    y = model(x)
```

## Model & data

```python
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        return self.body(x)           # return logits

loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
```

## The canonical loop

```python
model.train()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

model.eval()                          # AND no_grad for validation
with torch.no_grad():
    ...
```

## Save / load

```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()
```

---

## Decision Guide

| If you need… | Use… |
|--------------|------|
| Quick sequential model | `nn.Sequential` |
| Custom forward logic | subclass `nn.Module` |
| Batching/shuffling | `Dataset` + `DataLoader` |
| Freeze params / inference | `torch.no_grad()` + `model.eval()` |
| Optimizer | `torch.optim.AdamW` as a strong default |

---

## Gotchas

- ⚠️ Forgetting `zero_grad()` accumulates gradients across batches.
- ⚠️ `model.eval()` and `torch.no_grad()` are different things — validation needs **both**.
- ⚠️ `CrossEntropyLoss` wants logits + integer class labels.
- ⚠️ Tensors on different devices can't operate together — keep model & data on one device.

---

## Quick Links

- 📖 [PyTorch Essentials](../../04-deep-learning/lessons/pytorch.md) · [Regularization & Training](../../04-deep-learning/lessons/regularization-and-training.md)
- 🔗 [PyTorch docs](https://pytorch.org/docs/stable/)

---

## Navigation

- ⬆️ [Deep Learning Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
