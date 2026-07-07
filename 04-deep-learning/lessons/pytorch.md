---
title: PyTorch Essentials
description: Tensors, autograd, nn.Module, DataLoader, and the canonical training loop — the practical PyTorch toolkit you need to train real models.
type: lesson
domain: 04-deep-learning
tags: [deep-learning, pytorch, autograd, training-loop]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 50 min
prerequisites:
  - 04-deep-learning/lessons/regularization-and-training
---

# PyTorch Essentials

> **TL;DR:** PyTorch gives you NumPy-like tensors with GPU acceleration and automatic differentiation; you define models as `nn.Module` classes, feed them batches via `DataLoader`, and train them with an explicit loop you write yourself — which is exactly why it is so debuggable.

---

## Overview

Every deep learning concept in this module — layers, losses, gradients, regularization — becomes concrete in PyTorch. This lesson is the practical anchor: you learn the five building blocks (tensors, autograd, modules, data loading, the training loop) and assemble them into a complete, correct training script you can adapt for any project. The patterns here — device-agnostic code, train/eval modes, `state_dict` checkpoints — are the same ones used in production codebases.

**By the end, you will be able to:**
- Create and manipulate tensors on CPU or GPU, and move data between PyTorch and NumPy
- Explain how autograd builds a computation graph and when to disable it
- Write the canonical training loop from memory: zero grads, forward, loss, backward, step, and a validation pass under `torch.no_grad()`

---

## Intuition

Think of PyTorch as **NumPy with two superpowers**:

1. **Tensors can live on a GPU.** The same array math, but running on thousands of parallel cores. You choose where a tensor lives with `.to(device)`.
2. **Tensors remember their history.** When you set `requires_grad=True`, every operation on that tensor is recorded in a graph. Call `.backward()` on a scalar loss, and PyTorch walks the graph in reverse, applying the chain rule automatically. You never derive a gradient by hand.

Everything else is organization. `nn.Module` bundles parameters with the code that uses them. `DataLoader` turns a dataset into shuffled mini-batches. The training loop is just the gradient-descent recipe you already know, written explicitly: predict, measure error, compute gradients, nudge weights. PyTorch deliberately makes you write this loop yourself — five lines of ritual — so that when something goes wrong you can put a `print` or a breakpoint anywhere.

---

## Details

### Tensors: creation, dtype, and device

A `torch.Tensor` is an n-dimensional array with a **dtype** (e.g. `float32`) and a **device** (`cpu` or `cuda`).

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # from data; infers float32
z = torch.zeros(3, 4)                          # shape (3, 4)
r = torch.randn(2, 5)                          # standard normal
a = torch.arange(6).reshape(2, 3)              # int64 by default

print(x.shape, x.dtype, x.device)              # torch.Size([2, 2]) torch.float32 cpu

# Move to GPU if available (returns a *new* tensor)
if torch.cuda.is_available():
    x = x.to("cuda")
```

Common gotcha: operations require tensors on the **same device** and usually the same dtype. `x.to("cuda")` does not modify `x` in place — reassign the result.

**NumPy interop** is cheap on CPU because memory is shared:

```python
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)       # shares memory with arr (CPU only)
back = t.numpy()                # shares memory too; fails if t is on GPU
gpu_safe = t.detach().cpu().numpy()  # the always-works pattern
```

### Autograd: automatic differentiation

Set `requires_grad=True` and PyTorch records operations; `.backward()` computes gradients into `.grad`.

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)

y = w * x + b          # y = 7.0, and PyTorch recorded the graph
loss = (y - 10.0) ** 2  # scalar
loss.backward()         # chain rule, backwards through the graph

print(w.grad)  # dloss/dw = 2*(y-10)*x = -18.0
print(b.grad)  # dloss/db = 2*(y-10)   = -6.0
```

Two facts you must internalize:

- **Gradients accumulate.** A second `backward()` *adds* to `.grad`. This is why every training step starts with `optimizer.zero_grad()`.
- **`torch.no_grad()` disables recording.** Use it for inference and validation — it saves memory and time because no graph is built:

```python
with torch.no_grad():
    preds = w * x + b   # no history recorded
```

### Building models: `nn.Module` vs `nn.Sequential`

For a plain stack of layers, `nn.Sequential` is the quickest option:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 2),
)
```

For anything with branching, custom logic, or reusable structure, subclass `nn.Module`: define layers in `__init__`, define the computation in `forward`. PyTorch tracks all parameters automatically.

```python
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

Call the model as `model(x)`, never `model.forward(x)` directly — the call syntax runs hooks that PyTorch relies on.

### `Dataset` and `DataLoader`

A `Dataset` answers two questions: how many samples (`__len__`) and what is sample *i* (`__getitem__`). A `DataLoader` wraps it with batching, shuffling, and optional parallel loading.

```python
from torch.utils.data import Dataset, DataLoader

class ArrayDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X, self.y = X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]

train_loader = DataLoader(ArrayDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(ArrayDataset(X_val, y_val), batch_size=256, shuffle=False)
```

Shuffle the training set every epoch (`shuffle=True`); never shuffle validation — it wastes time and makes runs harder to compare.

### The canonical training loop

This is the pattern to memorize. Every line matters.

```python
def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()                          # enable dropout, batchnorm updates
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()              # clear stale gradients
        preds = model(xb)                  # forward pass
        loss = loss_fn(preds, yb)          # scalar loss
        loss.backward()                    # compute gradients
        optimizer.step()                   # update parameters
        total += loss.item() * len(xb)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()                           # disable dropout, freeze batchnorm stats
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += loss_fn(model(xb), yb).item() * len(xb)
    return total / len(loader.dataset)
```

The pairs to never separate: `model.train()` / `model.eval()`, and `optimizer.zero_grad()` before `loss.backward()` before `optimizer.step()`.

### Saving and loading: `state_dict`

Save the **parameters**, not the whole pickled object — it is smaller, safer, and survives code refactors:

```python
torch.save(model.state_dict(), "model.pt")

model = MLP(20, 64, 2)                     # rebuild the architecture first
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()                               # before inference!
```

### Device-agnostic code

Write once, run anywhere — define `device` at the top and use it everywhere:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(20, 64, 2).to(device)
```

## Worked Example

A complete, runnable binary-classification script tying every piece together:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic data: class depends on the sign of a linear score + noise
X = torch.randn(2000, 20)
true_w = torch.randn(20)
y = ((X @ true_w + 0.5 * torch.randn(2000)) > 0).long()
X_train, y_train, X_val, y_val = X[:1600], y[:1600], X[1600:], y[1600:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
    print(f"epoch {epoch + 1}: val acc = {correct / len(y_val):.3f}")

torch.save(model.state_dict(), "clf.pt")
```

Validation accuracy should climb well above chance within a few epochs — the data is nearly linearly separable by construction.

## Best Practices

- ✅ Define `device` once at the top and `.to(device)` both the model and every batch — never hard-code `"cuda"`.
- ✅ Always pair `model.train()` for training and `model.eval()` + `torch.no_grad()` for evaluation.
- ✅ Save `state_dict` checkpoints, not whole pickled models.
- ✅ Use `loss.item()` (not the tensor) when logging, so you do not keep the whole graph alive in memory.
- ✅ Seed with `torch.manual_seed()` when you need reproducible experiments.

## Common Mistakes

- ⚠️ **Forgetting `optimizer.zero_grad()`** — gradients accumulate across steps and training silently diverges. Zero at the start of every iteration.
- ⚠️ **Evaluating without `model.eval()`** — dropout stays active and batchnorm keeps updating, so validation metrics are wrong. Switch modes explicitly.
- ⚠️ **Calling `.numpy()` on a GPU or grad-tracking tensor** — it raises an error. Use `t.detach().cpu().numpy()`.
- ⚠️ **Device mismatch errors** ("expected all tensors on the same device") — move the batch to `device` inside the loop, not just the model.
- ⚠️ **Loading a `state_dict` into the wrong architecture** — keys will not match. Rebuild the exact model class first, then load.

## Industry Tips

- 💡 The explicit training loop is a feature: teams wrap it in helpers (or use libraries like PyTorch Lightning) only after the plain loop works, because the plain loop is where you debug.
- 💡 `map_location=device` in `torch.load` lets you load GPU-trained checkpoints on a CPU-only machine — essential for inference services.
- 💡 In real pipelines, `DataLoader(num_workers=...)` for parallel data loading is often the difference between a starved GPU and a busy one; profile before adding more GPUs.

## Real-World Use Cases

- Research prototyping: most modern deep learning papers ship reference implementations in PyTorch.
- Hugging Face Transformers models are `nn.Module` subclasses — everything in this lesson applies directly to fine-tuning LLMs.
- Vision, speech, and recommendation systems in production, exported via TorchScript or ONNX for serving.

---

## Summary

- Tensors are GPU-capable arrays; autograd records operations so `loss.backward()` computes every gradient via the chain rule.
- Models are `nn.Module` objects; `DataLoader` delivers shuffled batches; the canonical loop is zero-grad → forward → loss → backward → step, with validation under `no_grad` in eval mode.
- Save and load `state_dict`, and write device-agnostic code with a single `device` variable.

## Practice

- [ ] Exercises: [Module 4 Exercises](../exercises/README.md)
- [ ] Self-check: What goes wrong if you delete `optimizer.zero_grad()` from the loop, and why does the failure mode follow from how autograd accumulates gradients?

## Further Reading

- 📄 [PyTorch documentation](https://pytorch.org/docs/stable/)
- 📄 [PyTorch tutorials](https://pytorch.org/tutorials/)
- 📄 [TensorFlow documentation](https://www.tensorflow.org/)
- 📘 Dive into Deep Learning — Zhang, Lipton, Li & Smola (https://d2l.ai/)

## Related

- [Regularization and Training Techniques](regularization-and-training.md)
- [TensorFlow and Keras](tensorflow-and-keras.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
