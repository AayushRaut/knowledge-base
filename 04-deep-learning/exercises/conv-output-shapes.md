---
title: "Exercise: Conv Output Shapes"
description: Compute convolution and pooling output dimensions by hand, then verify with PyTorch.
type: exercise
domain: 04-deep-learning
tags: [deep-learning, cnn, convolution, shapes]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 04-deep-learning/lessons/cnn
---

# Exercise: Conv Output Shapes

> Practice for **[Convolutional Neural Networks](../lessons/cnn.md)**.

---

## Problem

Shape bugs are the most common CNN failure. Using the output-size formula

$$
o = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1
$$

($n$ input size, $p$ padding, $k$ kernel, $s$ stride), compute the output shape
after each layer of this stack for a `(3, 32, 32)` input — by hand first:

1. `Conv2d(3, 16, kernel_size=3, stride=1, padding=1)`
2. `MaxPool2d(2)`
3. `Conv2d(16, 32, kernel_size=5, stride=1, padding=0)`
4. `MaxPool2d(2)`

Then verify with PyTorch and state the flattened size entering a linear layer.

## Requirements

- [ ] Hand-computed `(C, H, W)` after each of the 4 layers.
- [ ] PyTorch verification with a dummy tensor.
- [ ] The correct `in_features` for the following `nn.Linear`.

---

## Hints

<details>
<summary>Hint 1</summary>

Channels come from the layer's `out_channels`; H/W come from the formula.
`MaxPool2d(2)` is kernel 2, stride 2, padding 0.

</details>

<details>
<summary>Hint 2</summary>

Verify with `model(torch.zeros(1, 3, 32, 32)).shape` layer by layer.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

**By hand:**

1. Conv 3×3, p=1, s=1: $o = \lfloor(32+2-3)/1\rfloor+1 = 32$ → **(16, 32, 32)**
2. Pool 2: $32/2$ → **(16, 16, 16)**
3. Conv 5×5, p=0, s=1: $o = \lfloor(16+0-5)/1\rfloor+1 = 12$ → **(32, 12, 12)**
4. Pool 2: $12/2$ → **(32, 6, 6)**

Flattened: $32 \times 6 \times 6 = \mathbf{1152}$.

```python
import torch
import torch.nn as nn

stack = nn.Sequential(
    nn.Conv2d(3, 16, 3, stride=1, padding=1),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 5, stride=1, padding=0),
    nn.MaxPool2d(2),
)
out = stack(torch.zeros(1, 3, 32, 32))
print(out.shape)                     # torch.Size([1, 32, 6, 6])
print(out.flatten(1).shape)          # torch.Size([1, 1152])
head = nn.Linear(1152, 10)           # correct in_features
```

**Explanation:** "Same" padding ($p=\lfloor k/2\rfloor$ for odd $k$, stride 1)
preserves H/W; unpadded convs shrink it by $k-1$; each 2×2 pool halves it.
Being able to run this arithmetic in your head is what lets you design
architectures and read others' code without trial-and-error.

</details>

---

## Navigation

- ⬆️ [Module 4 Exercises](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
