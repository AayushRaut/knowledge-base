---
title: "Debugging: Fix the Cheating Decoder"
description: A language model gets ~100% training accuracy but generates gibberish — the causal mask is missing.
type: exercise
domain: 06-transformers
tags: [transformers, debugging, causal-mask, data-leakage]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 06-transformers/lessons/masked-and-cross-attention
---

# Debugging: Fix the Cheating Decoder

> Practice for **[Masked and Cross-Attention](../lessons/masked-and-cross-attention.md)**.

---

## Problem

This decoder-only model trains to near-perfect next-token accuracy in a couple of
epochs, yet generates nonsense at inference. Explain the paradox and fix it.

```python
import torch
import torch.nn.functional as F

def self_attention(x, Wq, Wk, Wv):
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
    # BUG: no causal mask — every position sees the ENTIRE sequence,
    # including the tokens it is supposed to predict.
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

The training loss uses the standard shifted objective: predict token $t{+}1$ from
the sequence up to $t$.

## Requirements

- [ ] Explain why training accuracy is misleadingly high.
- [ ] Explain why generation fails despite that.
- [ ] Add the causal mask correctly.

---

## Hints

<details>
<summary>Hint 1</summary>

If position $t$ can attend to position $t{+}1$, and the target *is* token
$t{+}1$, the model can just read the answer.

</details>

<details>
<summary>Hint 2</summary>

Add an upper-triangular `-inf` mask to `scores` before the softmax.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import torch
import torch.nn.functional as F


def self_attention(x, Wq, Wk, Wv):
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    n = x.size(-2)
    scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
    # Forbid attending to future positions (strict upper triangle).
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

**The paradox explained:** without the mask, each position attends to *all*
positions — including the next token, which is exactly the training target. The
model learns the trivial "copy the token you can already see" shortcut, so
training accuracy shoots to ~100%. But at **generation** time the future tokens
don't exist yet (you're producing them one at a time), so the shortcut is
unavailable and the model, having learned nothing useful, emits gibberish. This
is **label leakage through attention** — the transformer analogue of the
train/test leakage bugs in earlier modules. The causal mask forces the model to
predict from the past alone, which is the only thing available at inference.

</details>

---

## Navigation

- ⬆️ [Module 6 Exercises](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
