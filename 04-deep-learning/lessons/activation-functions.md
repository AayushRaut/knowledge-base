---
title: Activation Functions
description: Why networks need nonlinearity, how sigmoid, tanh, ReLU, LeakyReLU, GELU, and softmax behave, and how to choose the right activation per layer and task.
type: lesson
domain: 04-deep-learning
tags: [deep-learning, activations, relu]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 40 min
prerequisites:
  - 04-deep-learning/lessons/neural-networks
---

# Activation Functions

> **TL;DR:** Activation functions inject the nonlinearity that gives neural networks their expressive power; ReLU-family functions dominate hidden layers, GELU powers transformers, and softmax/sigmoid live at output layers paired with the right loss.

---

## Overview

Every deep learning architecture you will build or debug — CNNs, transformers, LLMs — depends on activation functions in two ways: they make depth meaningful, and their gradient behavior determines whether training succeeds or stalls. Vanishing gradients, dead neurons, and saturated outputs are activation problems, and you will diagnose them in practice. Choosing activations correctly is a small decision with outsized impact on trainability.

**By the end, you will be able to:**
- Prove why stacked linear layers without activations collapse into a single linear layer
- Compare sigmoid, tanh, ReLU, LeakyReLU, and GELU by formula, gradient behavior, and failure modes
- Choose the right activation for hidden layers vs output layers given a task type

---

## Intuition

A linear layer can only stretch, rotate, and shift its input space. Chain ten of them together and you still have a single stretch-rotate-shift — like multiplying ten numbers to get one number. The composition never gets more interesting than its parts.

Activation functions break this. By bending the space between layers — clipping negatives to zero (ReLU), squashing into a band (sigmoid, tanh) — they let each subsequent layer work on a *fundamentally reshaped* version of the data. Folding a sheet of paper is a good mental model: each linear layer slides the sheet around, each activation folds it, and after enough folds you can bring together points that started far apart and separate points that started close.

The second thing to internalize: during training, gradients flow *backwards through* each activation, multiplied by its derivative. If that derivative is near zero (a "saturated" sigmoid, a "dead" ReLU), the gradient signal dies at that point and everything upstream stops learning. Most activation-function engineering is really gradient-flow engineering.

---

## Details

### Mathematics

**Why nonlinearity is required.** Let layers be $f_i(\mathbf{x}) = W_i\mathbf{x} + \mathbf{b}_i$ with weight matrices $W_i$ and biases $\mathbf{b}_i$. Composing two:

$$
f_2(f_1(\mathbf{x})) = W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)
$$

which is again a single affine map. By induction, any stack of linear layers equals one linear layer — no added expressive power without a nonlinearity between them.

**Sigmoid.** With input $z \in \mathbb{R}$:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,(1 - \sigma(z))
$$

Output range $(0, 1)$ — interpretable as a probability. The derivative peaks at $0.25$ (at $z=0$) and decays toward $0$ for large $|z|$: the function **saturates**. In a deep stack, backpropagation multiplies many such derivatives, each $\le 0.25$, so gradients shrink geometrically toward the early layers — the **vanishing gradient problem**. This is a core reason very deep sigmoid networks were historically hard to train.

**Tanh.**

$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}, \qquad \tanh'(z) = 1 - \tanh^2(z)
$$

Range $(-1, 1)$, zero-centered (unlike sigmoid), with maximum derivative $1$ at $z=0$. Zero-centered outputs help optimization, and tanh vanishes less aggressively than sigmoid — but it still saturates for large $|z|$.

**ReLU (Rectified Linear Unit).**

$$
\mathrm{ReLU}(z) = \max(0, z), \qquad \mathrm{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}
$$

For positive inputs the gradient is exactly $1$ — no shrinkage, no saturation on the active side — which is why ReLU made very deep networks practical. It is also nearly free to compute. Failure mode: **dying ReLU** — if a neuron's pre-activation becomes negative for every input (e.g. after a large gradient step), its output is always $0$, its gradient is always $0$, and it can never recover.

**LeakyReLU.** Fixes dying neurons by keeping a small slope $\alpha$ (typically $0.01$) for negatives:

$$
\mathrm{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \le 0 \end{cases}
$$

The gradient is never exactly zero, so units can recover.

**GELU (Gaussian Error Linear Unit).** Used in transformer architectures including BERT and the GPT family:

$$
\mathrm{GELU}(z) = z \cdot \Phi(z)
$$

where $\Phi(z)$ is the cumulative distribution function of the standard normal distribution. Intuition: instead of hard-gating the input by its sign (ReLU), GELU weights the input by the probability that a standard normal variable is below it — a smooth, slightly non-monotonic curve near zero that behaves like ReLU for large $|z|$ but with nonzero gradient everywhere. A common fast approximation uses tanh; PyTorch provides both exact and approximate forms.

**Softmax (output layers).** For a logit vector $\mathbf{z} \in \mathbb{R}^k$ over $k$ classes:

$$
\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
$$

Outputs are positive and sum to $1$ — a probability distribution over classes. Softmax is not a hidden-layer activation; it belongs at the output of a multi-class classifier and pairs with **cross-entropy loss**, whose gradient with respect to the logits simplifies to the clean form $\hat{\mathbf{y}} - \mathbf{y}$ (predicted probabilities minus one-hot targets). See [Loss Functions](loss-functions.md).

### Python implementation

Compute and compare the activations and their gradients over a grid — the numbers tell the saturation story without a plot:

```python
import torch
import torch.nn.functional as F

z = torch.linspace(-5.0, 5.0, steps=11, requires_grad=True)

activations = {
    "sigmoid":    torch.sigmoid,
    "tanh":       torch.tanh,
    "relu":       F.relu,
    "leaky_relu": lambda t: F.leaky_relu(t, negative_slope=0.01),
    "gelu":       F.gelu,
}

for name, fn in activations.items():
    y = fn(z)
    (grad,) = torch.autograd.grad(y.sum(), z, retain_graph=True)
    print(f"{name:>10} | f(-5)={y[0]:+.3f} f(0)={y[5]:+.3f} f(5)={y[-1]:+.3f}"
          f" | f'(-5)={grad[0]:.4f} f'(5)={grad[-1]:.4f}")

# Note sigmoid/tanh gradients ~0 at |z|=5 (saturation),
# relu gradient exactly 0 for z<0 (dying risk), leaky_relu keeps 0.01.
```

Using them in a model — activations as layers:

```python
import torch.nn as nn

hidden = nn.Sequential(
    nn.Linear(64, 128), nn.GELU(),      # transformer-style hidden block
    nn.Linear(128, 128), nn.ReLU(),     # classic hidden block
    nn.Linear(128, 10),                 # raw logits out — no softmax here:
)                                       # nn.CrossEntropyLoss applies it internally
```

## Worked Example

Trace one value through the stack to see saturation concretely. Suppose a hidden neuron receives pre-activation $z = 6$:

- **Sigmoid:** $\sigma(6) \approx 0.9975$, gradient $\sigma'(6) = 0.9975 \times 0.0025 \approx 0.0025$. During backprop, whatever gradient arrives at this neuron is multiplied by $0.0025$ — a 400× shrink at a *single* layer. Ten such saturated layers and the signal is effectively gone.
- **ReLU:** $\mathrm{ReLU}(6) = 6$, gradient $1$. The signal passes through untouched.
- **GELU:** $\mathrm{GELU}(6) \approx 6$ (since $\Phi(6) \approx 1$), gradient $\approx 1$ — ReLU-like for large positives, but smooth near zero.

Now $z = -2$:

- **ReLU:** output $0$, gradient $0$ — this example contributes nothing to learning through this unit.
- **LeakyReLU** ($\alpha = 0.01$): output $-0.02$, gradient $0.01$ — small but alive.
- **GELU:** output $\approx -2 \times \Phi(-2) \approx -0.045$, with a small negative-side gradient — also alive.

This is the entire practical trade-off, one number at a time: saturating functions choke gradients at extremes; ReLU is perfect on the positive side but stone-dead on the negative side; LeakyReLU and GELU keep a pulse everywhere.

## Best Practices

- ✅ Default to **ReLU** for hidden layers in MLPs/CNNs; switch to **GELU** when matching transformer-style architectures.
- ✅ Output raw **logits** from the final layer and let the loss (`CrossEntropyLoss`, `BCEWithLogitsLoss`) apply softmax/sigmoid internally — more numerically stable.
- ✅ If many units go silent (activations all zero across batches), try LeakyReLU or lower the learning rate — that pattern is dying ReLU.
- ✅ Pair your activation choice with the matching weight initialization (e.g. Kaiming/He initialization for ReLU-family), which PyTorch's `nn.Linear` defaults handle reasonably.

## Common Mistakes

- ⚠️ **Adding `nn.Softmax` before `nn.CrossEntropyLoss`** — the loss already applies log-softmax, so you softmax twice, wrecking gradients quietly. Fix: pass logits.
- ⚠️ **Using sigmoid/tanh in deep hidden stacks** — vanishing gradients stall early layers. Fix: ReLU-family for depth.
- ⚠️ **Ignoring silent neurons** — a network can "train" while half its ReLU units are permanently dead capacity. Fix: monitor the fraction of zero activations.
- ⚠️ **Softmax in hidden layers** — it couples all units and squashes information; it is an output-layer tool. Fix: keep it at the head only.

## Industry Tips

- 💡 Modern transformers (BERT, GPT-family) use GELU or its variants in their feed-forward blocks — when reproducing a paper, match its activation exactly; it affects results.
- 💡 Activation choice interacts with normalization: batch/layer norm keeps pre-activations in the non-saturating regime, which is part of why very deep networks train at all.
- 💡 In inference-optimized deployments, ReLU's cheapness matters — it fuses well into kernels and quantizes cleanly, which is one reason it persists in edge/mobile models.

## Real-World Use Cases

- GELU in the feed-forward layers of LLMs and BERT-style encoders
- ReLU throughout convolutional vision backbones (ResNet-style networks)
- Sigmoid outputs for multi-label tagging (each label an independent yes/no)
- Softmax heads in every multi-class classifier, from spam filters to next-token prediction

---

## Summary

- Without nonlinear activations, any depth of linear layers collapses algebraically into one linear layer — activations are what make deep networks deep.
- Sigmoid and tanh saturate, causing vanishing gradients in deep stacks; ReLU fixed this but can die on the negative side; LeakyReLU and GELU keep gradients alive everywhere.
- Hidden layers take ReLU/GELU; output layers take task-determined heads (softmax for multi-class, sigmoid for binary/multi-label, identity for regression) — applied inside the loss function, not the model.

## Practice

- [ ] Exercises: [Module 4 Exercises](../exercises/README.md)
- [ ] Self-check: Why does the derivative of sigmoid cause vanishing gradients in a 20-layer network, and why doesn't ReLU have the same problem?
- [ ] Self-check: When would you pick sigmoid over softmax at the output layer?

## Further Reading

- 📘 Deep Learning — Goodfellow, Bengio & Courville (https://www.deeplearningbook.org/) — Chapter 6 discusses activation choices
- 📘 Dive into Deep Learning — Zhang, Lipton, Li & Smola (https://d2l.ai/) — activation functions with runnable code
- 📄 [PyTorch documentation](https://pytorch.org/docs/stable/) — `torch.nn` non-linear activations reference
- ▶️ 3Blue1Brown — Neural networks series (https://www.youtube.com/@3blue1brown)
- ▶️ StatQuest (https://www.youtube.com/@statquest) — clear walk-throughs of ReLU and friends

## Related

- [Neural Networks and the Perceptron](neural-networks.md)
- [Loss Functions](loss-functions.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
