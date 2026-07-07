---
title: "Deep Learning — Interview Questions"
description: A bank of 20 deep-learning interview questions with answers.
type: interview
domain: 19-interview-preparation
tags: [deep-learning, interview, neural-networks]
category: deep-learning
difficulty: medium
status: complete
created: 2026-07-02
updated: 2026-07-02
companies: []
---

# Deep Learning — Interview Questions

> **Category:** deep-learning · **Difficulty:** mixed (easy → hard)
>
> A question bank tied to **[Module 4 — Deep Learning](../../04-deep-learning/README.md)**.
> Try to answer before reading the response.

---

## Contents

1. [Why nonlinear activations?](#1-why-nonlinear-activations)
2. [What is backpropagation?](#2-what-is-backpropagation)
3. [Vanishing & exploding gradients](#3-vanishing--exploding-gradients)
4. [Why ReLU over sigmoid?](#4-why-relu-over-sigmoid)
5. [SGD vs Adam](#5-sgd-vs-adam)
6. [What does the learning rate control?](#6-what-does-the-learning-rate-control)
7. [Batch vs mini-batch vs stochastic](#7-batch-vs-mini-batch-vs-stochastic)
8. [How dropout works](#8-how-dropout-works)
9. [Batch normalization](#9-batch-normalization)
10. [Why not initialize weights to zero?](#10-why-not-initialize-weights-to-zero)
11. [Why CNNs for images?](#11-why-cnns-for-images)
12. [Pooling and stride](#12-pooling-and-stride)
13. [ResNet skip connections](#13-resnet-skip-connections)
14. [Why LSTMs over vanilla RNNs?](#14-why-lstms-over-vanilla-rnns)
15. [Cross-entropy and logits in PyTorch](#15-cross-entropy-and-logits-in-pytorch)
16. [model.eval() vs torch.no_grad()](#16-modeleval-vs-torchno_grad)
17. [Autoencoders and their uses](#17-autoencoders-and-their-uses)
18. [How GANs train (and fail)](#18-how-gans-train-and-fail)
19. [Overfitting toolkit for deep nets](#19-overfitting-toolkit-for-deep-nets)
20. [First steps when loss won't decrease](#20-first-steps-when-loss-wont-decrease)

---

### 1. Why nonlinear activations?
**A:** A stack of linear layers is itself one linear map ($W_2(W_1x) = (W_2W_1)x$),
so without nonlinearities depth adds nothing. Activations between layers let the
network compose nonlinear functions and approximate complex mappings. See
[Activation Functions](../../04-deep-learning/lessons/activation-functions.md).

### 2. What is backpropagation?
**A:** An efficient application of the **chain rule** over the computation graph:
the forward pass caches intermediate values, then gradients flow backward from
the loss, reusing those caches so every parameter's gradient is computed in one
backward sweep. Frameworks automate it as autograd.

### 3. Vanishing & exploding gradients
**A:** Gradients are products of many Jacobians; if factors are consistently <1
they shrink exponentially (vanishing — early layers stop learning), if >1 they
blow up (exploding — NaNs). Remedies: ReLU-family activations, careful init,
batch norm, skip connections, gradient clipping, LSTM gating for recurrence.

### 4. Why ReLU over sigmoid?
**A:** Sigmoid saturates at both ends (gradient ≈ 0), causing vanishing gradients
and slower learning; it's also not zero-centered. ReLU keeps gradient 1 for
positive inputs, is cheap, and sparsifies activations. Watch for dying ReLUs
(use LeakyReLU if needed); reserve sigmoid for binary outputs.

### 5. SGD vs Adam
**A:** SGD (+momentum) uses one global LR; simple, strong generalization,
common in vision. **Adam** adapts a per-parameter step from first/second moment
estimates — faster to tune and the default for transformers/NLP (usually as
**AdamW**, which decouples weight decay). When unsure: AdamW with LR ~1e-3.

### 6. What does the learning rate control?
**A:** The step size along the negative gradient. Too high → divergence or
oscillation; too low → slow convergence and getting stuck. It's the single most
important hyperparameter; schedules (warmup, cosine/step decay) change it over
training.

### 7. Batch vs mini-batch vs stochastic
**A:** Full-batch: exact gradient, expensive, rare. Stochastic (1 sample): noisy,
poor hardware use. **Mini-batch** (32–512): the practical standard — good
gradient estimates, GPU-friendly, and the noise even helps escape sharp minima.

### 8. How dropout works
**A:** During **training**, each activation is zeroed with probability $p$ and
survivors are rescaled (inverted dropout), preventing co-adaptation — like
training an ensemble of subnetworks. At **eval** it's disabled entirely — hence
`model.eval()` matters.

### 9. Batch normalization
**A:** Normalizes each feature over the batch (then rescales with learned
$\gamma, \beta$), stabilizing and accelerating training and adding slight
regularization. Train mode uses batch statistics and updates running averages;
eval mode uses the stored running statistics — another reason mode toggling is
critical.

### 10. Why not initialize weights to zero?
**A:** All neurons in a layer would compute identical outputs and receive
identical gradients — they never differentiate (the symmetry problem). Random
init breaks symmetry; He (ReLU) / Xavier (tanh) scaling keeps activation and
gradient variance stable across depth.

### 11. Why CNNs for images?
**A:** Two inductive biases: **locality** (kernels look at neighborhoods) and
**parameter sharing** (the same kernel slides everywhere), giving translation
equivariance and far fewer parameters than dense layers on pixels — so they
generalize better on visual data. See [CNNs](../../04-deep-learning/lessons/cnn.md).

### 12. Pooling and stride
**A:** Both downsample feature maps. Max pooling keeps the strongest local
response, adds small translation invariance, and has no parameters; strided
convolution downsamples while learning how. Output size:
$\lfloor(n+2p-k)/s\rfloor + 1$.

### 13. ResNet skip connections
**A:** They add the block's input to its output ($y = F(x) + x$), so the block
learns a **residual**. Gradients flow directly through the identity path, curing
degradation in very deep nets and enabling hundreds of layers (He et al., 2015).

### 14. Why LSTMs over vanilla RNNs?
**A:** Vanilla RNNs suffer vanishing gradients through time, forgetting
long-range dependencies. LSTMs add a **cell state** with forget/input/output
gates — an additive highway through time that preserves gradients. GRUs get
similar benefits with fewer parameters. (Transformers have since superseded
both for most NLP.)

### 15. Cross-entropy and logits in PyTorch
**A:** `nn.CrossEntropyLoss` fuses log-softmax + NLL and therefore expects **raw
logits** and integer class labels. Adding your own softmax layer double-applies
it — a classic silent bug. For binary tasks prefer `BCEWithLogitsLoss` over
sigmoid + `BCELoss` for numerical stability.

### 16. model.eval() vs torch.no_grad()
**A:** Different jobs: `model.eval()` switches layer *behavior* (dropout off,
batch-norm uses running stats); `torch.no_grad()` disables gradient *tracking*
(saves memory/compute). Validation needs **both**.

### 17. Autoencoders and their uses
**A:** Encoder → bottleneck → decoder trained to reconstruct inputs; the
bottleneck forces a compressed representation. Uses: dimensionality reduction,
denoising, and **anomaly detection** (anomalies reconstruct poorly → high error).
VAEs make the latent space probabilistic so you can sample new data.

### 18. How GANs train (and fail)
**A:** A generator maps noise to fakes; a discriminator classifies real vs fake;
they play a minimax game. Failure modes: **mode collapse** (generator emits few
varieties), vanishing discriminator gradients, and non-convergent oscillation —
which is why diffusion models now dominate image generation.

### 19. Overfitting toolkit for deep nets
**A:** More data / augmentation, dropout, weight decay (AdamW), early stopping,
batch norm's mild regularization, smaller model, and transfer learning. Diagnose
first with train-vs-validation curves. See
[Regularization and Training](../../04-deep-learning/lessons/regularization-and-training.md).

### 20. First steps when loss won't decrease
**A:** (1) Overfit a tiny batch — if you can't, it's a bug, not a tuning issue.
(2) Check loss at init (≈ $\ln C$ for $C$-class CE). (3) Inspect one batch —
shapes, label range, input scale. (4) Verify the loop (`zero_grad`, modes).
(5) Only then sweep the learning rate.

---

## Common Mistakes

- ⚠️ Saying dropout is active at inference.
- ⚠️ Confusing `model.eval()` with `torch.no_grad()`.
- ⚠️ Softmax before `CrossEntropyLoss`.
- ⚠️ Blaming the architecture before checking the pipeline.

## Key Takeaways

- Interviewers probe *mechanics* (backprop, modes, losses) and *judgment*
  (what to try first, and why).
- Tie every technique to the failure mode it addresses.

## Related

- [Module 4 — Deep Learning](../../04-deep-learning/README.md)
- [Training Troubleshooting Cheat Sheet](../../20-cheat-sheets/deep-learning/training-troubleshooting.md)

---

## Navigation

- ⬆️ [Deep Learning Interview Questions](README.md)
- 📚 [Interview Preparation](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
