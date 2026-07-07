---
title: Loss Functions
description: How loss functions define the training objective — MSE and MAE for regression, cross-entropy for classification, and the PyTorch logits conventions that prevent numerical bugs.
type: lesson
domain: 04-deep-learning
tags: [deep-learning, loss-functions, cross-entropy]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 45 min
prerequisites:
  - 04-deep-learning/lessons/activation-functions
  - 02-mathematics-foundations/lessons/information-theory
---

# Loss Functions

> **TL;DR:** The loss function is the single number your network minimizes — it *is* the training objective. Use MSE/MAE for regression and cross-entropy (on logits, via `nn.CrossEntropyLoss` or `BCEWithLogitsLoss`) for classification.

---

## Overview

Everything a neural network learns, it learns by descending the gradient of a loss function — the loss defines what "good" means, and the optimizer merely chases it. Choosing the wrong loss (or feeding it the wrong inputs, like probabilities where logits are expected) is one of the most common and most silent deep learning bugs. As an AI engineer, you will pick, implement, and debug losses on every project, from regression heads to LLM pretraining objectives — next-token prediction is cross-entropy at scale.

**By the end, you will be able to:**
- Choose between MSE, MAE, binary cross-entropy, and categorical cross-entropy per task, and justify the choice
- Derive cross-entropy from maximum likelihood and explain why it beats MSE for classification
- Use PyTorch losses correctly — logits vs probabilities, `CrossEntropyLoss` vs `NLLLoss`, `BCEWithLogitsLoss` vs `BCELoss`

---

## Intuition

Training a network is like coaching by score alone: the network makes a prediction, the loss function hands back one number saying "this is how wrong you were," and gradient descent nudges every weight to shrink that number. The network never sees your intentions — only the loss. If the loss rewards the wrong thing, the network learns the wrong thing, perfectly.

Two intuitions carry most of this lesson:

1. **Regression losses measure distance.** Predicting 7 when the answer is 9 is "2 away." MSE squares that distance (so being far off hurts disproportionately — one terrible prediction dominates); MAE takes it straight (so all errors count proportionally — robust to outliers).
2. **Classification losses measure surprise.** If the model says "99% cat" and it's a dog, it should be punished brutally; if it says "55% cat," mildly. Cross-entropy does exactly this: it is the negative log of the probability assigned to the true class, so confidence in the wrong answer costs $-\log(\text{tiny number}) = $ a huge loss. This "surprise" framing is information theory speaking — see [Information Theory](../../02-mathematics-foundations/lessons/information-theory.md).

---

## Details

### Mathematics

Notation: $n$ examples, true target $y_i$, prediction $\hat{y}_i$.

**Mean Squared Error (MSE)** — regression:

$$
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Squaring penalizes large errors quadratically and gives a smooth gradient $\propto (\hat{y}_i - y_i)$. MSE corresponds to maximum likelihood under Gaussian-distributed noise. Sensitive to outliers: one wild target can dominate the gradient.

**Mean Absolute Error (MAE)** — regression:

$$
\mathcal{L}_{\mathrm{MAE}} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Linear penalty; robust to outliers (corresponds to Laplacian noise, and drives predictions toward the conditional median rather than the mean). Its gradient is constant $\pm 1$ regardless of error size, which can make final convergence less precise; Huber/smooth-L1 loss blends MSE near zero with MAE in the tails to get both behaviors.

**Binary cross-entropy (BCE)** — binary classification. With true label $y_i \in \{0, 1\}$ and predicted probability $p_i = \sigma(z_i) \in (0,1)$ (where $z_i$ is the model's raw logit and $\sigma$ is the sigmoid):

$$
\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\Big[\, y_i \log p_i + (1 - y_i)\log(1 - p_i) \,\Big]
$$

Exactly the negative log-likelihood of a Bernoulli model: minimizing BCE **is** maximum likelihood estimation.

**Categorical cross-entropy (CE)** — multi-class. With $k$ classes, one-hot target $\mathbf{y}_i$ (component $y_{i,c}$), and predicted distribution $\hat{\mathbf{p}}_i = \mathrm{softmax}(\mathbf{z}_i)$ over logits $\mathbf{z}_i \in \mathbb{R}^k$:

$$
\mathcal{L}_{\mathrm{CE}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{k} y_{i,c}\, \log \hat{p}_{i,c} \;=\; -\frac{1}{n}\sum_{i=1}^{n} \log \hat{p}_{i,\,c_i}
$$

where $c_i$ is the true class index. In information-theoretic terms this is the cross-entropy $H(\mathbf{y}, \hat{\mathbf{p}})$ between the target and predicted distributions; since the target is fixed, minimizing it minimizes the KL divergence $D_{\mathrm{KL}}(\mathbf{y}\,\|\,\hat{\mathbf{p}})$ — you are pushing the model's distribution toward the data's.

**Why CE + softmax, not MSE, for classification.** Two reasons:

1. **Gradient quality.** For softmax + CE, the gradient with respect to the logits is simply $\hat{\mathbf{p}} - \mathbf{y}$ — large exactly when the model is confidently wrong. With MSE on softmax outputs, the softmax derivative appears as a factor, and it is near zero when the model saturates — so a *confidently wrong* model gets a *tiny* gradient and barely corrects itself.
2. **Right statistical model.** CE is the negative log-likelihood of a categorical distribution — the correct probabilistic model for class labels. MSE assumes Gaussian targets, which class labels are not.

**Label smoothing** (brief). Replace the hard one-hot target with a softened one: $y_{c} = 1 - \epsilon$ for the true class and $\epsilon/(k-1)$ elsewhere, for small $\epsilon$ (e.g. $0.1$). This discourages the model from driving logits to extremes, typically improving calibration and generalization; widely used in vision and machine translation training recipes. In PyTorch: `nn.CrossEntropyLoss(label_smoothing=0.1)`.

### Python implementation

The critical PyTorch conventions, demonstrated:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---- Regression ----
pred = torch.tensor([2.5, 0.0, 2.0])
target = torch.tensor([3.0, -0.5, 2.0])
print("MSE:", F.mse_loss(pred, target).item())   # mean of squared errors
print("MAE:", F.l1_loss(pred, target).item())    # mean of absolute errors

# ---- Multi-class: CrossEntropyLoss expects RAW LOGITS ----
logits = torch.randn(4, 5)                # batch of 4, 5 classes — no softmax!
labels = torch.tensor([1, 0, 4, 2])       # class indices, NOT one-hot
ce = nn.CrossEntropyLoss()
print("CE:", ce(logits, labels).item())

# CrossEntropyLoss == log_softmax + NLLLoss, fused for numerical stability:
manual = F.nll_loss(F.log_softmax(logits, dim=1), labels)
assert torch.isclose(ce(logits, labels), manual)

# ---- Binary: prefer BCEWithLogitsLoss over sigmoid + BCELoss ----
z = torch.tensor([2.0, -1.0, 0.5])         # raw logits
y = torch.tensor([1.0, 0.0, 1.0])          # float targets in {0,1}
stable = nn.BCEWithLogitsLoss()(z, y)      # fuses sigmoid + BCE (log-sum-exp trick)
unstable = nn.BCELoss()(torch.sigmoid(z), y)
assert torch.isclose(stable, unstable)     # same value; the fused form
                                           # stays finite for extreme logits
```

Why the fused versions matter: computing `log(sigmoid(z))` in two steps underflows for very negative `z` (sigmoid rounds to 0, log gives `-inf`). `BCEWithLogitsLoss` and `CrossEntropyLoss` reformulate the math (log-sum-exp) so extreme logits stay finite.

**Choosing a loss per task:**

| Task | Output layer | PyTorch loss | Targets |
|------|--------------|--------------|---------|
| Regression (typical) | Linear (identity) | `nn.MSELoss` | float values |
| Regression (outlier-heavy) | Linear | `nn.L1Loss` or `nn.HuberLoss` | float values |
| Binary classification | Linear (1 logit) | `nn.BCEWithLogitsLoss` | float `{0., 1.}` |
| Multi-class (one label) | Linear ($k$ logits) | `nn.CrossEntropyLoss` | class indices (long) |
| Multi-label (many yes/no) | Linear ($k$ logits) | `nn.BCEWithLogitsLoss` | float multi-hot |
| Next-token prediction (LLMs) | Linear over vocab | `nn.CrossEntropyLoss` | token indices |

## Worked Example

Cross-entropy punishing confident wrongness, by hand. A 3-class model sees a true label of class 0 and produces two candidate logit vectors:

```python
import torch
import torch.nn.functional as F

y = torch.tensor([0])

confident_wrong = torch.tensor([[-2.0, 4.0, 1.0]])   # ~94% on class 1
hedged         = torch.tensor([[ 1.0, 1.2, 0.8]])    # ~31/38/26% split

for name, z in [("confident-wrong", confident_wrong), ("hedged", hedged)]:
    p = F.softmax(z, dim=1)
    loss = F.cross_entropy(z, y)
    print(f"{name:>15}: p(true class)={p[0, 0]:.4f}  CE={loss.item():.4f}")

# confident-wrong: p(true)~0.0024 -> CE ~ 6.05   (brutal)
#          hedged: p(true)~0.31   -> CE ~ 1.16   (mild)
```

The loss for the confident-wrong prediction is roughly five times higher, and its logit-gradient $\hat{\mathbf{p}} - \mathbf{y} \approx (-0.998, +0.94, +0.06)$ pushes hard to raise the true-class logit and slash the wrong one. Had you used MSE on the softmax outputs instead, the saturated softmax would have shrunk that corrective gradient — precisely the failure mode CE avoids.

## Best Practices

- ✅ Feed **raw logits** to `nn.CrossEntropyLoss` and `nn.BCEWithLogitsLoss`; never add a softmax/sigmoid layer before them.
- ✅ Pass class **indices** (dtype `long`) to `CrossEntropyLoss`, and **float** targets to BCE-family losses — the dtype requirements differ.
- ✅ Sanity-check your initial loss: a $k$-class classifier with random weights should start near $\ln k$ (e.g. $\ln 10 \approx 2.303$); a very different value means a pipeline bug.
- ✅ For imbalanced binary problems, use `BCEWithLogitsLoss(pos_weight=...)` before reaching for exotic losses.

## Common Mistakes

- ⚠️ **Softmax before `CrossEntropyLoss`** — the loss applies log-softmax internally; doubling it flattens gradients and quietly degrades training. Fix: output logits.
- ⚠️ **`BCELoss` on hand-applied sigmoids** — numerically unstable for extreme logits (`log(0) = -inf`, producing NaNs). Fix: `BCEWithLogitsLoss` on logits.
- ⚠️ **One-hot targets passed to `CrossEntropyLoss` expecting indices** — shape/dtype confusion; modern PyTorch accepts class probabilities as targets, but mixing the two conventions accidentally gives wrong losses. Fix: pick one convention and assert target shape/dtype.
- ⚠️ **MSE for classification** — pairs a Gaussian assumption with categorical data and yields vanishing gradients through saturated softmax. Fix: cross-entropy.

## Industry Tips

- 💡 LLM pretraining loss *is* categorical cross-entropy over the vocabulary at every token position — "the model's loss curve" in LLM papers is this quantity, and perplexity is $e^{\mathrm{CE}}$.
- 💡 Watch train loss in log scale during early training; the drop from the $\ln k$ baseline is the fastest signal your pipeline is wired correctly.
- 💡 Label smoothing (`label_smoothing=0.1`) is a cheap, common win for large-scale classification — but skip it when you need calibrated confidence scores downstream, as it biases probabilities away from extremes.

## Real-World Use Cases

- Next-token prediction in LLM pretraining and fine-tuning (cross-entropy over vocabulary)
- Fraud and churn scoring (BCE with `pos_weight` for heavy class imbalance)
- Demand and price forecasting (MSE/Huber regression heads)
- Multi-label content tagging — one `BCEWithLogitsLoss` across hundreds of independent tags

---

## Summary

- The loss function defines the objective: regression losses (MSE, MAE, Huber) measure distance to a numeric target; classification losses (cross-entropy) measure surprise under the predicted distribution and are equivalent to maximum likelihood.
- CE + softmax beats MSE for classification because its logit-gradient $\hat{\mathbf{p}} - \mathbf{y}$ stays large exactly when the model is confidently wrong.
- In PyTorch, always pass logits: `CrossEntropyLoss` fuses log-softmax + NLL, and `BCEWithLogitsLoss` fuses sigmoid + BCE — both for numerical stability.

## Practice

- [ ] Exercises: [Module 4 Exercises](../exercises/README.md)
- [ ] Self-check: Why does the gradient of MSE-through-softmax vanish for a confidently wrong prediction, while cross-entropy's does not?
- [ ] Self-check: Your 10-class classifier starts training with loss 15.2 instead of ~2.3 — name two likely causes.

## Further Reading

- 📘 Deep Learning — Goodfellow, Bengio & Courville (https://www.deeplearningbook.org/) — Chapters 5–6 on maximum likelihood and cost functions
- 📘 Dive into Deep Learning — Zhang, Lipton, Li & Smola (https://d2l.ai/) — softmax regression and cross-entropy derivations
- 📄 [PyTorch documentation](https://pytorch.org/docs/stable/) — `nn.CrossEntropyLoss`, `nn.BCEWithLogitsLoss`, `nn.HuberLoss`
- ▶️ 3Blue1Brown — Neural networks series (https://www.youtube.com/@3blue1brown)
- ▶️ StatQuest (https://www.youtube.com/@statquest) — cross-entropy explained step by step

## Related

- [Activation Functions](activation-functions.md)
- [Optimizers](optimizers.md)
- [Information Theory](../../02-mathematics-foundations/lessons/information-theory.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
