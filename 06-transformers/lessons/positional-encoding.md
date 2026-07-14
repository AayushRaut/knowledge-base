---
title: Positional Encoding
description: Why self-attention needs explicit position information and how sinusoidal, learned, and rotary encodings provide it.
type: lesson
domain: 06-transformers
tags: [transformers, positional-encoding, rope]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 45 min
prerequisites:
  - 06-transformers/lessons/self-attention
---

# Positional Encoding

> **TL;DR:** Self-attention treats its input as an unordered set, so "dog bites man" and "man bites dog" look identical to it. Positional encoding fixes this by injecting position information into the token embeddings — via fixed sinusoids, learned embeddings, or rotations of the query/key vectors (RoPE).

---

## Overview

Attention computes weighted averages over all tokens, and nothing in that computation depends on *where* a token sits in the sequence. This lesson shows why that is a problem, then walks through the three dominant fixes: the sinusoidal encoding from the original Transformer paper, learned positional embeddings (BERT/GPT-2 style), and rotary position embeddings (RoPE) used by Llama-family models.

**By the end, you will be able to:**
- Explain why self-attention is permutation-equivariant and demonstrate it in code.
- Implement the sinusoidal positional encoding from "Attention Is All You Need" and interpret its frequency structure.
- Compare sinusoidal, learned, and rotary encodings and say when each is used in practice.

---

## Intuition

Imagine handing someone a bag of word-cards: {"dog", "bites", "man"}. Without order information, they cannot tell who bit whom. Self-attention receives exactly such a bag: every token attends to every other token by content similarity alone, so permuting the input rows just permutes the output rows the same way — the model has no notion of "first", "next", or "far apart".

The fix is to stamp each token with a *position signature* before attention sees it. The sinusoidal scheme does this like an odometer built from smooth waves: the first pair of embedding dimensions oscillates fast (changing every position, like the ones digit), later pairs oscillate progressively slower (like tens, hundreds, ...). Think of binary counting — the lowest bit flips every step, the next bit every two steps — but with continuous sine/cosine waves instead of hard 0/1 flips. Two nearby positions get nearly identical signatures; distant positions get distinguishable ones.

RoPE takes a different angle, literally: instead of *adding* a position vector to the embedding, it *rotates* each query and key vector by an angle proportional to its position. When a rotated query meets a rotated key in a dot product, the absolute angles partially cancel and what survives depends on the *difference* of positions — attention becomes a function of relative distance.

---

## Details

### Mathematics

**Permutation equivariance.** Let $X \in \mathbb{R}^{n \times d}$ be the matrix of $n$ token embeddings of dimension $d$, and let $P$ be any $n \times n$ permutation matrix. Self-attention $\mathrm{Attn}(X) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ with $Q = XW_Q$, $K = XW_K$, $V = XW_V$ satisfies

$$\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X).$$

Shuffling the inputs shuffles the outputs identically — no output value ever changes, only its row. Word order carries no signal.

**Sinusoidal encoding** (Vaswani et al., 2017). Define a matrix $PE \in \mathbb{R}^{n \times d_{model}}$:

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

where:
- $pos \in \{0, 1, \dots, n-1\}$ is the token's position in the sequence,
- $i \in \{0, 1, \dots, d_{model}/2 - 1\}$ indexes the sine/cosine *pair* (dimension pair $2i$, $2i{+}1$),
- $d_{model}$ is the embedding dimension (512 in the original paper).

Each pair $(2i, 2i{+}1)$ traces a point on a circle rotating with wavelength $2\pi \cdot 10000^{2i/d_{model}}$: wavelengths form a geometric progression from $2\pi$ (fast) to $10000 \cdot 2\pi$ (slow). The encoding is simply **added** to the token embeddings: the input to the first layer is $X + PE$ (same shape).

**Key properties:**
- *Deterministic* — no parameters to train; computed once from position and dimension.
- *Extrapolates in principle* — defined for any $pos$, including positions longer than any training sequence (though quality at unseen lengths degrades in practice).
- *Relative offsets are linear.* For any fixed offset $k$, each pair satisfies a rotation identity: $\begin{pmatrix} \sin(\omega_i(pos+k)) \\ \cos(\omega_i(pos+k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \begin{pmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{pmatrix}$ where $\omega_i = 10000^{-2i/d_{model}}$ is the pair's angular frequency. So $PE_{pos+k}$ is a fixed linear transform of $PE_{pos}$, which the authors hypothesized makes relative positions easy to learn.

**Learned positional embeddings** (BERT, GPT-2). Replace $PE$ with a trainable lookup table $E_{pos} \in \mathbb{R}^{L_{max} \times d_{model}}$, where $L_{max}$ is a fixed maximum sequence length. Simple and flexible — the model learns whatever position features help — but it cannot represent positions beyond $L_{max}$ at all.

**RoPE — rotary position embeddings** (Su et al., 2021, "RoFormer"). Instead of adding anything to the embeddings, rotate each 2-D pair of the query and key vectors by a position-dependent angle: $q_{pos} \mapsto R(\theta \cdot pos)\, q_{pos}$, where $R(\alpha)$ is a 2-D rotation matrix by angle $\alpha$ and each pair has its own base frequency $\theta_i$. Because rotations are orthogonal, the attention score becomes

$$\langle R(\theta\, m)\, q,\; R(\theta\, n)\, k \rangle = \langle q,\; R(\theta (n - m))\, k \rangle,$$

with $m$, $n$ the query and key positions — the score depends only on the *relative* offset $n - m$. RoPE is used by the Llama family and many other modern LLMs. **ALiBi** is another relative scheme (it biases attention scores by distance); we only name it here.

### Python implementation

Sinusoidal PE in NumPy, plus a demonstration of permutation equivariance:

```python
import numpy as np

def sinusoidal_pe(n_positions: int, d_model: int) -> np.ndarray:
    """Return the (n_positions, d_model) sinusoidal positional encoding."""
    positions = np.arange(n_positions)[:, None]          # (n, 1)
    i = np.arange(d_model // 2)[None, :]                  # (1, d/2)
    angles = positions / (10000 ** (2 * i / d_model))     # (n, d/2)
    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(angles)   # even dims: sine
    pe[:, 1::2] = np.cos(angles)   # odd dims:  cosine
    return pe

pe = sinusoidal_pe(n_positions=50, d_model=8)
print(np.round(pe[:4], 3))  # first 4 positions, all 8 dims
```

```text
[[ 0.     1.     0.     1.     0.     1.     0.     1.   ]
 [ 0.841  0.54   0.1    0.995  0.01   1.     0.001  1.   ]
 [ 0.909 -0.416  0.199  0.98   0.02   1.     0.002  1.   ]
 [ 0.141 -0.99   0.296  0.955  0.03   1.     0.003  1.   ]]
```

Note how dims 0–1 change quickly across positions while dims 6–7 barely move — fast and slow "digits". Usage: with token embeddings `x` of shape `(n, d_model)`, the model input is simply `x + pe[:n]` — the encoding is added, not concatenated, so shapes are unchanged.

The problem it solves, demonstrated:

```python
def self_attention(x: np.ndarray) -> np.ndarray:
    """Toy self-attention with identity projections."""
    scores = x @ x.T / np.sqrt(x.shape[-1])
    weights = np.exp(scores) / np.exp(scores).sum(-1, keepdims=True)
    return weights @ x

rng = np.random.default_rng(0)
x = rng.normal(size=(3, 8))            # "dog bites man"
perm = [2, 1, 0]                        # "man bites dog"
out, out_perm = self_attention(x), self_attention(x[perm])
print(np.allclose(out[perm], out_perm))  # True — same outputs, just shuffled rows

# With PE, the shuffled sentence gets NEW position stamps -> different outputs:
y = x + pe[:3]                # original word order
y_perm = x[perm] + pe[:3]     # shuffled words, positions reassigned
print(np.allclose(self_attention(y)[perm], self_attention(y_perm)))  # False
```

## Worked Example

Take $d_{model} = 4$ and compute the encoding for position $pos = 3$. The two pairs have frequencies $\omega_0 = 10000^{0/4} = 1$ and $\omega_1 = 10000^{-2/4} = 0.01$:

- Pair 0 (dims 0, 1): $\sin(3 \cdot 1) = \sin(3) \approx 0.141$, $\cos(3) \approx -0.990$.
- Pair 1 (dims 2, 3): $\sin(3 \cdot 0.01) = \sin(0.03) \approx 0.030$, $\cos(0.03) \approx 1.000$.

So $PE_3 \approx (0.141, -0.990, 0.030, 1.000)$. Compare $PE_4 \approx (-0.757, -0.654, 0.040, 0.999)$: the fast pair changed dramatically, the slow pair barely moved. A model can read coarse position from slow dims and fine position from fast dims — and if the token "bites" has embedding $e$, the layer input at position 3 is just $e + PE_3$.

## Best Practices

- ✅ Add positional encodings *once*, at the input to the first layer, and scale token embeddings by $\sqrt{d_{model}}$ before adding (as in the original paper) so the embedding signal is not drowned out.
- ✅ Prefer RoPE for new decoder-style LLMs — it is the de facto standard (Llama family) and handles relative position natively.
- ✅ Cache the PE matrix (or RoPE sin/cos tables) as a non-trainable buffer rather than recomputing it every forward pass.

## Common Mistakes

- ⚠️ **Concatenating PE to the embedding instead of adding it.** The standard formulation adds: input shape stays `(n, d_model)`. Concatenation changes the model width and is not what the paper does.
- ⚠️ **Assuming sinusoidal PE gives free length generalization.** It is *defined* for any position, but models trained on short sequences typically degrade at much longer ones; extrapolation "in principle" is not extrapolation in practice.
- ⚠️ **Applying RoPE to the value vectors.** RoPE rotates queries and keys only — the relative-position property comes from the $q^\top k$ dot product; rotating $V$ breaks nothing mathematically but is not the method and wastes the design.

## Industry Tips

- 💡 Modern LLM stacks treat position handling as a *context-length lever*: RoPE frequency scaling tricks are the usual route to extending a pretrained model's context window, which is why understanding $\theta_i$ matters beyond theory.
- 💡 When debugging a from-scratch Transformer that produces order-insensitive outputs, check the PE addition first — forgetting it is one of the most common implementation bugs, and the model still trains (poorly) without it.

## Real-World Use Cases

- Machine translation and any encoder-decoder Transformer (original sinusoidal scheme).
- BERT-style encoders and GPT-2 (learned positional embeddings with a fixed maximum length).
- Llama-family and most current open-weight LLMs (RoPE), where relative-position attention supports long-context chat and code modeling.

---

## Summary

- Self-attention is permutation-equivariant: without injected position information, all word orders are indistinguishable.
- Sinusoidal encoding stamps each position with multi-frequency sine/cosine waves — deterministic, added to embeddings, with relative offsets expressible as linear (rotation) transforms.
- Learned embeddings trade extrapolation for flexibility; RoPE rotates queries/keys so attention scores depend on relative position, and it powers most modern LLMs.

## Practice

- [ ] Exercises: [Module 6 Exercises](../exercises/README.md)
- [ ] Self-check: Why does the dot product of two RoPE-rotated vectors depend only on the *difference* of their positions?

## Further Reading

- 📑 Attention Is All You Need — Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762)
- 📑 RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al., 2021 (https://arxiv.org/abs/2104.09864)
- 🌐 The Illustrated Transformer — Jay Alammar (https://jalammar.github.io/illustrated-transformer/)
- 🌐 The Annotated Transformer — Harvard NLP (https://nlp.seas.harvard.edu/annotated-transformer/)
- 📘 Dive into Deep Learning (https://d2l.ai/)

## Related

- [Self-Attention](self-attention.md)
- [The Transformer Architecture](transformer-architecture.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
