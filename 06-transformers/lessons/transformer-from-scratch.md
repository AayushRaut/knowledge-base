---
title: Building a Transformer from Scratch
description: Assemble a complete, runnable decoder-only (GPT-style) transformer in PyTorch — embeddings, pre-LN blocks, causal attention, the shifted cross-entropy objective, and a sampling loop.
type: lesson
domain: 06-transformers
tags: [transformers, implementation, pytorch]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 60 min
prerequisites:
  - 06-transformers/lessons/transformer-architecture
---

# Building a Transformer from Scratch

> **TL;DR:** You will build a minimal GPT-style decoder-only transformer in under 120 lines of PyTorch: token + positional embeddings, a stack of pre-LN blocks (causal multi-head attention, then a GELU feed-forward network, each wrapped in a residual), a final LayerNorm, and a linear head to vocabulary logits. Training is next-token cross-entropy with logits and targets shifted by one — the single most common implementation bug.

---

## Overview

Reading about transformers and implementing one are different skills. This lesson closes the gap: you assemble every component of a decoder-only transformer with explicit tensor-shape comments, count its parameters by hand, train it on a toy task, and sample from it. The reference points are Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — the canonical minimal GPT implementation — and, for the encoder-decoder equivalent, Harvard NLP's *The Annotated Transformer*.

**By the end, you will be able to:**
- Implement causal multi-head self-attention, pre-LN transformer blocks, and a full GPT-style model in PyTorch with correct shapes.
- Write the next-token training objective correctly, including the one-position shift between logits and targets.
- Implement autoregressive generation with temperature sampling.

---

## Intuition

A decoder-only transformer is a *next-token machine*. Feed it a sequence of $T$ token ids; it returns, **at every position** $t$, a probability distribution over what token $t{+}1$ should be. That "every position" is the key design win: one forward pass over a sequence of length $T$ yields $T$ training examples, because position 1 predicts token 2, position 2 predicts token 3, and so on. The causal mask is what makes this legal — it guarantees position $t$ never peeks at tokens after $t$, so each prediction is honest.

Everything else — embeddings, attention heads, feed-forward layers, LayerNorms — is machinery for making those $T$ predictions good. Build it bottom-up: attention head → block → stack → head-to-vocab.

---

## Details

### The configuration

A deliberately tiny config that trains in seconds on CPU:

| Hyperparameter | Symbol | Value |
|---|---|---|
| Vocabulary size | $V$ | 256 |
| Model width | $d_{model}$ | 64 |
| Attention heads | $h$ | 4 |
| Head dimension | $d_{head} = d_{model}/h$ | 16 |
| Layers | $L$ | 4 |
| FFN inner width | $d_{ff} = 4\,d_{model}$ | 256 |
| Max context length | $T_{max}$ | 128 |

### The full model

Every line carries its shape. `B` = batch, `T` = sequence length, `C` = $d_{model}$.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # fused Q, K, V projection
        self.proj = nn.Linear(d_model, d_model)      # output projection
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("mask", mask)           # (T_max, T_max), lower-triangular

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                                          # (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=2)                      # 3 x (B, T, C)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)     # (B, h, T, T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))     # causal mask
        att = F.softmax(att, dim=-1)                                 # rows sum to 1
        y = att @ v                                                  # (B, h, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)             # merge heads -> (B, T, C)
        return self.proj(y)                                          # (B, T, C)


class Block(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, max_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # (B, T, C) -> (B, T, 4C)
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),  # (B, T, 4C) -> (B, T, C)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual 1
        x = x + self.ffn(self.ln2(x))    # residual 2
        return x                          # (B, T, C)


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 4, max_len: int = 128) -> None:
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, d_model)     # (V, C)
        self.pos_emb = nn.Embedding(max_len, d_model)   # (T_max, C), learned positions
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, max_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)               # final LayerNorm
        self.lm_head = nn.Linear(d_model, vocab, bias=False)  # (C,) -> (V,)
        self.lm_head.weight = self.tok_emb.weight       # weight tying (common practice)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape                                 # (B, T) integer token ids
        pos = torch.arange(T, device=idx.device)         # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)        # (B, T, C) broadcast add
        for block in self.blocks:
            x = block(x)                                 # (B, T, C)
        x = self.ln_f(x)                                 # (B, T, C)
        return self.lm_head(x)                           # (B, T, V) logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]            # crop to context window
            logits = self(idx_cond)[:, -1, :]            # (B, V) — last position only
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)  # (B, 1) sample
            idx = torch.cat([idx, nxt], dim=1)           # (B, T+1)
        return idx
```

Notes on two deliberate choices:

- **Pre-LN** (LayerNorm *before* each sublayer) is used instead of the original paper's post-LN because it trains stably without a learning-rate warmup ritual — the convention GPT-2 popularized and nanoGPT follows.
- **Weight tying** shares the embedding matrix with the LM head. It saves $V \times d_{model}$ parameters and is standard practice in GPT-style models; for a 256-token toy vocab the saving is trivial, but at $V = 50{,}257$ (GPT-2's vocabulary) it is substantial.

### Parameter count, by hand

For the tiny config ($V{=}256$, $d_{model}{=}64$, $h{=}4$, $L{=}4$, $T_{max}{=}128$, $d_{ff}{=}256$):

| Component | Formula | Count |
|---|---|---|
| Token embedding | $V \cdot d_{model} = 256 \cdot 64$ | 16,384 |
| Positional embedding | $T_{max} \cdot d_{model} = 128 \cdot 64$ | 8,192 |
| Per block: QKV | $d_{model} \cdot 3d_{model} + 3d_{model}$ | 12,480 |
| Per block: attn output proj | $d_{model}^2 + d_{model}$ | 4,160 |
| Per block: 2 LayerNorms | $2 \cdot 2d_{model}$ | 256 |
| Per block: FFN | $(d_{model} d_{ff} + d_{ff}) + (d_{ff} d_{model} + d_{model})$ | 33,088 |
| Block total × 4 | $49{,}984 \cdot 4$ | 199,936 |
| Final LayerNorm | $2 d_{model}$ | 128 |
| LM head | tied — shares token embedding | 0 |
| **Total** | | **224,640** |

Verify: `sum(p.numel() for p in model.parameters())` returns 224,640.

### The training objective — and THE classic bug

The model outputs logits at every position; the target at position $t$ is the token at position $t{+}1$. The loss is average cross-entropy over all positions:

$$\mathcal{L} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P_\theta\!\left(x_{t+1} \mid x_{\le t}\right)$$

**You must shift by one.** Feed positions $1..T{-}1$ as input and use positions $2..T$ as targets. Forgetting the shift trains the model to *copy its input* — loss plummets toward zero, generation is garbage.

```python
def loss_fn(model: TinyGPT, idx: torch.Tensor) -> torch.Tensor:
    inputs, targets = idx[:, :-1], idx[:, 1:]   # THE shift: (B, T-1) each
    logits = model(inputs)                       # (B, T-1, V)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),     # (B*(T-1), V)
        targets.reshape(-1),                     # (B*(T-1),)
    )
```

## Worked Example

Train the model to continue a repeating byte pattern — a task tiny enough for CPU but real enough to prove every component works end to end:

```python
torch.manual_seed(0)
model = TinyGPT(vocab=256)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

pattern = torch.tensor(list(b"the quick brown fox jumps over the lazy dog. "))
data = pattern.repeat(20)  # (900,) byte-level "corpus"

for step in range(300):
    ix = torch.randint(0, len(data) - 65, (16,))                # 16 random offsets
    batch = torch.stack([data[i : i + 65] for i in ix])         # (16, 65)
    loss = loss_fn(model, batch)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step}: loss {loss.item():.3f}")

prompt = torch.tensor([list(b"the quick ")])                    # (1, 10)
out = model.generate(prompt, max_new_tokens=40, temperature=0.8)
print(bytes(out[0].tolist()).decode(errors="replace"))
```

Loss falls from around $\ln 256 \approx 5.5$ (uniform guessing over the byte vocabulary) toward near zero, and generation continues the sentence pattern. That initial-loss check — $\approx \ln V$ at step 0 — is a free correctness test: if your untrained loss is far from $\ln V$, something is miswired.

## Best Practices

- ✅ Comment the shape of every tensor at every step — shape bugs are the dominant failure mode, and `view`/`transpose` errors often fail *silently* by mixing data across heads.
- ✅ Sanity-check initial loss against $\ln V$ before training, and overfit a single small batch to near-zero loss before scaling up.
- ✅ Register the causal mask with `register_buffer` so it moves with `.to(device)` and is saved in `state_dict` without being a trainable parameter.
- ✅ Use `torch.manual_seed` during development so runs are reproducible while you debug.

## Common Mistakes

- ⚠️ **Forgetting the input/target shift** — the model learns the identity function and loss collapses deceptively fast. Fix: `inputs, targets = idx[:, :-1], idx[:, 1:]`, always.
- ⚠️ **Masking with 0 instead of $-\infty$** — a pre-softmax score of 0 still receives probability mass. Fix: `masked_fill(~mask, float("-inf"))` *before* softmax.
- ⚠️ **Wrong mask direction** — `torch.triu` instead of `torch.tril` lets every position attend to the *future*. Fix: keep the lower triangle; verify `mask[1, 0] == True` and `mask[0, 1] == False`.
- ⚠️ **Skipping `.contiguous()` after `transpose` before `view`** — PyTorch raises an error, or you reach for `reshape` and hide a layout mistake. Fix: `transpose(1, 2).contiguous().view(B, T, C)`.
- ⚠️ **Sampling from position 0's logits during generation** — you want the *last* position's prediction. Fix: `logits[:, -1, :]`.

## Industry Tips

- 💡 nanoGPT is the reference this lesson miniaturizes — read its `model.py` next; the structure maps one-to-one onto `TinyGPT`, plus production concerns (dropout, initialization scaling, flash attention via `F.scaled_dot_product_attention`).
- 💡 Real implementations replace the manual `softmax(QK^T/√d)V` with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, which fuses the operation and saves memory — write it manually once so you know what the fused kernel is doing.
- 💡 The naive `generate` loop above re-encodes the whole prefix every step; production inference caches K and V per layer (the "KV cache") so each new token costs one position of compute, not $T$.

## Real-World Use Cases

- Every GPT-family model — GPT-1 ("Improving Language Understanding by Generative Pre-Training", Radford et al., 2018) through today's LLMs — is architecturally this model scaled up: more layers, wider $d_{model}$, larger vocabulary and context.
- Character/byte-level toy models like this one are the standard vehicle for teaching and for debugging training infrastructure before burning money on large runs.
- Interview loops for ML engineering roles routinely ask candidates to sketch exactly this: causal attention, the loss shift, and a sampling loop.

---

## Summary

- A decoder-only transformer is embeddings → $L$ pre-LN blocks (causal attention + GELU FFN, each residual) → final LayerNorm → linear head to vocabulary logits; the toy config here has 224,640 parameters, countable by hand.
- The training objective is next-token cross-entropy over all positions simultaneously, and the logits/targets **one-position shift** is the classic bug to check first when loss behaves strangely.
- Generation is a loop: forward pass, take the last position's logits, divide by temperature, softmax, sample, append, repeat.

## Practice

- [ ] Exercises: [Module 6 Exercises](../exercises/README.md)
- [ ] Self-check: your untrained model's loss is 0.9 at step 0 with a 256-token vocabulary — what is almost certainly wrong, and why does $\ln V$ tell you?

## Further Reading

- 📑 [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — the original architecture this implementation adapts.
- 📑 "Improving Language Understanding by Generative Pre-Training" — Radford et al., 2018 (OpenAI) — the decoder-only GPT recipe.
- 📑 [nanoGPT — Andrej Karpathy](https://github.com/karpathy/nanoGPT) — the canonical minimal GPT implementation; see also his [YouTube channel](https://www.youtube.com/@AndrejKarpathy) for the build-it-from-scratch walkthroughs.
- 📄 [PyTorch documentation](https://pytorch.org/docs/stable/)
- 📄 [Hugging Face documentation](https://huggingface.co/docs)

## Related

- [The Transformer Architecture](transformer-architecture.md)
- [Architecture Variants](architecture-variants.md)
- [PyTorch Essentials](../../04-deep-learning/lessons/pytorch.md)
- [Char-Level Text Generator](../../04-deep-learning/projects/char-level-text-generator.md) — the RNN predecessor project this model supersedes.

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 6 — Transformers](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
