---
title: NLP Evaluation Metrics
description: Choose and compute the right metric for each NLP task — F1 variants, entity-level F1, BLEU, ROUGE, perplexity — and understand what each one hides.
type: lesson
domain: 05-nlp
tags: [nlp, evaluation, bleu, rouge, perplexity]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 40 min
prerequisites:
  - 05-nlp/lessons/sequence-to-sequence-tasks
---

# NLP Evaluation Metrics

> **TL;DR:** Classification tasks reuse precision/recall/F1; generation tasks need overlap metrics — BLEU for translation, ROUGE for summarization — and language models use perplexity. Every automatic metric is a proxy: it can rank systems, but only reading outputs tells you if they're good.

---

## Overview

Evaluating a classifier is easy: the prediction is right or wrong. Evaluating generated text is not — "The cat sat on the mat" and "A cat was sitting on the mat" are both fine translations, yet differ in every position. This lesson tours the standard metrics per task family, works through the math, implements two by hand, and ends with the golden rule that no metric replaces reading outputs.

**By the end, you will be able to:**
- Compute macro vs micro F1 and explain when they diverge, including entity-level F1 for NER.
- Explain how BLEU (n-gram precision + brevity penalty) and ROUGE (n-gram/LCS recall) work and where each fails.
- Interpret perplexity as exponentiated cross-entropy and know when to reach for embedding-based or LLM-as-judge evaluation.

---

## Intuition

Grading a math quiz is mechanical: compare to the answer key. Grading an essay is not — there is no single correct essay, only better and worse ones. NLP evaluation spans both worlds:

- **Classification** (sentiment, topic, NER) is the math quiz: one gold label, count matches.
- **Generation** (translation, summarization, open-ended QA) is the essay: **many valid outputs** exist, so we settle for measuring *overlap with reference outputs* written by humans — a useful but imperfect stand-in for quality.

Every generation metric in this lesson is some flavor of "how much does the system output overlap with the reference(s)?" Keep asking: *what kind of overlap, and what does it miss?*

---

## Details

### Classification: precision, recall, F1 — macro vs micro

You already know precision $P$, recall $R$, and $F1 = 2PR/(P+R)$ from [Model Evaluation Metrics](../../03-machine-learning/lessons/model-evaluation-metrics.md). With multiple classes there are two ways to aggregate:

- **Macro-F1:** compute F1 per class, then average. Every class counts equally — rare classes matter.
- **Micro-F1:** pool all decisions into global counts of true positives, false positives, and false negatives, then compute F1 once. Every *instance* counts equally — dominated by frequent classes. (For single-label multiclass classification, micro-F1 equals accuracy.)

**Tiny worked calculation.** 100 test sentences: 95 positive, 5 negative. A lazy model predicts *positive* for everything.

- Positive class: $P = 95/100 = 0.95$, $R = 95/95 = 1.0$, $F1 \approx 0.974$.
- Negative class: $P = 0$ (no negative predictions), $R = 0/5 = 0$, $F1 = 0$.
- **Macro-F1** $= (0.974 + 0)/2 \approx 0.487$ — exposes the failure.
- **Micro-F1** $= 95/100 = 0.95$ — hides it completely.

On imbalanced data, report macro-F1 (and per-class scores).

### Entity-level F1 for NER

Named entity recognition is evaluated at the **entity level**, not the token level, with an **exact-match convention**: a predicted entity counts as correct only if *both* its span boundaries *and* its type match the gold entity exactly. Predicting `[New York]LOC` when the gold is `[New York City]LOC` scores as both a false positive and a false negative. This is strict — a system can be "almost right" everywhere and still score poorly — but it keeps comparisons unambiguous.

### BLEU — machine translation (Papineni et al., 2002)

BLEU compares a candidate translation against one or more references using **modified n-gram precision**: for each n-gram in the candidate, count it as matched at most as many times as it appears in any reference (this "clipping" stops a system from spamming "the the the..."). Precisions $p_n$ are computed for $n = 1$ to $4$ and combined as a geometric mean.

Because precision alone rewards ultra-short outputs (a one-word candidate that appears in the reference has perfect precision), BLEU multiplies by a **brevity penalty**:

$$\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right), \qquad \text{BP} = \begin{cases} 1 & c > r \\ e^{\,1 - r/c} & c \le r \end{cases}$$

where $p_n$ is the modified precision for n-grams of size $n$, $w_n$ are weights (uniform $1/N$, typically $N=4$), $c$ is the candidate length, and $r$ is the reference length.

**Honest limitations:** BLEU sees only surface n-grams — it gives no credit for synonyms or paraphrases, correlates with human judgment at the *corpus* level but poorly on single sentences, and scores are not comparable across different tokenizations. Use the `sacrebleu` library, which standardizes tokenization, for reportable scores.

### ROUGE — summarization (Lin, 2004)

ROUGE flips the orientation: where BLEU asks "how much of the *candidate* is in the reference?" (precision), ROUGE-N classically asks "how much of the *reference* is in the candidate?" (**recall**) — natural for summarization, where covering the reference's content is the point.

- **ROUGE-N:** n-gram recall — overlapping n-grams divided by total n-grams in the reference (ROUGE-1 unigrams, ROUGE-2 bigrams). Modern implementations also report precision and F1.
- **ROUGE-L:** based on the **longest common subsequence** (LCS) between candidate and reference — rewards in-order content matches without requiring them to be contiguous.

Same caveat as BLEU: an abstractive summary that paraphrases perfectly scores low; a summary that copies reference wording but garbles a key number scores high.

### Perplexity — language models

Perplexity measures how "surprised" a language model is by held-out text. For a token sequence $w_1, \dots, w_N$:

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i \mid w_{<i})\right)$$

The exponent is the average negative log-likelihood per token — exactly the **cross-entropy** of the model's predictions against the data, so $\text{PPL} = e^{H}$ where $H$ is cross-entropy in nats (see [Information Theory](../../02-mathematics-foundations/lessons/information-theory.md)). Interpretation: a perplexity of $k$ means the model is, on average, as uncertain as if choosing uniformly among $k$ tokens. Lower is better. Crucially, perplexities are **only comparable between models sharing the same tokenizer and test set**.

### Newer approaches: embeddings and LLM-as-judge

Overlap metrics miss meaning, so two newer families have emerged:

- **Embedding-based metrics** compare candidate and reference in a semantic vector space (e.g., BERTScore-style token-embedding matching), giving credit for paraphrase that BLEU/ROUGE deny.
- **LLM-as-judge** prompts a strong LLM to grade outputs against a rubric. Increasingly common because it can assess reference-free qualities (faithfulness, helpfulness) — but it is newer, imperfect, and carries its own biases (verbosity preference, self-preference, prompt sensitivity). Treat judge scores as another proxy to validate against humans, not as ground truth. This becomes central in the AI evaluation practices covered in [Module 12 — AI Engineering](../../12-ai-engineering/README.md).

### The golden rule

**Automatic metrics are proxies.** They are cheap, reproducible, and good for *ranking* similar systems during development. They cannot tell you a summary hallucinated a name or a translation flipped a negation while preserving n-gram overlap. Before shipping or reporting, **read a sample of outputs** — every serious NLP team does.

---

## Details — Python implementations

Implement ROUGE-1 and macro-F1 by hand to demystify them.

```python
from collections import Counter


def rouge_1(candidate: str, reference: str) -> dict[str, float]:
    """ROUGE-1 precision, recall, and F1 via unigram overlap with clipping."""
    cand = Counter(candidate.lower().split())
    ref = Counter(reference.lower().split())
    overlap: int = sum(min(cand[w], ref[w]) for w in cand)
    precision = overlap / max(sum(cand.values()), 1)
    recall = overlap / max(sum(ref.values()), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Unweighted mean of per-class F1 scores."""
    classes = sorted(set(y_true))
    f1s: list[float] = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)
```

For real projects use maintained libraries: Hugging Face's `evaluate` package wraps ROUGE, BLEU, and more, and `sacrebleu` is the standard for reportable BLEU scores. Hand-rolled metrics drift from reference implementations in tokenization details.

## Worked Example

Score a machine summary and sanity-check the numbers by hand.

```python
reference = "the council approved forty kilometres of protected bike lanes"
candidate = "the council approved new bike lanes across the city"

scores = rouge_1(candidate, reference)
print(scores)  # precision=0.556, recall=0.556, f1≈0.556

# Macro-F1 on the imbalanced example from the lesson:
y_true = ["pos"] * 95 + ["neg"] * 5
y_pred = ["pos"] * 100
print(f"macro-F1 = {macro_f1(y_true, y_pred):.3f}")  # 0.487
```

Walk through ROUGE-1 manually: reference has 9 unigrams; candidate has 9. Overlapping (clipped) unigrams are `the` (2, both texts), `council`, `approved`, `bike`, `lanes` — total 5. Recall $= 5/9 \approx 0.556$, precision $= 5/9 \approx 0.556$. Now notice what the score *missed*: the candidate dropped "forty kilometres" (the key fact) and added "across the city" (unsupported) — yet still scores a middling 0.56. That gap between the number and the judgment is exactly why you read outputs.

---

## Best Practices

- ✅ Match metric to task: macro-F1 for imbalanced classification, entity-level F1 for NER, BLEU (via `sacrebleu`) for MT, ROUGE for summarization, perplexity for LMs.
- ✅ Report per-class scores alongside any aggregate — aggregates hide class-level failures.
- ✅ Use multiple references for BLEU/ROUGE when available; single-reference scores punish valid paraphrases hardest.
- ✅ Fix tokenizer, test set, and metric implementation before comparing systems — then never change them mid-comparison.
- ✅ Pair every automatic metric with a small human-read sample of outputs on every evaluation run.

## Common Mistakes

- ⚠️ **Comparing perplexities across different tokenizers.** A model with a larger vocabulary computes probability over different units. Fix: only compare PPL with identical tokenizer and test data.
- ⚠️ **Reporting micro-F1 (accuracy) on imbalanced data.** It rewards majority-class predictors. Fix: macro-F1 plus a confusion matrix.
- ⚠️ **Treating a +0.5 BLEU gain as meaningful.** Small deltas are often noise from tokenization or test-set choice. Fix: standardize with `sacrebleu` and check significance across test sets.
- ⚠️ **Optimizing ROUGE directly for summarization quality.** Systems learn to copy reference-like phrasing rather than be faithful. Fix: add faithfulness checks (human or LLM-judge) alongside ROUGE.

## Industry Tips

- 💡 In production, task metrics get replaced or supplemented by *business* metrics — deflection rate, edit distance from human post-editors, thumbs-up rate. Design the bridge between offline metric and online outcome early.
- 💡 A fixed, versioned evaluation set that the whole team trusts is worth more than a sophisticated metric on an ever-changing sample.
- 💡 LLM-as-judge pipelines need their own evaluation: periodically audit judge decisions against human labels before trusting trend lines.

## Real-World Use Cases

- Model selection: ranking candidate summarization checkpoints by ROUGE before a smaller human evaluation of the top two.
- MT vendor comparison: standardized `sacrebleu` scores across providers on a held-out in-domain test set.
- LM training: tracking validation perplexity across pretraining checkpoints to detect regressions.
- NER system audits: entity-level F1 per entity type to find which types (e.g., ORG vs PERSON) drag quality down.

---

## Summary

- Classification reuses P/R/F1 — macro treats classes equally, micro treats instances equally; on imbalanced data they diverge sharply, and NER adds an exact-match, entity-level convention.
- Generation metrics measure overlap with references: BLEU (clipped n-gram precision × brevity penalty) for translation, ROUGE (n-gram/LCS recall) for summarization — both blind to paraphrase and faithfulness; perplexity is exponentiated cross-entropy for language models.
- Embedding-based and LLM-as-judge evaluation address semantic blindness but are newer and imperfect; all automatic metrics are proxies — the golden rule is to read outputs.

## Practice

- [ ] Exercises: [Module 5 Exercises](../exercises/README.md)
- [ ] Self-check: A summarizer scores ROUGE-1 recall of 0.9 but invented a statistic not in the source. Which property did ROUGE fail to measure, and what would you add to catch it?

## Further Reading

- 📘 Speech and Language Processing — Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/)
- 📄 [Hugging Face documentation](https://huggingface.co/docs)
- 📄 BLEU: a Method for Automatic Evaluation of Machine Translation — Papineni et al. (2002)
- 📄 ROUGE: A Package for Automatic Evaluation of Summaries — Lin (2004)

## Related

- [Sequence-to-Sequence Tasks](sequence-to-sequence-tasks.md)
- [Sentiment Analysis](sentiment-analysis.md)
- [Information Theory](../../02-mathematics-foundations/lessons/information-theory.md)
- [Model Evaluation Metrics](../../03-machine-learning/lessons/model-evaluation-metrics.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
