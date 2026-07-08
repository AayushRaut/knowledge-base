---
title: Sentiment Analysis
description: Detect the emotional polarity of text with lexicon-based, classical ML, and transformer approaches — and know when each one breaks.
type: lesson
domain: 05-nlp
tags: [nlp, sentiment-analysis, classification]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 35 min
prerequisites:
  - 05-nlp/lessons/text-classification
---

# Sentiment Analysis

> **TL;DR:** Sentiment analysis classifies the emotional polarity of text (positive, negative, neutral). Lexicon tools like VADER need no training data, TF-IDF + logistic regression fits your domain cheaply, and transformer pipelines handle context best — pick based on data, budget, and how tricky your text is.

---

## Overview

Sentiment analysis is text classification specialized for *opinion*: given a review, tweet, or support ticket, decide whether the author feels positive, negative, or neutral. It looks simple — count happy words? — but negation, sarcasm, and domain-specific vocabulary make it one of the best case studies in why context matters in NLP.

**By the end, you will be able to:**
- Distinguish document-, sentence-, and aspect-level sentiment analysis and choose an appropriate label scheme.
- Compare lexicon-based (VADER), classical ML (TF-IDF + logistic regression), and transformer approaches with their trade-offs.
- Evaluate a sentiment model correctly with macro-F1 and diagnose the classic failure cases.

---

## Intuition

Imagine grading restaurant reviews by hand. Your first instinct is a mental word list: "delicious" is good, "awful" is bad. That is a **lexicon**. It works until you hit:

- *"The food was **not** bad at all."* — negation flips polarity.
- *"Oh great, another hour in the queue."* — sarcasm inverts the surface words.
- *"The plot was unpredictable."* — great for a movie, terrifying for a car's steering.

So sentiment lives on a ladder of sophistication: count polar words → learn domain-specific weights from labeled examples → read the whole sentence in context. Each rung costs more (data, compute) and fixes failures of the rung below. Your job as an engineer is to pick the *lowest rung that meets your accuracy bar*.

---

## Details

### Levels of analysis

| Level | Question answered | Example |
|-------|-------------------|---------|
| **Document** | What is the overall polarity of this text? | A product review is 4/5 positive |
| **Sentence** | What is the polarity of each sentence? | "Battery is great. Screen is dim." → +, − |
| **Aspect-based (ABSA)** | What is the polarity *toward each aspect*? | battery → positive, screen → negative |

Aspect-based sentiment analysis (ABSA) is the most useful in industry — a dashboard saying "customers love shipping speed but hate packaging" beats a single average score — and the hardest, because it couples aspect *extraction* with polarity *classification*.

### Label design

Decide the label space **before** collecting data:

- **Binary** (positive/negative) — simplest, but forces neutral text into a class it does not belong to.
- **3-class** (positive/neutral/negative) — the common default; neutral is genuinely frequent in real data.
- **Ordinal ratings** (1–5 stars) — richer, but adjacent classes (3 vs 4 stars) are noisy even for human annotators; consider ordinal regression rather than plain classification.

### Approach 1 — Lexicon-based: VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner; Hutto & Gilbert, 2014) ships with NLTK. It combines a human-curated lexicon of ~7,500 scored terms with heuristic rules: intensifiers ("**very** good" scores higher), negation within a window ("**not** good" flips), capitalization, punctuation ("great!!!"), and emoticons. It was tuned on social-media text.

- **Strengths:** zero training data, fast, interpretable, decent on short informal text.
- **Weaknesses:** no domain adaptation ("unpredictable" has one fixed score), blind to sarcasm, degrades on long formal documents.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# One-time: import nltk; nltk.download("vader_lexicon")
analyzer = SentimentIntensityAnalyzer()
scores: dict[str, float] = analyzer.polarity_scores("The food was not bad at all!")
print(scores)  # {'neg': ..., 'neu': ..., 'pos': ..., 'compound': ...}
```

The `compound` score is a normalized value in $[-1, 1]$; a common convention is positive if $\geq 0.05$, negative if $\leq -0.05$, neutral otherwise.

### Approach 2 — Classical ML: TF-IDF + logistic regression

When you have labeled in-domain data, a linear model over TF-IDF features is a strong, cheap baseline (see [Text Classification](text-classification.md)). It *learns* that "unpredictable" is positive in your movie-review corpus — something no general lexicon can know.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

texts: list[str] = ["loved it", "terrible acting", "an unpredictable, thrilling plot"]
labels: list[int] = [1, 0, 1]  # 1 = positive, 0 = negative

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),  # bigrams capture "not good"
    LogisticRegression(max_iter=1000),
)
model.fit(texts, labels)
print(model.predict(["the plot was unpredictable"]))  # [1]
```

Bigrams (`ngram_range=(1, 2)`) partially rescue negation: "not good" becomes its own feature with its own learned weight.

### Approach 3 — Transformers

A fine-tuned transformer reads the whole sequence with self-attention, so "not bad at all" is represented *in context* rather than as a bag of words. This is the strongest approach — and the costliest to serve.

```python
from transformers import pipeline

clf = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
print(clf("The food was not bad at all!"))
# [{'label': 'POSITIVE', 'score': ...}]
```

Always pin the model name explicitly — relying on the pipeline default makes results irreproducible when defaults change.

### The hard cases

- **Negation scope:** "I don't think the ending was good, but the start was" — the negation covers only part of the sentence. Bag-of-words models mangle this; transformers usually get it.
- **Sarcasm and irony:** "What a *fantastic* way to lose my luggage." Surface words say positive; pragmatics say negative. All approaches struggle; sarcasm remains an open research problem.
- **Comparative sentences:** "X is better than Y" is positive toward X and negative toward Y — a document-level label loses this entirely; you need aspect/entity-level analysis.
- **Domain-specific polarity:** "unpredictable" (movies +, cars −), "small" (phones +, portions −). This is the single strongest argument for training on in-domain data.

### Evaluation

Sentiment datasets are usually **imbalanced** (reviews skew positive). Accuracy is misleading; use **macro-F1** — the unweighted mean of per-class F1 — so the rare negative class counts as much as the dominant positive class:

$$\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C} F1_c$$

where $C$ is the number of classes and $F1_c$ is the F1 score of class $c$. See [NLP Evaluation Metrics](nlp-evaluation-metrics.md) for a worked calculation.

---

## Worked Example

Compare VADER against a transformer on deliberately tricky sentences.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

tricky: list[str] = [
    "The food was not bad at all.",            # negation
    "Oh great, another delayed flight.",       # sarcasm
    "The plot was completely unpredictable.",  # domain polarity
]

vader = SentimentIntensityAnalyzer()
transformer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

for text in tricky:
    v = vader.polarity_scores(text)["compound"]
    t = transformer(text)[0]
    print(f"{text}\n  VADER compound: {v:+.3f} | Transformer: {t['label']} ({t['score']:.3f})\n")
```

What to expect when you run it:

1. **Negation** — VADER's heuristic flip usually handles "not bad" correctly; so does the transformer. Tie.
2. **Sarcasm** — "Oh great" contains a strong positive lexicon entry, so VADER tends to score it positive. The transformer may or may not catch it — sarcasm was rare in its training data (SST-2 movie reviews). Neither is reliable here.
3. **Domain polarity** — VADER has one fixed score for "unpredictable"; a model fine-tuned on movie reviews has learned the movie-domain meaning. The lesson: match training domain to deployment domain.

Run the script and *read the outputs* — the disagreement pattern tells you which failure modes matter for your data.

---

## Best Practices

- ✅ Start with VADER or a pinned transformer pipeline as a baseline *before* labeling data — measure the gap first.
- ✅ Include a **neutral** class unless you have verified your data is genuinely bipolar.
- ✅ Evaluate with macro-F1 and a per-class confusion matrix, never accuracy alone.
- ✅ Sample and read misclassified examples — sentiment errors cluster (sarcasm, negation, one domain term) and reading reveals the cluster.
- ✅ Fine-tune or train on in-domain data when domain vocabulary diverges from general English.

## Common Mistakes

- ⚠️ **Using VADER on formal long-form text.** It was tuned for social media; on legal or clinical text it drifts badly. Fix: benchmark on a labeled in-domain sample before trusting it.
- ⚠️ **Reporting accuracy on a 90%-positive dataset.** A majority-class predictor gets 90%. Fix: macro-F1 plus per-class recall.
- ⚠️ **Treating star ratings as ground-truth sentiment.** A 3-star review can contain scathing text. Fix: spot-check label quality; consider text-based relabeling for a subset.
- ⚠️ **Ignoring aspect structure.** Averaging "battery great, screen awful" to *neutral* destroys the signal. Fix: use aspect-based analysis when opinions target multiple facets.

## Industry Tips

- 💡 Most production "sentiment" systems are really ABSA pipelines: extract aspects, then classify polarity per aspect — that is what product teams actually consume.
- 💡 A calibrated *confidence threshold* with a human-review queue for low-confidence predictions often matters more than 2 extra points of F1.
- 💡 Monitor for **domain drift**: a model trained on 2023 reviews degrades as slang and product vocabulary shift. Re-evaluate on fresh labeled samples quarterly.

## Real-World Use Cases

- Brand monitoring: track polarity of social mentions over time and alert on negative spikes.
- Customer support triage: route angry tickets to senior agents first.
- Product analytics: aspect-based dashboards ("shipping +0.7, packaging −0.4") from review streams.
- Finance: news and filing sentiment as a feature in trading and risk models (with heavy domain adaptation).

---

## Summary

- Sentiment analysis operates at document, sentence, and aspect level; aspect-based is the most valuable and hardest.
- Three tiers: VADER (no data, heuristic), TF-IDF + logistic regression (cheap, domain-fit), transformers (contextual, strongest, costliest) — choose the lowest tier that meets your bar.
- Negation, sarcasm, comparatives, and domain-specific polarity are the recurring failure modes; evaluate with macro-F1 on imbalanced data and read your errors.

## Practice

- [ ] Exercises: [Module 5 Exercises](../exercises/README.md)
- [ ] Self-check: Why does "The plot was unpredictable" defeat a general-purpose lexicon but not a model trained on movie reviews?

## Further Reading

- 📘 Speech and Language Processing — Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/)
- 📄 [Hugging Face documentation](https://huggingface.co/docs)
- 📄 VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text — Hutto & Gilbert (2014)
- 📄 [NLTK documentation](https://www.nltk.org/)

## Related

- [Text Classification](text-classification.md)
- [NLP Evaluation Metrics](nlp-evaluation-metrics.md)
- [Model Evaluation Metrics](../../03-machine-learning/lessons/model-evaluation-metrics.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
