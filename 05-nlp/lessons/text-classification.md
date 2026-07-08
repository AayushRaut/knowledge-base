---
title: Text Classification
description: The classical TF-IDF plus linear-classifier recipe for text classification, when to reach for embeddings or fine-tuning instead, and how to evaluate honestly.
type: lesson
domain: 05-nlp
tags: [nlp, text-classification, scikit-learn]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 40 min
prerequisites:
  - 05-nlp/lessons/text-representation
---

# Text Classification

> **TL;DR:** Text classification assigns a label (spam, topic, intent, toxicity) to a piece of text. The classical recipe — preprocess → TF-IDF → linear classifier — remains a strong, fast, interpretable baseline; embeddings and transformer fine-tuning earn their extra cost mainly on short texts, semantic nuance, and small label budgets.

---

## Overview

Text classification is the most common NLP task in industry: is this email spam, which department should this ticket go to, is this review positive, does this comment violate policy? You already know how to turn text into vectors; this lesson turns those vectors into decisions. You will build the classical scikit-learn pipeline, learn when the embedding-based and transformer-based recipes beat it, and — most importantly — learn to evaluate in a way that survives class imbalance.

**By the end, you will be able to:**
- Build a leak-free scikit-learn `Pipeline` (TF-IDF + linear classifier) and validate it with cross-validation
- Choose between the classical, embedding-based, and fine-tuning recipes based on data size, text length, and semantics
- Evaluate with macro-F1 and a confusion matrix, handle class imbalance, and debug by reading misclassified examples

---

## Intuition

How would you sort mail into "spam" and "not spam" by hand? You would look for telltale words: "winner", "free", "click here" push one way; a colleague's name or your project's jargon push the other. A linear classifier over TF-IDF vectors is exactly this intuition made mathematical: every word gets a learned weight (spam-ish positive, ham-ish negative), and the decision is a weighted vote of the words present.

This is why the classical recipe is hard to beat on many tasks: when the label is signaled by *which words appear* — topics, spam, most intents — a bag of weighted words captures nearly all the available signal. The recipe struggles when the signal lives *between* the words: sarcasm, negation ("not bad at all"), or two differently-worded texts meaning the same thing. That is the boundary where embeddings and fine-tuned transformers start paying for themselves.

---

## Details

### The task and its flavors

- **Binary** — spam vs ham, toxic vs clean
- **Multiclass** — one topic out of many (sports / politics / tech), one intent out of a bot's intent set
- **Multilabel** — a document can carry several tags at once (a support ticket that is both "billing" and "urgent")

All share the same shape: text in, label(s) out, learned from labeled examples.

### Recipe 1 — the classical baseline: TF-IDF + linear model

```text
raw text → (light) preprocessing → TF-IDF vectors → linear classifier → label
```

- **Preprocess lightly.** Lowercasing usually helps; aggressive stopword removal and stemming often do not — let TF-IDF down-weight common words instead.
- **Vectorize with TF-IDF** (see [Classical Text Representation](text-representation.md)). Word unigrams + bigrams (`ngram_range=(1, 2)`) capture short phrases like "not good".
- **Classify with a linear model.** **Multinomial Naive Bayes** is a fast, surprisingly strong baseline for word-count features. **Logistic Regression** usually edges it out, gives calibrated-ish probabilities, and its coefficients are directly interpretable (top positive/negative words per class). Linear SVMs are a comparable alternative.

Why linear models on sparse high-dimensional text? TF-IDF vectors have tens of thousands of dimensions, and in high dimensions classes are very often close to linearly separable. Linear models train in seconds, need little tuning, and resist overfitting with simple regularization. Always build this baseline first — it is your yardstick for anything fancier.

### Leak-free pipelines: fit on train only

The vectorizer *learns* from data (vocabulary, IDF weights). Fitting it on all your data before splitting leaks test-set statistics into training and inflates scores. scikit-learn's `Pipeline` solves this structurally: inside cross-validation, each fold fits the vectorizer on that fold's training portion only.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("model", LogisticRegression(max_iter=1000)),
])
# clf.fit(train_texts, train_labels) — the vectorizer never sees test data
```

### Handling class imbalance

Real datasets are lopsided — 2% spam, 0.5% toxic. Countermeasures, in the order to try them:

1. **Weight the classes**: `LogisticRegression(class_weight="balanced")` penalizes minority-class errors more. Cheapest fix, often sufficient.
2. **Pick the right metric** (below) — otherwise you cannot even *see* the problem.
3. **Tune the decision threshold** on predicted probabilities instead of accepting the 0.5 default, choosing the precision/recall trade-off the product needs.
4. **Resample** (oversample the minority or undersample the majority) — apply only to the *training* split, never to evaluation data.
5. **Get more minority-class labels** — frequently the highest-leverage move of all.

### Evaluation that doesn't lie

- **Accuracy misleads under imbalance**: with 2% spam, "always ham" scores 98%.
- **Macro-F1** averages per-class F1 with equal weight per class, so a model that ignores the minority class scores poorly — the right default headline metric for imbalanced problems.
- **Confusion matrix** shows *which* classes get confused with which — with 10 intents, discovering that two specific intents absorb most errors tells you exactly where to act.
- **Read misclassified examples.** Non-negotiable. Misclassifications reveal label noise, leakage, preprocessing bugs, and genuinely hard cases — things no aggregate metric shows. Budget time for it in every iteration.

For the general theory of these metrics, see [Model Evaluation Metrics](../../03-machine-learning/lessons/model-evaluation-metrics.md).

### Recipe 2 — sentence embeddings + classifier

Encode each text into a dense vector with a pretrained sentence-embedding model, then train a small classifier (logistic regression works fine) on those vectors. This recipe tends to win when:

- **Texts are short** — a one-line chat message gives TF-IDF almost no words to vote with, but an embedding still carries its meaning.
- **Semantics matter more than vocabulary** — "my parcel never arrived" and "still waiting on the delivery" share no content words but land near each other in embedding space.
- **Labels are few** — hundreds rather than tens of thousands of examples, because the pretrained encoder already did most of the representation learning.

Costs: an embedding model to run and serve, and the loss of per-word interpretability.

### Recipe 3 — transformer fine-tuning (the next step)

Fine-tuning a pretrained transformer with a classification head — updating the whole network on your labeled data — is generally the accuracy ceiling for text classification, especially for subtle distinctions (implicit toxicity, sarcasm, fine-grained intents). It costs GPU training, slower inference, and more MLOps surface. The standard tooling lives in the [Hugging Face docs](https://huggingface.co/docs); the mechanics are covered in [Module 11 — Fine-Tuning](../../11-fine-tuning/README.md). The pragmatic path: baseline with Recipe 1, and escalate only when the measured gap justifies the cost.

## Worked Example

An end-to-end pipeline with cross-validation on a small intent-classification dataset.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Toy dataset — in practice load hundreds+ of labeled examples per class.
texts: list[str] = [
    "I want my money back for this order",
    "refund the charge on my card please",
    "this purchase was a mistake, please reverse it",
    "I was billed twice for the same item",
    "where is my package",
    "my order still has not arrived",
    "tracking says delivered but nothing came",
    "how long does shipping take",
    "how do I reset my password",
    "I cannot log into my account",
    "the app keeps crashing on startup",
    "the website shows an error when I check out",
]
labels: list[str] = ["refund"] * 4 + ["delivery"] * 4 + ["tech-support"] * 4

clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

# Cross-validation: each fold re-fits the whole pipeline — no leakage.
scores = cross_val_score(clf, texts, labels, cv=3, scoring="f1_macro")
print(f"macro-F1: {scores.mean():.2f} (+/- {scores.std():.2f})")

# A held-out split for the final report and error reading.
X_tr, X_te, y_tr, y_te = train_test_split(
    texts, labels, test_size=0.25, stratify=labels, random_state=0
)
clf.fit(X_tr, y_tr)
preds = clf.predict(X_te)
print(classification_report(y_te, preds))
print(confusion_matrix(y_te, preds))

# The debugging habit that matters most: read the misses.
for text, gold, pred in zip(X_te, y_te, preds):
    if gold != pred:
        print(f"MISS: {text!r}  gold={gold}  pred={pred}")
```

With 12 examples the scores are noise — the point is the *shape*: pipeline, cross-validated macro-F1, per-class report, confusion matrix, then eyes on the misses. Swap in a real dataset and the workflow is unchanged.

## Best Practices

- ✅ Always build the TF-IDF + logistic regression baseline first; measure everything fancier against it.
- ✅ Put every learned step (vectorizer included) inside a `Pipeline` so cross-validation and deployment use identical, leak-free processing.
- ✅ Use `stratify=` in `train_test_split` so class proportions survive the split.
- ✅ Report macro-F1 and the confusion matrix, not accuracy alone.
- ✅ Inspect `LogisticRegression` coefficients per class — the top-weighted n-grams are a free sanity check on what the model learned.

## Common Mistakes

- ⚠️ **Fitting the vectorizer on all data before splitting.** Test-set vocabulary and IDF statistics leak into training. Fix: pipeline + cross-validation, as above.
- ⚠️ **Celebrating high accuracy on imbalanced data.** "Always majority class" scores high too. Fix: macro-F1, per-class recall, confusion matrix.
- ⚠️ **Skipping error analysis.** Aggregates hide label noise and systematic failure modes. Fix: read a sample of misclassifications every iteration.
- ⚠️ **Jumping straight to transformer fine-tuning.** You pay GPU and latency costs before knowing whether a 30-second baseline was already within a point or two. Fix: baseline first, escalate on evidence.
- ⚠️ **Oversampling before splitting.** Duplicated minority examples land in both train and test, inflating scores. Fix: resample the training split only.

## Industry Tips

- 💡 In production, most "model" wins come from the data: fixing mislabeled examples and adding coverage for confused classes usually beats swapping algorithms.
- 💡 Latency and cost keep linear TF-IDF models in production far longer than papers suggest — a logistic regression serves thousands of requests per second on a CPU.
- 💡 Log low-confidence predictions and route them to human review; the reviewed items become your next batch of high-value training labels.

## Real-World Use Cases

- Spam and phishing filtering in email systems
- Support-ticket routing and chatbot intent detection
- Content moderation: toxicity, policy-violation, and NSFW text flagging
- Tagging news articles and product reviews by topic or department

---

## Summary

- Text classification maps text to labels; the classical recipe (preprocess → TF-IDF → Naive Bayes / logistic regression) is fast, interpretable, and remains the baseline every alternative must beat.
- scikit-learn's `Pipeline` prevents train/test leakage by re-fitting the vectorizer inside each fold; class imbalance is handled with class weights, threshold tuning, and metrics that see minority classes.
- Escalate to sentence embeddings (short texts, semantic paraphrase, few labels) or transformer fine-tuning (accuracy ceiling, subtle distinctions) only when the measured gap over the baseline justifies the cost — and always evaluate with macro-F1, a confusion matrix, and your own eyes on the misclassified examples.

## Practice

- [ ] Exercises: [Module 5 Exercises](../exercises/README.md)
- [ ] Self-check: Your toxicity classifier reports 97% accuracy on a dataset that is 3% toxic. What single number do you ask for next, and what result would prove the model is useless?

## Further Reading

- 📘 Speech and Language Processing — Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/)
- 📄 [scikit-learn documentation](https://scikit-learn.org/stable/)
- 📄 [Hugging Face documentation](https://huggingface.co/docs)
- 📄 [spaCy documentation](https://spacy.io/)

## Related

- [Classical Text Representation](text-representation.md)
- [Sentiment Analysis](sentiment-analysis.md)
- [Model Evaluation Metrics — Module 3 Machine Learning](../../03-machine-learning/lessons/model-evaluation-metrics.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
