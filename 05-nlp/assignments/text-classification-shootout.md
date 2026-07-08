---
title: "Assignment: Text-Classification Shoot-Out"
description: Compare bag-of-words, TF-IDF, and sentence-embedding representations on one dataset with honest evaluation.
type: assignment
domain: 05-nlp
tags: [nlp, text-classification, tf-idf, embeddings, evaluation]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 4–6 hours
covers:
  - 05-nlp/lessons/text-preprocessing
  - 05-nlp/lessons/text-representation
  - 05-nlp/lessons/sentence-embeddings
  - 05-nlp/lessons/text-classification
  - 05-nlp/lessons/nlp-evaluation-metrics
---

# Assignment: Text-Classification Shoot-Out

> A larger, assessed task that integrates multiple concepts from
> **[Module 5](../README.md)**.

---

## Context

"Which representation should I use?" is the first question of every applied NLP
project. You will answer it empirically: hold the dataset, splits, and classifier
constant, and vary **only** the text representation.

## Objectives

- Run a controlled comparison of BoW, TF-IDF, and sentence embeddings.
- Keep preprocessing/vectorization leak-free inside pipelines.
- Evaluate with the right metrics and read the errors.

---

## Tasks

1. **Dataset & protocol** — Pick a public labeled text dataset (topic or
   sentiment, ≥2 classes). Fix one stratified train/test split and one CV scheme
   up front. See [Text Classification](../lessons/text-classification.md).
2. **Contender A: Bag of Words** — `CountVectorizer` + logistic regression in a
   `Pipeline`; tune only n-gram range via CV.
3. **Contender B: TF-IDF** — `TfidfVectorizer` + the same classifier and tuning
   budget. See [Classical Text Representation](../lessons/text-representation.md).
4. **Contender C: Sentence embeddings** — `sentence-transformers` encodings +
   the same classifier. See [Sentence Embeddings](../lessons/sentence-embeddings.md).
5. **Verdict** — Report macro-F1 (CV mean ± std, then one test-set run each),
   a confusion matrix for the winner, and 10 misclassified examples with your
   analysis of *why*. Discuss cost/latency differences qualitatively.

## Constraints

- Same classifier, splits, and tuning budget across contenders — only the
  representation varies.
- All fitting inside `Pipeline`s; test set touched once per contender.
- Macro-F1 (not accuracy) as the headline metric; justify if you deviate.

---

## Deliverables

- [ ] One script/notebook producing the full comparison table.
- [ ] Confusion matrix + error analysis for the best model.
- [ ] A short write-up: which representation won, where, and why.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Experimental control | 35% | Identical protocol across contenders; no leakage. |
| Evaluation | 25% | Correct metrics, CV variability reported, honest test use. |
| Error analysis | 25% | Misclassifications examined and explained. |
| Communication | 15% | Clear table, reproducible code. |

---

## Submission

Push to a branch `add/module-5-classification-shootout` and open a pull request.

## Further Reading

- Speech and Language Processing — Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/)
- [scikit-learn text features](https://scikit-learn.org/stable/)

---

## Navigation

- ⬆️ [Module 5 Assignments](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
