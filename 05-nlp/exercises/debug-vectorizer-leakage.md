---
title: "Debugging: Fix the Vectorizer Leakage"
description: Find why a text classifier's reported score is inflated — the vectorizer saw the test set.
type: exercise
domain: 05-nlp
tags: [nlp, debugging, tf-idf, data-leakage]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 05-nlp/lessons/text-classification
---

# Debugging: Fix the Vectorizer Leakage

> Practice for **[Text Classification](../lessons/text-classification.md)**.

---

## Problem

This text classifier reports a great score, but it degrades badly on new data.
The evaluation is dishonest — find the leakage and fix it.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_and_score(texts: list[str], labels: list[int]) -> float:
    # BUG: the vectorizer is fit on ALL texts — including the test set.
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=0, stratify=labels
    )
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    return clf.score(X_test, y_test)
```

## Requirements

- [ ] Explain what leaks (two things: vocabulary and IDF statistics).
- [ ] Fix it so the test set never influences fitting.
- [ ] Prefer a `Pipeline` so cross-validation is also safe.

---

## Hints

<details>
<summary>Hint 1</summary>

`fit_transform(texts)` learns the vocabulary **and** IDF weights from every
document — including the ones you'll later call "unseen."

</details>

<details>
<summary>Hint 2</summary>

Split the raw texts first; put the vectorizer and classifier in one `Pipeline`
fitted on training texts only.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def train_and_score(texts: list[str], labels: list[int]) -> float:
    # Split RAW texts first — test documents stay truly unseen.
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=0, stratify=labels
    )
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)     # vocabulary + IDF learned from train only
    return model.score(X_test, y_test)
```

**What leaked:** (1) the **vocabulary** — test-only words got columns, so at
"inference" the model had features it could never have in production, and
(2) the **IDF statistics** — document frequencies included test documents,
shifting every weight. Both make the test score optimistic. The `Pipeline` fix
also makes `cross_val_score`/`GridSearchCV` leak-free, since each fold refits
the vectorizer on its own training portion — the same principle as the
[scaler-leakage bug](../../03-machine-learning/exercises/debug-cv-leakage.md)
in Module 3, wearing NLP clothes.

</details>

---

## Navigation

- ⬆️ [Module 5 Exercises](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
