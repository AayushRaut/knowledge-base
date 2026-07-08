---
title: "Debugging: Fix the Train/Inference Mismatch"
description: A model that scored well offline performs terribly in production — the preprocessing differs.
type: exercise
domain: 05-nlp
tags: [nlp, debugging, preprocessing, production]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 05-nlp/lessons/text-preprocessing
  - 05-nlp/lessons/text-classification
---

# Debugging: Fix the Train/Inference Mismatch

> Practice for **[Text Preprocessing](../lessons/text-preprocessing.md)**.

---

## Problem

A sentiment model was trained on cleaned text but serves raw text in production.
Offline F1 was strong; live performance is terrible. Spot every mismatch.

```python
# ---------- training (offline) ----------
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z' ]", " ", text)      # strip punctuation/digits
    return re.sub(r"\s+", " ", text).strip()

train_texts = [clean(t) for t in raw_train_texts]
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(train_texts, y_train)
joblib.dump(model, "sentiment.joblib")

# ---------- serving (production) ----------
model = joblib.load("sentiment.joblib")

def predict(text: str) -> int:
    return int(model.predict([text])[0])       # BUG: raw text goes straight in
```

## Requirements

- [ ] Explain exactly why live inputs mismatch training inputs.
- [ ] Fix it so preprocessing is *guaranteed* identical in both places.
- [ ] Say how you'd catch this class of bug with a test.

---

## Hints

<details>
<summary>Hint 1</summary>

At serving time, "GREAT product!! https://bit.ly/x" keeps its case, URL, and
punctuation — the vectorizer's vocabulary was built from cleaned tokens, so most
live tokens simply don't match.

</details>

<details>
<summary>Hint 2</summary>

Either ship `clean()` inside the model artifact (custom transformer /
`TfidfVectorizer(preprocessor=clean)`) or call the same imported function in
`predict`. One source of truth.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
# One source of truth: bake the cleaner INTO the pipeline itself.
model = make_pipeline(
    TfidfVectorizer(preprocessor=clean),   # clean() runs inside the artifact
    LogisticRegression(max_iter=1000),
)
model.fit(raw_train_texts, y_train)        # fit on RAW text — cleaner is inside
joblib.dump(model, "sentiment.joblib")

# ---------- serving ----------
model = joblib.load("sentiment.joblib")

def predict(text: str) -> int:
    return int(model.predict([text])[0])   # raw text is now CORRECT input
```

**Why it broke:** the model's vocabulary and weights assume lowercased,
de-punctuated, URL-free tokens. Raw production text ("GREAT", "product!!",
URLs) tokenizes differently, so most features are unseen and predictions
collapse toward the majority class. **The fix's key idea:** preprocessing
belongs *inside* the persisted artifact (or a single shared function), so
train/serve can't drift.

**The test that catches it:** a parity test — assert
`served_pipeline.predict([raw]) == offline_pipeline.predict([raw])` for a fixed
set of raw strings, run in CI. (The same idea as train/serve-skew tests in the
[image-classification service](../../18-projects/intermediate/image-classification-service.md).)

</details>

---

## Navigation

- ⬆️ [Module 5 Exercises](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
