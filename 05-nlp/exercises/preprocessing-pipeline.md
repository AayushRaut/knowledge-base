---
title: "Exercise: Build a Preprocessing Pipeline"
description: Compose a reusable, testable text-cleaning pipeline from small pure functions.
type: exercise
domain: 05-nlp
tags: [nlp, preprocessing, pipeline]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 05-nlp/lessons/text-preprocessing
---

# Exercise: Build a Preprocessing Pipeline

> Practice for **[Text Preprocessing](../lessons/text-preprocessing.md)**.

---

## Problem

Build `preprocess(text)` from small pure steps: lowercase → strip URLs → strip
HTML tags → remove extra whitespace → tokenize on words → drop stopwords. Each
step must be independently testable, and the pipeline must be the *single* place
preprocessing lives (used identically at train and inference time).

```python
preprocess("Check <b>THIS</b> out!! https://x.co  it's great")
# -> ["check", "great"]        (with a tiny stopword list incl. "this", "out", "it's")
```

## Requirements

- [ ] Each step is a pure function `str -> str` (or `str -> list[str]` at the end).
- [ ] Steps compose in an explicit ordered list.
- [ ] Unit-testable: at least one assert per step.

---

## Hints

<details>
<summary>Hint 1</summary>

Regexes: URLs `r"https?://\S+"`, HTML tags `r"<[^>]+>"`, words `r"[a-z']+"`.

</details>

<details>
<summary>Hint 2</summary>

Compose with `functools.reduce`, or a simple loop over a list of functions.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import re

STOPWORDS = {"the", "a", "an", "this", "that", "it", "its", "it's", "out", "is"}


def lowercase(text: str) -> str:
    return text.lower()

def strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", " ", text)

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text)

def drop_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def preprocess(text: str) -> list[str]:
    """The ONE preprocessing entry point — used at train AND inference."""
    for step in (lowercase, strip_urls, strip_html, normalize_ws):
        text = step(text)
    return drop_stopwords(tokenize(text))


assert strip_urls("go https://x.co now") == "go   now"
assert strip_html("<b>hi</b>") == " hi "
assert preprocess("Check <b>THIS</b> out!! https://x.co  it's great") == ["check", "great"]
```

**Explanation:** Small pure steps are individually testable and reorderable, and
one canonical `preprocess()` guarantees train/inference symmetry — the property
whose absence causes the silent accuracy drops you'll debug in
[the mismatch exercise](debug-preprocessing-mismatch.md).

</details>

---

## Navigation

- ⬆️ [Module 5 Exercises](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
