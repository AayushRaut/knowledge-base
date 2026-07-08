---
title: "Exercise: TF-IDF from Scratch"
description: Implement term frequency–inverse document frequency in pure Python and sanity-check against scikit-learn.
type: exercise
domain: 05-nlp
tags: [nlp, tf-idf, vectorization]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 05-nlp/lessons/text-representation
---

# Exercise: TF-IDF from Scratch

> Practice for **[Classical Text Representation](../lessons/text-representation.md)**.

---

## Problem

Implement TF-IDF for a small corpus using the textbook formula

$$
\text{tfidf}(t, d) = \text{tf}(t, d)\cdot\log\frac{N}{\text{df}(t)}
$$

where $\text{tf}(t,d)$ is the count of term $t$ in document $d$, $N$ the number
of documents, and $\text{df}(t)$ the number of documents containing $t$. Rank
each document's most distinctive terms.

## Requirements

- [ ] `tfidf(corpus: list[list[str]]) -> list[dict[str, float]]`.
- [ ] Terms appearing in **every** document get weight 0 (log 1 = 0).
- [ ] Show the top-2 terms per document on a 3-document toy corpus.
- [ ] Note (in a comment) how scikit-learn's smoothed formula differs.

---

## Hints

<details>
<summary>Hint 1</summary>

Compute `df` once with a `Counter` over each document's *set* of terms.

</details>

<details>
<summary>Hint 2</summary>

scikit-learn uses `idf = ln((1+N)/(1+df)) + 1` (smoothing, never zero) and then
L2-normalizes rows — so exact numbers differ; the *ranking* intuition is the same.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import math
from collections import Counter


def tfidf(corpus: list[list[str]]) -> list[dict[str, float]]:
    n_docs = len(corpus)
    df = Counter(term for doc in corpus for term in set(doc))
    out: list[dict[str, float]] = []
    for doc in corpus:
        tf = Counter(doc)
        out.append(
            {t: count * math.log(n_docs / df[t]) for t, count in tf.items()}
        )
    return out


corpus = [
    "the cat sat on the mat".split(),
    "the dog sat on the log".split(),
    "cats and dogs are pets".split(),
]
for i, weights in enumerate(tfidf(corpus)):
    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:2]
    print(f"doc {i}: {top}")
# "the"/"sat"/"on" score low or 0 (common); "cat", "mat", "dog", "pets" rank top.
# NOTE: sklearn's TfidfVectorizer uses idf = ln((1+N)/(1+df)) + 1 and L2-normalizes,
# so its numbers differ — but the distinctive-term ranking matches.
```

**Explanation:** TF rewards terms frequent *in this document*; IDF discounts
terms frequent *across all documents*. Their product surfaces what makes each
document distinctive — which is why TF-IDF remains a strong baseline for search
and classification decades after its invention.

</details>

---

## Navigation

- ⬆️ [Module 5 Exercises](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
