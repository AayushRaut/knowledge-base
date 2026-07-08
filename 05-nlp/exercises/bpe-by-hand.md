---
title: "Exercise: BPE Merges by Hand"
description: Run the byte-pair-encoding merge algorithm manually on a tiny corpus, then code it.
type: exercise
domain: 05-nlp
tags: [nlp, tokenization, bpe, subword]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 30 min
reinforces:
  - 05-nlp/lessons/tokenization
---

# Exercise: BPE Merges by Hand

> Practice for **[Tokenization](../lessons/tokenization.md)**.

---

## Problem

Byte-Pair Encoding builds a subword vocabulary by repeatedly merging the most
frequent adjacent symbol pair. For the toy corpus (with word frequencies):

```text
low : 5     lower : 2     newest : 6     widest : 3
```

Start from characters (plus an end-of-word marker `_`), perform the **first 4
merges by hand**, then implement the merge loop in Python and confirm it produces
the same merges.

## Requirements

- [ ] Show the pair counts and chosen merge for each of the first 4 steps.
- [ ] Implement `learn_bpe(corpus, num_merges)` returning the merge list.
- [ ] Code output matches your hand-computed merges.

---

## Hints

<details>
<summary>Hint 1</summary>

Represent words as symbol tuples: `l o w _`, `n e w e s t _`, … weighted by
frequency. Count adjacent pairs across all words.

</details>

<details>
<summary>Hint 2</summary>

`e s` appears in newest (6) + widest (3) = 9 times — a strong first-merge
candidate. After merging, recount pairs (`es t` now exists).

</details>

---

## Solution

<details>
<summary>Show solution</summary>

**By hand (one valid sequence; ties may be broken differently):**

1. `e s` (9: newest 6 + widest 3) → `es`
2. `es t` (9) → `est`
3. `est _` (9) → `est_`
4. `l o` (7: low 5 + lower 2) → `lo`

```python
from collections import Counter


def learn_bpe(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    words = {tuple(w) + ("_",): f for w, f in corpus.items()}
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs: Counter = Counter()
        for sym, freq in words.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)          # most frequent adjacent pair
        merges.append(best)
        merged = {}
        for sym, freq in words.items():           # apply the merge everywhere
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            merged[tuple(out)] = freq
        words = merged
    return merges


corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
print(learn_bpe(corpus, 4))
# [('e','s'), ('es','t'), ('est','_'), ('l','o')]
```

**Explanation:** BPE greedily compresses the corpus: frequent character
sequences become single vocabulary items (`est_` captures the shared suffix of
*newest/widest*), while rare words stay decomposable into pieces. This is why
subword tokenizers have no out-of-vocabulary problem — exactly the mechanism
inside GPT- and BERT-family tokenizers.

</details>

---

## Navigation

- ⬆️ [Module 5 Exercises](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
