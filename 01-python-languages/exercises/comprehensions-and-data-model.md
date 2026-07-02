---
title: "Exercise: Refactor Loops into Comprehensions"
description: Rewrite imperative loops as readable comprehensions and generator expressions.
type: exercise
domain: 01-python-languages
tags: [python, comprehensions, data-model]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 15 min
reinforces:
  - 01-python-languages/lessons/python-refresher
---

# Exercise: Refactor Loops into Comprehensions

> Practice for **[Python Refresher for AI Engineering](../lessons/python-refresher.md)**.

---

## Problem

You are given three imperative snippets from a data-prep script. Rewrite each
using a comprehension (list/dict/set) or a generator expression, without changing
behavior.

```python
# 1. Squares of even numbers 0..n-1
def even_squares(n):
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(i * i)
    return out

# 2. Map token -> length, skipping empty tokens
def token_lengths(tokens):
    result = {}
    for t in tokens:
        if t:
            result[t] = len(t)
    return result

# 3. Sum of squares over a large range WITHOUT building a list in memory
def sum_of_squares(n):
    total = 0
    values = [i * i for i in range(n)]  # wastes memory for large n
    for v in values:
        total += v
    return total
```

## Requirements

- [ ] `even_squares` uses a single list comprehension.
- [ ] `token_lengths` uses a dict comprehension with the same filter.
- [ ] `sum_of_squares` uses a **generator expression** so no intermediate list is built.

---

## Hints

<details>
<summary>Hint 1</summary>

A comprehension can carry an `if` filter: `[expr for x in it if cond]`.

</details>

<details>
<summary>Hint 2</summary>

`sum(...)` accepts any iterable, including a generator expression written without
brackets: `sum(x for x in it)`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
def even_squares(n: int) -> list[int]:
    return [i * i for i in range(n) if i % 2 == 0]


def token_lengths(tokens: list[str]) -> dict[str, int]:
    return {t: len(t) for t in tokens if t}


def sum_of_squares(n: int) -> int:
    # Generator expression: values are produced lazily, so memory stays O(1).
    return sum(i * i for i in range(n))
```

**Explanation:** Comprehensions express *what* you want rather than *how* to
accumulate it, which is both shorter and faster (the loop runs in optimized C).
The generator expression in `sum_of_squares` avoids materializing a list of `n`
integers — important when `n` is large, exactly the kind of streaming you want in
data pipelines.

</details>

---

## Navigation

- ⬆️ [Module 1 Exercises](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
