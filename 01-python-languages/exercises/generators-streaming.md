---
title: "Exercise: Stream Batches with a Generator"
description: Build a memory-efficient generator that yields fixed-size batches from any iterable.
type: exercise
domain: 01-python-languages
tags: [python, generators, batching, data-pipeline]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 01-python-languages/lessons/iterators-and-generators
---

# Exercise: Stream Batches with a Generator

> Practice for **[Iterators and Generators](../lessons/iterators-and-generators.md)**.

---

## Problem

Training and inference loops rarely fit a whole dataset in memory. Write a
generator `batched(iterable, size)` that yields lists of up to `size` items,
consuming the source **lazily** (it must work on an infinite generator too).

```python
list(batched([1, 2, 3, 4, 5], 2))
# -> [[1, 2], [3, 4], [5]]
```

## Requirements

- [ ] Works on any iterable (list, generator, file object).
- [ ] Never holds more than `size` items in memory at once.
- [ ] The final partial batch is still yielded.
- [ ] Raises `ValueError` if `size < 1`.

---

## Hints

<details>
<summary>Hint 1</summary>

Get an iterator with `iter(iterable)` and pull items with `next(...)`, catching
`StopIteration`.

</details>

<details>
<summary>Hint 2</summary>

Python 3.12+ ships `itertools.batched`. Implement it yourself here, then compare.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield lists of up to `size` items, consuming `iterable` lazily."""
    if size < 1:
        raise ValueError("size must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, size)):  # islice pulls at most `size` items
        yield batch
```

**Explanation:** `islice(it, size)` advances the shared iterator by at most
`size` items each call, so memory stays bounded and infinite sources work. The
walrus operator (`:=`) assigns the batch and tests for emptiness in one step; an
empty list ends the loop, which also yields the final partial batch naturally.

</details>

---

## Navigation

- ⬆️ [Module 1 Exercises](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
