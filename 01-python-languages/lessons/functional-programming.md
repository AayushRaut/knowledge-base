---
title: Functional Programming in Python
description: First-class functions, purity, and the functools/itertools toolkit for building clear, composable data pipelines.
type: lesson
domain: 01-python-languages
tags: [python, functional, itertools]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 35 min
prerequisites:
  - 01-python-languages/lessons/python-refresher
---

# Functional Programming in Python

> **TL;DR:** Functions in Python are values you can pass, return, and compose; leaning on pure functions, `functools`, and `itertools` makes data-transformation code shorter, testable, and easy to reason about.

---

## Overview
Data pipelines are chains of transformations, and functional programming (FP) is a natural fit: each step is a small, pure function you can test and compose. Python is not a pure FP language, but it gives you first-class functions and a strong `functools`/`itertools` toolkit. This lesson shows the FP patterns worth using and when they beat a comprehension.

**By the end, you will be able to:**
- Use functions as first-class and higher-order values.
- Write pure functions and explain why immutability simplifies reasoning and testing.
- Apply `map`/`filter`/`sorted`, `functools`, and `itertools` to real data tasks.

---

## Intuition
Treat a function like any other value — a number you can hand to another function or return from one. A data pipeline then becomes a recipe: "take these records, apply clean, then map to features, then filter valid ones." Each verb is a small function; the pipeline is their composition. Pure functions (same input → same output, no side effects) make each step a black box you can trust and test in isolation.

---

## Details

### First-class and higher-order functions
Functions are objects: you can assign them, store them in lists, pass them as arguments, and return them. A higher-order function takes or returns a function.

```python
def make_scaler(factor: float):
    """Return a function that scales its input — a closure over `factor`."""
    def scale(x: float) -> float:
        return x * factor
    return scale

normalize = make_scaler(1 / 255)     # image pixel normalizer
print(normalize(128))                # 0.501...
```

### Pure functions and immutability
A pure function depends only on its arguments and produces no side effects (no mutation, no I/O). Pure functions are trivial to test, safe to cache, and safe to run in parallel. Prefer returning new values over mutating inputs.

```python
def add_bias(row: tuple[float, ...], bias: float) -> tuple[float, ...]:
    """Return a new row; the input tuple is never mutated."""
    return row + (bias,)
```

### map, filter, sorted with keys
`map` applies a function lazily to each item; `filter` keeps items where a predicate is true; `sorted` orders by a `key` function. The `key` argument is where FP shines in day-to-day code.

```python
records = [{"id": "a", "loss": 0.9}, {"id": "b", "loss": 0.2}]
by_loss = sorted(records, key=lambda r: r["loss"])          # ascending loss
losses = list(map(lambda r: r["loss"], records))
good = list(filter(lambda r: r["loss"] < 0.5, records))
```

### lambda
`lambda` creates a small anonymous function inline. Keep it to a single expression; if it grows or needs a name/docstring, use `def`.

### functools: reduce, partial, lru_cache
`reduce` folds a sequence into one value. `partial` pre-fills some arguments to specialize a function. `lru_cache` memoizes results of a pure function.

```python
from functools import reduce, partial, lru_cache

total = reduce(lambda acc, x: acc + x, [1, 2, 3, 4], 0)     # 10

def scale(factor: float, x: float) -> float:
    return x * factor
to_percent = partial(scale, 100)                            # factor fixed
print(to_percent(0.25))                                     # 25.0

@lru_cache(maxsize=None)
def token_cost(n_tokens: int) -> float:
    """Pure and deterministic, so caching is safe."""
    return n_tokens * 0.000002
```

### itertools highlights
`itertools` builds fast, memory-lazy iterators. The ones you reach for most in data work:

```python
from itertools import chain, islice, groupby, count

stream = chain([1, 2], [3, 4])          # concatenate iterables lazily
first_three = list(islice(stream, 3))   # take a slice without materializing all

rows = [("cat", 1), ("cat", 2), ("dog", 3)]
for key, group in groupby(rows, key=lambda r: r[0]):   # requires sorted input
    print(key, [v for _, v in group])

ids = count(start=1000)                 # infinite id generator
```

### Comprehensions vs FP style
For simple transforms, a comprehension is usually clearer and faster than `map`/`filter` + `lambda`. Reach for `map`/`filter` when you already have a named function, and for `itertools` when you need laziness over large or infinite sequences.

```python
# Comprehension — preferred for a simple inline transform
squared = [x * x for x in range(5)]
# map with an existing named function — also fine
squared = list(map(abs, [-1, -2, 3]))
```

### When FP helps data pipelines
FP shines when you compose independent, pure steps over a stream: normalize → featurize → filter → batch. Purity means each stage is testable alone, and laziness (`map`, `itertools`) keeps memory flat regardless of dataset size.

## Worked Example
A lazy preprocessing pipeline that normalizes text, keeps non-empty items, and counts tokens — composed from small pure functions, streaming end to end.

```python
from functools import partial
from itertools import islice

def normalize(text: str) -> str:
    return text.strip().lower()

def is_nonempty(text: str) -> bool:
    return bool(text)

def token_count(text: str) -> int:
    return len(text.split())

raw = ["  Hello World ", "   ", "Functional PYTHON rocks", ""]

# Compose lazily: each stage yields one item at a time.
normalized = map(normalize, raw)
valid = filter(is_nonempty, normalized)
counts = map(token_count, valid)

print(list(islice(counts, 10)))     # [2, 3]
```

Because every stage is a pure function over an iterator, nothing materializes until `list(...)` pulls it, and you could swap `raw` for a million-line file unchanged.

## Best Practices
- ✅ Keep transformation functions pure — no mutation of inputs, no hidden I/O.
- ✅ Use `partial` to specialize a general function instead of writing thin wrappers.
- ✅ Cache only pure functions with `lru_cache`; caching an impure function hides bugs.

## Common Mistakes
- ⚠️ `groupby` groups only consecutive equal keys — sort by the same key first, or you will get fragmented groups.
- ⚠️ Applying `lru_cache` to a function with unhashable arguments (e.g. a list) raises `TypeError` — pass tuples instead.
- ⚠️ Forcing FP style (`reduce`, nested `lambda`) where a comprehension or plain loop is clearer hurts readability — FP is a tool, not a mandate.

## Industry Tips
- 💡 `lru_cache` on a deterministic embedding or tokenization helper can cut repeated compute dramatically in serving code — just size `maxsize` for your memory budget.
- 💡 Building pipelines from `map`/`filter`/`itertools` keeps peak memory constant, which is what lets you preprocess datasets larger than RAM.

## Real-World Use Cases
- Streaming preprocessing of large corpora that will not fit in memory.
- Memoizing expensive deterministic calls (tokenization, feature hashing).
- Configurable pipelines where each transform is injected as a function.

---

## Summary
- Functions are first-class values; higher-order functions and closures let you build and specialize behavior.
- Pure, immutable transformations are testable, cacheable, and parallel-safe.
- `functools` and `itertools` provide the composable, lazy building blocks for real data pipelines.

## Practice
- [ ] Exercises: [Module 1 Exercises](../exercises/README.md)
- [ ] Self-check: Why must the input to `itertools.groupby` be sorted by the grouping key?

## Further Reading
- 📘 Fluent Python, Luciano Ramalho
- 📄 [functools — Higher-order functions](https://docs.python.org/3/library/functools.html)
- 📄 [itertools — Functions creating iterators](https://docs.python.org/3/library/itertools.html)
- 🌐 Real Python — https://realpython.com/

## Related
- [Iterators and Generators](iterators-and-generators.md)
- [Decorators](decorators.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
