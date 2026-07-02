---
title: Python Refresher for AI Engineering
description: A fast, dense refresher on Python's data model and core idioms for people who already know basic Python.
type: lesson
domain: 01-python-languages
tags: [python, fundamentals, data-model]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 30 min
prerequisites: []
---

# Python Refresher for AI Engineering

> **TL;DR:** Python treats everything as an object with identity, type, and value, and rewards you for using its built-in idioms — comprehensions, unpacking, EAFP, and the right container — over hand-rolled loops.

---

## Overview
AI engineering code spends most of its time wrangling data before a model ever sees it. Fluency with Python's data model and idioms keeps that glue code short, correct, and fast. This lesson is a dense refresher: it assumes you can already write a `for` loop and defines only the ideas that trip people up in practice.

**By the end, you will be able to:**
- Explain the difference between mutable and immutable objects and why it matters for defaults and caching.
- Choose the right built-in container and comprehension for a given data task.
- Apply idiomatic patterns: EAFP, unpacking, `enumerate`/`zip`, f-strings, and `pathlib`.

---

## Intuition
Think of every Python name as a sticky label, not a box. When you write `x = [1, 2]`, the label `x` points at a list object living somewhere in memory. Assignment moves labels; it never copies objects. Most confusing bugs — shared mutable defaults, aliasing, "why did my other variable change?" — dissolve once you picture labels pointing at objects.

---

## Details

### The data model: identity, type, value
Every object has an identity (`id(obj)`, its address), a type (`type(obj)`), and a value. `is` compares identity; `==` compares value. Use `is` only for singletons like `None`.

```python
a = [1, 2, 3]
b = a          # same object: b is a -> True
c = a[:]       # a shallow copy: c == a but c is not a
print(a is b, a is c, a == c)  # True False True
```

### Mutable vs immutable
Immutable objects (`int`, `float`, `str`, `tuple`, `frozenset`, `bytes`) cannot change after creation; mutable objects (`list`, `dict`, `set`) can. This distinction drives two rules: only immutable (hashable) objects can be dict keys or set members, and you must never use a mutable object as a default argument.

```python
def add_feature(x: float, buffer: list[float] | None = None) -> list[float]:
    """Append x to buffer. Uses None sentinel to avoid a shared mutable default."""
    if buffer is None:
        buffer = []  # fresh list per call, not one shared across calls
    buffer.append(x)
    return buffer
```

### Core containers
- **list** — ordered, mutable, allows duplicates. Your default sequence.
- **tuple** — ordered, immutable. Use for fixed records and as dict keys.
- **dict** — hash map, insertion-ordered since 3.7. The workhorse for feature maps and configs.
- **set** — unordered unique elements. Use for membership tests and deduplication (`O(1)` average lookup vs `O(n)` for a list).

```python
labels = ["cat", "dog", "cat", "bird"]
unique = set(labels)                     # {'cat', 'dog', 'bird'}
counts = {name: labels.count(name) for name in unique}
```

### Comprehensions
Comprehensions build a container in one expression and read as "output for item in iterable if condition". They are faster and clearer than an append loop for simple transforms.

```python
tokens = ["  Hello ", "WORLD", " ai "]
cleaned = [t.strip().lower() for t in tokens if t.strip()]
lengths = {t: len(t) for t in cleaned}   # dict comprehension
```

### Truthiness
Empty containers, `0`, `""`, and `None` are falsy; almost everything else is truthy. Test emptiness with the object itself, not `len(x) == 0`.

```python
batch: list[int] = []
if not batch:            # idiomatic empty check
    print("nothing to process")
```

### EAFP vs LBYL
LBYL ("look before you leap") checks preconditions first; EAFP ("easier to ask forgiveness than permission") just tries and catches the exception. Python favors EAFP because it avoids race conditions and redundant checks.

```python
# EAFP: attempt the access, handle the miss
config = {"lr": 0.001}
try:
    lr = config["learning_rate"]
except KeyError:
    lr = 0.01
# LBYL equivalent, often less idiomatic:
lr = config["learning_rate"] if "learning_rate" in config else 0.01
```

### f-strings, unpacking, enumerate, zip
f-strings interpolate expressions inline and support format specs. Starred assignment splits sequences; `enumerate` pairs index with value; `zip` walks sequences in parallel.

```python
epoch, loss = 3, 0.1234
print(f"epoch={epoch} loss={loss:.3f}")   # loss=0.123

first, *rest = [10, 20, 30, 40]            # first=10, rest=[20, 30, 40]

for i, name in enumerate(["a", "b"], start=1):
    print(i, name)

for feature, weight in zip(["age", "income"], [0.4, 0.6]):
    print(feature, weight)
```

### pathlib basics
`pathlib.Path` replaces brittle string paths with an object that works across operating systems.

```python
from pathlib import Path

data_dir = Path("datasets") / "train"     # OS-correct separator
for csv in data_dir.glob("*.csv"):         # lazy iteration over matches
    text = csv.read_text(encoding="utf-8")
```

## Worked Example
Suppose you receive raw label rows and need clean, deduplicated `(id, label)` records plus a per-label count for a class-balance check.

```python
from collections import Counter

raw = [
    ("001", " Cat "),
    ("002", "dog"),
    ("003", "CAT"),
    ("002", "dog"),   # duplicate row
]

# Normalize, then deduplicate by id using a dict (last write wins).
records = {row_id: label.strip().lower() for row_id, label in raw}
class_counts = Counter(records.values())

print(records)        # {'001': 'cat', '002': 'dog', '003': 'cat'}
print(class_counts)   # Counter({'cat': 2, 'dog': 1})
```

This uses a dict comprehension for O(n) dedup by key, `str` methods for cleaning, and `Counter` for the tally — no manual loop needed.

## Best Practices
- ✅ Use `None` as the sentinel for optional mutable arguments, then create the object inside the function.
- ✅ Reach for a comprehension when the loop body is a single transform or filter.
- ✅ Use `is` only for `None`, `True`, `False`; use `==` for value comparison.
- ✅ Prefer `pathlib.Path` over string concatenation for filesystem paths.

## Common Mistakes
- ⚠️ Mutable default argument (`def f(x, acc=[])`) silently shares state across calls — use the `None` sentinel instead.
- ⚠️ Assuming `b = a` copies a list — it aliases it; use `a[:]`, `list(a)`, or `copy.deepcopy` for nested structures.
- ⚠️ Membership tests against a large list are O(n) — convert to a `set` first.

## Industry Tips
- 💡 In data pipelines, dict insertion order is guaranteed (3.7+), so you can rely on it for stable feature ordering without an `OrderedDict`.
- 💡 `Counter`, `defaultdict`, and `namedtuple` from `collections` remove a lot of boilerplate in preprocessing code — learn them early.

## Real-World Use Cases
- Cleaning and deduplicating scraped or labeled datasets before training.
- Building feature dictionaries and config objects for ML experiments.
- Fast membership checks (e.g. filtering a stream against a stop-word or blocklist set).

---

## Summary
- Names are labels bound to objects; assignment moves labels, never copies.
- Mutability decides hashability, default-argument safety, and aliasing behavior.
- Idiomatic Python (comprehensions, EAFP, unpacking, `enumerate`/`zip`, `pathlib`) is shorter and usually faster than hand-written loops.

## Practice
- [ ] Exercises: [Module 1 Exercises](../exercises/README.md)
- [ ] Self-check: Why does `def f(x, acc=[])` accumulate values across separate calls, and how do you fix it?

## Further Reading
- 📘 Fluent Python, Luciano Ramalho
- 📄 [The Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- 🌐 Real Python — https://realpython.com/
- ▶️ Real Python — https://realpython.com/

## Related
- [Object-Oriented Programming in Python](oop.md)
- [Iterators and Generators](iterators-and-generators.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
