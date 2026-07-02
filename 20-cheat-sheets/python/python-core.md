---
title: Python Core Cheat Sheet
description: Fast reference for Python's data model, containers, comprehensions, functions, and classes.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [python, core, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Python Core Cheat Sheet

> Fast reference for core Python. For depth, see
> [Module 1 lessons](../../01-python-languages/lessons/README.md).

---

## Core Concepts

| Term | Definition |
|------|------------|
| Mutable | Value can change in place: `list`, `dict`, `set`. |
| Immutable | Cannot change: `int`, `float`, `str`, `tuple`, `frozenset`. |
| Iterable | Has `__iter__`; can be looped (list, str, generator). |
| Iterator | Has `__next__`; produces items one at a time. |
| EAFP | "Easier to Ask Forgiveness than Permission" — `try/except` over pre-checks. |

---

## Containers

```python
nums = [1, 2, 3]            # list  — ordered, mutable
point = (1, 2)             # tuple — ordered, immutable
seen = {1, 2, 3}           # set   — unique, unordered
ages = {"a": 1, "b": 2}    # dict  — key -> value

ages.get("c", 0)            # 0 (safe default, no KeyError)
ages.setdefault("c", []).append(1)
```

## Comprehensions

```python
squares = [x * x for x in range(10) if x % 2 == 0]   # list
by_len = {w: len(w) for w in words}                  # dict
uniq = {c for c in text}                             # set
lazy = (x * x for x in range(10))                    # generator (no brackets)
```

## Functions & idioms

```python
def f(a, b=0, *args, **kwargs): ...   # positional, default, var-args
first, *rest = [1, 2, 3]              # unpacking -> 1, [2, 3]
for i, v in enumerate(seq): ...       # index + value
for a, b in zip(xs, ys): ...          # parallel iteration
name = user or "anonymous"            # truthiness fallback
```

## Classes

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    def norm(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5
```

---

## Decision Guide

| If you need… | Use… | Notes |
|--------------|------|-------|
| Ordered, changeable sequence | `list` | Default container. |
| Fixed record / dict key | `tuple` | Hashable if items are. |
| Membership tests / dedup | `set` | O(1) average lookup. |
| Lazy, memory-bounded stream | generator | `yield` or `(… for …)`. |
| Simple data holder | `@dataclass` | Auto `__init__`/`__repr__`/`__eq__`. |

---

## Gotchas

- ⚠️ **Mutable default arguments** (`def f(x=[])`) persist across calls — use `None` + create inside.
- ⚠️ **Late-binding closures** in loops capture the variable, not its value — bind with `lambda i=i:`.
- ⚠️ `is` checks identity, `==` checks equality — use `==` for values, `is` only for `None`.
- ⚠️ Copying: `b = a` aliases; use `copy.deepcopy` for nested structures.

---

## Quick Links

- 📖 [Python Refresher](../../01-python-languages/lessons/python-refresher.md)
- 📖 [OOP](../../01-python-languages/lessons/oop.md) · [Functional Programming](../../01-python-languages/lessons/functional-programming.md)
- 🔗 [Python docs](https://docs.python.org/3/) · [PEP 8](https://peps.python.org/pep-0008/)

---

## Navigation

- ⬆️ [Python Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
