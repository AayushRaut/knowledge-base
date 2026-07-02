---
title: "Debugging: Fix the Mutable-Default & Closure Bug"
description: Diagnose two classic Python gotchas — the mutable default argument and the late-binding closure.
type: exercise
domain: 01-python-languages
tags: [python, debugging, closures, gotchas]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 01-python-languages/lessons/python-refresher
  - 01-python-languages/lessons/functional-programming
---

# Debugging: Fix the Mutable-Default & Closure Bug

> Practice for **[Python Refresher](../lessons/python-refresher.md)** and
> **[Functional Programming in Python](../lessons/functional-programming.md)**.

---

## Problem

Two functions behave in surprising ways. Explain each bug and fix it.

```python
# Bug A: accumulator "remembers" values across calls
def add_item(item, bucket=[]):
    bucket.append(item)
    return bucket

# add_item(1) -> [1]
# add_item(2) -> [1, 2]   # expected [2]

# Bug B: all callbacks print 3, not 0,1,2
def make_printers():
    printers = []
    for i in range(3):
        printers.append(lambda: print(i))
    return printers

# for p in make_printers(): p()  ->  3 3 3
```

## Requirements

- [ ] Explain why each behaves as it does.
- [ ] Fix `add_item` so each call starts fresh unless a bucket is passed.
- [ ] Fix `make_printers` so the printers output `0`, `1`, `2`.

---

## Hints

<details>
<summary>Hint 1</summary>

Default argument values are evaluated **once**, when the function is defined — not
on each call.

</details>

<details>
<summary>Hint 2</summary>

Closures capture the *variable* `i`, not its value at creation time. Bind the
current value as a default argument, or use `functools.partial`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
from collections.abc import Callable


def add_item(item: object, bucket: list | None = None) -> list:
    # Create a new list per call; the shared mutable default is the bug.
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket


def make_printers() -> list[Callable[[], None]]:
    # Bind the current value of i via a default argument (evaluated now).
    return [lambda i=i: print(i) for i in range(3)]
```

**Why they broke:**

- **Bug A:** the default `[]` is created once at definition time and reused by
  every call that doesn't pass `bucket`, so it accumulates. The `None` sentinel
  pattern gives each call a fresh list.
- **Bug B:** each `lambda` closes over the *variable* `i`, which equals `2` after
  the loop ends. `lambda i=i:` captures the current value as a default at each
  iteration, so the printers see `0`, `1`, `2`.

</details>

---

## Navigation

- ⬆️ [Module 1 Exercises](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
