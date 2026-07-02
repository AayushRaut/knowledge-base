---
title: "Exercise: Build a Timing Context Manager"
description: Implement a reusable block timer as a class and with contextlib.
type: exercise
domain: 01-python-languages
tags: [python, context-managers, profiling]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 01-python-languages/lessons/context-managers
---

# Exercise: Build a Timing Context Manager

> Practice for **[Context Managers](../lessons/context-managers.md)**.

---

## Problem

You want to time arbitrary blocks of code — for example, how long a preprocessing
step or a model call takes. Implement a `Timer` context manager **two ways** and
confirm both behave identically:

```python
with Timer("preprocess") as t:
    do_work()
print(t.elapsed)  # seconds, as a float
```

## Requirements

- [ ] Version A: a class with `__enter__`/`__exit__`.
- [ ] Version B: a generator function decorated with `@contextlib.contextmanager`.
- [ ] Both record wall-clock elapsed seconds even if the block raises.
- [ ] Use `time.perf_counter` (monotonic, high-resolution).

---

## Hints

<details>
<summary>Hint 1</summary>

`__exit__` runs even on exceptions. Returning a falsy value from `__exit__` lets
the exception propagate (what you want here).

</details>

<details>
<summary>Hint 2</summary>

In the `@contextmanager` version, put the "before" code above `yield` and the
"after" code in a `finally` block so it runs on errors too.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import contextlib
import time
from collections.abc import Iterator


class Timer:
    """Class-based context manager measuring wall-clock time of a block."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed = time.perf_counter() - self._start
        return False  # do not suppress exceptions


@contextlib.contextmanager
def timer(label: str = "") -> Iterator[dict[str, float]]:
    """Generator-based equivalent; the yielded dict exposes `elapsed`."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:  # runs even if the block raises
        result["elapsed"] = time.perf_counter() - start
```

**Explanation:** Both capture a start time on entry and compute elapsed time on
exit. The class stores it on `self`; the generator stores it in a mutable dict it
yields. `finally` (and `__exit__` always running) guarantees timing is recorded
even when the block errors — essential for profiling flaky I/O.

</details>

---

## Navigation

- ⬆️ [Module 1 Exercises](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
