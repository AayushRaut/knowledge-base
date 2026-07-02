---
title: "Python for AI Engineering — Interview Questions"
description: A bank of 20 Python interview questions with answers, focused on AI/ML engineering roles.
type: interview
domain: 19-interview-preparation
tags: [python, interview, coding]
category: coding
difficulty: medium
status: complete
created: 2026-07-02
updated: 2026-07-02
companies: []
---

# Python for AI Engineering — Interview Questions

> **Category:** coding · **Difficulty:** mixed (easy → hard)
>
> A question bank tied to **[Module 1 — Python for AI Engineering](../../01-python-languages/README.md)**.
> Try to answer before reading the response.

---

## Contents

1. [Mutable vs immutable types](#1-mutable-vs-immutable-types)
2. [`is` vs `==`](#2-is-vs-)
3. [Mutable default arguments](#3-mutable-default-arguments)
4. [List vs generator](#4-list-vs-generator)
5. [What does `yield` do?](#5-what-does-yield-do)
6. [Decorators](#6-decorators)
7. [`*args` and `**kwargs`](#7-args-and-kwargs)
8. [The GIL](#8-the-gil)
9. [Threading vs multiprocessing vs asyncio](#9-threading-vs-multiprocessing-vs-asyncio)
10. [`async`/`await`](#10-asyncawait)
11. [Context managers](#11-context-managers)
12. [Shallow vs deep copy](#12-shallow-vs-deep-copy)
13. [Why vectorize with NumPy?](#13-why-vectorize-with-numpy)
14. [NumPy broadcasting](#14-numpy-broadcasting)
15. [`loc` vs `iloc` and `SettingWithCopyWarning`](#15-loc-vs-iloc-and-settingwithcopywarning)
16. [`apply` vs vectorized ops](#16-apply-vs-vectorized-ops)
17. [Data leakage in preprocessing](#17-data-leakage-in-preprocessing)
18. [Why type hints?](#18-why-type-hints)
19. [`__slots__` and `dataclass`](#19-__slots__-and-dataclass)
20. [Testing non-deterministic code](#20-testing-non-deterministic-code)

---

### 1. Mutable vs immutable types
**Q:** Which built-in types are mutable, which are immutable, and why does it matter?

**A:** Immutable: `int`, `float`, `str`, `tuple`, `frozenset`, `bytes`. Mutable:
`list`, `dict`, `set`, most objects. It matters because immutables are hashable
(usable as dict keys / set members) and safe to share; mutables can be aliased and
changed in place, causing action-at-a-distance bugs. See
[Python Refresher](../../01-python-languages/lessons/python-refresher.md).

### 2. `is` vs `==`
**Q:** What's the difference?

**A:** `==` compares **values** (calls `__eq__`); `is` compares **identity**
(same object in memory). Use `==` for value checks and reserve `is` for
singletons, especially `x is None`. Small-int and string interning can make `is`
*appear* to work for values — don't rely on it.

### 3. Mutable default arguments
**Q:** What's wrong with `def add(x, items=[])`?

**A:** The default list is created **once** at function-definition time and shared
across all calls, so it accumulates state. Fix with the `None` sentinel:
`def add(x, items=None): items = [] if items is None else items`.

### 4. List vs generator
**Q:** When would you return a generator instead of a list?

**A:** When the sequence is large or infinite, or you only need to iterate once.
Generators are **lazy** — O(1) memory and they start producing immediately.
Return a list when you need random access, `len()`, or to iterate multiple times.

### 5. What does `yield` do?
**Q:** Explain generator functions.

**A:** A function containing `yield` returns a generator. Each `next()` runs until
the next `yield`, returns that value, and **suspends** — locals are preserved.
This enables streaming pipelines that process data without materializing it. See
[Iterators and Generators](../../01-python-languages/lessons/iterators-and-generators.md).

### 6. Decorators
**Q:** What is a decorator and why use `functools.wraps`?

**A:** A decorator is a callable that takes a function and returns a replacement,
used to add cross-cutting behavior (timing, caching, retry, logging) without
editing the function. `functools.wraps` copies the wrapped function's name,
docstring, and metadata to the wrapper so introspection and debugging still work.

### 7. `*args` and `**kwargs`
**Q:** What do they mean?

**A:** In a signature, `*args` collects extra positional arguments into a tuple and
`**kwargs` collects extra keyword arguments into a dict. At a call site, `*` and
`**` unpack an iterable/mapping into arguments. They're essential for writing
wrappers and decorators that forward arbitrary arguments.

### 8. The GIL
**Q:** What is the Global Interpreter Lock and how does it affect concurrency?

**A:** In CPython, the GIL allows only one thread to execute Python bytecode at a
time, so threads don't give true parallelism for **CPU-bound** work. Threads still
help **I/O-bound** work (the GIL is released during I/O). For CPU parallelism, use
processes (`multiprocessing`). See
[Concurrency](../../01-python-languages/lessons/concurrency.md).

### 9. Threading vs multiprocessing vs asyncio
**Q:** How do you choose?

**A:** **I/O-bound, many tasks** → `asyncio` (cooperative, single thread, highly
scalable). **I/O-bound, simpler / blocking libraries** → threads. **CPU-bound** →
`multiprocessing` (sidesteps the GIL with separate processes). Match the tool to
whether the bottleneck is waiting or computing.

### 10. `async`/`await`
**Q:** What does `await` actually do?

**A:** `await` suspends the current coroutine and yields control to the event loop,
which can run other coroutines while this one waits on I/O. It gives concurrency
on a single thread. It only helps if the awaited call is genuinely async — a
blocking call (e.g. `time.sleep`, sync HTTP) stalls the whole loop.

### 11. Context managers
**Q:** What problem do they solve and how do you write one?

**A:** They guarantee setup/teardown (closing files, releasing locks, restoring
state) even when exceptions occur. Implement `__enter__`/`__exit__`, or decorate a
generator with `@contextlib.contextmanager` and put cleanup in a `finally`. See
[Context Managers](../../01-python-languages/lessons/context-managers.md).

### 12. Shallow vs deep copy
**Q:** Difference between `copy.copy` and `copy.deepcopy`?

**A:** A shallow copy duplicates the outer container but shares references to the
inner objects; a deep copy recursively duplicates everything. Mutating a nested
object after a shallow copy affects both copies. Deep copies are safer but slower.

### 13. Why vectorize with NumPy?
**Q:** Why is `arr * 2` preferred over a Python loop?

**A:** Vectorized ops run in optimized, compiled C over contiguous memory, avoiding
per-element Python interpreter overhead, and they're typically much faster and more
readable. (State the benefit qualitatively unless you have measured numbers.)

### 14. NumPy broadcasting
**Q:** What are the rules?

**A:** Align shapes from the trailing dimension; two dimensions are compatible if
they're equal or one is 1 (which is stretched). Broadcasting avoids explicit tiling
and temporary copies. Example: `(n, 1) + (1, m) -> (n, m)`. See
[NumPy](../../01-python-languages/lessons/numpy.md).

### 15. `loc` vs `iloc` and `SettingWithCopyWarning`
**Q:** Explain both.

**A:** `.loc` selects by **label**, `.iloc` by **integer position**. The warning
appears when you assign to what might be a copy from chained indexing
(`df[mask]["col"] = ...`); do a single `df.loc[mask, "col"] = ...` instead.

### 16. `apply` vs vectorized ops
**Q:** Why avoid `df.apply(f, axis=1)`?

**A:** Row-wise `apply` runs a Python callable per row (slow). Prefer built-in
vectorized operations or `groupby` aggregations, which run in C over whole columns.
Reserve `apply` for genuinely irregular logic that can't be vectorized.

### 17. Data leakage in preprocessing
**Q:** How does scaling cause leakage, and how do you prevent it?

**A:** Fitting a scaler/encoder on the full dataset lets test-set statistics leak
into training, inflating scores. Fit transforms on **train only**, then transform
test — ideally inside a scikit-learn `Pipeline` so it's automatic and CV-safe. See
[Feature Engineering](../../01-python-languages/lessons/feature-engineering.md).

### 18. Why type hints?
**Q:** Do type hints affect runtime? Why use them?

**A:** They don't change runtime behavior (they're not enforced by the interpreter),
but static checkers (`mypy`, `pyright`) and IDEs use them to catch bugs early —
`None` handling, wrong shapes/args — and to document intent. Invaluable in large AI
codebases. See [Type Hints](../../01-python-languages/lessons/type-hints.md).

### 19. `__slots__` and `dataclass`
**Q:** What do they give you?

**A:** `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from typed
fields — great for config and record objects. `__slots__` declares a fixed set of
attributes, removing the per-instance `__dict__` to save memory and speed attribute
access — useful when you create millions of small objects.

### 20. Testing non-deterministic code
**Q:** How do you test code that calls an LLM/API or uses randomness?

**A:** Isolate nondeterminism: **mock** the external call (`unittest.mock` /
`monkeypatch`) so tests are deterministic and fast; **seed** RNGs; and compare
floats with tolerance (`pytest.approx`, `numpy.testing.assert_allclose`). Test your
logic, not the third-party service. See
[Testing](../../01-python-languages/lessons/testing.md).

---

## Common Mistakes

- ⚠️ Answering "the GIL makes Python slow" — it specifically limits CPU-bound
  *threading*, not all performance.
- ⚠️ Claiming type hints are enforced at runtime — they are not, by default.
- ⚠️ Forgetting that generators are single-pass.

## Key Takeaways

- Know *why*, not just *what* — interviewers probe trade-offs.
- Tie answers to real AI/ML scenarios (batching, embeddings, API calls, leakage).

## Related

- [Module 1 — Python for AI Engineering](../../01-python-languages/README.md)
- [Python Core Cheat Sheet](../../20-cheat-sheets/python/python-core.md)

---

## Navigation

- ⬆️ [Coding Interview Questions](README.md)
- 📚 [Interview Preparation](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
