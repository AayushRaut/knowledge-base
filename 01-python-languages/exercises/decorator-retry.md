---
title: "Exercise: Write a retry Decorator"
description: Build a parameterized decorator that retries flaky calls with exponential backoff.
type: exercise
domain: 01-python-languages
tags: [python, decorators, resilience, api]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 01-python-languages/lessons/decorators
---

# Exercise: Write a `retry` Decorator

> Practice for **[Decorators](../lessons/decorators.md)**.

---

## Problem

Calls to external services (LLM APIs, databases) fail transiently. Write a
decorator factory `retry(times, exceptions, base_delay)` that retries the wrapped
function on the given exceptions, sleeping `base_delay * 2**attempt` between tries
(exponential backoff), and re-raises the last exception if all attempts fail.

```python
@retry(times=3, exceptions=(ConnectionError,), base_delay=0.5)
def call_model(prompt: str) -> str:
    ...
```

## Requirements

- [ ] Preserves the wrapped function's name and docstring.
- [ ] Retries only on the specified exception types.
- [ ] Sleeps with exponential backoff between attempts.
- [ ] Re-raises the final exception after exhausting retries.

---

## Hints

<details>
<summary>Hint 1</summary>

A decorator that takes arguments is three nested functions: `retry(...)` returns
`decorator(func)` which returns `wrapper(*args, **kwargs)`.

</details>

<details>
<summary>Hint 2</summary>

Use `functools.wraps(func)` on the wrapper so metadata is preserved. Import
`time.sleep` for the delay.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import functools
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(
    times: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    base_delay: float = 0.5,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry the wrapped call on `exceptions` with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            last_exc: Exception | None = None
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # only catch the requested types
                    last_exc = exc
                    if attempt < times - 1:
                        time.sleep(base_delay * 2**attempt)
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator
```

**Explanation:** The outer `retry` captures configuration; `decorator` binds the
target function; `wrapper` runs the retry loop. Catching `exceptions` (a tuple of
types) means unrelated errors propagate immediately. Backoff grows as
`base_delay * 2**attempt`, and the final failure is re-raised so callers still see
real errors.

</details>

---

## Navigation

- ⬆️ [Module 1 Exercises](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
