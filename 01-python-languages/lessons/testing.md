---
title: Testing Python Code with pytest
description: Write fast, deterministic tests for AI code using pytest fixtures, parametrization, mocking, and coverage.
type: lesson
domain: 01-python-languages
tags: [python, testing, pytest, quality]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 50 min
prerequisites:
  - 01-python-languages/lessons/type-hints
---

# Testing Python Code with pytest

> **TL;DR:** Tests give you the confidence to refactor and ship; `pytest` makes them concise with plain `assert`, fixtures, and parametrization, and mocking lets you test LLM code deterministically without hitting a real API.

---

## Overview
AI systems are full of code that is expensive, non-deterministic, or slow to run for real: LLM API calls, embedding jobs, data pipelines. Tests let you verify the *logic around* those calls — parsing, retries, prompt assembly, numeric transforms — without paying for or waiting on the real thing. A tested codebase is one you can refactor and extend without fear of silent regressions.

**By the end, you will be able to:**
- Write and run `pytest` tests using plain `assert`, fixtures, and parametrization.
- Mock external dependencies (like an LLM API) so tests are fast and deterministic.
- Compare floating-point and array results correctly, and measure coverage.

---

## Intuition
A test suite is a safety net stretched under a tightrope. Every time you change the code, you walk the rope again — but the net catches you if you slip. Without it, every refactor is a gamble; with it, you move fast because mistakes surface instantly and locally instead of in production.

The **testing pyramid** describes the healthy shape of that net: many fast **unit** tests at the base (one function/class in isolation), fewer **integration** tests in the middle (components working together), and a small number of slow **end-to-end** tests at the top (the whole system). Keep the base wide.

---

## Details

### Why test
- **Catch regressions** — a passing suite proves your change did not break existing behavior.
- **Refactor with confidence** — you can restructure internals freely as long as tests stay green.
- **Document behavior** — a test names the expected contract of a function.
- **Design pressure** — code that is hard to test is usually too coupled; tests push you toward clean boundaries.

### pytest basics
`pytest` discovers files named `test_*.py`, functions named `test_*`, and uses Python's plain `assert` (it rewrites assertions to give rich failure output). See the [pytest docs](https://docs.pytest.org/).

```python
# src/textutils.py
def normalize(text: str) -> str:
    """Lowercase and collapse surrounding whitespace."""
    return " ".join(text.lower().split())
```

```python
# tests/test_textutils.py
from textutils import normalize


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  Hello   WORLD ") == "hello world"
```

Run the suite:

```bash
pytest -q
```

### Fixtures
A **fixture** provides reusable setup (and optional teardown) to tests. Declare it once, request it by naming it as a test argument.

```python
import pytest


@pytest.fixture
def sample_prompts() -> list[str]:
    """Reusable prompt corpus for tests."""
    return ["Summarize this.", "Translate to French."]


def test_batch_size(sample_prompts: list[str]) -> None:
    assert len(sample_prompts) == 2
```

Use `yield` in a fixture when you need cleanup after the test runs:

```python
@pytest.fixture
def temp_index(tmp_path):
    """Create a vector index in a temp dir, then remove it."""
    path = tmp_path / "index.bin"
    path.write_bytes(b"")  # setup
    yield path
    # anything after yield runs as teardown (tmp_path auto-cleans too)
```

### Parametrization
`@pytest.mark.parametrize` runs the same test body across many inputs, each reported separately — far better than a loop.

```python
import pytest
from textutils import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A B", "a b"),
        ("  x  ", "x"),
        ("MiXeD\tCase", "mixed case"),
    ],
)
def test_normalize_cases(raw: str, expected: str) -> None:
    assert normalize(raw) == expected
```

### Mocking external calls
LLM and network calls are slow, costly, and non-deterministic — never call them for real in a unit test. Replace them with a **mock** using `unittest.mock` or pytest's `monkeypatch`.

Suppose your code calls a client:

```python
# src/summarizer.py
def summarize(client, text: str) -> str:
    """Return a one-line summary using the given LLM client."""
    resp = client.complete(prompt=f"Summarize: {text}")
    return resp.strip()
```

Test it with a mock client so the result is deterministic and free:

```python
from unittest.mock import Mock
from summarizer import summarize


def test_summarize_strips_response() -> None:
    client = Mock()
    client.complete.return_value = "  a short summary  "

    result = summarize(client, "long article text")

    assert result == "a short summary"
    client.complete.assert_called_once()  # verify the call happened
```

Use `monkeypatch` to replace a module-level function or environment variable:

```python
def test_reads_model_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-model")
    import os

    assert os.environ["LLM_MODEL"] == "test-model"
```

### Testing numeric and data code
Floating-point results are almost never exactly equal. Use `pytest.approx` for scalars and `numpy.testing` for arrays.

```python
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_cosine_similarity_of_identical_vectors() -> None:
    v = np.array([1.0, 2.0, 3.0])
    sim = float(v @ v / (np.linalg.norm(v) ** 2))
    assert sim == pytest.approx(1.0)


def test_normalized_vector_has_unit_norm() -> None:
    v = np.array([3.0, 4.0])
    unit = v / np.linalg.norm(v)
    assert_allclose(np.linalg.norm(unit), 1.0, rtol=1e-7)
```

### Coverage
Coverage measures which lines your tests execute. Use `pytest-cov` (built on `coverage.py`):

```bash
pytest --cov=src --cov-report=term-missing
```

Treat coverage as a guide, not a goal — 100% coverage of trivial code proves little, while a well-chosen 80% covering critical paths is valuable.

### Property-based testing
Instead of hand-picking examples, **Hypothesis** (https://hypothesis.readthedocs.io/) generates many inputs and checks that a *property* always holds, shrinking any failure to a minimal case.

```python
from hypothesis import given, strategies as st
from textutils import normalize


@given(st.text())
def test_normalize_is_idempotent(s: str) -> None:
    # Normalizing twice equals normalizing once.
    assert normalize(normalize(s)) == normalize(s)
```

### Running tests in CI
Run the suite on every push/PR so regressions never merge. A minimal GitHub Actions step:

```yaml
- name: Run tests
  run: pytest --cov=src -q
```

## Worked Example
A retry wrapper around a flaky LLM call, tested without any network access.

```python
# src/robust_call.py
def call_with_retry(client, prompt: str, retries: int = 2) -> str:
    """Call client.complete, retrying on exception up to `retries` times."""
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            return client.complete(prompt=prompt)
        except Exception as exc:  # noqa: BLE001 - retry on any failure
            last_exc = exc
    raise RuntimeError("all retries failed") from last_exc
```

```python
# tests/test_robust_call.py
from unittest.mock import Mock
import pytest
from robust_call import call_with_retry


def test_succeeds_after_one_failure() -> None:
    client = Mock()
    # First call raises, second returns a value.
    client.complete.side_effect = [ValueError("boom"), "ok"]

    result = call_with_retry(client, "hi", retries=2)

    assert result == "ok"
    assert client.complete.call_count == 2


def test_raises_when_all_retries_fail() -> None:
    client = Mock()
    client.complete.side_effect = ValueError("boom")

    with pytest.raises(RuntimeError, match="all retries failed"):
        call_with_retry(client, "hi", retries=1)
```

## Best Practices
- ✅ Keep unit tests fast and isolated — mock every network, disk, and clock dependency.
- ✅ Use `parametrize` instead of loops so each case reports separately.
- ✅ Name tests for the behavior they assert (`test_raises_when_all_retries_fail`).
- ✅ Compare floats with `pytest.approx` and arrays with `numpy.testing`.
- ✅ Run the suite in CI on every pull request.

## Common Mistakes
- ⚠️ Calling a real LLM/API in a unit test → mock the client; tests become flaky and expensive otherwise.
- ⚠️ Asserting exact float equality (`assert x == 0.1 + 0.2`) → use `pytest.approx`.
- ⚠️ One giant test asserting many things → split so a failure pinpoints the cause.
- ⚠️ Chasing 100% coverage on trivial code → target critical paths and edge cases.
- ⚠️ Tests that depend on run order or shared mutable state → make each test self-contained via fixtures.

## Industry Tips
- 💡 Store a few real LLM responses as fixtures ("golden files") to test parsing against realistic payloads without live calls.
- 💡 Mark slow or networked tests (`@pytest.mark.slow`) and exclude them from the fast CI lane.
- 💡 Add Hypothesis for parsers and numeric utilities — it finds edge cases you would never write by hand.

## Real-World Use Cases
- Verifying prompt-template assembly and response parsing without API cost.
- Regression-testing a data-cleaning pipeline before retraining.
- Ensuring an embedding normalization produces unit vectors within tolerance.

---

## Summary
- Tests catch regressions and let you refactor with confidence; keep the pyramid base-heavy.
- `pytest` uses plain `assert`, with fixtures for setup and `parametrize` for many cases.
- Mock external calls (LLM APIs) for deterministic, free unit tests.
- Compare floats/arrays with tolerance; measure coverage and run tests in CI.

## Practice
- [ ] Exercises: [Module 1 Exercises](../exercises/README.md)
- [ ] Self-check: Why should you mock an LLM API call in a unit test instead of calling it for real?

## Further Reading
- 📘 Architecture Patterns with Python, Harry Percival & Bob Gregory
- 📄 [pytest documentation](https://docs.pytest.org/)
- 📄 [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- 🌐 Real Python — https://realpython.com/
- ▶️ ArjanCodes (YouTube) — https://www.youtube.com/@ArjanCodes

## Related
- [Type Hints and Static Typing](type-hints.md)
- [Python Project Structure and Clean Architecture](project-structure.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
