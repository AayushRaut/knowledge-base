---
title: "Assignment: Async Batch Inference Client"
description: Build a concurrent, resilient client that fans out many I/O-bound calls with backoff and config.
type: assignment
domain: 01-python-languages
tags: [python, asyncio, decorators, resilience, configuration]
difficulty: advanced
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 3–5 hours
covers:
  - 01-python-languages/lessons/async-programming
  - 01-python-languages/lessons/decorators
  - 01-python-languages/lessons/context-managers
  - 01-python-languages/lessons/logging-and-configuration
  - 01-python-languages/lessons/testing
---

# Assignment: Async Batch Inference Client

> A larger, assessed task that integrates multiple concepts from
> **[Module 1](../README.md)**.

---

## Context

Batch jobs often send thousands of I/O-bound requests (embedding calls, model
inference, API lookups). Doing this sequentially is slow; doing it without limits
overwhelms the service. You will build an **async batch client** that processes a
list of inputs concurrently, with a concurrency limit, retries with backoff, and
structured logging — using a **mock** async service so no real API or key is
required.

## Objectives

- Use `asyncio` to run many I/O-bound calls concurrently and correctly.
- Bound concurrency and handle transient failures gracefully.
- Keep configuration and secrets out of the code.

---

## Tasks

1. **Mock the service** — Write an `async def fake_infer(text) -> str` that sleeps
   briefly and randomly raises a transient error, standing in for a real endpoint.
   (Vary randomness by input, not by a global seed you cannot control.)
2. **Add resilience** — Wrap calls with a `retry` decorator (exponential backoff);
   reuse your solution from the
   [retry exercise](../exercises/decorator-retry.md). See
   [Decorators](../lessons/decorators.md).
3. **Bound concurrency** — Use an `asyncio.Semaphore` so at most `N` calls run at
   once; gather all results with `asyncio.gather`. See
   [Async Programming](../lessons/async-programming.md).
4. **Config & logging** — Read `MAX_CONCURRENCY` and log level from the
   environment; log start/finish and per-item failures (never log secrets). Use a
   [context manager](../lessons/context-managers.md) to time the whole batch.
5. **Test it** — Write `pytest` tests using `pytest.mark.asyncio` (or
   `asyncio.run`) and mocking to prove retries happen and results preserve input
   order. See [Testing](../lessons/testing.md).

## Constraints

- Standard library `asyncio` only for concurrency (no external async framework).
- Results must be returned in the same order as inputs.
- Failures after all retries are collected and reported, not silently dropped.

---

## Deliverables

- [ ] A runnable module exposing `async def run_batch(items) -> list[Result]`.
- [ ] Retry decorator, semaphore-bounded concurrency, and timing context manager.
- [ ] `pytest` tests covering success, retry, and final-failure paths.
- [ ] `README.md` with usage and configuration.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Correctness | 40% | Concurrent, order-preserving, retries work; no blocking-call misuse. |
| Design & clarity | 25% | Clean async structure, reusable decorator/context manager. |
| Documentation | 20% | README, docstrings, clear config. |
| Going further | 15% | Robust error reporting, tests for edge cases, metrics/logging. |

---

## Submission

Push to a branch `add/module-1-async-client-assignment` and open a pull request.

## Further Reading

- [asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [pytest documentation](https://docs.pytest.org/)

---

## Navigation

- ⬆️ [Module 1 Assignments](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
