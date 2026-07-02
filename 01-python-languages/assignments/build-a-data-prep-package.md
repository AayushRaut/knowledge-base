---
title: "Assignment: Build a Reusable Data-Prep Package"
description: Package a small, tested, installable data-cleaning library with clean structure and logging.
type: assignment
domain: 01-python-languages
tags: [python, packaging, testing, pandas, project-structure]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 3–4 hours
covers:
  - 01-python-languages/lessons/project-structure
  - 01-python-languages/lessons/packaging
  - 01-python-languages/lessons/testing
  - 01-python-languages/lessons/data-cleaning-and-wrangling
  - 01-python-languages/lessons/type-hints
  - 01-python-languages/lessons/logging-and-configuration
---

# Assignment: Build a Reusable Data-Prep Package

> A larger, assessed task that integrates multiple concepts from
> **[Module 1](../README.md)**.

---

## Context

Every ML project re-implements the same cleaning steps. You will package a small,
reusable, installable library — `dataprep` — that turns a raw tabular dataset into
a clean, model-ready `DataFrame`, with tests and logging. This mirrors how teams
factor shared preprocessing out of notebooks and into maintained packages.

## Objectives

- Apply a clean, installable project structure (`src/` layout + `pyproject.toml`).
- Write typed, documented, testable functions.
- Configure logging and read settings from the environment.

---

## Tasks

1. **Scaffold the project** — Create a `src/`-layout package with `pyproject.toml`,
   a virtual environment, `README.md`, and a `.env.example`. Make it
   `pip install -e .`-able. See [Packaging](../lessons/packaging.md) and
   [Project Structure](../lessons/project-structure.md).
2. **Implement the API** — Provide pure, type-hinted functions:
   `load(path) -> DataFrame`, `clean(df) -> DataFrame` (missing values, dtypes,
   duplicates), and `add_features(df) -> DataFrame`. Compose them in a
   `run_pipeline(path)` entry point. See
   [Data Cleaning](../lessons/data-cleaning-and-wrangling.md) and
   [Feature Engineering](../lessons/feature-engineering.md).
3. **Add logging & config** — Configure the `logging` module once; read the input
   path and log level from environment variables (never hard-code secrets). See
   [Logging & Configuration](../lessons/logging-and-configuration.md).
4. **Test it** — Write `pytest` unit tests (including a fixture with a small
   sample DataFrame and at least one `parametrize`d case) and reach meaningful
   coverage of the cleaning logic. See [Testing](../lessons/testing.md).

## Constraints

- Python 3.11+; dependencies limited to `pandas`, `pydantic-settings` (or
  `python-dotenv`), and `pytest`.
- All public functions have type hints and docstrings.
- No leakage of computed statistics between "fit" and "apply" steps.

---

## Deliverables

- [ ] An installable package (`pip install -e .` succeeds).
- [ ] `src/dataprep/` with typed modules and a CLI entry point.
- [ ] `tests/` passing with `pytest`.
- [ ] `README.md` documenting install, usage, and configuration.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Correctness | 40% | Pipeline runs end-to-end and produces a clean frame; tests pass. |
| Design & clarity | 25% | Clean structure, separation of concerns, readable typed code. |
| Documentation | 20% | Clear README, docstrings, `.env.example`. |
| Going further | 15% | Config via env, logging, coverage, CLI polish. |

---

## Submission

Push to a Git branch named `add/module-1-dataprep-assignment` and open a pull
request with the package, tests, and README.

## Further Reading

- [Python Packaging User Guide](https://packaging.python.org/)
- [pytest documentation](https://docs.pytest.org/)

---

## Navigation

- ⬆️ [Module 1 Assignments](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
