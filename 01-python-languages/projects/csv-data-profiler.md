---
title: "Mini Project: CSV Data Profiler CLI"
description: Build a command-line tool that profiles any CSV and prints a data-quality report.
type: project
domain: 01-python-languages
tags: [python, pandas, cli, data-quality]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 3–4 hours
prerequisites:
  - 01-python-languages/lessons/pandas
  - 01-python-languages/lessons/project-structure
tech_stack: [python, pandas, argparse]
---

# Mini Project: CSV Data Profiler CLI

> **What you'll build:** A `profile-csv` command that reads any CSV and prints a
> per-column data-quality report (types, missing %, unique counts, basic stats).

---

## Objective

Before modeling, you must understand your data. You will build a small,
reusable CLI that turns "what's in this file?" into a one-command answer — the
kind of utility every data scientist keeps in their toolbelt.

## Learning Goals

- Structure a small installable CLI with clean separation of concerns.
- Use Pandas to compute per-column summaries.
- Handle files, arguments, and errors robustly.

---

## Prerequisites

- [Pandas for Data Work](../lessons/pandas.md)
- [Python Project Structure](../lessons/project-structure.md)
- A CSV file to test with (any public dataset).

## Architecture

```mermaid
flowchart LR
  A[CSV path arg] --> B[load with pandas]
  B --> C[per-column profiler]
  C --> D[format report]
  D --> E[stdout / optional JSON]
```

The CLI layer parses arguments and delegates to a pure `profile(df)` function
that returns a structured report; a formatter renders it. Keeping `profile`
pure makes it trivial to unit-test.

---

## Steps

### 1. Setup
Create a `src/`-layout package with `pyproject.toml` and a console entry point
`profile-csv`. Install with `pip install -e .`.

### 2. Load
Read the CSV with `pandas.read_csv`, handling a missing/invalid path with a clear
error message and non-zero exit code.

### 3. Profile
Write `profile(df) -> dict` returning, per column: dtype, non-null count, missing
percentage, unique count, and (for numeric columns) min/max/mean/std.

### 4. Report
Render the profile as an aligned table to stdout; add a `--json` flag to emit the
report as JSON for piping into other tools.

### 5. Test & Validate
Unit-test `profile` with a small in-memory DataFrame fixture; confirm the CLI
returns a non-zero exit code on a bad path.

---

## Deliverables

- [ ] Installable `profile-csv` CLI.
- [ ] Pure, tested `profile(df)` function.
- [ ] Human-readable table output plus `--json`.
- [ ] `README.md` with usage examples.

## Success Criteria

Running `profile-csv data.csv` on any well-formed CSV prints a correct report;
tests pass; bad input fails gracefully with a helpful message.

---

## Extensions (Optional)

- 🚀 Detect likely categorical vs continuous columns and suggest encodings.
- 🚀 Flag columns with >X% missing or a single dominant value.

## Further Reading

- [pandas documentation](https://pandas.pydata.org/docs/)
- [argparse documentation](https://docs.python.org/3/library/argparse.html)

---

## Navigation

- ⬆️ [Module 1 Mini Projects](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
