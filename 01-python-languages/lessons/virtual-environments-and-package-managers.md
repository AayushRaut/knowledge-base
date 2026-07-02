---
title: Virtual Environments and Package Managers
description: Isolate dependencies and reproduce ML environments reliably with venv, pip, uv, and Poetry.
type: lesson
domain: 01-python-languages
tags: [python, environments, pip, uv, poetry, tooling]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 30 min
prerequisites:
  - 01-python-languages/lessons/python-refresher
---

# Virtual Environments and Package Managers

> **TL;DR:** A virtual environment isolates one project's packages from every other project and from system Python. Combined with a lockfile, it makes an ML experiment reproducible months later on another machine.

---

## Overview
AI projects pull in fast-moving, tightly coupled libraries — a specific `torch` built against a specific CUDA version, a `transformers` release that expects a particular `tokenizers`. Without isolation, installing one project breaks another. Virtual environments give each project its own package set; package managers install those packages and record exactly what was installed so a teammate (or a training run six months later) gets the same result.

**By the end, you will be able to:**
- Create and activate isolated environments with `venv`.
- Install, pin, and freeze dependencies with `pip` and `requirements.txt`.
- Use `uv` and `Poetry` for faster installs and true lockfiles.
- Choose the right tool for a given ML project and reproduce it exactly.

---

## Intuition
Think of system Python as a shared kitchen. If every project dumps its ingredients on the same counter, recipes collide — one needs flour version 1, another version 2, and the last one to install wins. A virtual environment gives each recipe its own counter. A lockfile is the exact shopping receipt: not "some flour," but "this brand, this size, this batch," so you can rebuild the identical dish anywhere.

---

## Details

### The problem: dependency conflicts and drift
Two failure modes dominate. **Conflicts:** project A needs `numpy==1.26` and project B needs `numpy>=2.0`; a single shared install can only satisfy one. **Drift:** you `pip install transformers` today and get 4.44; a colleague installs next month and gets 4.46 with a changed default, and your fine-tuning script behaves differently. Isolation solves the first; pinning and lockfiles solve the second.

### `venv` + `pip` + `requirements.txt`
`venv` ships with the standard library. It creates a directory holding an isolated interpreter and `site-packages`.

```bash
# Create an isolated environment in ./.venv (a common convention)
python -m venv .venv

# Activate it (the shell now uses this interpreter and its packages)
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate          # Windows PowerShell

# Install into the active environment only
pip install torch transformers datasets

# Leave the environment when done
deactivate
```

### Pinning and `pip freeze`
`pip freeze` prints the exact installed versions. Redirect it to a file so the environment can be recreated.

```bash
# Capture exact versions of everything currently installed
pip freeze > requirements.txt

# Recreate the same set later, in a fresh environment
pip install -r requirements.txt
```

A `requirements.txt` may hold loose constraints (`transformers>=4.44`) or exact pins (`transformers==4.44.2`). For reproducible ML runs, prefer exact pins. Note that `pip freeze` records *installed* versions but not the resolution logic, and it flattens direct and transitive dependencies together — you lose the distinction between what you asked for and what came along. Lockfile-based tools fix that.

### `uv` — fast installer, resolver, and project manager
`uv` is a Python package and project manager written in Rust by Astral. It works as a drop-in-fast replacement for `pip`/`venv` workflows and also manages projects via `pyproject.toml` plus a `uv.lock` lockfile. See https://docs.astral.sh/uv/.

```bash
# Start a new project (writes pyproject.toml)
uv init my-ml-project
cd my-ml-project

# Add a dependency: resolves, installs, and updates uv.lock
uv add torch transformers

# Create/sync the environment exactly from the lockfile
uv sync

# Run a command inside the project environment
uv run python train.py
```

`uv` also speaks the classic interface — `uv venv` creates an environment and `uv pip install -r requirements.txt` installs into it — so you can adopt it incrementally.

### `Poetry` — dependency management with a lockfile
`Poetry` manages dependencies and packaging through `pyproject.toml` and resolves them into a `poetry.lock` file for reproducible installs. See https://python-poetry.org/.

```bash
# Create a new project scaffold
poetry new my-ml-project
cd my-ml-project

# Add a dependency: updates pyproject.toml and poetry.lock
poetry add scikit-learn pandas

# Install exactly what the lockfile specifies
poetry install

# Run inside the managed environment
poetry run python evaluate.py
```

A lockfile records the full resolved graph — every direct and transitive package with an exact version (and often a hash) — so `install`/`sync` reproduces the same environment on any machine.

### Comparison

| Tool | Speed | Lockfile | Scope |
|------|-------|----------|-------|
| `venv` + `pip` | Baseline (pure-Python resolver) | No native lockfile (`requirements.txt` is a manual pin list) | Environment creation + package install |
| `Poetry` | Moderate | Yes (`poetry.lock`) | Dependency management + packaging + environments |
| `uv` | Fast (Rust resolver/installer) | Yes (`uv.lock`) | Environments + install + project & packaging management |

---

## Diagram

```mermaid
flowchart LR
    A[pyproject.toml / requirements.txt] -->|resolve| B[Dependency graph]
    B -->|write| C[Lockfile: uv.lock / poetry.lock]
    C -->|sync / install| D[Isolated .venv]
    D -->|uv run / poetry run| E[Reproducible ML run]
```

---

## Worked Example
You are handed a colleague's fine-tuning repo and must reproduce their results.

```bash
# 1. Clone and enter the project
git clone https://example.com/team/llm-finetune.git
cd llm-finetune

# 2. If it uses uv, one command builds the exact environment from uv.lock
uv sync

# 3. Run their training entrypoint inside that environment
uv run python -m finetune.train --config configs/lora.yaml
```

Because `uv.lock` pins every transitive dependency, the `torch`, `transformers`, and `peft` versions match theirs exactly — so a divergent loss curve points to data or hardware, not silent dependency drift.

---

## Best Practices
- ✅ One environment per project; never install project packages into system Python.
- ✅ Commit your lockfile (`uv.lock` / `poetry.lock`) so runs are reproducible.
- ✅ Use exact pins for anything that affects model behavior (`torch`, `transformers`, `numpy`).
- ✅ Add `.venv/` to `.gitignore` — recreate it from the lockfile, don't commit it.
- ✅ Record the Python version too (e.g. `requires-python` in `pyproject.toml`).

## Common Mistakes
- ⚠️ Running `pip install` without an active environment installs globally and pollutes system Python — activate first, or use `uv run`/`poetry run`.
- ⚠️ Committing only a loose `requirements.txt` with `>=` ranges — resolution drifts over time; pin exact versions or commit a lockfile.
- ⚠️ Editing `pyproject.toml` by hand and forgetting to re-lock — always run `uv add`/`poetry add` (or re-sync) so the lockfile stays consistent.
- ⚠️ Committing the `.venv/` directory — it is large, machine-specific, and non-portable.

## Industry Tips
- 💡 In CI and Docker builds, `uv sync --frozen` (or `poetry install`) against a committed lockfile gives byte-for-byte reproducible dependency layers and much faster cold builds than resolving fresh.
- 💡 For GPU deep-learning stacks, pin `torch` explicitly and follow the official install index for your CUDA build; a generic resolver may pick a CPU-only wheel.

## Real-World Use Cases
- Reproducing a published paper's results, where an unpinned dependency silently changes a default.
- Standing up identical training environments across a team and a cluster.
- Building slim, cache-friendly Docker images for model serving.

---

## Summary
- Virtual environments isolate a project's packages; lockfiles make the exact set reproducible.
- `venv` + `pip` is the built-in baseline; `requirements.txt` pins but is not a true lockfile.
- `uv` (Rust, fast) and `Poetry` add real lockfiles and project management via `pyproject.toml`.
- For ML, pin behavior-affecting packages and commit the lockfile.

## Practice
- [ ] Exercises: [Module 1 Exercises](../exercises/README.md)
- [ ] Self-check: Why does a committed `poetry.lock` reproduce an environment more reliably than a hand-written `requirements.txt` with `>=` ranges?

## Further Reading
- 📘 *Python Packaging* guidance — Python Packaging Authority (PyPA)
- 📄 [venv — Creation of virtual environments](https://docs.python.org/3/library/venv.html)
- 📄 [Python Packaging User Guide](https://packaging.python.org/)
- 📄 [uv documentation](https://docs.astral.sh/uv/)
- 📄 [Poetry documentation](https://python-poetry.org/)
- 🌐 Real Python — https://realpython.com/

## Related
- [Packaging Python Projects](packaging.md)
- [Python Project Structure and Clean Architecture](project-structure.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
