---
title: Packaging Python Projects
description: Turn AI/ML code into an installable, distributable package with pyproject.toml, src layout, wheels, and PyPI publishing.
type: lesson
domain: 01-python-languages
tags: [python, packaging, pyproject, distribution]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 40 min
prerequisites:
  - 01-python-languages/lessons/virtual-environments-and-package-managers
---

# Packaging Python Projects

> **TL;DR:** Packaging turns a folder of scripts into an installable distribution defined by `pyproject.toml`. It gives you clean imports, versioned releases, and a `pip install`-able artifact you can share on PyPI or a private index.

---

## Overview
On an ML team, the same preprocessing, feature, and evaluation code is used by training scripts, notebooks, a serving API, and CI. Copy-pasting it drifts; a proper package makes it `import`-able everywhere from one source of truth. Modern packaging is standardized: a single `pyproject.toml` declares metadata and the build system, and standard tools build a wheel you can install or publish.

**By the end, you will be able to:**
- Write a `pyproject.toml` with project metadata and a build backend.
- Lay out a project with the `src/` layout and expose a console script.
- Explain the difference between an sdist and a wheel.
- Build and publish a package to PyPI or a private index.

---

## Intuition
A loose folder of `.py` files is like ingredients on a counter — usable only if you're standing in that kitchen. A package is a sealed, labeled meal kit: it declares its name, version, and required ingredients, so anyone can install it and it works the same in a notebook, a Docker image, or a teammate's laptop. `pyproject.toml` is the label; the wheel is the sealed kit.

---

## Details

### Why package code
Packaging gives you three things: **reuse** (one canonical copy imported by many consumers), **distribution** (a single artifact installable with `pip`), and **clean imports** (`from mypkg.features import build` instead of fragile `sys.path` hacks and relative-path guessing).

### `pyproject.toml`: metadata and build system
`pyproject.toml` is the standard configuration file. Two standards matter:

- **PEP 517/518** define the *build system* — the `[build-system]` table that names the build backend (the tool that turns your source into a wheel) and its requirements. See https://peps.python.org/pep-0517/ and https://peps.python.org/pep-0518/.
- **PEP 621** defines *project metadata* — the `[project]` table (name, version, dependencies, etc.). See https://peps.python.org/pep-0621/.

Common build backends include `setuptools` and `hatchling`; both are widely used and read the same standard `[project]` table.

```toml
# pyproject.toml
[build-system]
# The build backend turns source into sdist/wheel; PEP 517/518 make this pluggable.
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ml-features"
version = "0.1.0"                     # bump on each release (see versioning below)
description = "Reusable feature engineering for our ML pipelines"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.0",
]

[project.optional-dependencies]
# Extras: `pip install ml-features[dev]` pulls test/lint tools only when needed.
dev = ["pytest>=8", "mypy>=1.10"]

[project.scripts]
# Exposes a CLI command `build-features` mapped to a function (see entry points).
build-features = "ml_features.cli:main"
```

### The `src/` layout
Place your import package inside a top-level `src/` directory:

```text
ml-features/
├── pyproject.toml
├── README.md
└── src/
    └── ml_features/
        ├── __init__.py
        ├── features.py
        └── cli.py
```

The `src/` layout prevents Python from importing your package straight from the working directory. That forces you to test against the *installed* package, catching missing files or bad packaging before your users do.

### sdist vs wheel
Building produces two artifact types:

- **sdist** (source distribution, `.tar.gz`): the source plus metadata; the consumer's machine builds it. Needed when there is a compile step.
- **wheel** (`.whl`): a pre-built, ready-to-install artifact. Installs fast because no build step runs on the target. Pure-Python packages ship a single wheel.

Publish both: the wheel for fast installs, the sdist as a buildable fallback.

### Entry points / console scripts
The `[project.scripts]` table maps a command name to a `module:function` target. After install, `pip` creates an executable so users can run `build-features` on the command line instead of `python -m ml_features.cli`. This is how tools expose CLIs (e.g. a `train` or `evaluate` command for your pipeline).

```python
# src/ml_features/cli.py
def main() -> None:
    """Entry point invoked by the `build-features` console script."""
    print("Building features...")
```

### Versioning basics
Use **semantic versioning** — `MAJOR.MINOR.PATCH`: bump PATCH for backward-compatible fixes, MINOR for backward-compatible features, MAJOR for breaking changes. Downstream consumers rely on this to write safe constraints like `ml-features>=0.1,<0.2`.

### Building and publishing
Build a wheel and sdist, then upload to an index.

```bash
# Option A: standard build + twine
python -m build              # writes dist/*.whl and dist/*.tar.gz
python -m twine upload dist/*   # uploads to PyPI (prompts for credentials/token)

# Option B: uv builds and publishes with the same standard artifacts
uv build                     # writes dist/*.whl and dist/*.tar.gz
uv publish                   # uploads to a configured index
```

Test first against TestPyPI (`https://test.pypi.org/`) so a mistake never lands on the real index. Authenticate with an API token, not a password.

### Internal / private packages
Teams often keep packages off public PyPI. Publish to a private index (a self-hosted or hosted package registry) and point your installer at it. Consumers install by name exactly as with public packages, but resolution goes to the private index URL. This shares proprietary preprocessing or model code across repos without making it public.

---

## Worked Example
Package the feature code above and install it into a training project.

```bash
# 1. From the ml-features/ root, build the artifacts
uv build            # or: python -m build

# 2. Install the built wheel into your training environment
pip install dist/ml_features-0.1.0-py3-none-any.whl

# 3. Import it cleanly from anywhere, and use the console script
python -c "from ml_features.features import build; print(build)"
build-features      # runs ml_features.cli:main
```

Now the training repo, the serving API, and CI all `import ml_features` from the same versioned artifact — no copy-pasted preprocessing.

---

## Best Practices
- ✅ Use the `src/` layout so tests run against the installed package.
- ✅ Declare `requires-python` and pin/constrain dependencies deliberately.
- ✅ Ship both a wheel and an sdist; upload with an API token.
- ✅ Follow semantic versioning and bump the version on every release.
- ✅ Validate on TestPyPI before publishing to production PyPI.

## Common Mistakes
- ⚠️ Forgetting to bump `version` before publishing — PyPI rejects re-uploading an existing version; increment it.
- ⚠️ Flat layout that "works" only because you're in the project directory — switch to `src/` to catch packaging gaps.
- ⚠️ Putting heavy `dev`/test tools in core `dependencies` — move them to `optional-dependencies` extras.
- ⚠️ Committing built `dist/` artifacts or secrets — build in CI and inject tokens from the environment.

## Industry Tips
- 💡 Automate `build` + `publish` in CI on a Git tag so releases are reproducible and never depend on a laptop's local state.
- 💡 Split large ML monorepos into small internal packages (features, data, serving) with independent versions to keep dependency graphs and CI fast.

## Real-World Use Cases
- A shared `preprocessing` package imported by training, batch inference, and a real-time API.
- A private model-client package distributed to product teams via an internal index.
- A CLI tool exposed as a console script for reproducible data preparation.

---

## Summary
- `pyproject.toml` (PEP 517/518/621) declares the build backend and project metadata in one standard file.
- The `src/` layout forces testing against the installed package.
- A wheel is pre-built and fast; an sdist is buildable source — publish both.
- Build with `build`/`uv build` and publish to PyPI or a private index with a token.

## Practice
- [ ] Exercises: [Module 1 Exercises](../exercises/README.md)
- [ ] Self-check: Why is a wheel faster to install than an sdist, and when do you still need to ship an sdist?

## Further Reading
- 📘 *Publishing Python Packages* — Dane Hillard
- 📄 [Packaging Python Projects tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- 📄 [PEP 621 — Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- 📄 [PEP 517 — A build-system independent format for source trees](https://peps.python.org/pep-0517/)
- 📄 [uv — Building and publishing](https://docs.astral.sh/uv/)
- 🌐 Real Python — https://realpython.com/

## Related
- [Virtual Environments and Package Managers](virtual-environments-and-package-managers.md)
- [Python Project Structure and Clean Architecture](project-structure.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
