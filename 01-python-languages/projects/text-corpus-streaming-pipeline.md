---
title: "Mini Project: Streaming Text Corpus Pipeline"
description: Process a larger-than-memory text corpus with lazy generators and functional composition.
type: project
domain: 01-python-languages
tags: [python, generators, itertools, data-pipeline, nlp]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 4–5 hours
prerequisites:
  - 01-python-languages/lessons/iterators-and-generators
  - 01-python-languages/lessons/functional-programming
tech_stack: [python]
---

# Mini Project: Streaming Text Corpus Pipeline

> **What you'll build:** A memory-bounded pipeline that reads a large text file,
> cleans and tokenizes it lazily, and yields fixed-size batches ready for
> embedding or training.

---

## Objective

Real corpora don't fit in RAM. You will build a pipeline of composable generators
that stream a file line-by-line, apply cleaning steps lazily, and emit batches —
the exact pattern used to feed data loaders and embedding jobs.

## Learning Goals

- Compose lazy generators into a readable pipeline.
- Keep memory usage constant regardless of file size.
- Separate pure transformation steps from I/O.

---

## Prerequisites

- [Iterators and Generators](../lessons/iterators-and-generators.md)
- [Functional Programming in Python](../lessons/functional-programming.md)
- A large-ish `.txt` file (any public-domain book works).

## Architecture

```mermaid
flowchart LR
  A[read_lines] --> B[normalize]
  B --> C[filter empties]
  C --> D[tokenize]
  D --> E[batched]
  E --> F[consumer: count / embed]
```

Each stage is a generator that consumes the previous stage lazily, so only one
line (plus the current batch) is ever in memory.

---

## Steps

### 1. Setup
Create the package and a `pipeline.py` module. Use only the standard library.

### 2. Source stage
Write `read_lines(path)` that yields stripped lines from the file (a generator, so
it never loads the whole file).

### 3. Transformation stages
Write pure generator stages: `normalize` (lowercase, collapse whitespace),
`drop_empty`, and `tokenize` (split into tokens). Compose them by feeding one into
the next.

### 4. Batching
Reuse the `batched` generator from the
[streaming exercise](../exercises/generators-streaming.md) to yield token batches
of a configurable size.

### 5. Test & Validate
Prove correctness on a tiny sample, then run on the large file and confirm memory
stays flat (e.g., observe with a simple resident-memory check). Add unit tests for
each pure stage.

---

## Deliverables

- [ ] A composable generator pipeline (`read_lines → … → batched`).
- [ ] A consumer that computes corpus stats (line/token counts, vocabulary size).
- [ ] Unit tests for each pure stage.
- [ ] `README.md` explaining the design and memory behavior.

## Success Criteria

The pipeline processes a file far larger than a single batch without loading it
entirely into memory, and per-stage tests pass.

---

## Extensions (Optional)

- 🚀 Add an `async` source so multiple files stream concurrently.
- 🚀 Emit batches to disk as shards for later training.

## Further Reading

- [itertools documentation](https://docs.python.org/3/library/itertools.html)
- [Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html)

---

## Navigation

- ⬆️ [Module 1 Mini Projects](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
