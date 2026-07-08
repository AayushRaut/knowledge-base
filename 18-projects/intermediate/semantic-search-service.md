---
title: "Portfolio Project: Semantic Search Service"
description: A production-grade semantic search API — embed, index, and query a document corpus behind FastAPI in Docker.
type: project
domain: 18-projects
tags: [nlp, semantic-search, embeddings, fastapi, docker, portfolio]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 12–18 hours
prerequisites:
  - 05-nlp
  - 01-python-languages
tech_stack: [python, sentence-transformers, numpy, fastapi, pytest, docker]
---

# Portfolio Project: Semantic Search Service

> **What you'll build:** `search-service` — an installable package + API that
> ingests documents, embeds and indexes them, and serves meaning-based search
> with evaluated quality. The direct precursor to a
> [RAG](../../09-rag/README.md) retriever, built to production standards.

---

## Objective

The capstone of [Module 5](../../05-nlp/README.md): take semantic search from a
notebook demo to a tested, containerized service — ingestion, indexing,
querying, evaluation, and serving as one coherent system.

## Learning Goals

- Design an ingest → embed → index → query pipeline with persistence.
- Serve retrieval behind a clean API with validation and logging.
- Evaluate retrieval quality (Recall@k, MRR) and ship the numbers.

---

## Prerequisites

- [Module 5 — NLP](../../05-nlp/README.md), especially
  [Sentence Embeddings](../../05-nlp/lessons/sentence-embeddings.md).
- [Module 1](../../01-python-languages/README.md) engineering practices.

## Architecture

```mermaid
flowchart TB
  A[documents in] --> B[chunk + clean]
  B --> C[embed once<br/>sentence-transformers]
  C --> D[(index: vectors + metadata,<br/>persisted to disk)]
  Q[POST /search query] --> E[embed query]
  E --> F[top-k cosine over index]
  D --> F
  F --> G[ranked results + scores]
  H[eval set] --> I[Recall@k, MRR report]
  D --> I
```

The index (vectors + document metadata) persists to disk and reloads at startup;
the embedding model name is stored *with* the index so query-time and index-time
models can never diverge.

---

## Steps

### 1. Scaffold
`src/`-layout package with `pyproject.toml`, config via environment, `tests/`,
and two entry points: `index` (CLI) and `serve` (API).

### 2. Ingestion & chunking
Load a folder of text/markdown files; split into overlapping chunks with stable
ids; store metadata (source file, position).

### 3. Embed & index
Embed chunks in batches (normalized) with `sentence-transformers`; persist
vectors (NumPy) + metadata (JSON/SQLite) + the model name as one artifact.

### 4. Query path
`POST /search {query, k}` → embed → top-k dot product → results with scores,
snippets, and source references. Validate inputs; log queries (never log
secrets).

### 5. Evaluate
Author ≥20 judged queries; compute Recall@k and MRR against a TF-IDF baseline;
ship the comparison in the README.

### 6. Quality gates & Docker
`pytest` (chunking, index round-trip, API contract, model-name parity),
`ruff`/`mypy`, GitHub Actions CI, and a `Dockerfile` so
`docker run search-service` serves the API.

---

## Deliverables

- [ ] `index` CLI producing a persisted, reloadable index artifact.
- [ ] `POST /search` endpoint with validation and ranked results.
- [ ] Recall@k / MRR evaluation vs a lexical baseline.
- [ ] Tests, CI, `Dockerfile`, and a results-bearing `README.md`.

## Success Criteria

A reviewer can index a sample corpus with one command, `docker run` the service,
query it, and see evaluated retrieval quality — with green CI behind it.

---

## Extensions (Optional)

- 🚀 Add a FAISS index and compare latency at 100k+ chunks.
- 🚀 Add hybrid retrieval (TF-IDF + embeddings, score fusion) — a preview of
  [RAG retrieval techniques](../../09-rag/README.md).

## Further Reading

- Sentence-Transformers documentation (https://www.sbert.net/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)

---

## Navigation

- ⬆️ [Intermediate Projects](README.md)
- 🗂️ [Projects](../README.md)
- 📚 [Module 5 — NLP](../../05-nlp/README.md)
- 🏠 [Knowledge Base Home](../../README.md)
