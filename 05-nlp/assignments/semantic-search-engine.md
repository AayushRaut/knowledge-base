---
title: "Assignment: Semantic Search Engine"
description: Build and evaluate a semantic search engine over a real document corpus, compared against a BM25/TF-IDF baseline.
type: assignment
domain: 05-nlp
tags: [nlp, semantic-search, embeddings, retrieval, evaluation]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 4–6 hours
covers:
  - 05-nlp/lessons/sentence-embeddings
  - 05-nlp/lessons/text-representation
  - 05-nlp/lessons/nlp-evaluation-metrics
---

# Assignment: Semantic Search Engine

> A larger, assessed task that integrates multiple concepts from
> **[Module 5](../README.md)**.

---

## Context

Semantic retrieval is the beating heart of RAG and modern search. You will build
a small but complete search engine over a corpus you choose, evaluate it against
a lexical baseline with a query set you author, and learn why "it feels better"
isn't an evaluation.

## Objectives

- Build an embed-index-query retrieval loop over a real corpus.
- Author a small evaluation set and compute retrieval metrics.
- Compare semantic vs lexical retrieval honestly.

---

## Tasks

1. **Corpus** — Assemble 200+ text chunks (e.g. documentation sections, news
   paragraphs, FAQ entries). Keep chunks reasonably sized and record their ids.
2. **Lexical baseline** — TF-IDF retrieval (cosine over `TfidfVectorizer`
   vectors). See [Classical Text Representation](../lessons/text-representation.md).
3. **Semantic engine** — `sentence-transformers` embeddings, normalized, top-k by
   dot product; embed the corpus once. See
   [Sentence Embeddings](../lessons/sentence-embeddings.md).
4. **Evaluation set** — Write ≥20 queries with judged relevant chunks (your own
   annotations are fine — document the guideline you used). Compute
   **Recall@k** and **MRR** for both systems; a short worked explanation of each
   metric is part of the deliverable. See
   [NLP Evaluation Metrics](../lessons/nlp-evaluation-metrics.md).
5. **Analysis** — Show queries where semantic wins (synonyms/paraphrase), where
   lexical wins (exact identifiers, rare names), and reflect on when a hybrid
   would help (link forward: [RAG](../../09-rag/README.md)).

## Constraints

- Corpus embedded once at index time; queries embedded at query time.
- Evaluation metrics computed by your own code (no metric libraries).
- Report both systems on the identical query set.

---

## Deliverables

- [ ] Indexing + query code for both systems.
- [ ] The query set with relevance judgments and the annotation guideline.
- [ ] Recall@k / MRR table + win/loss query analysis.
- [ ] `README.md` with setup, results, and the hybrid-retrieval reflection.

## Grading Rubric

| Criterion | Weight | Description |
|-----------|:------:|-------------|
| Correctness | 30% | Both retrievers work; embedding done once; scores sane. |
| Evaluation rigor | 30% | Sensible judgments, correct Recall@k/MRR implementations. |
| Analysis | 25% | Insightful win/loss cases; honest limitations. |
| Communication | 15% | Clear README and reproducible code. |

---

## Submission

Push to a branch `add/module-5-semantic-search` and open a pull request.

## Further Reading

- Sentence-Transformers documentation (https://www.sbert.net/)
- Speech and Language Processing — Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/)

---

## Navigation

- ⬆️ [Module 5 Assignments](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
