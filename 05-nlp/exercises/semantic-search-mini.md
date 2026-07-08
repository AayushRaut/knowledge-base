---
title: "Exercise: Semantic Search in 30 Lines"
description: Build a tiny semantic search engine with sentence-transformers and cosine similarity.
type: exercise
domain: 05-nlp
tags: [nlp, semantic-search, sentence-transformers, embeddings]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 05-nlp/lessons/sentence-embeddings
---

# Exercise: Semantic Search in 30 Lines

> Practice for **[Sentence Embeddings and Semantic Similarity](../lessons/sentence-embeddings.md)**.

---

## Problem

Build `SemanticSearch` over a list of documents: embed the corpus once, then
answer queries with top-k cosine similarity. Prove it beats keyword matching by
finding a document that shares **no words** with the query.

```python
docs = [
    "The chef prepared a delicious pasta dinner.",
    "Quarterly revenue exceeded analyst expectations.",
    "The goalkeeper made a stunning save in extra time.",
]
search.query("financial results beat forecasts", k=1)
# -> the revenue document (zero word overlap with the query)
```

## Requirements

- [ ] Embed the corpus **once** at index time (not per query).
- [ ] Normalize embeddings so cosine = dot product.
- [ ] Return `(doc, score)` pairs, best first.
- [ ] Demonstrate the zero-word-overlap match.

---

## Hints

<details>
<summary>Hint 1</summary>

`SentenceTransformer("all-MiniLM-L6-v2")` with
`encode(..., normalize_embeddings=True)` gives unit vectors directly.

</details>

<details>
<summary>Hint 2</summary>

With normalized vectors, scores for all docs are one matrix–vector product:
`corpus_emb @ query_emb`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, docs: list[str], model: str = "all-MiniLM-L6-v2") -> None:
        self.docs = docs
        self.model = SentenceTransformer(model)
        # Embed once; normalized so cosine similarity == dot product.
        self.emb = self.model.encode(docs, normalize_embeddings=True)

    def query(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        q = self.model.encode([text], normalize_embeddings=True)[0]
        scores = self.emb @ q                      # (n_docs,)
        top = np.argsort(scores)[::-1][:k]
        return [(self.docs[i], float(scores[i])) for i in top]


docs = [
    "The chef prepared a delicious pasta dinner.",
    "Quarterly revenue exceeded analyst expectations.",
    "The goalkeeper made a stunning save in extra time.",
]
search = SemanticSearch(docs)
print(search.query("financial results beat forecasts", k=1))
# [('Quarterly revenue exceeded analyst expectations.', ~0.5+)]
```

**Explanation:** The query shares no tokens with the winning document — TF-IDF
would score it zero — but their *meanings* are close, so their embeddings are
close. Embedding the corpus once and normalizing makes each query a single
matrix–vector product. Scale this up with a vector index and you have the
retriever inside every [RAG system](../../09-rag/README.md).

</details>

---

## Navigation

- ⬆️ [Module 5 Exercises](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
