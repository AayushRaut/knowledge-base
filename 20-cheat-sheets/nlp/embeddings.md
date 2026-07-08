---
title: Text Embeddings Cheat Sheet
description: Choosing between TF-IDF, word embeddings, and sentence embeddings.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [nlp, embeddings, tf-idf, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Text Embeddings Cheat Sheet

> Fast reference. For depth, see
> [Word Embeddings](../../05-nlp/lessons/word-embeddings.md) and
> [Sentence Embeddings](../../05-nlp/lessons/sentence-embeddings.md).

---

## Representation comparison

| Method | Type | Semantics | OOV | Best for |
|--------|------|-----------|-----|----------|
| BoW / TF-IDF | sparse | none (lexical) | new words ignored | Strong classical baseline, exact terms |
| Word2Vec | dense, static | word-level | ✗ (no vector) | Word similarity/analogy, features |
| GloVe | dense, static | word-level | ✗ | Same as Word2Vec (global co-occurrence) |
| FastText | dense, static | word + subword | ✅ (char n-grams) | Morphology-rich languages, typos |
| Sentence-Transformers | dense, contextual | sentence-level | ✅ (subword) | Semantic search, STS, clustering, RAG |

## Key formulas

$$
\text{tfidf}(t,d) = \text{tf}(t,d)\cdot\log\tfrac{N}{\text{df}(t)}
\qquad
\text{cos}(\mathbf{a},\mathbf{b}) = \tfrac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}
$$

## Snippets

```python
# TF-IDF (fit on train only)
from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer().fit_transform(train_texts)

# Word2Vec (gensim)
from gensim.models import Word2Vec
w2v = Word2Vec(sentences=token_lists, vector_size=100, sg=1)
w2v.wv.most_similar("king")

# Sentence embeddings + semantic search
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(docs, normalize_embeddings=True)
scores = emb @ model.encode([query], normalize_embeddings=True)[0]
```

---

## Decision Guide

| If you need… | Use… | Notes |
|--------------|------|-------|
| Fast baseline classifier | TF-IDF + linear model | Hard to beat per compute. |
| Exact keyword/ID matching | Lexical (TF-IDF/BM25) | Semantics can miss rare literals. |
| Meaning-based retrieval | Sentence embeddings | The RAG retriever. |
| OOV robustness (typos, morphology) | FastText / subword models | Char n-grams. |
| Word analogies / lexical semantics | Word2Vec / GloVe | Static vectors suffice. |

---

## Gotchas

- ⚠️ Static word vectors give **one** vector per word — no polysemy ("bank").
- ⚠️ Averaging word vectors for sentences loses word order and emphasis.
- ⚠️ Normalize embeddings before cosine/dot-product search.
- ⚠️ Embedding model at index time must equal the one at query time.

---

## Quick Links

- 📖 [Word Embeddings](../../05-nlp/lessons/word-embeddings.md) · [Sentence Embeddings](../../05-nlp/lessons/sentence-embeddings.md) · [Classical Representation](../../05-nlp/lessons/text-representation.md)
- 🔗 [sbert.net](https://www.sbert.net/) · [gensim](https://radimrehurek.com/gensim/)

---

## Navigation

- ⬆️ [NLP Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
