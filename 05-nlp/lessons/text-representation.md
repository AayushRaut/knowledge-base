---
title: Classical Text Representation
description: Turn text into numeric vectors with one-hot encoding, Bag of Words, n-grams, and TF-IDF — and understand why their lack of semantics motivates embeddings.
type: lesson
domain: 05-nlp
tags: [nlp, bag-of-words, tf-idf, n-grams]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 35 min
prerequisites:
  - 05-nlp/lessons/text-preprocessing
---

# Classical Text Representation

> **TL;DR:** Machine learning needs numbers, not strings. Bag of Words counts words, n-grams recover local order, and TF-IDF weights words by how distinctive they are — simple, strong baselines whose blindness to meaning is exactly what embeddings later fix.

---

## Overview

Every model — logistic regression or transformer — consumes vectors. Classical representations (counts, n-grams, TF-IDF) are how NLP turned text into vectors for decades, and they remain excellent baselines: fast, interpretable, and surprisingly hard to beat on small datasets. For AI engineering they matter twice — as production tools (search ranking, spam filters) and as the conceptual foundation that makes embeddings comprehensible.

**By the end, you will be able to:**
- Build Bag-of-Words and n-gram feature matrices and explain their trade-offs
- Compute TF-IDF by hand and explain each term in the formula
- Measure document similarity with cosine over TF-IDF vectors using scikit-learn

---

## Intuition

Imagine describing a document only by *which words appear and how often* — dump the words in a bag, shake it, count. "Dog bites man" and "man bites dog" become identical bags, so order is lost, but for many tasks (is this email spam? is this review positive?) the word counts alone carry most of the signal.

Raw counts have a flaw: the most frequent words ("the", "is") say the least about what a document is *about*. The fix is intuitive — weight each word by how *rare* it is across the collection. A word that appears often in *this* document but in few *other* documents ("hemoglobin") is a strong identifier. That is TF-IDF: term frequency × inverse document frequency.

---

## Details

### One-hot encoding and sparsity

The simplest scheme: with a vocabulary of $V$ words, represent word $i$ as a vector of $V$ zeros with a single 1 at position $i$. Two problems:

- **Sparsity:** with $V = 50{,}000$, each vector is 50,000 dimensions with one nonzero entry.
- **No similarity:** every pair of distinct words is equally far apart — the dot product between any two different one-hot vectors is 0.

One-hot encodes *identity*, nothing more. It is the degenerate baseline everything else improves on.

### Bag of Words (BoW)

Represent a *document* as the sum of its words' one-hot vectors — i.e., a vector of word counts. Formally, document $d$ becomes $\mathbf{x}_d \in \mathbb{R}^{V}$ where $x_{d,t}$ is the count of term $t$ in $d$.

What you gain: fixed-size vectors any classifier can use. What you lose: **all word order** ("not good, just bad" ≈ "not bad, just good") and grammar.

### n-grams: recovering local order

An **n-gram** is a contiguous sequence of $n$ tokens. Using bigrams ($n=2$) as features, "not good" becomes a single feature distinct from "good" — local order is partially recovered.

The cost is combinatorial: the number of possible n-grams grows roughly with $V^n$, so feature matrices explode and most n-grams appear once. In practice, unigrams + bigrams (`ngram_range=(1, 2)`) with a frequency cutoff (`min_df`) is the sweet spot.

### TF-IDF

**TF-IDF** down-weights ubiquitous words and up-weights distinctive ones:

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \log\frac{N}{\text{df}(t)}
$$

where:

- $t$ — a term (word or n-gram); $d$ — a document
- $\text{tf}(t, d)$ — **term frequency**: how many times $t$ occurs in $d$ (raw count in the simplest variant)
- $N$ — total number of documents in the collection
- $\text{df}(t)$ — **document frequency**: the number of documents containing $t$
- $\log\frac{N}{\text{df}(t)}$ — **inverse document frequency (IDF)**: 0 for a term in every document ($\text{df} = N$), large for rare terms

A word like "the" (in every document) gets IDF $\approx 0$ and vanishes; "hemoglobin" (in 3 of 10,000 documents) gets a large IDF and dominates.

> **Note:** scikit-learn's `TfidfVectorizer` uses a smoothed variant by default: $\text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1$, and then L2-normalizes each document vector. Your hand-computed numbers will differ slightly from scikit-learn's — that is expected, not a bug.

### Document similarity with cosine

With documents as TF-IDF vectors, similarity is the **cosine** of the angle between them:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}
$$

Cosine ignores document *length* (a tweet and an essay about the same topic point the same direction) and ranges from 0 (no shared terms, for non-negative vectors) to 1 (identical direction). See [Vectors](../../02-mathematics-foundations/lessons/vectors.md) for the geometry.

### Python: scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train_docs: list[str] = [
    "the engine failed during the test flight",
    "the pilot reported engine trouble mid flight",
    "chocolate cake recipe with dark chocolate",
]

# Bag of Words with unigrams + bigrams
bow = CountVectorizer(ngram_range=(1, 2), min_df=1)
X_bow = bow.fit_transform(train_docs)          # sparse matrix, shape (3, n_features)
print(X_bow.shape, "features:", len(bow.get_feature_names_out()))

# TF-IDF — fit on TRAINING data only
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_docs)

# New (test/inference) documents: transform ONLY — never re-fit
query = ["engine problems on a flight"]
X_query = tfidf.transform(query)

sims = cosine_similarity(X_query, X_train)[0]
for doc, s in sorted(zip(train_docs, sims), key=lambda p: -p[1]):
    print(f"{s:.3f}  {doc}")
```

The fit/transform split is the representation-level version of the identical-pipeline rule from [Text Preprocessing](text-preprocessing.md): `fit` learns the vocabulary and IDF weights from training data; `transform` applies them unchanged everywhere else. Re-fitting on test data leaks information and changes the feature space.

### The limitation that motivates embeddings

TF-IDF vectors have **no semantics**. "car" and "automobile" occupy different dimensions, so two documents using different synonyms for the same concept have cosine similarity 0 on those terms — they are *orthogonal* by construction. Every classical representation shares this flaw: similarity exists only through exact (sub)string overlap. Dense [word embeddings](word-embeddings.md) fix precisely this, by learning vectors where synonyms end up close together.

## Worked Example

Three-document collection ($N = 3$):

```text
d1: "the cat sat on the mat"
d2: "the dog sat on the log"
d3: "cats and dogs and rain"
```

Compute TF-IDF for two terms in $d1$ using the lecture formula $\text{tf} \cdot \log(N/\text{df})$ (natural log):

- **"the"**: $\text{tf}(\text{the}, d1) = 2$; appears in $d1, d2$ so $\text{df} = 2$.
  $\text{tfidf} = 2 \cdot \ln(3/2) \approx 2 \times 0.405 = 0.81$
- **"cat"**: $\text{tf}(\text{cat}, d1) = 1$; appears only in $d1$ so $\text{df} = 1$.
  $\text{tfidf} = 1 \cdot \ln(3/1) \approx 1.10$

Despite occurring twice as often, "the" scores *lower* than "cat" — the IDF term recognizes that "the" appears in most documents and says little, while "cat" identifies what $d1$ is about. Note "cats" in $d3$ did not count toward "cat"'s document frequency: without lemmatization they are different strings, a small preview of the no-semantics limitation.

## Best Practices

- ✅ Fit vectorizers on training data only; call `transform` (never `fit_transform`) on validation, test, and production inputs.
- ✅ Start with `TfidfVectorizer(ngram_range=(1, 2), min_df=2)` as a baseline before anything neural — it is a strong, cheap yardstick.
- ✅ Keep matrices sparse (scipy CSR, as scikit-learn returns) — never call `.toarray()` on a large corpus.
- ✅ Persist the fitted vectorizer with the model (e.g., in one scikit-learn `Pipeline`) so vocabulary and weights ship together.

## Common Mistakes

- ⚠️ Calling `fit_transform` on the test set — this leaks test statistics and silently changes feature indices; use `transform`.
- ⚠️ Using unbounded n-grams (`ngram_range=(1, 4)` with no `min_df`) — feature count explodes with mostly-once features; cap `n` and set `min_df`.
- ⚠️ Comparing TF-IDF documents with Euclidean distance — length differences dominate; use cosine similarity.
- ⚠️ Expecting synonym matching ("car" vs "automobile") from TF-IDF — impossible by construction; switch to embeddings when paraphrase robustness matters.
- ⚠️ Hand-verifying scikit-learn TF-IDF values against the textbook formula and "finding a bug" — scikit-learn smooths IDF and L2-normalizes; the numbers legitimately differ.

## Industry Tips

- 💡 TF-IDF + logistic regression trains in seconds and is interpretable (inspect coefficients per word) — often the right first model, and sometimes the right final one for narrow domains.
- 💡 Lexical (TF-IDF-style) and semantic (embedding) retrieval are complementary: hybrid search that combines both typically beats either alone, which is why production RAG stacks often keep a lexical signal such as BM25 (a refined TF-IDF-family ranking function).
- 💡 `HashingVectorizer` trades the stored vocabulary for a hash function — useful for streaming pipelines where the vocabulary cannot be held or shipped.

## Real-World Use Cases

- Spam and abuse filtering with TF-IDF + linear classifiers
- Search engines: TF-IDF/BM25 lexical ranking of documents against queries
- Near-duplicate detection and clustering of news articles or support tickets
- Keyword extraction: a document's top TF-IDF terms serve as cheap auto-tags

---

## Summary

- BoW turns text into fixed-size count vectors at the price of word order; n-grams buy back local order at the price of vocabulary blow-up.
- TF-IDF weights terms by within-document frequency times across-collection rarity; combined with cosine similarity it powers strong lexical search and classification baselines.
- Classical vectors match strings, not meanings — synonyms are orthogonal — which is the direct motivation for dense embeddings.

## Practice

- [ ] Exercises: [Module 5 Exercises](../exercises/README.md)
- [ ] Self-check: Two documents describe the same event, one saying "car crash" and the other "automobile accident". What is their TF-IDF cosine similarity on those terms, and why?

## Further Reading

- 📘 Speech and Language Processing — Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/)
- 📄 [scikit-learn text feature extraction](https://scikit-learn.org/stable/)
- 📄 [spaCy documentation](https://spacy.io/)
- 📄 [NLTK documentation](https://www.nltk.org/)

## Related

- [Word Embeddings](word-embeddings.md)
- [Text Classification](text-classification.md)
- [Vectors](../../02-mathematics-foundations/lessons/vectors.md) — cosine similarity geometry

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
