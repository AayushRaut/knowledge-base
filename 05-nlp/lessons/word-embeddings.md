---
title: "Word Embeddings: Word2Vec, GloVe and FastText"
description: How dense word vectors learned from co-occurrence capture meaning, and how Word2Vec, GloVe and FastText differ in training signal and OOV handling.
type: lesson
domain: 05-nlp
tags: [nlp, embeddings, word2vec, glove, fasttext]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 50 min
prerequisites:
  - 05-nlp/lessons/text-representation
---

# Word Embeddings: Word2Vec, GloVe and FastText

> **TL;DR:** Word embeddings map each word to a dense vector such that words used in similar contexts end up close together. Word2Vec learns them by predicting context, GloVe by factorizing global co-occurrence counts, and FastText by composing character n-grams — which lets it embed words it has never seen.

---

## Overview

Sparse representations like bag-of-words and TF-IDF treat every word as its own dimension, so "excellent" and "great" are as unrelated as "excellent" and "carburetor". Word embeddings fix this by learning a low-dimensional dense vector for each word from raw text, with geometry that reflects meaning. This lesson covers the distributional hypothesis, the three classic embedding families, and training your own vectors with gensim.

**By the end, you will be able to:**
- Explain the distributional hypothesis and why dense vectors beat sparse ones for capturing similarity
- Contrast Word2Vec (CBOW vs skip-gram), GloVe, and FastText by training signal and OOV behavior
- Train and query a Word2Vec model with gensim, including similarity and analogy queries

---

## Intuition

Linguist J. R. Firth summarized the core idea in 1957: **"You shall know a word by the company it keeps."** You can guess what an unfamiliar word means from its neighbors. If you read "she poured the *tezgüino* into a glass and drank it", you infer *tezgüino* is a beverage — purely from context.

Embedding algorithms mechanize this. They scan a large corpus and force words that appear in similar contexts to have similar vectors. Nobody tells the model that "cat" and "dog" are related; it discovers this because both occur near "pet", "fur", "vet", and "food".

Think of the result as a map of meaning: each word is a point in, say, 100-dimensional space. Distance encodes relatedness, and some *directions* encode relations — moving from "man" to "woman" is roughly the same displacement as moving from "king" to "queen".

Contrast with sparse vectors:

- **Sparse (one-hot / TF-IDF):** dimension = vocabulary size (10⁴–10⁶), almost all zeros, every pair of distinct words is orthogonal — similarity between words is always 0.
- **Dense (embeddings):** dimension 50–1000, all entries used, similarity is graded and learned from data.

---

## Details

### The distributional hypothesis, formalized

Let $w$ be a target word and $c$ a context word (a word within a window of $\pm k$ positions around $w$). The distributional hypothesis says the meaning of $w$ is characterized by its distribution over contexts $p(c \mid w)$. Two words are similar if their context distributions are similar. Every method below is a different way of compressing that distribution into a vector.

### Word2Vec: learn by predicting

Word2Vec (Mikolov et al., 2013) trains a tiny neural network on a fake task; the embeddings are the by-product. There are two architectures:

- **CBOW (continuous bag-of-words):** average the context vectors and predict the center word. Fast; smooths over rare words.
- **Skip-gram:** use the center word to predict each context word. Slower, but works better for rare words and small corpora.

Skip-gram maximizes the average log-probability of contexts over a corpus of $T$ tokens with window size $c$:

$$
\frac{1}{T}\sum_{t=1}^{T} \sum_{\substack{-c \le j \le c \\ j \neq 0}} \log p(w_{t+j} \mid w_t)
$$

where $p(w_{t+j} \mid w_t)$ is a softmax over the whole vocabulary — too expensive to compute exactly for large vocabularies. **Negative sampling** replaces it with a cheap binary task: for each observed (word, context) pair, also draw $k$ random "negative" words and train the model to score the real pair high and the fakes low. Instead of normalizing over the entire vocabulary, each update touches only $k+1$ words.

**Analogy arithmetic.** A famous property: $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$, i.e. some relations correspond to roughly constant vector offsets. Honest caveats:

- It works for a curated set of relations (gender, capital–country, verb tense) far better than for arbitrary ones.
- Standard evaluation *excludes the input words* from candidate answers; without that exclusion, the nearest neighbor of the query point is often just "king" again.
- Quality depends heavily on corpus size and frequency of the words involved.

### GloVe: learn from global counts

GloVe (Pennington, Socher & Manning, 2014) starts from the **global co-occurrence matrix** $X$, where $X_{ij}$ counts how often word $j$ appears in the context of word $i$ across the whole corpus. Rather than streaming windows like Word2Vec, it fits vectors so that their dot product matches the log co-occurrence count, via weighted least squares:

$$
J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2
$$

where $V$ is the vocabulary size, $w_i$ and $\tilde{w}_j$ are word and context vectors, $b_i, \tilde{b}_j$ are biases, and $f$ is a weighting function that caps the influence of very frequent pairs and ignores zero counts. The key insight of the paper: *ratios* of co-occurrence probabilities carry meaning (ice co-occurs with "solid" much more than steam does), and this objective encodes those ratios as vector differences. Pretrained GloVe vectors are distributed via the Stanford project page (see Further Reading).

### FastText: words are made of parts

Word2Vec and GloVe treat each word as an atomic unit: "run", "running", and "runner" get independent vectors, and an unseen word gets *no* vector. FastText (Bojanowski et al., 2016) represents each word as a **bag of character n-grams** (typically 3–6 characters) plus the word itself. With boundary markers, `where` with $n=3$ yields `<wh, whe, her, ere, re>` and the special token `<where>`. The word vector is the sum of its n-gram vectors:

$$
\vec{v}_w = \sum_{g \in G_w} \vec{z}_g
$$

where $G_w$ is the set of n-grams of $w$ and $\vec{z}_g$ are learned n-gram vectors. Consequences:

- **OOV handling:** an unseen word like "runnable" can still be embedded by summing its n-grams.
- **Morphology:** shared substrings ("run") pull related word forms together — especially valuable for morphologically rich languages (Finnish, Turkish, German).

### Comparison

| | Word2Vec | GloVe | FastText |
|---|---|---|---|
| Training signal | Predict local context windows (streaming) | Factorize global co-occurrence counts | Predict context, like Word2Vec, but from subword n-grams |
| Unit | Whole word | Whole word | Character n-grams + word |
| OOV words | No vector | No vector | Composable from n-grams |
| Typos / rare morphology | Poor | Poor | Robust |
| When to use | Solid default when training your own vectors with gensim | Good pretrained general-purpose vectors | Noisy text, rare words, morphologically rich languages |

### Training Word2Vec with gensim

```python
from gensim.models import Word2Vec

corpus: list[list[str]] = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "rug"],
    ["cats", "and", "dogs", "are", "popular", "pets"],
    ["the", "king", "ruled", "the", "kingdom"],
    ["the", "queen", "ruled", "the", "kingdom"],
    ["a", "man", "and", "a", "woman", "walked", "the", "dog"],
]

model = Word2Vec(
    sentences=corpus,
    vector_size=50,   # embedding dimension
    window=3,         # context window size
    min_count=1,      # keep every word (tiny corpus)
    sg=1,             # 1 = skip-gram, 0 = CBOW
    negative=5,       # negative samples per positive pair
    epochs=200,       # tiny corpus needs many passes
    seed=42,
)

vec = model.wv["cat"]                 # numpy array, shape (50,)
print(model.wv.similarity("cat", "dog"))
print(model.wv.most_similar("king", topn=3))

# Analogy: king - man + woman ≈ ?
print(model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3))
```

On six sentences the numbers are noise — real embeddings need millions of tokens. The API, however, is identical at scale, and `most_similar` already applies the standard trick of excluding input words from the candidates.

## Worked Example

Suppose you build a support-ticket search tool and a user searches for "refund". With TF-IDF, tickets saying "money back" or "reimbursement" score zero overlap. With embeddings:

1. Train (or load) vectors over your ticket corpus, where "refund", "reimbursement", and "money" co-occur with "returned", "charged", "payment".
2. Because these words share contexts, their vectors land close together: `similarity("refund", "reimbursement")` is high, `similarity("refund", "keyboard")` is low.
3. Represent the query and each ticket, then rank by vector similarity: the "money back" ticket now surfaces even with zero shared words.

Step 3's naive version — averaging word vectors over a document — works surprisingly often but loses word order and emphasis; the [Sentence Embeddings](sentence-embeddings.md) lesson covers the proper fix.

## Best Practices

- ✅ Prefer pretrained vectors (GloVe, FastText) unless your domain vocabulary is unusual — training good embeddings needs large corpora.
- ✅ Use skip-gram (`sg=1`) for small corpora and rare words; CBOW for speed on large corpora.
- ✅ Pick FastText when your text has typos, hashtags, or rich morphology.
- ✅ Compare vectors with cosine similarity, not Euclidean distance — vector norms correlate with word frequency.

## Common Mistakes

- ⚠️ Training on a few thousand sentences and expecting clean analogies — embeddings need data. Fix: use pretrained vectors or aggressively lower `vector_size`.
- ⚠️ Forgetting `min_count` filters rare words (default 5), then hitting `KeyError` on lookup. Fix: check `word in model.wv.key_to_index` or lower `min_count`.
- ⚠️ Treating analogy arithmetic as a reliable reasoning tool. It is a diagnostic curiosity that works for selected relations, not a general inference method.
- ⚠️ Averaging word vectors as your only sentence representation for similarity-critical tasks. Fix: use sentence embeddings (next lesson).

## Industry Tips

- 💡 Static embeddings are far from obsolete: they are tiny, fast, and run with a dictionary lookup — useful for autocomplete, lightweight matching, and features in tabular models where a transformer is overkill.
- 💡 gensim's `KeyedVectors` lets you load pretrained vectors and drop the training machinery — much lower memory footprint at inference time.
- 💡 The biggest limitation to remember in interviews: **one vector per word means no polysemy.** "Bank" gets a single vector blending river banks and financial banks. Contextual models (see [Transformers](../../06-transformers/README.md)) produce a different vector per occurrence and were the direct answer to this weakness.

## Real-World Use Cases

- Semantic search and query expansion (matching "money back" to "refund")
- Recommendation features: embedding item descriptions or user queries
- Deduplication and clustering of short texts (support tickets, product titles)
- Initializing the embedding layer of task-specific neural models

---

## Summary

- The distributional hypothesis — words in similar contexts have similar meanings — is the foundation of all embedding methods.
- Word2Vec learns by predicting local context (CBOW/skip-gram with negative sampling); GloVe factorizes global co-occurrence counts; FastText builds word vectors from character n-grams, which handles OOV words and morphology.
- Static embeddings assign one vector per word regardless of context, so they cannot represent polysemy — the motivation for contextual embeddings from transformers.

## Practice

- [ ] Exercises: [Module 5 Exercises](../exercises/README.md)
- [ ] Self-check: Why can FastText produce a vector for a word it never saw during training, while Word2Vec cannot?

## Further Reading

- 📘 Speech and Language Processing — Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/)
- 📄 Mikolov et al. 2013, "Efficient Estimation of Word Representations in Vector Space" (https://arxiv.org/abs/1301.3781)
- 📄 Pennington, Socher & Manning 2014, GloVe project page (https://nlp.stanford.edu/projects/glove/)
- 📄 Bojanowski et al. 2016, "Enriching Word Vectors with Subword Information" (https://arxiv.org/abs/1607.04606)
- 📄 gensim documentation (https://radimrehurek.com/gensim/)

## Related

- [Classical Text Representation](text-representation.md)
- [Sentence Embeddings](sentence-embeddings.md)
- [Transformers](../../06-transformers/README.md) — contextual embeddings that resolve polysemy

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
