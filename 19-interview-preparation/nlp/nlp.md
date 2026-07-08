---
title: "Natural Language Processing — Interview Questions"
description: A bank of 20 NLP interview questions with answers.
type: interview
domain: 19-interview-preparation
tags: [nlp, interview, embeddings, tokenization]
category: ml-concepts
difficulty: medium
status: complete
created: 2026-07-02
updated: 2026-07-02
companies: []
---

# Natural Language Processing — Interview Questions

> **Category:** NLP · **Difficulty:** mixed (easy → hard)
>
> A question bank tied to **[Module 5 — NLP](../../05-nlp/README.md)**.
> Try to answer before reading the response.

---

## Contents

1. [Stemming vs lemmatization](#1-stemming-vs-lemmatization)
2. [Should you always remove stopwords?](#2-should-you-always-remove-stopwords)
3. [Why subword tokenization?](#3-why-subword-tokenization)
4. [How BPE works](#4-how-bpe-works)
5. [BoW vs TF-IDF](#5-bow-vs-tf-idf)
6. [What TF-IDF cannot capture](#6-what-tf-idf-cannot-capture)
7. [The distributional hypothesis](#7-the-distributional-hypothesis)
8. [Word2Vec: CBOW vs skip-gram](#8-word2vec-cbow-vs-skip-gram)
9. [FastText's advantage](#9-fasttexts-advantage)
10. [Static vs contextual embeddings](#10-static-vs-contextual-embeddings)
11. [Why Sentence-BERT?](#11-why-sentence-bert)
12. [Semantic search mechanics](#12-semantic-search-mechanics)
13. [The BIO tagging scheme](#13-the-bio-tagging-scheme)
14. [Evaluating NER](#14-evaluating-ner)
15. [What dependency parsing gives you](#15-what-dependency-parsing-gives-you)
16. [Extractive vs abstractive summarization](#16-extractive-vs-abstractive-summarization)
17. [BLEU vs ROUGE](#17-bleu-vs-rouge)
18. [Perplexity](#18-perplexity)
19. [Hard cases in sentiment analysis](#19-hard-cases-in-sentiment-analysis)
20. [Train/serve preprocessing skew](#20-trainserve-preprocessing-skew)

---

### 1. Stemming vs lemmatization
**A:** Both reduce words to a base form. **Stemming** chops suffixes with rules
(fast, crude — "studies"→"studi"); **lemmatization** uses vocabulary and POS to
return real dictionary forms ("studies"→"study"; "better"→"good") — slower but
cleaner. Prefer lemmatization when interpretability matters; skip both for
transformer inputs.

### 2. Should you always remove stopwords?
**A:** No. It helps sparse lexical models (TF-IDF) by cutting noise, but it
destroys negation ("not good" → "good") and hurts transformers, which use
function words for context. Decide per representation and validate empirically.

### 3. Why subword tokenization?
**A:** Word-level vocabularies explode and still hit out-of-vocabulary words;
character-level makes sequences too long. Subwords (BPE/WordPiece/SentencePiece)
keep a compact vocabulary while composing any unseen word from pieces — which is
why every modern LLM uses them. See [Tokenization](../../05-nlp/lessons/tokenization.md).

### 4. How BPE works
**A:** Start from characters; repeatedly count adjacent symbol pairs across the
corpus and merge the most frequent pair into a new symbol; stop after N merges.
Frequent words become single tokens, rare words decompose into subwords.

### 5. BoW vs TF-IDF
**A:** Bag of Words uses raw counts, so frequent-everywhere terms dominate.
TF-IDF multiplies term frequency by inverse document frequency
($\log N/\text{df}$), down-weighting ubiquitous terms and surfacing distinctive
ones — usually a strictly better sparse baseline.

### 6. What TF-IDF cannot capture
**A:** Semantics and word order: "car" and "automobile" are orthogonal
dimensions; "dog bites man" equals "man bites dog". Embeddings fix the first;
sequence models the second.

### 7. The distributional hypothesis
**A:** "You shall know a word by the company it keeps" (Firth): words in similar
contexts have similar meanings. It's the training signal behind Word2Vec, GloVe,
and ultimately language-model pretraining.

### 8. Word2Vec: CBOW vs skip-gram
**A:** **CBOW** predicts the center word from its context (fast, good for
frequent words); **skip-gram** predicts context words from the center word
(slower, better for rare words/small corpora). Both are trained efficiently with
negative sampling.

### 9. FastText's advantage
**A:** It represents words as sums of character n-gram vectors, so it (a) builds
embeddings for **unseen** words and (b) captures morphology ("teach"/"teacher"
share n-grams) — valuable for morphology-rich languages and noisy text.

### 10. Static vs contextual embeddings
**A:** Static (Word2Vec/GloVe/FastText) assign **one** vector per word, so
polysemy is collapsed ("bank"). Contextual models (transformers) produce a
different vector per occurrence, disambiguating by context — the key upgrade
behind modern NLP.

### 11. Why Sentence-BERT?
**A:** Vanilla BERT doesn't yield good sentence vectors (and pairwise scoring is
O(n²) cross-encoding). SBERT trains a siamese network so a single `encode()`
gives vectors whose **cosine similarity is meaningful** — enabling fast semantic
search, clustering, and STS (Reimers & Gurevych, 2019).

### 12. Semantic search mechanics
**A:** Embed the corpus once (normalized), index the vectors; at query time embed
the query and score all docs by dot product/cosine, returning top-k. Evaluate
with Recall@k and MRR. It's the retriever inside RAG; hybrid lexical+semantic
covers exact identifiers too.

### 13. The BIO tagging scheme
**A:** Token-level NER labels: **B-**egin marks an entity's first token, **I-**nside
its continuation, **O** everything else — e.g. "Satya/B-PER Nadella/I-PER
visited/O". It turns span extraction into per-token classification and cleanly
handles adjacent entities.

### 14. Evaluating NER
**A:** Entity-level precision/recall/F1 with **exact match**: a predicted entity
counts only if both its span and type are right. Partial overlaps count as
errors under the strict convention — token accuracy would grossly flatter the
model since most tokens are O.

### 15. What dependency parsing gives you
**A:** Head→dependent syntactic relations (nsubj, dobj, amod…), i.e. *who did
what to whom*. It powers interpretable relation extraction and aspect-based
sentiment even in the transformer era. See
[Dependency Parsing](../../05-nlp/lessons/dependency-parsing.md).

### 16. Extractive vs abstractive summarization
**A:** Extractive selects existing sentences (guaranteed faithful, often
choppy); abstractive generates new text (fluent, but can **hallucinate**
unsupported facts — the central risk to check). Evaluate with ROUGE plus a
human/factuality pass.

### 17. BLEU vs ROUGE
**A:** Both count n-gram overlap with references. **BLEU** (translation) is
precision-oriented with a brevity penalty; **ROUGE** (summarization) is
recall-oriented (ROUGE-N, ROUGE-L for longest common subsequence). Both punish
valid paraphrases and aren't comparable across datasets/tokenizations.

### 18. Perplexity
**A:** $\text{PPL} = \exp$(average negative log-likelihood) — the exponentiated
cross-entropy of a language model on text. Intuition: the model's effective
branching factor per token; lower is better. Only comparable under the same
tokenizer and data.

### 19. Hard cases in sentiment analysis
**A:** Negation scope ("not exactly great"), sarcasm/irony, comparative
sentences ("better than X but…"), mixed aspect-level polarity ("food great,
service awful"), and domain-dependent polarity ("unpredictable" plot vs
steering). Aspect-based sentiment and contextual models address parts of these.

### 20. Train/serve preprocessing skew
**A:** If serving applies different cleaning/tokenization than training, the
model receives out-of-distribution features and quietly degrades. Fix: one
canonical preprocessing function baked *into* the persisted pipeline artifact,
plus a parity test in CI. See
[the mismatch debug exercise](../../05-nlp/exercises/debug-preprocessing-mismatch.md).

---

## Common Mistakes

- ⚠️ Treating BLEU/ROUGE as ground truth rather than proxies.
- ⚠️ Claiming Word2Vec handles polysemy — it can't (one vector per word).
- ⚠️ Reporting token accuracy for NER.
- ⚠️ Forgetting that vectorizers are *learned* and must be fit on train only.

## Key Takeaways

- Know which representation fits which task — and its failure modes.
- Every metric answer should include what the metric *misses*.

## Related

- [Module 5 — NLP](../../05-nlp/README.md)
- [NLP Tasks & Metrics Cheat Sheet](../../20-cheat-sheets/nlp/tasks-and-metrics.md)

---

## Navigation

- ⬆️ [NLP Interview Questions](README.md)
- 📚 [Interview Preparation](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
