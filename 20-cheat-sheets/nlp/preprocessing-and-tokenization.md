---
title: NLP Preprocessing & Tokenization Cheat Sheet
description: Text cleaning, normalization, and tokenization choices at a glance.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [nlp, preprocessing, tokenization, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# NLP Preprocessing & Tokenization Cheat Sheet

> Fast reference. For depth, see
> [Text Preprocessing](../../05-nlp/lessons/text-preprocessing.md) and
> [Tokenization](../../05-nlp/lessons/tokenization.md).

---

## Preprocessing steps

| Step | Tool | Note |
|------|------|------|
| Lowercase | `text.lower()` | Skip for cased transformer models. |
| Strip URLs/HTML | regex | `r"https?://\S+"`, `r"<[^>]+>"` |
| Stopwords | NLTK/spaCy lists | **Keep** for transformers & when negation matters. |
| Stemming | NLTK `PorterStemmer` | Fast, crude ("studies"→"studi"). |
| Lemmatization | spaCy `token.lemma_` | Slower, real words ("studies"→"study"). |
| Unicode | `unicodedata.normalize("NFKC", s)` | Normalize confusable characters. |

**Golden rule:** one `preprocess()` function, used identically at train **and**
inference (bake it into the pipeline artifact).

## Tokenization

| Level | Pros | Cons | Used by |
|-------|------|------|---------|
| Word | Interpretable | OOV problem, huge vocab | Classical NLP |
| Character | No OOV, tiny vocab | Long sequences, weak units | Some niche models |
| **Subword** (BPE/WordPiece/SentencePiece) | No OOV, compact vocab | Pieces not always intuitive | GPT, BERT, LLMs |

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
tok.tokenize("unbelievable")   # ['un', '##bel', '##iev', '##able']
```

## spaCy quickstart

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple hired Sam in London.")
[(t.text, t.pos_, t.lemma_, t.dep_) for t in doc]
[(e.text, e.label_) for e in doc.ents]     # [('Apple','ORG'), ...]
```

---

## Gotchas

- ⚠️ Different preprocessing at train vs serve silently destroys accuracy.
- ⚠️ Removing stopwords deletes negation ("not good" → "good").
- ⚠️ Don't stem/lemmatize before a transformer tokenizer — it expects raw-ish text.
- ⚠️ Fit vectorizers/tokenizers on training data only.

---

## Quick Links

- 📖 [Text Preprocessing](../../05-nlp/lessons/text-preprocessing.md) · [Tokenization](../../05-nlp/lessons/tokenization.md)
- 🔗 [spaCy](https://spacy.io/) · [NLTK](https://www.nltk.org/)

---

## Navigation

- ⬆️ [NLP Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
