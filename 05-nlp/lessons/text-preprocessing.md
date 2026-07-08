---
title: Text Preprocessing
description: Clean and normalize raw text into a consistent form that downstream NLP models can consume reliably.
type: lesson
domain: 05-nlp
tags: [nlp, preprocessing, normalization]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 35 min
prerequisites:
  - 01-python-languages/lessons/python-refresher
---

# Text Preprocessing

> **TL;DR:** Raw text is noisy and inconsistent; preprocessing normalizes it into a predictable form. Which steps you apply depends on the task — and training and inference must use the *identical* pipeline.

---

## Overview

Every NLP system starts with messy input: mixed casing, HTML fragments, URLs, emojis, typos, and Unicode quirks. Preprocessing decides what your model actually sees, so it silently shapes every downstream result. For AI engineering, the pipeline matters as much as the model: a mismatch between how you cleaned training data and how you clean production input is one of the most common (and hardest to debug) failure modes.

**By the end, you will be able to:**
- Explain what each common preprocessing step does and when to skip it
- Choose between stemming and lemmatization based on the accuracy/speed trade-off
- Build a reusable, testable `preprocess(text)` pipeline with NLTK and spaCy

---

## Intuition

Think of preprocessing as standardizing ingredients before cooking. "Apple", "apple", "APPLES!!!", and "🍎 apples" all refer to roughly the same thing, but to a computer they are four completely different strings. Preprocessing collapses these surface variations so the model can focus on meaning instead of formatting accidents.

But standardization destroys information. Lowercasing erases the difference between "US" (the country) and "us" (the pronoun). Removing stopwords turns "not good" into "good". So preprocessing is not a fixed checklist — it is a set of *dials* you tune per task. A classical bag-of-words classifier wants aggressive normalization; a modern transformer wants almost none, because its tokenizer and training data already handle raw text.

---

## Details

### Lowercasing and Unicode normalization

Lowercasing shrinks the vocabulary ("The" and "the" merge), which helps small-data classical models. Python's `str.casefold()` is a more aggressive, Unicode-aware version of `str.lower()` (e.g., German `ß` → `ss`).

Unicode is subtler: the same visible character can have multiple byte encodings. `é` can be one code point (U+00E9) or two (`e` + combining accent U+0301). Normalize with `unicodedata.normalize`:

- **NFC** — canonical composition (safe default; preserves appearance)
- **NFKC** — compatibility composition (also folds things like `ﬁ` ligature → `fi`, full-width digits → ASCII digits; lossy but useful for search/matching)

```python
import unicodedata

s1: str = "café"        # 'café' as one code point
s2: str = "café"       # 'café' as e + combining accent
print(s1 == s2)                              # False
print(unicodedata.normalize("NFC", s1) ==
      unicodedata.normalize("NFC", s2))      # True
```

### Removing punctuation, URLs, and HTML

Web text is full of markup and links that rarely help classical models. Strip them with regular expressions (or an HTML parser like `BeautifulSoup` for real HTML):

```python
import re

def strip_noise(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"[^\w\s']", " ", text)                # punctuation (keeps apostrophes)
    return re.sub(r"\s+", " ", text).strip()
```

Caution: punctuation can carry signal. "?" and "!" help sarcasm/sentiment detection, and "$3.50" loses meaning if you delete the period. Decide per task.

### Stopwords — and when NOT to remove them

Stopwords are high-frequency function words ("the", "is", "at") that add little topical signal to bag-of-words models, so removing them shrinks feature space and can reduce noise.

Do **not** remove stopwords when:

- **Negation matters.** "not", "no", "never" are in most stopword lists; dropping them flips sentiment ("not good" → "good").
- **You feed a transformer.** BERT-style and LLM tokenizers were trained on natural text; stopwords carry syntactic structure the model uses. Removing them *hurts*.
- **Phrases matter.** "to be or not to be" is almost entirely stopwords.

```python
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP: set[str] = set(stopwords.words("english")) - {"not", "no", "nor"}  # keep negation

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP]
```

### Stemming vs. lemmatization

Both reduce inflected words to a base form, trading precision for a smaller vocabulary:

- **Stemming** chops suffixes with rules. The classic **Porter stemmer** is fast but crude: it produces non-words ("studies" → "studi") and over-stems ("organization" → "organ").
- **Lemmatization** maps a word to its dictionary form (*lemma*) using vocabulary and part-of-speech context: "studies" → "study", "was" → "be". It is more accurate but slower, since it needs POS tagging or a lookup model.

Rule of thumb: stemming for quick-and-dirty search/IR-style matching at scale; lemmatization when the output is user-facing or accuracy matters; neither for transformers.

```python
from nltk.stem import PorterStemmer
import spacy

stemmer = PorterStemmer()
print([stemmer.stem(w) for w in ["running", "studies", "better"]])
# ['run', 'studi', 'better']

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
doc = nlp("He was running better studies")
print([tok.lemma_ for tok in doc])
# ['he', 'be', 'run', 'well', 'study']
```

### Emojis and numbers

- **Emojis** carry sentiment (😀 vs 😡). For sentiment tasks, keep them or convert to text with the `emoji` package (`emoji.demojize("😀")` → `":grinning_face:"`); for topic classification, removing them is usually fine.
- **Numbers** are often noise for topic models but critical for extraction tasks. A common compromise is replacing each number with a placeholder token like `<NUM>` so the model learns "a number was here" without exploding the vocabulary.

### A reusable, testable pipeline

Wrap the steps in one function with explicit flags, so the exact configuration is visible, versionable, and unit-testable:

```python
import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()
_STOP: set[str] = set(stopwords.words("english")) - {"not", "no", "nor"}

def preprocess(
    text: str,
    *,
    lower: bool = True,
    strip_urls_html: bool = True,
    remove_stop: bool = False,
    stem: bool = False,
) -> list[str]:
    """Normalize text and return a list of cleaned tokens."""
    text = unicodedata.normalize("NFKC", text)
    if strip_urls_html:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    if lower:
        text = text.casefold()
    text = re.sub(r"[^\w\s']", " ", text)
    tokens: list[str] = text.split()
    if remove_stop:
        tokens = [t for t in tokens if t not in _STOP]
    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]
    return tokens

# Unit-testable:
assert preprocess("Visit https://example.com NOW!") == ["visit", "now"]
```

### The identical-pipeline rule

Whatever pipeline produced your training features **must** run, byte-for-byte identically, on every inference input. If training lowercased but production doesn't, "Great" becomes an out-of-vocabulary token and predictions silently degrade. In practice: package the pipeline as code (a function/class with pinned dependency versions), serialize it alongside the model (e.g., inside a scikit-learn `Pipeline`), and write tests that pin its outputs.

## Worked Example

Take a raw product review and run it through the pipeline in two configurations:

```python
raw = 'Loved it!!! <br> NOT worth returning 😀 — see https://shop.example.com/item?id=42'

# Config A: for a bag-of-words classifier
print(preprocess(raw, remove_stop=True, stem=True))
# ['love', 'not', 'worth', 'return']

# Config B: minimal cleaning (e.g., before a transformer tokenizer)
print(preprocess(raw, remove_stop=False, stem=False))
# ['loved', 'it', 'not', 'worth', 'returning']
```

Config A collapses "Loved"/"loved"/"loving" into one feature `love` and keeps the negation `not` because we exempted it from the stopword list — exactly the signal a sentiment classifier needs. Config B preserves the natural word forms for models that handle morphology themselves.

## Best Practices

- ✅ Decide each step *per task*: cleaning that helps bag-of-words often hurts transformers.
- ✅ Keep negation words when sentiment or meaning polarity matters.
- ✅ Normalize Unicode (NFC/NFKC) before any other string comparison or matching.
- ✅ Encapsulate the pipeline in one versioned, unit-tested function.
- ✅ Serialize preprocessing with the model so they cannot drift apart.

## Common Mistakes

- ⚠️ Removing stopwords before sentiment analysis and losing "not"/"no" — exempt negation words or skip stopword removal.
- ⚠️ Applying heavy cleaning (stemming, stopword removal) before a transformer tokenizer — pass near-raw text instead; the tokenizer expects it.
- ⚠️ Training/serving skew: cleaning notebooks apply one pipeline, the API applies another — share a single code path.
- ⚠️ Lowercasing when case is a feature (NER: "Apple" vs "apple") — keep case for entity-sensitive tasks.
- ⚠️ Forgetting Unicode normalization, so visually identical strings never match — normalize first.

## Industry Tips

- 💡 Log a sample of preprocessed production inputs; comparing them against preprocessed training samples catches pipeline drift early.
- 💡 Prefer spaCy over NLTK in production for lemmatization and tokenization — one dependency, fast, and actively maintained; NLTK remains great for teaching and quick experiments.
- 💡 The `disable=["parser", "ner"]` trick when loading spaCy pipelines cuts processing time substantially when you only need tokens/lemmas.

## Real-World Use Cases

- Normalizing user reviews and support tickets before classical sentiment/intent classifiers
- Search indexing, where stemming boosts recall ("running shoes" matches "run shoe")
- Deduplicating scraped web corpora (Unicode + whitespace normalization) before LLM training
- Cleaning HTML/boilerplate out of documents before chunking them for RAG

---

## Summary

- Preprocessing is a set of task-dependent dials, not a fixed checklist; aggressive cleaning helps classical models and hurts transformers.
- Stemming is fast and crude; lemmatization is slower and accurate; neither is needed for modern subword-tokenized models.
- Training and inference must share one identical, versioned, tested pipeline — skew here silently destroys accuracy.

## Practice

- [ ] Exercises: [Module 5 Exercises](../exercises/README.md)
- [ ] Self-check: Why does removing stopwords improve a bag-of-words topic classifier but degrade a BERT-based one?

## Further Reading

- 📘 Speech and Language Processing — Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/)
- 📄 [spaCy documentation](https://spacy.io/)
- 📄 [NLTK documentation](https://www.nltk.org/)

## Related

- [Tokenization](tokenization.md)
- [Classical Text Representation](text-representation.md)
- [Data Cleaning](../../01-python-languages/lessons/data-cleaning-and-wrangling.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
