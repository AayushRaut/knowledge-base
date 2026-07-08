---
title: "Exercise: Extract Entities and SVO Triples"
description: Use spaCy to pull named entities and subject–verb–object triples out of raw sentences.
type: exercise
domain: 05-nlp
tags: [nlp, ner, dependency-parsing, spacy, information-extraction]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 25 min
reinforces:
  - 05-nlp/lessons/pos-tagging-and-ner
  - 05-nlp/lessons/dependency-parsing
---

# Exercise: Extract Entities and SVO Triples

> Practice for **[POS Tagging and NER](../lessons/pos-tagging-and-ner.md)** and
> **[Dependency Parsing](../lessons/dependency-parsing.md)**.

---

## Problem

Given news-style sentences, extract (a) all named entities with their labels and
(b) simple **(subject, verb, object)** triples using the dependency tree.

```python
extract("Microsoft acquired GitHub. Satya Nadella praised the deal in Seattle.")
# entities: [("Microsoft", "ORG"), ("GitHub", "ORG"),
#            ("Satya Nadella", "PERSON"), ("Seattle", "GPE")]
# triples:  [("Microsoft", "acquired", "GitHub"),
#            ("Satya Nadella", "praised", "deal")]
```

## Requirements

- [ ] Entities via `doc.ents` with labels.
- [ ] Triples via the dependency tree: for each verb, find its `nsubj` and
      `dobj` children.
- [ ] Handle sentences with no triple gracefully (skip).

---

## Hints

<details>
<summary>Hint 1</summary>

Load `en_core_web_sm` once at module level. Iterate `doc.sents`, then tokens
with `token.pos_ == "VERB"`.

</details>

<details>
<summary>Hint 2</summary>

`[c for c in verb.children if c.dep_ == "nsubj"]` gives the subject; use
`c.dep_ in ("dobj", "obj")` for the object. For multi-word entities/nouns,
`token.subtree` or the entity span text reads better than the bare token.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import spacy

nlp = spacy.load("en_core_web_sm")   # small English pipeline (install once)


def extract(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    triples: list[tuple[str, str, str]] = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB":
                continue
            subs = [c for c in token.children if c.dep_ == "nsubj"]
            objs = [c for c in token.children if c.dep_ in ("dobj", "obj")]
            for s in subs:
                for o in objs:
                    # Expand to full noun phrases / entity spans where possible
                    subj = next((e.text for e in doc.ents if s in e), s.text)
                    obj = next((e.text for e in doc.ents if o in e), o.text)
                    triples.append((subj, token.lemma_, obj))
    return entities, triples
```

**Explanation:** NER gives you *who/what is mentioned*; the dependency tree gives
you *how they relate*. Walking from each verb to its `nsubj` and `dobj` children
turns free text into structured facts — the core move of information extraction
and knowledge-graph population. Real systems add passive voice (`nsubjpass`),
conjunctions, and clausal objects, but this skeleton is the honest 80%.

</details>

---

## Navigation

- ⬆️ [Module 5 Exercises](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
