---
title: Dependency Parsing
description: How dependency parsing reveals who-did-what-to-whom structure through head-dependent relations, and how to navigate parse trees with spaCy.
type: lesson
domain: 05-nlp
tags: [nlp, dependency-parsing, syntax, spacy]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 35 min
prerequisites:
  - 05-nlp/lessons/pos-tagging-and-ner
---

# Dependency Parsing

> **TL;DR:** A dependency parse connects every word to the word it modifies (its head) with a labeled arc, turning a flat sentence into a tree that answers "who did what to whom." spaCy exposes the tree through `token.dep_`, `token.head`, and `token.children`, which is all you need for interpretable relation extraction.

---

## Overview

POS tags tell you *what* each word is; dependency parsing tells you *how the words relate*. The parse is a tree: every word points to exactly one head, and the arcs carry labels like `nsubj` (subject) and `dobj` (direct object). This structure is the workhorse behind rule-based relation extraction, aspect-based sentiment, and question understanding — anywhere you need to read grammatical relationships off a sentence programmatically.

**By the end, you will be able to:**
- Read a dependency tree: identify the root, heads, dependents, and the meaning of common labels (`nsubj`, `dobj`, `amod`, `prep`/`pobj`)
- Explain why parsing matters for extraction tasks, and where it stands relative to end-to-end transformer models
- Navigate a spaCy parse with `token.dep_`, `token.head`, and `token.children` to extract (subject, verb, object) triples

---

## Intuition

Take the sentence "*The hungry cat chased a mouse.*" You instinctively know that *hungry* describes *cat* (not *mouse*), that *cat* is the one chasing, and that *mouse* is the one being chased. Word order alone doesn't encode this reliably — "*A mouse was chased by the hungry cat*" scrambles the order but keeps the relationships.

Dependency grammar captures the relationships directly: draw an arrow from each word to the word it depends on. *Hungry* → *cat* ("I modify you"). *Cat* → *chased* ("I am your subject"). *Mouse* → *chased* ("I am your object"). Every word gets exactly one outgoing arrow except one — the main verb, which anchors the whole sentence and is called the **root**. The result is a tree with the root at the top and modifiers hanging below the words they modify.

Once a sentence is a tree, questions like "what is the subject of this verb?" become tree lookups instead of brittle regex over word order.

---

## Details

### Heads, dependents, and labels

Each arc connects a **head** (the governing word) to a **dependent** (the word that modifies or completes it) and carries a **dependency label** naming the grammatical relation. The most useful labels (spaCy's English models use a variant of the Universal Dependencies / ClearNLP label set):

| Label | Relation | In the example below |
|-------|----------|----------------------|
| `nsubj` | nominal subject | *cat* → *chased* |
| `dobj` | direct object | *mouse* → *chased* |
| `amod` | adjectival modifier | *hungry* → *cat* |
| `det` | determiner | *the* → *cat* |
| `prep` | prepositional modifier | *in* → *chased* |
| `pobj` | object of a preposition | *garden* → *in* |
| `ROOT` | sentence root | *chased* |

Text-diagrammed example — "*The hungry cat chased a mouse in the garden.*":

```text
chased (ROOT)
├── cat (nsubj)          "who chased?"
│   ├── The (det)
│   └── hungry (amod)    "which cat? the hungry one"
├── mouse (dobj)         "chased what?"
│   └── a (det)
└── in (prep)            "chased where?"
    └── garden (pobj)
        └── the (det)
```

Read arcs as questions: `nsubj` answers *who?*, `dobj` answers *what?*, `prep`+`pobj` answers *where/when/how?*. Note the two-step pattern for prepositional phrases: the preposition attaches to the verb (`prep`), and the noun attaches to the preposition (`pobj`).

### Dependency trees vs constituency trees

Two mainstream ways to represent syntax:

- **Constituency (phrase-structure) trees** group words into nested phrases: sentence → noun phrase + verb phrase → … Useful in linguistics and grammar-driven applications.
- **Dependency trees** skip phrases and link words directly with typed relations.

Dependency parses dominate applied NLP because word-to-word relations map directly onto extraction questions ("find the subject of *acquired*"), they handle free-word-order languages more naturally, and they are compact — one node per word. Jurafsky & Martin cover both formalisms in depth.

### How parsers work (conceptual)

The most influential family is **transition-based parsing**. Picture a stack and a buffer of remaining words. The parser repeatedly chooses one of a few actions — *shift* the next word onto the stack, or attach the top stack items with a *left-arc*/*right-arc* — until the buffer is empty and a full tree is built. A trained classifier (today, a neural network over contextual token representations) picks each action. This makes parsing fast — roughly linear in sentence length — which is why spaCy's parser is practical to run on millions of documents. Graph-based parsers, the other family, score all candidate head-dependent pairs and find the best tree globally.

You will rarely implement a parser; you will *consume* parses. Knowing the transition-based idea mainly helps you understand parser errors: a wrong early attachment can cascade.

### Do you still need parsing in the transformer era?

An honest note: modern transformer models learn syntax-like structure *implicitly* from raw text, and for many end-to-end tasks (classification, QA, generation) you never need an explicit parse. But explicit parsing remains valuable when you need:

- **Interpretability** — a rule "subject of *acquire* where object is an ORG" can be read, audited, and fixed; a fine-tuned model's decision cannot.
- **No training data** — dependency rules work out of the box on a new extraction task.
- **Precision-first extraction** — in legal, medical, and financial pipelines, high-precision syntactic rules are often preferred over recall-oriented learned extractors.

### Python: navigating the parse in spaCy

Every spaCy token carries its parse information:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The hungry cat chased a mouse in the garden.")

for token in doc:
    print(f"{token.text:<8} {token.dep_:<8} head={token.head.text:<8} "
          f"children={[c.text for c in token.children]}")
```

Key attributes:

- `token.dep_` — the dependency label of the arc from this token to its head
- `token.head` — the token this one attaches to (the root's head is itself)
- `token.children` — iterator over this token's direct dependents
- `token.subtree` — this token plus everything below it (great for extracting whole phrases)

Visualize the arcs while learning — it is the fastest way to build intuition:

```python
from spacy import displacy

displacy.render(doc, style="dep")   # displacy.serve(...) outside notebooks
```

## Worked Example

Extract (subject, verb, object) triples — the core of "who did what to whom" relation extraction.

```python
import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")

def extract_svo(doc: Doc) -> list[tuple[str, str, str]]:
    """Extract (subject, verb, object) triples from a parsed doc."""
    triples: list[tuple[str, str, str]] = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        subjects = [c for c in token.children if c.dep_ == "nsubj"]
        objects = [c for c in token.children if c.dep_ == "dobj"]
        for subj in subjects:
            for obj in objects:
                # Use the subtree to capture full noun phrases,
                # e.g. "The Berlin startup" instead of just "startup".
                subj_phrase = " ".join(t.text for t in subj.subtree)
                obj_phrase = " ".join(t.text for t in obj.subtree)
                triples.append((subj_phrase, token.lemma_, obj_phrase))
    return triples

text = (
    "Anthropic released a new model. "
    "The Berlin startup acquired its biggest competitor."
)
for sent in nlp(text).sents:
    for subj, verb, obj in extract_svo(sent.as_doc()):
        print(f"({subj!r}, {verb!r}, {obj!r})")

# ('Anthropic', 'release', 'a new model')
# ('The Berlin startup', 'acquire', 'its biggest competitor')
```

Walkthrough for the second sentence: the parser marks *acquired* as `ROOT`; *startup* attaches to it as `nsubj` (with *The* and *Berlin* below it), and *competitor* attaches as `dobj` (with *its* and *biggest* below it). The function finds the verb, follows the `nsubj` and `dobj` children, and expands each to its full subtree — three tree lookups replacing an unwriteable regex.

This basic extractor misses passives ("*was acquired by*" uses `nsubjpass` and an *agent* phrase) and clausal objects; production systems add those patterns one labeled case at a time.

## Best Practices

- ✅ Parse and extract **per sentence** (`doc.sents`) — cross-sentence arcs do not exist, and per-sentence processing keeps rules simple.
- ✅ Use `token.subtree` to capture complete phrases instead of single head words.
- ✅ Inspect parses with `displacy` while developing rules; write rules against what the parser *actually* outputs, not what grammar books say.
- ✅ Match on `token.lemma_` (e.g. `acquire`) rather than surface forms so one rule covers *acquires/acquired/acquiring*.

## Common Mistakes

- ⚠️ **Confusing the arc direction.** `token.dep_` labels the relation *to the head*: in "*cat chased*", *cat*'s dep is `nsubj` and *cat*'s head is *chased* — the verb has no "subject" attribute; you find subjects by scanning the verb's children. 
- ⚠️ **Forgetting passive voice.** "*The mouse was chased*" has no `nsubj` on *chased* — the mouse is `nsubjpass`. Extraction rules that only check `nsubj`/`dobj` silently drop passives; add the passive patterns.
- ⚠️ **Trusting parses on noisy text.** Parsers are trained mostly on edited text; accuracy drops on tweets, ASR transcripts, and fragments. Spot-check before building rules on such data.
- ⚠️ **Reinventing tree search with string offsets.** If you find yourself matching words by position, step back and use `head`/`children`/`subtree` — that is what the tree is for.

## Industry Tips

- 💡 A common production pattern: dependency rules for high-precision extraction, with a learned model as a recall backstop — the rules also generate weak labels for training that model.
- 💡 Combine the parser with NER: "`nsubj` of *acquire* must be an `ORG`" filters out most false positives in one condition.
- 💡 Disable the parser (`nlp.pipe(texts, disable=["parser"])`) when you only need tagging or NER — it is one of the more expensive pipeline components.

## Real-World Use Cases

- Relation extraction for knowledge graphs: (company, *acquired*, company), (person, *founded*, org)
- Aspect-based sentiment: attaching opinion words to the product feature they modify ("the *battery* is **terrible**, the *screen* is **great**")
- Question understanding: mapping "Who founded SpaceX?" to a query over subject/verb/object structure
- Compliance and contract analysis: finding which party carries an obligation expressed by verbs like *shall pay*

---

## Summary

- A dependency parse links every word to its head with a labeled arc (`nsubj`, `dobj`, `amod`, `prep`/`pobj`, …), forming a tree rooted at the main verb — grammar you can traverse in code.
- Transition-based parsers build the tree with a fast sequence of shift/attach actions chosen by a trained classifier; you consume the result via `token.dep_`, `token.head`, `token.children`, and `token.subtree`.
- Transformers learn syntax implicitly, but explicit parses remain the tool of choice for interpretable, training-data-free, precision-first extraction.

## Practice

- [ ] Exercises: [Module 5 Exercises](../exercises/README.md)
- [ ] Self-check: In "*The startup in Berlin raised new funding*", what is the head of *Berlin*, and via which two arcs is *Berlin* connected to *startup*?

## Further Reading

- 📘 Speech and Language Processing — Jurafsky & Martin (https://web.stanford.edu/~jurafsky/slp3/)
- 📄 [spaCy documentation](https://spacy.io/)

## Related

- [POS Tagging and NER](pos-tagging-and-ner.md)

---

## Navigation

- ⬆️ [Lessons](README.md)
- 📚 [Module 5 — Natural Language Processing](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
