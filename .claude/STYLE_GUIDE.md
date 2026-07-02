# Style Guide

The rules that keep thousands of files consistent and readable. When in doubt,
match existing, reviewed content and prefer clarity.

---

## 1. Files & folders

- **Naming:** `kebab-case` for everything (`word-embeddings.md`, not
  `Word_Embeddings.md`).
- **Numeric prefixes** (`NN-`) only for ordered top-level domains.
- **Folder index:** every folder has a `README.md` that links its children.
- **One concept per file.** If a file exceeds ~400 lines, split it.
- **Assets** go in [`assets/`](../assets/) (`images/`, `diagrams/`), referenced
  with relative paths.

---

## 2. Frontmatter

Every content file begins with a YAML frontmatter block. Fields vary by type
(see [templates/](../templates/)), but these are common:

```yaml
---
title: Word Embeddings
description: Dense vector representations of words and how they are learned.
domain: 05-nlp
tags: [nlp, embeddings, word2vec, glove]
difficulty: beginner   # beginner | intermediate | advanced
status: draft          # draft | review | complete
created: 2026-07-02
updated: 2026-07-02
prerequisites:
  - 05-nlp/text-preprocessing
---
```

- Use ISO dates (`YYYY-MM-DD`).
- Keep `tags` lowercase and specific.
- Update `updated` on every substantive edit.

---

## 3. Document structure

1. **One H1 (`#`)** per file — the title, matching frontmatter `title`.
2. Then `##` sections, `###` subsections. Never skip levels.
3. Recommended lesson flow:
   `Overview → Intuition → Details → Example → Pitfalls → Summary → Further reading`.
4. End substantial notes with a **Summary** and a **Further reading** list.

---

## 4. Writing style

- **Voice:** clear, direct, second person ("you"). Active voice.
- **Tone:** professional and friendly; explain, don't lecture.
- **Concision:** short paragraphs (2–5 sentences). Prefer lists and tables for
  structured information.
- **Define before you use.** Introduce a term, then its acronym: "Retrieval-
  Augmented Generation (RAG)".
- **No fluff.** Cut "As we all know", "Obviously", and filler.
- **Accuracy:** never fabricate numbers, benchmarks, or citations. Mark
  uncertainty explicitly.

---

## 5. Code

- Always specify the language in fenced blocks:

  ````
  ```python
  def cosine_similarity(a, b):
      return (a @ b) / (norm(a) * norm(b))
  ```
  ````

- Keep examples **minimal and runnable**. Show imports when non-obvious.
- Prefer standard, current libraries. Note versions when behavior is
  version-sensitive.
- Comment the *why*, not the obvious *what*.

---

## 6. Math

- Inline math with `$...$`, display math with `$$...$$` (KaTeX/Obsidian syntax).
- Define every symbol the first time it appears.

```
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
```

---

## 7. Links & cross-references

- Use **relative paths** for internal links: `[chunking](../09-rag/chunking-strategies/README.md)`.
- Obsidian `[[wiki-links]]` are allowed for note-to-note links; ensure they also
  make sense on GitHub where possible.
- Every external claim gets a source link.

---

## 8. Diagrams & images

- Prefer **Mermaid** for diagrams (renders on GitHub and Obsidian):

  ````
  ```mermaid
  flowchart LR
    Q[Query] --> R[Retriever] --> C[Context] --> L[LLM] --> A[Answer]
  ```
  ````

- Store static images in [`assets/images/`](../assets/images/); always add alt text.

---

## 9. Admonitions / callouts

Use blockquote callouts sparingly for emphasis:

```
> [!NOTE] Useful context.
> [!TIP] A practical shortcut.
> [!WARNING] A common mistake or risk.
```

---

## 10. Status & review

- Mark maturity with `status` frontmatter and the legend:
  🟢 complete · 🟡 in progress · 🔴 placeholder.
- Add a **Last reviewed** date to fast-moving topics.
- Prefer evergreen phrasing; when citing "latest", pin a date or version.

---

## 11. Consistency checklist

- [ ] kebab-case filename in the correct folder.
- [ ] Complete, valid frontmatter.
- [ ] Single H1; ordered headings.
- [ ] Language-tagged code blocks.
- [ ] Relative internal links that resolve.
- [ ] Sources cited; no fabricated data.
- [ ] Parent `README.md` updated.
