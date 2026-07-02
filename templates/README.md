# Templates

> Canonical starting points for every kind of content in this knowledge base.
> **Always copy from here** — do not hand-roll structure.

**Status:** 🟢 Complete
**Last reviewed:** 2026-07-02

---

## Available templates

| Template | Use it for | Typical location |
|----------|-----------|------------------|
| [lesson-template.md](lesson-template.md) | A single concept explained end-to-end | `NN-domain/subtopic/` |
| [exercise-template.md](exercise-template.md) | Short practice with hints + solution | Alongside its lesson |
| [assignment-template.md](assignment-template.md) | A larger, graded/assessed task with a rubric | Alongside a module |
| [project-template.md](project-template.md) | A hands-on build with deliverables | [`18-projects/`](../18-projects/README.md) |
| [interview-template.md](interview-template.md) | An interview question with answers + follow-ups | [`19-interview-preparation/`](../19-interview-preparation/README.md) |
| [cheat-sheet-template.md](cheat-sheet-template.md) | A dense, fast-reference summary | [`20-cheat-sheets/`](../20-cheat-sheets/README.md) |

---

## How to use

```bash
# Copy the template into the correct folder, then edit it.
cp templates/lesson-template.md 05-nlp/word-embeddings/word2vec.md
```

1. Fill in the **frontmatter** completely (see the [Style Guide](../.claude/STYLE_GUIDE.md#2-frontmatter)).
2. Write the body, keeping to **one concept per file**.
3. Link the new file from its folder's `README.md`.

---

## Changing a template

Templates are shared infrastructure. If you need a new field or section, update
the template itself (and note it in your PR) rather than diverging in one file.
See [Repository Rules §5](../.claude/REPOSITORY_RULES.md).

---

## Navigation

- 🏠 [Knowledge Base Home](../README.md)
- 📐 [Style Guide](../.claude/STYLE_GUIDE.md)
