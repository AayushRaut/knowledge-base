# Assets

> Shared images and diagrams referenced by content across the knowledge base.

**Status:** 🟢 Ready for use
**Last reviewed:** 2026-07-02

---

## Structure

| Folder | Contents |
|--------|----------|
| [`images/`](images/) | Screenshots, photos, static illustrations (PNG/JPG/SVG) |
| [`diagrams/`](diagrams/) | Exported diagram sources and images |

---

## Conventions

- **Naming:** `kebab-case`, prefixed by domain when useful —
  `rag-retrieval-flow.png`, `transformer-attention.svg`.
- **Prefer Mermaid** (inline in Markdown) for diagrams so they render on GitHub
  and Obsidian and stay diff-friendly. Use static images only when Mermaid can't
  express it.
- **Always add alt text:** `![RAG retrieval flow](../assets/diagrams/rag-retrieval-flow.png)`.
- **Optimize** large images before committing; never commit huge binaries
  (see [`.gitignore`](../.gitignore)).
- **Reference with relative paths** from the content file.

---

## Navigation

- 🏠 [Knowledge Base Home](../README.md)
- 📐 [Style Guide](../.claude/STYLE_GUIDE.md#8-diagrams--images)
