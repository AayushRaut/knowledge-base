# Repository Rules

Hard constraints that govern structure and change. These are stricter than the
[Style Guide](STYLE_GUIDE.md) (which covers *how content reads*); this document
covers *how the repository is organized and evolves*.

---

## 1. Structure is sacred

- The **numbered top-level domains** (`00-` … `20-`, plus `resources/`,
  `templates/`, `assets/`, `.claude/`) are the canonical taxonomy.
- **Do not add, rename, or renumber top-level domains** without a proposal in
  [ROADMAP.md](ROADMAP.md) and maintainer approval.
- Subtopic folders may be added freely **within** a domain, provided each new
  folder gets a `README.md` index and is linked from its parent.

---

## 2. Every folder has an index

- Each folder **must** contain a `README.md` that:
  - States what the folder covers.
  - Links to every child file/folder.
  - Links back to its parent (`../README.md`).
- A folder without an index is considered broken.

---

## 3. One concept per file

- Files are small and single-purpose. Split when a file grows past ~400 lines or
  covers more than one idea.
- Reuse via linking, not duplication. If two notes overlap, extract the shared
  idea into its own note and link to it.

---

## 4. Naming

- `kebab-case` for all files and folders.
- Numeric prefixes (`NN-`) **only** for ordered top-level domains — not for
  individual lessons.
- No spaces, no uppercase, no special characters in paths.

---

## 5. Templates are mandatory

- All new content is created from [`templates/`](../templates/).
- Do not invent alternative document shapes. If a template is missing a field
  you need, update the template (and note it) rather than diverging ad hoc.

---

## 6. Content integrity

- **No fabricated facts, numbers, benchmarks, or citations.** Ever.
- Every non-trivial claim has a source.
- Mark uncertainty explicitly; never present a guess as fact.
- Fast-moving topics carry a review date and version notes.

---

## 7. What must never be committed

- Secrets, API keys, tokens, credentials, `.env` files.
- Large binaries: model weights, datasets, checkpoints (see [`.gitignore`](../.gitignore)).
- Generated caches, virtual environments, or editor state.
- Copyrighted material reproduced without permission — summarize and link instead.

---

## 8. Change management

- **Branch per topic.** Naming: `add/`, `fix/`, `refactor/`, `chore/`
  (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
- **Conventional Commits** for messages.
- **Small, reviewable PRs.** Update the parent `README.md` in the same PR that
  adds a file.
- The default branch is `main`.

---

## 9. Links

- Internal links use **relative paths** and must resolve.
- Obsidian `[[wiki-links]]` are permitted for note graphs but should not be the
  *only* way to reach a file — the folder `README.md` must still link it.
- Broken links are treated as bugs.

---

## 10. Roles

| Role | Responsibility |
|------|----------------|
| **Maintainer** | Owns taxonomy, approves structural changes, reviews PRs. |
| **Contributor** | Adds/edits content within the rules. |
| **AI assistant** | Maintains structure, drafts from templates on request, never invents facts or structure. See [CLAUDE.md](CLAUDE.md). |

---

## 11. Precedence

When guidance conflicts, follow this order:

1. **REPOSITORY_RULES.md** (this file)
2. [STYLE_GUIDE.md](STYLE_GUIDE.md)
3. [CONTRIBUTING.md](../CONTRIBUTING.md)
4. Existing reviewed content

Anything not covered here defaults to "keep it consistent with what already
exists, and ask."
