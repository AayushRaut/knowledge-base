# CLAUDE.md — Working in this Repository

This file orients AI assistants (Claude Code and others) and human maintainers
who use AI tooling. Read it before making changes.

---

## What this repository is

A **long-term AI Engineering Knowledge Base**: a structured collection of
lessons, exercises, projects, interview prep, and cheat sheets spanning the full
AI engineering stack. It is an [Obsidian](https://obsidian.md) vault that also
renders as plain Markdown on GitHub.

It is **content-first**, not a software project. There is no build step, no test
suite, and no application to run. The "product" is well-organized Markdown.

---

## Golden rules

1. **Do not invent structure.** Use the existing folder hierarchy and the
   [templates](../templates/). If a new location seems needed, propose it via
   [ROADMAP.md](ROADMAP.md) rather than creating ad-hoc folders.
2. **One concept per file**, kebab-case names, and a `README.md` index in every
   folder.
3. **Never fabricate facts, benchmarks, or citations.** If unsure, mark it
   clearly and link a source. Technical accuracy is the top priority.
4. **Match the [Style Guide](STYLE_GUIDE.md) exactly** — frontmatter, headings,
   tone, and code-block conventions.
5. **Keep changes small and reviewable.** One topic per branch/PR.
6. **Update the parent `README.md`** whenever you add or move a file.
7. **No secrets or large binaries.** Respect [`.gitignore`](../.gitignore).

---

## Repository layout

```
knowledge-base/
├── 00-getting-started/        Orientation, learning paths, FAQ
├── 01-python-languages/       Programming foundations (pre-existing)
├── 02-mathematics-foundations/
├── 03-machine-learning/
├── 04-deep-learning/
├── 05-nlp/
├── 06-transformers/
├── 07-large-language-models/
├── 08-prompt-engineering/
├── 09-rag/
├── 10-ai-agents/
├── 11-fine-tuning/
├── 12-ai-engineering/
├── 13-mlops/
├── 14-llmops/
├── 15-deployment/
├── 16-architecture/
├── 17-production-applications/
├── 18-projects/
├── 19-interview-preparation/
├── 20-cheat-sheets/
├── resources/                 Papers, books, courses, datasets
├── templates/                 Canonical templates for all content types
├── assets/                    Images and diagrams
└── .claude/                   Governance: this file + vision, style, rules, roadmap
```

Each numbered domain contains subtopic folders, each with its own `README.md`.

---

## How to add content (the standard task)

1. Identify the correct `NN-domain/subtopic/` folder.
2. Copy the matching file from [`templates/`](../templates/).
3. Fill in the frontmatter and body per the [Style Guide](STYLE_GUIDE.md).
4. Add a link to the new file in the folder's `README.md`.
5. Keep it self-contained; cross-link related notes with relative links or
   Obsidian `[[wiki-links]]`.

**Do not generate lesson content unless explicitly asked.** The default posture
is to maintain and extend *structure*, templates, and indexes.

---

## Ideas, brainstorming, and the backlog

Ideas and future enhancements go through [BACKLOG.md](BACKLOG.md), not directly
into the canonical docs:

1. **Record brainstorming ideas first in [BACKLOG.md](BACKLOG.md)** as `Proposed`
   items — never edit the curriculum or standards just to capture an idea.
2. **[MASTER_CURRICULUM.md](MASTER_CURRICULUM.md) is updated only after a backlog
   item is `Approved`.** It stays the canonical source of truth in the meantime.
3. **[COURSE_VISION.md](COURSE_VISION.md) remains stable** — it is not changed as
   part of routine idea capture.
4. **Temporary discussions must not modify the repository** unless the change is
   explicitly approved and made through the normal process in
   [REPOSITORY_RULES.md](REPOSITORY_RULES.md).

---

## Conventions cheat-sheet

| Thing | Rule |
|-------|------|
| File/folder names | `kebab-case` |
| Ordered domains | numeric prefix `NN-` |
| Folder index | `README.md` |
| Headings | one `#` H1 per file, then `##`/`###` |
| Frontmatter | YAML block per template |
| Links | relative paths; `[[wiki-links]]` allowed for Obsidian |
| Status markers | 🟢 done · 🟡 in progress · 🔴 placeholder |
| Dates | ISO `YYYY-MM-DD` |

---

## Related governance docs

- **[MASTER_CURRICULUM.md](MASTER_CURRICULUM.md)** — the canonical curriculum specification (all modules, requirements, tools, roadmap).
- **[COURSE_VISION.md](COURSE_VISION.md)** — the "why" and long-term goals.
- **[STYLE_GUIDE.md](STYLE_GUIDE.md)** — how content must look and read.
- **[REPOSITORY_RULES.md](REPOSITORY_RULES.md)** — hard constraints and structure.
- **[ROADMAP.md](ROADMAP.md)** — what is being built and in what order.
- **[BACKLOG.md](BACKLOG.md)** — planning-only capture of ideas and enhancements (not a source of truth).
