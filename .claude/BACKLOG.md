# Backlog

> A lightweight capture space for ideas, future enhancements, feature requests,
> and brainstorming — **kept separate from the official curriculum and standards.**

**Status of this document:** 🟡 Living · **Last reviewed:** 2026-07-02

---

## Overview

This file exists so ideas can be recorded **without** touching the canonical
governance documents. It is a **planning document only**:

- ❌ It is **not** a source of truth.
- ❌ It is **not** part of the curriculum.
- ✅ It **is** a place to park, triage, and prioritize ideas.

Workflow:

1. Anyone adds an idea here as a **Proposed** item.
2. The maintainer triages it and sets a status (see below).
3. **Only after an item is `Approved`** does it graduate into the real docs —
   [MASTER_CURRICULUM.md](MASTER_CURRICULUM.md) for scope,
   [ROADMAP.md](ROADMAP.md) for build order, or the relevant content folder.
4. [COURSE_VISION.md](COURSE_VISION.md) stays stable and is not edited from here.

Nothing in this file changes the repository until it is explicitly approved and
implemented through the normal process in
[REPOSITORY_RULES.md](REPOSITORY_RULES.md).

---

## Status

Every item uses exactly one of these statuses:

| Status | Meaning |
|--------|---------|
| **Proposed** | Captured, not yet reviewed. |
| **Approved** | Accepted; may now graduate into the curriculum/roadmap/content. |
| **In Progress** | Being implemented. |
| **Completed** | Done and merged. |
| **Rejected** | Considered and declined (keep for the record + reason). |
| **Deferred** | Good idea, not now — revisit later. |

**Priority** is one of: `High` · `Medium` · `Low`.

---

## Item format

Each backlog item is recorded with these fields. IDs are sequential (`BL-001`,
`BL-002`, …) and never reused.

```
### BL-000 — <Title>
- **Description:** <what the idea is>
- **Reason:** <why it's worth doing>
- **Priority:** High | Medium | Low
- **Status:** Proposed | Approved | In Progress | Completed | Rejected | Deferred
- **Affected Modules:** <module numbers / areas, or "—">
- **Notes:** <anything else>
- **Decision Date:** <YYYY-MM-DD, optional — set when status changes>
```

---

## Categories

### Curriculum Improvements

#### BL-001 — Add per-module estimated study hours
- **Description:** Populate the "estimated study hours" field (already required by the curriculum's Output Format) across all module READMEs as they are built.
- **Reason:** Helps learners plan; the curriculum already mandates it for each module.
- **Priority:** Medium
- **Status:** Proposed
- **Affected Modules:** All
- **Notes:** Module 1 already shows an estimate; standardize the rest.
- **Decision Date:** —

### New Topics

#### BL-002 — Lesson on `uv` workspaces for multi-package repos
- **Description:** Add a Module 1 (or Module 17) note covering `uv` workspaces and monorepo dependency management.
- **Reason:** Modern, increasingly common; complements the existing environments lesson.
- **Priority:** Low
- **Status:** Proposed
- **Affected Modules:** 01-python, 17-mlops-llmops
- **Notes:** Keep vendor-neutral; pair with the packaging lesson.
- **Decision Date:** —

### Module Enhancements

#### BL-003 — Expand Module 1 with more exercises and split long lessons
- **Description:** Add more coding/debugging exercises to Module 1 and split any lesson that grows past the ~400-line guideline.
- **Reason:** Requested by the owner as a future enhancement; strengthens practice depth.
- **Priority:** Medium
- **Status:** Deferred
- **Affected Modules:** 01-python
- **Notes:** Owner said this can be done later. Revisit after more modules exist.
- **Decision Date:** 2026-07-02

### New Projects

#### BL-004 — RAG PDF Q&A application (engineering-labs)
- **Description:** An end-to-end "chat with your PDFs" app (chunking → embeddings → vector store → retrieval → LLM answer).
- **Reason:** Flagship portfolio project; exercises Module 9 end-to-end.
- **Priority:** High
- **Status:** Proposed
- **Affected Modules:** 09-rag, 13-ai-application-development
- **Notes:** Lives in the separate `engineering-labs` repo, not here. Link back to the RAG lessons.
- **Decision Date:** —

### Documentation Improvements

#### BL-005 — Add a glossary to 00-getting-started
- **Description:** A single glossary of recurring terms/acronyms with links to the lessons that define them.
- **Reason:** Reduces repeated definitions; improves navigation for beginners.
- **Priority:** Medium
- **Status:** Proposed
- **Affected Modules:** 00-getting-started
- **Notes:** Could be auto-cross-linked with Obsidian aliases.
- **Decision Date:** —

### Repository Improvements

#### BL-006 — CI link-checker for internal Markdown links
- **Description:** A GitHub Action that fails a PR if any relative link is broken.
- **Reason:** As the repo scales to thousands of files, broken links become a real risk.
- **Priority:** Medium
- **Status:** Proposed
- **Affected Modules:** — (repo-wide)
- **Notes:** A local script already validated Module 1's links; formalize it in CI.
- **Decision Date:** —

### AI Frameworks

#### BL-007 — Decide framework coverage depth (LangGraph, DSPy, etc.)
- **Description:** Decide how deeply to cover specific frameworks vs. framework-agnostic concepts in Modules 9–11.
- **Reason:** Frameworks move fast; over-committing to one risks staleness.
- **Priority:** Low
- **Status:** Proposed
- **Affected Modules:** 09-rag, 10-ai-agents, 11-fine-tuning
- **Notes:** Curriculum already names several; this is about *depth*, not adding scope.
- **Decision Date:** —

### Future Technologies

#### BL-008 — Track notable new model / tooling releases
- **Description:** A watchlist of significant releases that may warrant lesson updates, with review dates.
- **Reason:** Keeps fast-moving topics current without churn.
- **Priority:** Low
- **Status:** Proposed
- **Affected Modules:** 07-large-language-models, 12-multimodal
- **Notes:** Capture here first; only update lessons when a change is durable.
- **Decision Date:** —

### Tooling

#### BL-009 — Add markdownlint + spell-check
- **Description:** Introduce markdownlint and a spell-checker to keep formatting/prose consistent.
- **Reason:** Enforces the Style Guide mechanically at scale.
- **Priority:** Low
- **Status:** Proposed
- **Affected Modules:** — (repo-wide)
- **Notes:** Must be tuned so it doesn't fight Obsidian/Mermaid syntax.
- **Decision Date:** —

### Interview Preparation

#### BL-010 — Grow the interview bank toward the 100-question target
- **Description:** Expand interview Q&A across categories to reach the curriculum's 100-question goal.
- **Reason:** Curriculum sets 100 Q&A as a portfolio target; Module 1 contributes 20.
- **Priority:** Medium
- **Status:** Proposed
- **Affected Modules:** 19-interview-preparation
- **Notes:** Grow in step with each module so questions stay grounded in taught material.
- **Decision Date:** —

### Learning Resources

#### BL-011 — Curate a foundational papers list in resources/
- **Description:** A vetted, dated reading list of seminal papers per domain.
- **Reason:** Learners repeatedly ask "what should I read?"; centralize verified sources.
- **Priority:** Low
- **Status:** Proposed
- **Affected Modules:** resources/, 06-transformers, 07-large-language-models
- **Notes:** Cite accurately (title, authors, year); no fabricated references.
- **Decision Date:** —

### Ideas

#### BL-012 — Automate theory↔practice cross-linking between the two repos
- **Description:** A small tool that keeps links between `knowledge-base` lessons and `engineering-labs` projects in sync.
- **Reason:** The two-repo split needs reliable bi-directional navigation.
- **Priority:** Low
- **Status:** Proposed
- **Affected Modules:** — (cross-repo)
- **Notes:** Could be a generated "implemented in" table. Purely exploratory for now.
- **Decision Date:** —

---

## Navigation

- 🏠 [Knowledge Base Home](../README.md)
- 🧭 [CLAUDE.md](CLAUDE.md) · 📖 [Master Curriculum](MASTER_CURRICULUM.md) · 🗺️ [Roadmap](ROADMAP.md)

---
*Planning document only. Items change the repository **only** after they are
`Approved` and implemented through the normal process.*
