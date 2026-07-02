# Vision

> The north star for the AI Engineering Knowledge Base.

---

## This document vs. the curriculum

Two documents govern the project and play distinct, non-overlapping roles:

| Document | Role |
|----------|------|
| **[MASTER_CURRICULUM.md](MASTER_CURRICULUM.md)** | The **canonical curriculum specification** — the authoritative definition of *what* is taught: every module, skill, technology, project, deliverable, and requirement. |
| **COURSE_VISION.md** (this file) | Describes the project's **goals, mission, success criteria, and scope** — the *why* and the standards the work is held to. |

This file intentionally **does not duplicate the curriculum**. For the full list
of modules and requirements, always refer to
[MASTER_CURRICULUM.md](MASTER_CURRICULUM.md).

> [!IMPORTANT]
> **All future content generation must align with
> [MASTER_CURRICULUM.md](MASTER_CURRICULUM.md).** It is the single source of
> truth for scope and requirements; this vision defines the bar that content
> must meet. Where the two are read together, the curriculum specifies *what*
> to build and this document specifies *how well*.

---

## Mission

Build the **most practical, well-structured, and durable open knowledge base for
AI Engineering** — one that takes a motivated learner from programming
fundamentals all the way to shipping production-grade, LLM-powered systems, and
that remains a trusted daily reference for working practitioners.

---

## Why this exists

AI engineering knowledge today is scattered across papers, blog posts, docs,
and courses that go stale quickly and rarely connect end-to-end. This repository
consolidates that knowledge into a **single, coherent, navigable structure** that:

- Connects theory to production practice.
- Stays current as the field moves.
- Is small-grained enough to reference in seconds, yet complete enough to learn from.

---

## Who it's for

| Audience | What they get |
|----------|---------------|
| **Beginners** | A clear path from Python and math to modern AI. |
| **ML/AI engineers** | Deep dives, patterns, and production checklists. |
| **Software engineers moving into AI** | Bridges from familiar engineering to ML/LLM systems. |
| **Interview candidates** | Structured prep across concepts, system design, and coding. |
| **Teams** | A shared reference and onboarding resource. |

---

## Principles

1. **Practical over academic.** Every topic answers "how would I use this?"
2. **Production-minded.** Deployment, cost, observability, and safety are
   first-class, not afterthoughts.
3. **Small, composable units.** One concept per file; build understanding by
   linking, not by writing monoliths.
4. **Accuracy first.** No hand-waving, no fabricated numbers. Cite sources.
5. **Consistency scales.** Shared templates and a strict style guide keep
   thousands of files coherent.
6. **Evergreen where possible, dated where necessary.** Fast-moving topics carry
   review dates and version notes.
7. **Learn by building.** Concepts are paired with exercises, assignments, and
   real projects.

---

## Scope

The **authoritative, detailed scope** — every module, skill, technology,
project, and requirement — is defined in
[MASTER_CURRICULUM.md](MASTER_CURRICULUM.md). This section states the scope only
at the level of principle; it does not restate the curriculum.

**In scope:** the full AI engineering stack, from absolute fundamentals
(Python, math) through to advanced, production-grade Generative AI systems, plus
the projects, interview preparation, and reference material needed to make a
learner job-ready. See [MASTER_CURRICULUM.md](MASTER_CURRICULUM.md) for the
definitive module and requirement list.

**Out of scope (for now):** vendor-specific certification cramming, unmaintained
framework tutorials, and anything that duplicates official docs without adding
synthesis or practical guidance.

---

## What "done" looks like for a topic

A topic is considered mature when it has:

- [ ] Conceptual lessons that build from intuition to detail.
- [ ] At least one worked example or exercise.
- [ ] A cheat-sheet entry for fast recall.
- [ ] Links to authoritative sources and further reading.
- [ ] A review date and an owner.

---

## Long-term direction

- Reference architectures and case studies for real production systems.
- A capstone project track that integrates multiple domains.
- Interview tracks tailored to specific roles (ML engineer, LLM engineer, MLOps).
- Optional tooling (link checking, search) as the content base grows.

See the [ROADMAP.md](ROADMAP.md) for the concrete build order.
