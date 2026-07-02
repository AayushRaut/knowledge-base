# Contributing Guide

Thank you for helping build the **AI Engineering Knowledge Base**. This document
explains how to add and maintain content so the repository stays consistent as
it scales to thousands of files.

> **Read first:** [Style Guide](.claude/STYLE_GUIDE.md) ·
> [Repository Rules](.claude/REPOSITORY_RULES.md) ·
> [Vision](.claude/COURSE_VISION.md)

---

## 1. Ground rules

1. **One concept per file.** Keep lessons small, focused, and composable.
2. **Start from a template.** Never hand-roll structure — copy from
   [`templates/`](templates/).
3. **Follow the naming convention.** kebab-case for files and folders; numbered
   prefixes only for ordered, top-level domains.
4. **Every folder has a `README.md`** that indexes its children.
5. **Cite your sources.** Link papers, docs, and references.
6. **No secrets, no large binaries.** See [`.gitignore`](.gitignore).

---

## 2. What can I contribute?

| Type | Template | Lives in |
|------|----------|----------|
| Lesson / concept | [lesson-template.md](templates/lesson-template.md) | The relevant `NN-domain/subtopic/` |
| Exercise | [exercise-template.md](templates/exercise-template.md) | Alongside the lesson it reinforces |
| Assignment | [assignment-template.md](templates/assignment-template.md) | Alongside the lesson/module |
| Project | [project-template.md](templates/project-template.md) | [`18-projects/`](18-projects/README.md) |
| Interview Q&A | [interview-template.md](templates/interview-template.md) | [`19-interview-preparation/`](19-interview-preparation/README.md) |
| Cheat sheet | [cheat-sheet-template.md](templates/cheat-sheet-template.md) | [`20-cheat-sheets/`](20-cheat-sheets/README.md) |

---

## 3. Workflow

```bash
# 1. Create a branch
git checkout -b add/<domain>-<short-topic>

# 2. Copy the right template into the correct folder
cp templates/lesson-template.md 05-nlp/word-embeddings/word2vec.md

# 3. Write the content, following the Style Guide

# 4. Update the folder README.md index to link your new file

# 5. Commit with a clear message (see below) and open a PR
```

### Branch naming

- `add/<area>-<topic>` — new content
- `fix/<area>-<topic>` — corrections
- `refactor/<area>` — restructuring
- `chore/<thing>` — tooling, meta, housekeeping

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(nlp): add word2vec lesson
fix(rag): correct chunking diagram
docs(readme): update domain map
chore(templates): tighten lesson frontmatter
```

---

## 4. Content checklist (before opening a PR)

- [ ] File created from the correct template.
- [ ] Frontmatter is complete and accurate.
- [ ] Placed in the correct `NN-domain/subtopic/` folder.
- [ ] Parent `README.md` index updated with a link.
- [ ] Internal links use relative paths and resolve.
- [ ] Prose follows the [Style Guide](.claude/STYLE_GUIDE.md).
- [ ] Sources cited.
- [ ] No secrets, credentials, or large binaries committed.
- [ ] Spelling and grammar checked.

---

## 5. Quality bar

Content should be **accurate, concise, and practical**. Prefer:

- Clear explanations over exhaustive ones.
- Runnable, minimal code examples.
- Diagrams for anything with structure or flow.
- Explicit prerequisites and "further reading" links.

If you are unsure, open a draft PR and ask for feedback early.

---

## 6. Reviews

Every PR is reviewed for correctness, structure, and adherence to the Style
Guide. Reviewers may request changes; keep discussion in the PR thread. Small,
focused PRs are merged fastest.

---

Questions? Open an issue or start a discussion. Happy writing! 🧠
