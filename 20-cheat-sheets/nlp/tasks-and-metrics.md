---
title: NLP Tasks & Metrics Cheat Sheet
description: Task-to-approach map and the evaluation metric for each NLP task.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [nlp, tasks, metrics, bleu, rouge, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# NLP Tasks & Metrics Cheat Sheet

> Fast reference. For depth, see
> [NLP Evaluation Metrics](../../05-nlp/lessons/nlp-evaluation-metrics.md).

---

## Task → approach → metric

| Task | Strong baseline | Modern approach | Headline metric |
|------|----------------|-----------------|-----------------|
| Text classification | TF-IDF + LogReg | Fine-tuned transformer | macro-F1 |
| Sentiment | VADER (no training) | Transformer pipeline | macro-F1 |
| NER | spaCy pretrained | Fine-tuned transformer | entity-level F1 |
| Summarization | Extractive (top sentences) | Abstractive seq2seq | ROUGE (+ human check) |
| Translation | — | Neural MT | BLEU / chrF |
| Extractive QA | — | Span-prediction model | exact match + F1 |
| Semantic search | TF-IDF/BM25 | Sentence embeddings | Recall@k, MRR |
| Language modeling | n-gram | Transformer LM | perplexity |

## Metric crib notes

- **Macro-F1** — average F1 over classes, each class equal weight → use when imbalanced.
- **Micro-F1** — pool all decisions → dominated by frequent classes.
- **Entity-level F1** — an entity counts only if span *and* type match exactly.
- **BLEU** — modified n-gram *precision* × brevity penalty (translation).
- **ROUGE-N / ROUGE-L** — n-gram / longest-common-subsequence *recall* (summarization).
- **Perplexity** — $\exp(\text{avg negative log-likelihood})$; lower = better; ties to cross-entropy.
- **Recall@k / MRR** — did a relevant doc appear in top-k / how high did the first one rank.

## Hugging Face task pipelines

```python
from transformers import pipeline
pipeline("sentiment-analysis")("I loved it")
pipeline("summarization")(long_text, max_length=60)
pipeline("question-answering")(question=q, context=ctx)
pipeline("translation_en_to_fr")("Hello world")
```

---

## Gotchas

- ⚠️ Automatic metrics are **proxies** — always read a sample of outputs.
- ⚠️ BLEU/ROUGE punish valid paraphrases; don't compare across datasets/tokenizations.
- ⚠️ Accuracy on imbalanced sentiment/NER data is misleading — use macro/entity F1.
- ⚠️ Abstractive summaries can be fluent **and** unfaithful — check factuality.

---

## Quick Links

- 📖 [NLP Evaluation Metrics](../../05-nlp/lessons/nlp-evaluation-metrics.md) · [Seq2Seq Tasks](../../05-nlp/lessons/sequence-to-sequence-tasks.md)
- 🔗 [Hugging Face docs](https://huggingface.co/docs)

---

## Navigation

- ⬆️ [NLP Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
