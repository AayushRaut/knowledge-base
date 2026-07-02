---
title: "Exercise: Aggregate with groupby"
description: Use split-apply-combine to build per-group summary features from a DataFrame.
type: exercise
domain: 01-python-languages
tags: [python, pandas, groupby, aggregation]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 01-python-languages/lessons/pandas
---

# Exercise: Aggregate with `groupby`

> Practice for **[Pandas for Data Work](../lessons/pandas.md)**.

---

## Problem

You have an events log. For each `user_id`, compute: the number of events, the
total `amount`, and the mean `amount`. Then keep only users whose total spend
exceeds 100, sorted by total spend descending.

```python
import pandas as pd

events = pd.DataFrame(
    {
        "user_id": [1, 1, 2, 2, 2, 3],
        "amount": [40.0, 80.0, 5.0, 5.0, 5.0, 250.0],
    }
)
```

## Requirements

- [ ] Produce one row per user with columns `n_events`, `total`, `mean`.
- [ ] Filter to `total > 100`.
- [ ] Sort by `total` descending.
- [ ] No Python loops — use `groupby` + `agg`.

---

## Hints

<details>
<summary>Hint 1</summary>

`groupby("user_id")["amount"].agg([...])` can take named aggregations.

</details>

<details>
<summary>Hint 2</summary>

Named aggregation keeps column names clean:
`groupby("user_id").agg(total=("amount", "sum"), ...)`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import pandas as pd


def summarize_users(events: pd.DataFrame) -> pd.DataFrame:
    summary = (
        events.groupby("user_id")
        .agg(
            n_events=("amount", "size"),
            total=("amount", "sum"),
            mean=("amount", "mean"),
        )
        .reset_index()
    )
    return (
        summary[summary["total"] > 100]
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )
```

**Explanation:** `groupby(...).agg(name=(column, func))` is the split-apply-combine
pattern with readable output column names. Filtering the aggregated frame (not the
raw rows) is what "keep users with total > 100" means. This exact shape — grouping
raw events into per-entity features — is the backbone of tabular feature
engineering.

</details>

---

## Navigation

- ⬆️ [Module 1 Exercises](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
