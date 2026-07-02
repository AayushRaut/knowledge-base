---
title: Pandas Cheat Sheet
description: Fast reference for pandas Series/DataFrame selection, groupby, joins, and missing data.
type: cheat-sheet
domain: 20-cheat-sheets
tags: [python, pandas, dataframes, reference]
status: complete
created: 2026-07-02
updated: 2026-07-02
---

# Pandas Cheat Sheet

> Fast reference for pandas. For depth, see
> [Pandas for Data Work](../../01-python-languages/lessons/pandas.md).

---

## Core Concepts

| Term | Definition |
|------|------------|
| `Series` | 1-D labeled array (one column). |
| `DataFrame` | 2-D labeled table (columns of Series). |
| Index | Row labels; enables alignment and fast lookup. |
| `loc` / `iloc` | Label-based / integer-position-based selection. |
| Vectorized op | Column-wide operation, far faster than `apply`/loops. |

---

## Load & inspect

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head(); df.info(); df.describe()
df.shape; df.dtypes; df.columns
```

## Select & filter

```python
df["col"]; df[["a", "b"]]              # column(s)
df.loc[df["age"] > 30, ["name", "age"]]   # label + boolean mask
df.iloc[0:5, :2]                       # position-based
df.query("age > 30 and city == 'NY'")  # expression filter
```

## Transform, group, join

```python
df["scaled"] = (df["x"] - df["x"].mean()) / df["x"].std()   # vectorized
df.groupby("user").agg(total=("amt", "sum"), n=("amt", "size"))
pd.merge(a, b, on="id", how="left", validate="many_to_one")
df.assign(y=lambda d: d["a"] + d["b"])   # method chaining
```

## Missing data

```python
df.isna().sum()                        # count nulls per column
df.dropna(subset=["x"])                # drop rows missing x
df["x"] = df["x"].fillna(df["x"].median())
```

---

## Decision Guide

| If you need… | Use… | Notes |
|--------------|------|-------|
| Label-based selection | `.loc` | Row/column labels. |
| Position-based selection | `.iloc` | Integer indices. |
| Per-group summaries | `groupby().agg()` | Split-apply-combine. |
| Combine tables | `merge`/`join` | Set `validate=` to catch bad keys. |
| Save memory on categories | `astype("category")` | Great for low-cardinality strings. |

---

## Gotchas

- ⚠️ `SettingWithCopyWarning`: assign with a single `.loc[...]`, not chained `df[a][b] = ...`.
- ⚠️ Prefer vectorized ops over `df.apply(..., axis=1)` (row-wise apply is slow).
- ⚠️ `merge` without `how`/`validate` can silently duplicate or drop rows.
- ⚠️ Chained boolean filters need parentheses: `df[(df.a > 1) & (df.b < 2)]`.

---

## Quick Links

- 📖 [Pandas for Data Work](../../01-python-languages/lessons/pandas.md)
- 📖 [Data Cleaning and Wrangling](../../01-python-languages/lessons/data-cleaning-and-wrangling.md)
- 🔗 [pandas documentation](https://pandas.pydata.org/docs/)

---

## Navigation

- ⬆️ [Python Cheat Sheets](README.md)
- 🏠 [Knowledge Base Home](../../README.md)
