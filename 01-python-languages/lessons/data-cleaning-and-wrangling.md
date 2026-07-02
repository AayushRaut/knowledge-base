---
title: Data Cleaning and Wrangling
description: Profile, repair, reshape, and validate messy tabular data into a clean DataFrame using a reproducible pandas pipeline.
type: lesson
domain: 01-python-languages
tags: [python, pandas, data-cleaning, etl]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 50 min
prerequisites:
  - 01-python-languages/lessons/pandas
---

# Data Cleaning and Wrangling

> **TL;DR:** Real data arrives dirty — missing values, wrong dtypes, duplicates, inconsistent strings. Profile it first, fix each problem deliberately, then wrap the fixes in one reproducible function that turns a raw DataFrame into a clean one.

---

## Overview
Most of the effort in an AI project is spent turning raw data into something a model can consume. Cleaning is where correctness is won or lost: a silent dtype error or a leaked duplicate can quietly ruin a model's metrics. This lesson walks the full loop — profile, repair, reshape, join, validate — and shows how to package it as a reproducible pipeline so the same raw input always yields the same clean output.

**By the end, you will be able to:**
- Profile a DataFrame and decide how to handle missing values, dtypes, and duplicates.
- Reshape and join datasets with `melt`, `pivot`, `stack`/`unstack`, and `merge`.
- Write a single, testable cleaning function that takes a raw DataFrame and returns a validated clean one.

---

## Intuition
Treat cleaning like triage in an emergency room. First you *assess* the patient (profile the data) instead of guessing. Then you treat problems in order of severity, and — critically — you record what you did so it can be repeated. A cleaning step you cannot rerun is not a fix; it is a one-off patch that will rot the moment new data arrives. Aim for a function you can point at next month's raw file and trust.

---

## Details

### Profiling: look before you clean
Never clean blind. Inspect shape, dtypes, missingness, and per-column value ranges first.

```python
import pandas as pd

df = pd.read_csv("raw_customers.csv")

df.shape                 # (rows, cols)
df.info()                # dtypes + non-null counts per column
df.describe(include="all")  # summary stats for numeric and object columns
df.isna().sum()          # missing count per column
df.duplicated().sum()    # exact duplicate rows
```

### Handling missing values: drop vs. impute
There is no default answer — it depends on how much is missing and why.

- **Drop rows** (`df.dropna(subset=[...])`) when missingness is rare and rows are cheap.
- **Drop columns** when a column is mostly empty and low-value.
- **Impute** (fill) when you cannot afford to lose rows: median for skewed numerics, mean for symmetric ones, a constant or `"unknown"` for categoricals.

The trade-off: dropping loses information and can bias your data if missingness is not random; imputing invents values and can dampen variance. Whatever you choose, do it explicitly.

```python
df["age"] = df["age"].fillna(df["age"].median())     # robust to outliers
df["city"] = df["city"].fillna("unknown")            # explicit sentinel
df = df.dropna(subset=["target"])                    # never impute the label
```

### Fixing dtypes
Numbers read as strings, dates read as objects, and low-cardinality strings wasting memory are all common. Cast them intentionally.

```python
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
df["plan"] = df["plan"].astype("category")   # memory + speed for low-cardinality
```

`errors="coerce"` turns unparseable values into `NaT`/`NaN` so you can find and handle them rather than crashing.

### Removing duplicates
Distinguish *exact* duplicate rows from duplicates on a key (e.g. two rows for one user id).

```python
df = df.drop_duplicates()                              # exact copies
df = df.drop_duplicates(subset=["user_id"], keep="last")  # latest record per key
```

### Detecting outliers
Outliers may be errors or genuine rare events — flag before you delete. Two common qualitative rules:

- **IQR rule:** values outside `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` are suspect.
- **z-score:** values more than ~3 standard deviations from the mean are suspect (assumes roughly normal data).

```python
q1, q3 = df["revenue"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["revenue"] < lower) | (df["revenue"] > upper)]
```

Investigate outliers; do not reflexively drop them, especially before you know whether they are valid.

### String and text cleaning
Inconsistent casing and stray whitespace create phantom categories. Use the vectorized `.str` accessor.

```python
df["email"] = df["email"].str.strip().str.lower()
df["state"] = df["state"].str.upper().str.replace(r"\s+", "", regex=True)
```

### Reshaping: wide ↔ long
Models and plots want *tidy* data (one observation per row). Reshaping tools:

- **`melt`** — wide → long (many measurement columns become key/value pairs).
- **`pivot`** — long → wide (spread a category column into columns).
- **`stack`/`unstack`** — move an index level to/from columns.

```python
# Wide: one column per month. Long: one row per (id, month).
long = df.melt(id_vars="user_id", var_name="month", value_name="spend")
wide = long.pivot(index="user_id", columns="month", values="spend")
```

### Joining datasets
Combine tables with `merge`. Always be explicit about the join type and check row counts afterward — a bad key silently duplicates or drops rows.

```python
orders = orders.merge(customers, on="user_id", how="left", validate="many_to_one")
```

`validate="many_to_one"` raises if the key relationship is not what you assumed — a cheap safeguard against accidental row explosions.

### Validating results
Assertions turn silent corruption into loud failures. Encode your expectations.

```python
assert df["target"].notna().all(), "target has missing values"
assert df["user_id"].is_unique, "duplicate user_id after cleaning"
assert df["age"].between(0, 120).all(), "age out of range"
```

## Worked Example
Wrap the whole loop in one reproducible function. Given a raw DataFrame, it returns a clean, validated one — and never mutates its input.

```python
import pandas as pd


def clean_customers(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw customers DataFrame.

    Steps: normalize strings, fix dtypes, impute/drop missing values,
    deduplicate by user, and validate invariants before returning.
    """
    df = raw.copy()  # never mutate the caller's data

    # 1. Normalize text so categories don't fragment on case/whitespace.
    for col in ["email", "city"]:
        df[col] = df[col].str.strip().str.lower()

    # 2. Coerce dtypes; unparseable values become NaN/NaT for later handling.
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # 3. Handle missing values deliberately.
    df["age"] = df["age"].fillna(df["age"].median())  # robust central value
    df["city"] = df["city"].fillna("unknown")
    df = df.dropna(subset=["user_id", "target"])      # cannot impute keys/labels

    # 4. Deduplicate, keeping the most recent record per user.
    df = df.sort_values("signup_date").drop_duplicates(
        subset=["user_id"], keep="last"
    )

    # 5. Validate invariants; fail loudly if an assumption is violated.
    assert df["user_id"].is_unique, "duplicate user_id after cleaning"
    assert df["age"].between(0, 120).all(), "age out of plausible range"

    return df.reset_index(drop=True)
```

Because it is a pure function of its input, you can unit-test it, rerun it on new files, and drop it straight into an ETL job.

## Best Practices
- ✅ Profile first (`info`, `describe`, `isna().sum()`) so every fix targets a real, observed problem.
- ✅ Work on a `.copy()` and return a new DataFrame — never mutate the caller's data.
- ✅ Make missing-value and dtype decisions explicit; leave a comment on the *why*.
- ✅ End every pipeline with assertions that encode your data contract.

## Common Mistakes
- ⚠️ Imputing the target/label column — this leaks information and inflates metrics; drop those rows instead.
- ⚠️ Chained assignment (`df[df.x > 0]["y"] = ...`) may silently no-op — use `.loc[...]` for assignment.
- ⚠️ Merging without checking row counts — a duplicated key can multiply your rows; use `validate=`.
- ⚠️ Parsing dates without `errors="coerce"`, so one bad value crashes the whole job.

## Industry Tips
- 💡 Keep the raw file immutable and write cleaning as a function from raw → clean; reproducibility beats a pile of notebook cells.
- 💡 `df.astype("category")` on low-cardinality string columns can cut memory dramatically on large datasets.
- 💡 Log row counts before and after each major step; unexpected changes are the fastest way to catch a bad join.

## Real-World Use Cases
- Preparing training data: harmonizing dtypes and labels across sources before feature engineering.
- ETL jobs that ingest daily exports and must handle the same defects reproducibly.
- Deduplicating and normalizing scraped or user-submitted records.

---

## Summary
- Profile before you clean; every repair should target an observed defect.
- Missing values, dtypes, duplicates, outliers, and messy strings each need a deliberate, documented decision.
- Reshape with `melt`/`pivot`/`stack`, join with `merge` (and `validate=`), and end with assertions.
- Package the whole loop as one pure function so cleaning is reproducible and testable.

## Practice
- [ ] Exercises: [Module 1 Exercises](../exercises/README.md)
- [ ] Self-check: Why should you never impute the target column, and what should you do instead?

## Further Reading
- 📘 Python for Data Analysis, Wes McKinney
- 📄 [pandas documentation](https://pandas.pydata.org/docs/)
- 🌐 Real Python — https://realpython.com/

## Related
- [Pandas for Data Work](pandas.md)
- [Feature Engineering](feature-engineering.md)

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 1 — Python for AI Engineering](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
