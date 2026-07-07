---
title: "Exercise: Regression Metrics from Scratch"
description: Implement MAE, MSE, RMSE, and R-squared and match scikit-learn's results.
type: exercise
domain: 03-machine-learning
tags: [machine-learning, regression, metrics, evaluation]
difficulty: beginner
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 20 min
reinforces:
  - 03-machine-learning/lessons/model-evaluation-metrics
---

# Exercise: Regression Metrics from Scratch

> Practice for **[Model Evaluation Metrics](../lessons/model-evaluation-metrics.md)**.

---

## Problem

Implement the four core regression metrics in NumPy and verify them against
scikit-learn:

$$
\text{MAE}=\frac{1}{n}\sum|y_i-\hat{y}_i|,\quad
\text{RMSE}=\sqrt{\frac{1}{n}\sum (y_i-\hat{y}_i)^2},\quad
R^2 = 1 - \frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}
$$

## Requirements

- [ ] Implement `mae`, `mse`, `rmse`, `r2` with NumPy.
- [ ] Match `sklearn.metrics` to within floating-point tolerance.
- [ ] Explain what a negative $R^2$ means.

---

## Hints

<details>
<summary>Hint 1</summary>

$R^2$ compares your model's squared error to the error of predicting the mean
$\bar{y}$ every time.

</details>

<details>
<summary>Hint 2</summary>

Compare against `mean_absolute_error`, `mean_squared_error`, `r2_score`.

</details>

---

## Solution

<details>
<summary>Show solution</summary>

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def mse(y, yhat): return float(np.mean((y - yhat) ** 2))
def rmse(y, yhat): return float(np.sqrt(mse(y, yhat)))
def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


rng = np.random.default_rng(0)
y = rng.normal(size=100)
yhat = y + rng.normal(scale=0.3, size=100)
assert np.isclose(mae(y, yhat), mean_absolute_error(y, yhat))
assert np.isclose(rmse(y, yhat), mean_squared_error(y, yhat) ** 0.5)
assert np.isclose(r2(y, yhat), r2_score(y, yhat))
```

**Explanation:** RMSE penalizes large errors more than MAE (squares before
averaging), so it is sensitive to outliers. $R^2$ is the fraction of variance
explained relative to the mean baseline; it is **negative** when the model does
worse than simply predicting $\bar{y}$.

</details>

---

## Navigation

- ⬆️ [Module 3 Exercises](README.md)
- 📚 [Module 3 — Machine Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
