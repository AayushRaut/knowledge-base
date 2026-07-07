---
title: TensorFlow and Keras
description: Keras's three model-building styles, the compile/fit workflow, callbacks, and an honest comparison with PyTorch to guide framework choice.
type: lesson
domain: 04-deep-learning
tags: [deep-learning, tensorflow, keras]
difficulty: intermediate
status: complete
created: 2026-07-02
updated: 2026-07-02
estimated_time: 45 min
prerequisites:
  - 04-deep-learning/lessons/pytorch
---

# TensorFlow and Keras

> **TL;DR:** Keras is a high-level API for building and training networks — declare the model, `compile` it with a loss and optimizer, and `fit` it; TensorFlow is the numerical engine underneath, and knowing both PyTorch and Keras lets you read any team's codebase.

---

## Overview

You just wrote PyTorch training loops by hand. Keras takes the opposite stance: the framework owns the loop, and you configure it. This lesson covers the TensorFlow/Keras relationship, the three ways to define a Keras model, the `compile`/`fit`/`evaluate`/`predict` workflow with callbacks, and a fair comparison with PyTorch so you can navigate either ecosystem — a genuine requirement in industry, where legacy and greenfield systems use both.

**By the end, you will be able to:**
- Build the same model three ways — `Sequential`, Functional API, and subclassing — and know when each fits
- Train, evaluate, and checkpoint a model with `compile`/`fit` and callbacks like `EarlyStopping`
- Compare PyTorch and TensorFlow/Keras accurately and justify a framework choice for a given team

---

## Intuition

The mental model: **PyTorch hands you an engine and a wrench; Keras hands you a car.**

In PyTorch you wrote the loop — zero grads, forward, backward, step — yourself. In Keras you *describe* what you want (this architecture, this loss, this optimizer, stop early if validation stalls) and call `model.fit(...)`. The framework runs the loop, tracks metrics, and invokes your callbacks at the right moments. Less control by default, far less boilerplate.

The layering is worth keeping straight:

- **TensorFlow** is the numerical engine: tensors, automatic differentiation, GPU/TPU execution, and deployment tooling (SavedModel, TF Serving, TensorFlow Lite/LiteRT).
- **Keras** is the high-level modeling API — layers, models, losses, `fit`. It has been TensorFlow's official high-level API since TF 2.x, and since **Keras 3** it is multi-backend again: the same Keras code can run on TensorFlow, PyTorch, or JAX.

Since TensorFlow 2.x, TF runs **eagerly** by default (operations execute immediately, like PyTorch and NumPy), with `tf.function` available to compile code into graphs for speed. The old TF 1.x world of manually built static graphs and sessions is legacy.

---

## Details

### Three ways to build the same model

**1. Sequential** — a plain stack of layers, one input, one output. Simplest, least flexible.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(20,)),
    layers.Dropout(0.2),
    layers.Dense(2, activation="softmax"),
])
```

**2. Functional API** — layers are called like functions on symbolic tensors. Handles multiple inputs/outputs, branches, and shared layers, and Keras can validate and plot the whole graph.

```python
inputs = keras.Input(shape=(20,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

**3. Subclassing** — the PyTorch-like style: define layers in `__init__`, computation in `call`. Maximum flexibility (loops, conditionals, custom logic), but Keras can no longer inspect the architecture as a static graph.

```python
class MLP(keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = layers.Dense(64, activation="relu")
        self.drop = layers.Dropout(0.2)
        self.out = layers.Dense(2, activation="softmax")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.hidden(x)
        x = self.drop(x, training=training)
        return self.out(x)

model = MLP()
```

Rule of thumb: start with **Sequential**, move to **Functional** when the topology branches, reach for **subclassing** only when the forward pass needs real Python control flow. Note the `training` flag in `call` — Keras passes it automatically so dropout behaves correctly; this replaces PyTorch's `model.train()`/`model.eval()` toggle.

### The `compile` / `fit` / `evaluate` / `predict` workflow

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",   # integer labels
    metrics=["accuracy"],
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
)

test_loss, test_acc = model.evaluate(X_test, y_test)
probs = model.predict(X_new)     # class probabilities, shape (n, 2)
```

`compile` attaches the optimizer, loss, and metrics. `fit` runs the training loop — batching, shuffling, the gradient steps, validation each epoch — and returns a `history` object with per-epoch metrics you can plot. Use `"sparse_categorical_crossentropy"` for integer labels and `"categorical_crossentropy"` for one-hot labels; mixing these up is a classic error.

### Callbacks: hooks into the loop

Callbacks run at defined points during `fit` — this is how you get early stopping and checkpointing without writing the loop:

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "best.keras", monitor="val_loss", save_best_only=True
    ),
]

model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=100, callbacks=callbacks)
```

`EarlyStopping` halts training after `patience` epochs without improvement and (with `restore_best_weights=True`) rolls back to the best epoch. `ModelCheckpoint` writes the best model to disk as training runs. These two cover most regularization-by-callback needs from the previous lesson.

### `tf.data` input pipelines (briefly)

For data that does not fit in memory or needs preprocessing, `tf.data.Dataset` is TensorFlow's counterpart to PyTorch's `DataLoader`:

```python
ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=1024)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)   # overlap preprocessing with training
)
model.fit(ds, epochs=10)
```

The chained style builds a pipeline; `prefetch` keeps the accelerator fed while the CPU prepares the next batch.

### Saving models

```python
model.save("model.keras")                  # architecture + weights + optimizer state
restored = keras.models.load_model("model.keras")

model.save_weights("weights.h5")           # weights only, like a state_dict
model.export("saved_model_dir")            # TF SavedModel for serving (Keras 3)
```

The `.keras` format restores the full model in one call — no need to rebuild the architecture first, unlike PyTorch's `state_dict`. The SavedModel export is what deployment tools such as TF Serving consume.

## Worked Example

The same binary classifier as the PyTorch lesson, end to end in Keras — compare line counts and where the complexity lives:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 20)).astype("float32")
true_w = rng.normal(size=20).astype("float32")
y = ((X @ true_w + 0.5 * rng.normal(size=2000)) > 0).astype("int64")
X_train, y_train, X_val, y_val = X[:1600], y[:1600], X[1600:], y[1600:]

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(20,)),
    layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=64,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)],
    verbose=2,
)

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"val acc = {acc:.3f}")
model.save("clf.keras")
```

The entire training loop from the PyTorch lesson collapsed into one `fit` call — that trade of control for convenience is the essence of the framework difference.

### PyTorch vs TensorFlow/Keras — an honest comparison

| Aspect | PyTorch | TensorFlow / Keras |
|--------|---------|--------------------|
| Training loop | You write it explicitly | `fit` owns it; custom loops possible via `GradientTape` |
| Execution | Eager by default; `torch.compile` for graph optimization | Eager by default since TF 2.x; `tf.function` for graphs |
| API feel | Pythonic, imperative, close to NumPy | Declarative and layered; subclassing gives an imperative style |
| Research adoption | Dominant in recent academic work and open-source model releases (e.g. Hugging Face defaults to PyTorch) | Present but less common in new research code |
| Production tooling | TorchServe, ONNX export, `torch.compile`; ecosystem matured over time | Long-standing pipeline: TF Serving, TensorFlow Lite/LiteRT (mobile/edge), TF.js (browser), TPU support |
| Debugging | Standard Python debugger works everywhere | Easy in eager mode; harder inside `tf.function` graphs |
| Multi-backend | PyTorch only | Keras 3 runs on TF, PyTorch, or JAX backends |

Avoid absolutist claims here: both frameworks are actively developed, both run eagerly, both deploy to production at scale. The historical stereotype — "PyTorch for research, TF for production" — reflects real ecosystem histories, but the gap has narrowed from both sides.

**When a team picks which:** existing codebase and team expertise dominate the decision. Teams fine-tuning open-source LLMs usually land in PyTorch because that is where the model releases are; teams with mobile/edge targets or existing TF Serving infrastructure often stay with TensorFlow; teams that want one modeling API over multiple backends can use Keras 3.

## Best Practices

- ✅ Prefer the Functional API once a model has branches or multiple inputs — you get graph validation and `model.summary()` for free.
- ✅ Always pass `validation_data` to `fit` and monitor `val_loss`, not training loss, in callbacks.
- ✅ Use `EarlyStopping(restore_best_weights=True)` so the model you keep is the best one, not the last one.
- ✅ Save in the native `.keras` format for checkpoints; `export` to SavedModel only for serving.

## Common Mistakes

- ⚠️ **Wrong loss for the label format** — `sparse_categorical_crossentropy` expects integer labels, `categorical_crossentropy` expects one-hot. Mismatches raise shape errors or silently mistrain. Check your label shape first.
- ⚠️ **Forgetting to `compile` before `fit`** — Keras raises an error; compile attaches the optimizer and loss, so it must come first (and re-compiling resets optimizer state).
- ⚠️ **Monitoring `loss` instead of `val_loss` in `EarlyStopping`** — training loss almost always decreases, so early stopping never triggers. Monitor a validation metric.
- ⚠️ **Assuming a subclassed model can be summarized or plotted like a Functional one** — its graph is opaque until called on data; use Functional when you need introspection.

## Industry Tips

- 💡 Read job postings and codebases, not framework wars: many companies run both, and the concepts (layers, losses, callbacks, checkpoints) transfer almost one-to-one.
- 💡 `model.fit` covers 90% of supervised training; for research-style custom losses or training schemes, Keras supports custom training loops with `tf.GradientTape` — learn `fit` first.
- 💡 If your target is mobile or embedded inference, TensorFlow's Lite/LiteRT toolchain is a major reason teams choose the TF stack.

## Real-World Use Cases

- On-device ML — mobile apps and embedded systems using the TensorFlow Lite/LiteRT runtime.
- Tabular and structured-data models in enterprises with established TF Serving deployments.
- Rapid prototyping in applied teams where `compile`/`fit` plus callbacks covers the whole workflow.

---

## Summary

- Keras is the high-level modeling API (Sequential, Functional, subclassing); TensorFlow is the engine and deployment stack beneath it, and Keras 3 also runs on PyTorch and JAX backends.
- The workflow is declare → `compile` → `fit` → `evaluate`/`predict`, with callbacks like `EarlyStopping` and `ModelCheckpoint` hooking into the managed loop.
- PyTorch and TF/Keras are both eager, capable, production-ready frameworks; choose based on team expertise, the models you build on, and deployment targets.

## Practice

- [ ] Exercises: [Module 4 Exercises](../exercises/README.md)
- [ ] Self-check: You need a model with two input branches that merge before the output layer — which Keras style do you use, and why not `Sequential`?

## Further Reading

- 📄 [TensorFlow documentation](https://www.tensorflow.org/)
- 📄 [Keras documentation](https://keras.io/)
- 📄 [PyTorch documentation](https://pytorch.org/docs/stable/)
- 📘 Hands-On Machine Learning — Aurélien Géron (covers Keras and TensorFlow in depth)
- 📘 Dive into Deep Learning — Zhang, Lipton, Li & Smola (https://d2l.ai/)

## Related

- [PyTorch Essentials](pytorch.md)
- [Module 15 — Deployment](../../15-deployment/README.md) — serving trained models in production

---

## Navigation
- ⬆️ [Lessons](README.md)
- 📚 [Module 4 — Deep Learning](../README.md)
- 🏠 [Knowledge Base Home](../../README.md)
