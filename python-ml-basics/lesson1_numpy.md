# Lesson: NumPy — The Math Engine of ML

**Date taught:** 2026-09-26
**Folder:** python-ml-basics/
**Status:** ✅ Taught

---

## Analogy

Imagine you have 1000 students' exam scores. You want the average.

**Without NumPy (plain Python):**
```python
total = 0
for score in scores:
    total += score
average = total / len(scores)
```
You loop through each score one by one. Slow.

**With NumPy:**
```python
average = np.mean(scores)
```
NumPy operates on the ENTIRE list at once — like a calculator that processes all 1000 numbers simultaneously. That's why ML uses it: neural networks multiply millions of numbers and NumPy does it in microseconds.

> NumPy = math on entire arrays at once, not one number at a time.

---

## Why NumPy Matters for ML

Every single operation in a neural network is NumPy under the hood:
```
input × weights    →  matrix multiplication (np.dot)
add bias           →  array addition
activation (ReLU)  →  np.maximum(0, x)
loss               →  np.mean(...)
```

---

## Core Concepts & Diagram

```
PYTHON LIST         →        NUMPY ARRAY
[1, 2, 3]           →        np.array([1, 2, 3])
  ↓                               ↓
  slow loop math          fast vectorized math
  no shape info           has .shape, .dtype
  mixed types             single type (float, int)
```

### Array Shapes (very important)

```
1D array  →  shape (3,)       [1, 2, 3]
2D array  →  shape (3, 4)     matrix: 3 rows, 4 cols
3D array  →  shape (2, 3, 4)  batch of 2 matrices

In ML:
  Input data    →  shape (batch_size, features)   e.g. (32, 784)
  Weights       →  shape (features, neurons)       e.g. (784, 128)
  Output        →  shape (batch_size, neurons)     e.g. (32, 128)
```

### Broadcasting

```
NumPy can operate on arrays of different shapes by "stretching" the smaller one:

  [1, 2, 3]        shape (3,)
+     5            shape ()   ← scalar
= [6, 7, 8]

  [[1,2,3],        shape (2,3)
   [4,5,6]]
+  [10,20,30]      shape (3,)  ← stretches across rows
= [[11,22,33],
   [14,25,36]]
```

---

## Code — Everything You Need

```python
import numpy as np

# ── Creating Arrays ───────────────────────────────────
a = np.array([1, 2, 3, 4, 5])           # 1D
b = np.array([[1,2,3],[4,5,6]])          # 2D (2 rows, 3 cols)
z = np.zeros((3, 4))                     # all zeros
o = np.ones((2, 3))                      # all ones
r = np.random.randn(3, 4)               # random (normal dist)
i = np.eye(3)                            # identity matrix

# ── Shape & Info ──────────────────────────────────────
print(b.shape)        # (2, 3)
print(b.dtype)        # int64
print(b.ndim)         # 2
print(b.size)         # 6 (total elements)

# ── Indexing & Slicing ────────────────────────────────
a[0]          # first element: 1
b[0, :]       # first row: [1, 2, 3]
b[:, 1]       # second column: [2, 5]
b[0:2, 1:3]   # submatrix

# ── Math Operations (all vectorized) ─────────────────
a + 10        # [11, 12, 13, 14, 15]
a * 2         # [2, 4, 6, 8, 10]
a ** 2        # [1, 4, 9, 16, 25]
np.sqrt(a)    # square root of each
np.exp(a)     # e^x for each (used in softmax)
np.log(a)     # log of each (used in cross-entropy loss)

# ── Aggregations ──────────────────────────────────────
np.sum(a)         # 15
np.mean(a)        # 3.0
np.max(a)         # 5
np.min(a)         # 1
np.std(a)         # standard deviation
np.argmax(a)      # index of max value (used in classification!)

# ── Matrix Multiplication (THE most used in ML) ───────
W = np.random.randn(3, 4)  # weights matrix
x = np.random.randn(4)     # input vector
out = W @ x                # matrix multiply → shape (3,)
# or: np.dot(W, x)

A = np.random.randn(2, 3)
B = np.random.randn(3, 4)
C = A @ B                  # shape (2, 4)

# ── Reshape (very common in ML) ───────────────────────
a = np.arange(12)           # [0,1,2,...,11]
a.reshape(3, 4)             # 3 rows, 4 cols
a.reshape(2, 2, 3)          # 3D
a.reshape(-1, 4)            # -1 means "figure it out" → (3,4)

# ── Transpose ─────────────────────────────────────────
b.T                        # flip rows and cols: (2,3) → (3,2)

# ── Boolean Masking (used in ReLU, masking, etc.) ─────
a = np.array([-3, -1, 0, 2, 5])
a[a > 0]                   # [2, 5]
np.maximum(0, a)           # [0, 0, 0, 2, 5]  ← this IS ReLU

# ── Stacking arrays ───────────────────────────────────
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
np.vstack([x1, x2])        # [[1,2,3],[4,5,6]]  ← stack vertically
np.hstack([x1, x2])        # [1,2,3,4,5,6]      ← stack horizontally

# ── Random seeds (for reproducibility) ───────────────
np.random.seed(42)
np.random.randn(3)         # same result every time
```

---

## Key Operations for ML

| Operation | NumPy code | When used |
|-----------|-----------|-----------|
| Matrix multiply | `A @ B` | Every forward pass |
| Element-wise ops | `a * b`, `a + b` | Bias addition, loss |
| ReLU | `np.maximum(0, x)` | Activation |
| Softmax numerator | `np.exp(x)` | Output layer |
| Cross-entropy | `np.log(p)` | Loss function |
| Argmax | `np.argmax(output)` | Getting predicted class |
| Reshape | `x.reshape(-1, 1)` | Fixing dimension mismatches |
| Transpose | `W.T` | Backprop weight gradients |

---

## Exercise
1. Create a (4, 3) random matrix and a (3,) vector. Multiply them. What shape is the output?
2. Create array `[-5, -2, 0, 3, 7]`. Apply ReLU using NumPy in one line.
3. Create a (6,) array. Reshape to (2, 3). Then transpose it.
4. Find the index of the max value in `[0.1, 0.7, 0.05, 0.15]` (like picking the predicted class).

---

## What to Remember Forever
> NumPy arrays = math on millions of numbers at once.
> Shape is everything — always check `.shape` when something breaks.
> `@` = matrix multiply. Used in EVERY forward pass.
