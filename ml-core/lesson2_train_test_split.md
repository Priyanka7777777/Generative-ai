# Lesson: Train/Test Split & Cross-Validation

**Date taught:** 2026-09-26
**Folder:** ml-core/
**Status:** ✅ Taught

---

## Analogy

You're studying for a math exam.

You have a textbook with 1000 practice problems and answers.

**Wrong approach:** Study ALL 1000 problems, memorize all answers. On exam day, you recognize the exact problems and ace it — but you learned nothing. If the exam has new problems, you fail.

**Right approach:**
- Study from 800 problems (training set)
- Test yourself on 200 problems you've never seen (test set)

If you score well on those 200 unseen problems, you genuinely learned math — not just memorized the textbook.

> Training set = textbook. Test set = real exam. Never peek at the exam during studying.

---

## Diagram

```
Full Dataset (1000 samples)
│
├──── Train Set (800 samples = 80%)  ← model learns from this
└──── Test Set  (200 samples = 20%)  ← model evaluated on this
                                        (never seen during training)

More careful split:
│
├──── Train Set  (700 samples = 70%)  ← model learns from this
├──── Val Set    (150 samples = 15%)  ← tune hyperparameters
└──── Test Set   (150 samples = 15%)  ← final evaluation only

Val set = practice exam  (you can adjust your study based on it)
Test set = real exam     (look at it only ONCE at the very end)
```

### K-Fold Cross-Validation
```
Split data into 5 equal folds:
  [Fold1][Fold2][Fold3][Fold4][Fold5]

Round 1:  TRAIN on [2][3][4][5],  TEST on [1]
Round 2:  TRAIN on [1][3][4][5],  TEST on [2]
Round 3:  TRAIN on [1][2][4][5],  TEST on [3]
Round 4:  TRAIN on [1][2][3][5],  TEST on [4]
Round 5:  TRAIN on [1][2][3][4],  TEST on [5]

Final score = average of 5 test scores
→ More reliable than a single split, especially with small datasets
```

---

## Core Concepts

### Overfitting
```
Model memorizes training data, fails on new data.

Train accuracy: 99%
Test accuracy:  62%  ← red flag — huge gap

Symptoms:
  - Loss keeps going down on train but UP on val
  - Model learns noise in training data, not true patterns
```

### Underfitting
```
Model too simple — can't learn the patterns.

Train accuracy: 64%
Test accuracy:  62%  ← both low, small gap

Symptoms:
  - Loss stays high even after many epochs
  - Model is too simple for the problem
```

### The Goldilocks Zone
```
  Underfitting ←────────────────────────→ Overfitting
  (too simple)         ↑                  (too complex)
                  Just right
                  Train ≈ Val ≈ high
```

---

## Data Leakage — The Silent Killer

```
WRONG (data leakage):
  1. Normalize entire dataset using mean/std of ALL data
  2. Split into train/test
  
  Problem: test set influenced train set's statistics → inflated scores

CORRECT:
  1. Split into train/test FIRST
  2. Compute mean/std on TRAIN set only
  3. Apply same mean/std to BOTH train and test

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)   ← learns stats from train
  X_test  = scaler.transform(X_test)         ← applies same stats to test
                  ↑
            NOT fit_transform — never fit on test data!
```

---

## Code

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ── Basic Train/Test Split ─────────────────────────────
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # reproducible split
    stratify=y            # keep class balance in both splits
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class balance: {y_train.mean():.2f}")
print(f"Test class balance:  {y_test.mean():.2f}")   # should be similar

# ── Train/Val/Test Split ───────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
# 0.176 of 0.85 ≈ 0.15 of total → gives ~70/15/15 split

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ── Correct way: normalize AFTER splitting ────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform
X_val_scaled   = scaler.transform(X_val)          # transform ONLY
X_test_scaled  = scaler.transform(X_test)         # transform ONLY

# ── Using Pipeline (handles this automatically) ────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
print(f"Train accuracy: {pipeline.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {pipeline.score(X_test,  y_test):.3f}")

# ── K-Fold Cross-Validation ───────────────────────────
model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")

# ── Visualizing Train vs Val Loss ─────────────────────
# (simulated — in real training you'd log these)
epochs = range(1, 31)
train_acc = [0.60 + 0.02*i - 0.0003*i**2 + np.random.uniform(-0.01, 0.01)
             for i in epochs]
val_acc   = [0.58 + 0.018*i - 0.0004*i**2 + np.random.uniform(-0.02, 0.02)
             for i in epochs]

plt.figure(figsize=(8, 4))
plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
plt.plot(epochs, val_acc,   'r--', label='Val Accuracy')
plt.axvline(x=20, color='green', linestyle=':', label='Early stopping point')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend(); plt.grid(True)
plt.show()
```

---

## Key Rules

```
Rule 1: Split BEFORE any preprocessing (normalization, encoding)
Rule 2: fit() the scaler on train only, transform() on val/test
Rule 3: Never look at test set until final evaluation
Rule 4: Use stratify=y to preserve class balance in splits
Rule 5: Small dataset → use K-Fold CV for reliable scores
Rule 6: Big gap between train/val = overfitting → regularize or get more data
```

---

## Exercise
1. Split `make_classification(n_samples=500)` into 70/15/15. Print sizes.
2. Train logistic regression, get train AND test accuracy. Is there overfitting?
3. Try `test_size=0.5` (50% test). Does accuracy drop? Why?
4. Run 10-fold CV instead of 5-fold. Does the score change much?

---

## What to Remember Forever
> Never preprocess before splitting — always split first.
> Train accuracy high + val accuracy low = overfitting.
> K-Fold CV gives more reliable scores on small datasets.
> Test set = touch it once, at the very end. That's your honest score.
