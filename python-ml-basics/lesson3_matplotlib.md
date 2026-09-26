# Lesson: Matplotlib — Visualizing Data

**Date taught:** 2026-09-26
**Folder:** python-ml-basics/
**Status:** ✅ Taught

---

## Analogy

Matplotlib is the **printer for your data** — it turns numbers into pictures.

In ML you need to see your data because:
- Numbers lie, pictures reveal patterns
- You can't understand 10,000 rows of numbers — but you can see a scatter plot
- Training curves tell you if the model is learning, overfitting, or broken

> "A loss curve going up means something is very wrong. You'd never know from the numbers alone."

---

## When to Use Which Chart

```
┌────────────────────┬──────────────────────────────────────────┐
│ Chart type         │ When to use                              │
├────────────────────┼──────────────────────────────────────────┤
│ Line plot          │ Training/validation loss over epochs     │
│ Scatter plot       │ Feature relationships, 2D clusters       │
│ Histogram          │ Distribution of a feature or loss values │
│ Bar chart          │ Comparing categories (accuracy per class)│
│ Heatmap/imshow     │ Confusion matrix, weight visualization   │
│ Subplots           │ Multiple charts side by side             │
└────────────────────┴──────────────────────────────────────────┘
```

---

## Diagram

```
plt.figure()            ← create canvas
    ↓
plt.plot() / plt.scatter() / plt.hist()   ← draw on canvas
    ↓
plt.xlabel(), plt.ylabel(), plt.title()  ← label it
    ↓
plt.legend()            ← add legend
    ↓
plt.show()              ← display it
```

---

## Code — Everything You Need

```python
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Line Plot — Training Curve (most used in ML) ───
epochs     = list(range(1, 21))
train_loss = [1.0, 0.85, 0.72, 0.61, 0.52, 0.45, 0.39, 0.34,
              0.30, 0.27, 0.24, 0.22, 0.20, 0.18, 0.17, 0.16,
              0.15, 0.14, 0.135, 0.13]
val_loss   = [1.1, 0.90, 0.76, 0.65, 0.57, 0.52, 0.48, 0.46,
              0.45, 0.44, 0.44, 0.45, 0.46, 0.47, 0.49, 0.51,
              0.53, 0.56, 0.58, 0.61]   # ← starts going up = overfitting!

plt.figure(figsize=(8, 4))
plt.plot(epochs, train_loss, label='Train Loss', color='blue')
plt.plot(epochs, val_loss,   label='Val Loss',   color='red', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── 2. Scatter Plot — Feature Relationships ───────────
np.random.seed(42)
class0_x = np.random.randn(50) - 1
class0_y = np.random.randn(50) - 1
class1_x = np.random.randn(50) + 1
class1_y = np.random.randn(50) + 1

plt.figure(figsize=(6, 5))
plt.scatter(class0_x, class0_y, label='Class 0', color='blue', alpha=0.6)
plt.scatter(class1_x, class1_y, label='Class 1', color='red',  alpha=0.6)
plt.xlabel('Feature 1');  plt.ylabel('Feature 2')
plt.title('Feature Space (2 classes)')
plt.legend()
plt.show()

# ── 3. Histogram — Data Distribution ─────────────────
data = np.random.randn(1000)   # normal distribution

plt.figure(figsize=(6, 4))
plt.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Value');  plt.ylabel('Count')
plt.title('Distribution of Feature')
plt.show()

# ── 4. Bar Chart — Class Accuracy ────────────────────
classes   = ['Cat', 'Dog', 'Bird', 'Fish']
accuracy  = [0.92, 0.88, 0.75, 0.95]

plt.figure(figsize=(6, 4))
plt.bar(classes, accuracy, color=['#4CAF50','#2196F3','#FF9800','#9C27B0'])
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy')
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.01, f'{v:.0%}', ha='center', fontweight='bold')
plt.show()

# ── 5. Subplots — Multiple Charts Together ────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, train_loss, 'b-', label='Train')
axes[0].plot(epochs, val_loss,   'r--', label='Val')
axes[0].set_title('Loss Curves')
axes[0].legend()

axes[1].scatter(class0_x, class0_y, c='blue', alpha=0.5, label='Class 0')
axes[1].scatter(class1_x, class1_y, c='red',  alpha=0.5, label='Class 1')
axes[1].set_title('Feature Space')
axes[1].legend()

plt.tight_layout()
plt.show()

# ── 6. Heatmap — Confusion Matrix ────────────────────
confusion = np.array([[85, 10,  5],
                       [ 8, 78, 14],
                       [ 3, 12, 85]])
classes = ['Cat', 'Dog', 'Bird']

plt.figure(figsize=(5, 4))
plt.imshow(confusion, cmap='Blues')
plt.colorbar()
plt.xticks(range(3), classes)
plt.yticks(range(3), classes)
plt.xlabel('Predicted');  plt.ylabel('Actual')
plt.title('Confusion Matrix')
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(confusion[i,j]), ha='center', va='center',
                 color='white' if confusion[i,j] > 60 else 'black')
plt.tight_layout()
plt.show()
```

---

## Reading a Training Curve — Critical Skill

```
Good training:
  Train loss ↘  Val loss ↘  (both going down together)

Overfitting:
  Train loss ↘  Val loss ↘ then ↗  (val starts going back up)
  → model memorizing training data, not generalizing

Underfitting:
  Train loss high  Val loss high  (both staying high)
  → model not learning enough

Learning rate too high:
  Loss bounces up and down wildly or goes to NaN
```

---

## Exercise
1. Plot a training curve where overfitting starts at epoch 10
2. Create a scatter plot with 3 clusters of 30 points each in different colors
3. Plot a histogram of 500 random values from `np.random.exponential(scale=2)`
4. Create a 1×3 subplot: line plot, scatter, histogram side by side

---

## What to Remember Forever
> Always plot your training/validation loss — it tells you everything about your model's health.
> If val loss starts going up while train loss goes down: overfitting. Stop training or regularize.
