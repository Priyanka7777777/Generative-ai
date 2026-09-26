# Lesson: Supervised vs Unsupervised Learning

**Date taught:** 2026-09-26
**Folder:** ml-core/
**Status:** ✅ Taught

---

## Analogy

### Supervised Learning
A teacher gives you flashcards:
- Front: photo of an animal
- Back: the answer ("Cat", "Dog", "Bird")

You study with answers. At the exam, you predict unseen animals.
**You learned with labels.**

### Unsupervised Learning
You're given 1000 photos with NO labels. No teacher.
You notice some photos look similar — pointy ears, whiskers, furry.
You group them together. You discover structure on your own.
**You learned patterns without labels.**

---

## Diagram

```
SUPERVISED LEARNING
                                         ┌── Classification
  Labeled data → Model → Predictions ───┤
  (X, y)                                 └── Regression

UNSUPERVISED LEARNING
                                         ┌── Clustering
  Unlabeled data → Model → Patterns ────┤   (K-Means, DBSCAN)
  (X only)                               ├── Dimensionality Reduction
                                         │   (PCA, t-SNE, UMAP)
                                         └── Anomaly Detection

SEMI-SUPERVISED
  Small labeled + large unlabeled → Model  (common in practice)

SELF-SUPERVISED (key for LLMs!)
  Data labels itself:
  "The cat sat on the ___"  → model learns to predict next word
  No human labels needed → how GPT/BERT are trained
```

---

## Supervised Learning Types

### Classification — predict a category

```
Input  →  [age, salary, experience]
Output →  "Will this person buy?" (Yes / No)
         "Is this email spam?" (Spam / Not spam)
         "What digit is this?" (0-9)

Output is a discrete class.
Loss function: Cross-entropy
```

### Regression — predict a number

```
Input  →  [bedrooms, location, area]
Output →  "What is the house price?" ($350,000)

Output is a continuous number.
Loss function: MSE (Mean Squared Error)
```

---

## Unsupervised Learning Types

### Clustering — group similar things

```
Input  →  1000 customers with [age, spend, frequency]
Output →  Group A: young, high spend (premium)
          Group B: older, low spend (budget)
          Group C: occasional buyers (seasonal)
No labels — model discovers the groups.
```

### Dimensionality Reduction — compress while preserving structure

```
Input  →  784-dimensional image (28×28 pixels)
Output →  2D representation (for visualization)

High-dim data                2D representation
(can't visualize)     PCA    (can visualize clusters)
     •                →          •    •
    • •                         •  •
      •                           •
```

---

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─── SUPERVISED: Classification ───────────────────────
# Generate fake labeled data
X, y = make_classification(
    n_samples=200,
    n_features=2,        # 2 features (so we can plot)
    n_classes=2,
    random_state=42
)

# Train a logistic regression classifier
model = LogisticRegression()
model.fit(X, y)

# Predict on new data
X_new = np.array([[1.5, -0.5], [-1.0, 1.2]])
preds = model.predict(X_new)
probs = model.predict_proba(X_new)
print("Predictions:", preds)           # [1, 0]
print("Probabilities:", probs)         # [[0.1, 0.9], [0.8, 0.2]]

# Plot
plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1', alpha=0.6)
plt.title('Supervised: Classification')
plt.legend(); plt.show()

# ─── UNSUPERVISED: Clustering ─────────────────────────
# Generate unlabeled data with natural clusters
X_raw, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# K-Means: find 3 groups (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_raw)

plt.scatter(X_raw[:, 0], X_raw[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='X', s=200, c='red', label='Centroids')
plt.title('Unsupervised: K-Means Clustering')
plt.legend(); plt.show()

# ─── UNSUPERVISED: Dimensionality Reduction (PCA) ─────
from sklearn.datasets import load_digits
digits = load_digits()                  # 1797 images, each 64 features (8x8 pixels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(digits.data)

# Reduce from 64 dimensions → 2 dimensions
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print(f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}")  # ~18%

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_2d[:,0], X_2d[:,1],
                      c=digits.target, cmap='tab10', alpha=0.6, s=15)
plt.colorbar(scatter)
plt.title('PCA: 64D → 2D (digits dataset)')
plt.show()
```

---

## Quick Reference

| | Supervised | Unsupervised |
|--|-----------|--------------|
| Has labels? | Yes (X, y) | No (X only) |
| Goal | Predict y for new X | Find structure in X |
| Example tasks | Spam detection, price prediction | Customer segmentation, compression |
| Common algorithms | Linear regression, decision trees, neural nets | K-Means, PCA, autoencoders |
| Evaluation | Accuracy, MSE, F1 | Silhouette score, visual inspection |

---

## Self-Supervised = How LLMs Are Trained

```
Text: "The cat sat on the mat"

Task auto-generated:
  Input:  "The cat sat on the ___"
  Target: "mat"

The data labels ITSELF — no human annotation needed.
Train on billions of such predictions → model learns language.

This is how GPT, BERT, Claude are pre-trained.
```

---

## Exercise
1. Run the classification example. Change `n_classes=3`. Does logistic regression still work?
2. Run K-Means with `n_clusters=2` on the 3-cluster data — what happens?
3. Run PCA on digits with `n_components=10`. How much variance is explained? More than 2 components?

---

## What to Remember Forever
> Supervised = learning with answers (labeled data).
> Unsupervised = finding patterns without answers.
> LLMs are self-supervised — text predicts itself, no human labeling needed.
