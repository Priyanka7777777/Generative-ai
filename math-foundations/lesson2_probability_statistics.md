# Lesson: Probability & Statistics for ML

**Date taught:** 2026-09-26
**Folder:** math-foundations/
**Status:** ✅ Taught

---

## Analogy

**Probability** = how likely is something to happen?
Flip a coin → 50% heads, 50% tails.

**Statistics** = given data that already happened, what can we learn from it?
You flipped a coin 1000 times and got 600 heads → is it a fair coin?

In ML:
- **Probability** is used in model outputs ("70% chance this is a cat")
- **Statistics** is used in data analysis and evaluating model performance

---

## Diagrams

### Normal (Gaussian) Distribution
```
        ↑ probability
        │        ╭───╮
        │      ╭─╯   ╰─╮
        │    ╭─╯         ╰─╮
        │  ╭─╯               ╰─╮
        └──────────────────────────→ value
             μ-2σ  μ-σ  μ  μ+σ  μ+2σ

μ (mu)    = mean (center of the bell)
σ (sigma) = standard deviation (width of the bell)

68% of data falls within 1σ of mean
95% within 2σ
99.7% within 3σ
```

### Probability vs Likelihood
```
Probability: Fixed distribution, unknown outcome
  "What's the chance of rolling a 6?" → 1/6

Likelihood: Fixed data, unknown distribution
  "Given I rolled [4,6,2,6,6], what die is most likely?"
  → Used in training: adjust model params to maximize likelihood of seeing the training data
```

---

## Core Concepts + Code

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Basic Statistics ───────────────────────────────────
data = np.array([23, 25, 28, 22, 27, 30, 24, 26, 29, 21])

mean   = np.mean(data)          # average: 25.5
median = np.median(data)        # middle value: 25.5
std    = np.std(data)           # spread: how far from mean on average
var    = np.var(data)           # variance = std²
minimum = np.min(data)
maximum = np.max(data)

# ── Distributions ─────────────────────────────────────
# Normal distribution
normal_data = np.random.normal(loc=0, scale=1, size=1000)
# loc = mean (μ), scale = std (σ)

# Uniform distribution
uniform_data = np.random.uniform(low=0, high=1, size=1000)

# Visualize
plt.hist(normal_data, bins=30, density=True, alpha=0.7)
plt.title('Normal Distribution')
plt.show()

# ── Probability Basics ─────────────────────────────────
# P(A and B) = P(A) × P(B)  [if independent]
p_rain  = 0.3
p_cold  = 0.4
p_both  = p_rain * p_cold      # 0.12

# P(A or B) = P(A) + P(B) - P(A and B)
p_either = p_rain + p_cold - p_both   # 0.58

# ── Bayes Theorem ─────────────────────────────────────
# P(A|B) = P(B|A) × P(A) / P(B)
# "What's probability of A given B happened?"

# Example: spam detection
# P(spam) = 0.2          ← prior: 20% of emails are spam
# P("free"|spam) = 0.8   ← 80% of spam contains "free"
# P("free") = 0.3        ← 30% of all emails contain "free"

p_spam = 0.2
p_free_given_spam = 0.8
p_free = 0.3

p_spam_given_free = (p_free_given_spam * p_spam) / p_free
print(f"P(spam | 'free') = {p_spam_given_free:.2f}")  # 0.53

# ── Cross-Entropy Loss (most important formula in ML) ──
# Used for classification. Measures distance between predicted and true distributions.

def cross_entropy_loss(y_true, y_pred):
    # y_pred: model output probabilities (after softmax)
    # y_true: one-hot encoded true labels
    eps = 1e-9   # avoid log(0)
    return -np.sum(y_true * np.log(y_pred + eps))

# Example: 3-class classification
y_true = np.array([0, 1, 0])       # true class is class 1
y_pred_good = np.array([0.1, 0.8, 0.1])   # model is confident & correct
y_pred_bad  = np.array([0.1, 0.1, 0.8])   # model is confident & wrong

print(cross_entropy_loss(y_true, y_pred_good))  # low loss  ~0.22
print(cross_entropy_loss(y_true, y_pred_bad))   # high loss ~2.3

# ── Softmax (converts logits to probabilities) ─────────
def softmax(x):
    e = np.exp(x - np.max(x))   # subtract max for numerical stability
    return e / e.sum()

logits = np.array([2.0, 1.0, 0.1])
probs  = softmax(logits)
print(probs)          # [0.659, 0.242, 0.099] — sum = 1.0

# ── Sigmoid (binary classification output) ────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(0))     # 0.5 (uncertain)
print(sigmoid(3))     # 0.95 (confident positive)
print(sigmoid(-3))    # 0.047 (confident negative)

# ── Standard Normalization (Z-score) ──────────────────
# Why: neural networks train much better when inputs are ~N(0,1)
X = np.array([100, 200, 150, 300, 250], dtype=float)
X_normalized = (X - X.mean()) / X.std()
print(X_normalized)   # values now centered at 0, spread ~1

# ── Correlation ───────────────────────────────────────
# How related are two features?
age    = np.array([25, 30, 35, 40, 45])
salary = np.array([40, 55, 65, 80, 95])
corr   = np.corrcoef(age, salary)[0, 1]
print(f"Correlation: {corr:.2f}")   # close to 1.0 = strong positive
```

---

## Key Concepts for ML

| Concept | Formula | ML Usage |
|---------|---------|----------|
| Mean | `Σx / n` | Data normalization, baseline |
| Std deviation | `√(Σ(x-μ)² / n)` | Normalization, weight init |
| Normal dist | `N(μ, σ²)` | Weight initialization, data assumptions |
| Softmax | `eˣⁱ / Σeˣ` | Multi-class output probabilities |
| Sigmoid | `1/(1+e⁻ˣ)` | Binary classification output |
| Cross-entropy | `-Σy·log(ŷ)` | Classification loss function |
| Bayes theorem | `P(A\|B) = P(B\|A)P(A)/P(B)` | Naive Bayes, understanding priors |
| Z-score norm | `(x - μ) / σ` | Feature scaling before training |

---

## Why Z-score Normalization Matters

```
Without normalization:          With normalization:
  age:    [25, 30, 45]            age:    [-1.2, 0.0, 1.2]
  salary: [40k, 80k, 120k]       salary: [-1.2, 0.0, 1.2]

The model tries to learn weights for both.
Salary is 1000× larger → its gradient dominates → age gets ignored!
After normalization: both features on same scale → fair learning.
```

---

## Exercise
1. Generate 500 random numbers from `N(mean=10, std=2)`. Verify `np.mean()` ≈ 10 and `np.std()` ≈ 2
2. Compute softmax of `[3.0, 1.0, 0.2]`. Verify they sum to 1.0
3. Compute cross-entropy loss for a perfect prediction `y_pred=[1.0, 0.0, 0.0]`, `y_true=[1,0,0]`
4. Normalize `[100, 200, 300, 400, 500]` using z-score. What are the new values?

---

## What to Remember Forever
> Softmax = convert raw scores to probabilities (multi-class)
> Sigmoid = convert raw score to 0-1 probability (binary)
> Cross-entropy = how wrong your probabilities are (lower = better)
> Always normalize input features — same scale, better training.
