# Lesson: Loss Functions

**Date taught:** 2026-09-26
**Folder:** ml-core/
**Status:** ✅ Taught

---

## Analogy

You're learning to throw darts. The bullseye is the correct answer.

After each throw, your coach tells you how far off you were:
- "You missed by 3cm" → small loss
- "You missed by 20cm" → large loss
- "Perfect bullseye!" → zero loss

The way your coach measures distance is the **loss function**.

Different problems need different rulers:
- How far your dart landed = MSE (regression)
- Did you hit the right zone? = Cross-entropy (classification)

> Loss = the number that tells your model how wrong it was. Training = minimize this number.

---

## Diagram: Loss Functions at a Glance

```
REGRESSION (predict a number)
  MSE   → big errors punished harder (squares the error)
  MAE   → all errors treated equally (absolute value)

BINARY CLASSIFICATION (yes/no, 0/1)
  Binary Cross-Entropy → measures wrong probabilities for 2 classes

MULTI-CLASS CLASSIFICATION (cat/dog/bird)
  Categorical Cross-Entropy → measures wrong probabilities for N classes

GENERATION (LLMs, next-word prediction)
  Cross-Entropy on token probabilities → used in all LLM training

Flow:
  Model prediction → Loss function → Single number → Backprop → Update weights
```

---

## Loss Function 1: MSE (Mean Squared Error)

**Use for:** regression (predicting continuous values like price, temperature)

```
MSE = (1/n) × Σ(y_pred - y_true)²

If you predicted 10, true is 8:  error = (10-8)² = 4
If you predicted 15, true is 8:  error = (15-8)² = 49  ← punishes big errors!
```

```python
import numpy as np

y_true = np.array([3.0, 5.0, 7.0, 2.0])
y_pred = np.array([2.5, 5.1, 6.8, 2.2])

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))   # same units as original

print(f"MSE:  {mse(y_true, y_pred):.4f}")
print(f"RMSE: {rmse(y_true, y_pred):.4f}")  # easier to interpret
```

---

## Loss Function 2: MAE (Mean Absolute Error)

**Use for:** regression when you want to be less sensitive to outliers

```
MAE = (1/n) × Σ|y_pred - y_true|

If you predicted 10, true is 8:  error = |10-8| = 2
If you predicted 15, true is 8:  error = |15-8| = 7  ← linear, not squared
```

```python
def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

# MSE vs MAE on outlier
y_true_out = np.array([3.0, 5.0, 7.0, 2.0, 3.0])
y_pred_out = np.array([2.5, 5.1, 6.8, 2.2, 100.0])  # massive outlier!

print(f"MSE with outlier: {mse(y_true_out, y_pred_out):.1f}")   # ~1940 — huge!
print(f"MAE with outlier: {mae(y_true_out, y_pred_out):.1f}")   # ~19 — more robust
```

---

## Loss Function 3: Binary Cross-Entropy

**Use for:** binary classification (spam/not spam, yes/no, 0/1)

```
BCE = -(1/n) × Σ [y×log(p) + (1-y)×log(1-p)]

Where:
  y = true label (0 or 1)
  p = predicted probability

If true=1, predicted=0.9 → loss = -log(0.9) = 0.10  (small, correct)
If true=1, predicted=0.1 → loss = -log(0.1) = 2.30  (large, wrong)
If true=1, predicted=0.5 → loss = -log(0.5) = 0.69  (uncertain)
```

```python
def binary_cross_entropy(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)   # avoid log(0)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

y_true = np.array([1, 1, 0, 0, 1])
y_pred_good = np.array([0.9, 0.8, 0.1, 0.2, 0.85])  # confident & correct
y_pred_bad  = np.array([0.1, 0.3, 0.8, 0.7, 0.2])   # confident & wrong

print(f"Good predictions BCE: {binary_cross_entropy(y_true, y_pred_good):.4f}")  # ~0.17
print(f"Bad predictions BCE:  {binary_cross_entropy(y_true, y_pred_bad):.4f}")   # ~1.63
```

---

## Loss Function 4: Categorical Cross-Entropy

**Use for:** multi-class classification (3+ classes: cat/dog/bird)

```
CCE = -(1/n) × Σ Σ y_ij × log(p_ij)

Where y is one-hot encoded and p is softmax output.

True class = Dog (index 1)
y_true = [0, 1, 0]   (one-hot)
y_pred = [0.1, 0.7, 0.2]   (softmax probabilities)
loss = -log(0.7) = 0.36   (correct class probability)

If model is uncertain:
y_pred = [0.33, 0.34, 0.33]
loss = -log(0.34) = 1.08   (higher loss for uncertainty)
```

```python
def categorical_cross_entropy(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1.0)
    return -np.sum(y_true * np.log(y_pred))   # for one sample

# 3-class example: cat=0, dog=1, bird=2
y_true = np.array([0, 1, 0])           # true: dog

y_pred_correct   = np.array([0.1, 0.8, 0.1])   # confident correct
y_pred_wrong     = np.array([0.8, 0.1, 0.1])   # confident wrong
y_pred_uncertain = np.array([0.33, 0.34, 0.33]) # uncertain

print(f"Correct:   {categorical_cross_entropy(y_true, y_pred_correct):.3f}")    # 0.223
print(f"Wrong:     {categorical_cross_entropy(y_true, y_pred_wrong):.3f}")      # 2.303
print(f"Uncertain: {categorical_cross_entropy(y_true, y_pred_uncertain):.3f}")  # 1.079
```

---

## Key Terms

| Term | Definition |
|------|-----------|
| Loss function | A formula that produces a single number measuring how wrong the model's prediction was. Lower = better. Training = minimize this number. |
| MSE | Mean Squared Error. Average of squared differences between prediction and truth. Penalises large errors heavily (squares them). Used for regression. |
| MAE | Mean Absolute Error. Average of absolute differences. Treats all errors equally. More robust to outliers than MSE. |
| RMSE | Root Mean Squared Error. Square root of MSE. Same units as the target — easier to interpret. |
| Cross-entropy | Measures the distance between two probability distributions. The model's output probabilities vs the true one-hot labels. Standard loss for classification. |
| Binary cross-entropy (BCE) | Cross-entropy for 2-class problems. Output layer uses sigmoid. Formula: `-[y·log(p) + (1-y)·log(1-p)]` |
| Categorical cross-entropy (CCE) | Cross-entropy for 3+ class problems. Output layer uses softmax. Formula: `-Σ y_i · log(p_i)` |
| Logit | Raw, unnormalised score output by the last layer before any activation function. Passed to sigmoid (binary) or softmax (multi-class). |
| One-hot encoding | Representing a class as a binary vector. Class 1 out of 3 → [0, 1, 0]. Required format for CCE. |
| Softmax | Converts a vector of logits into probabilities that sum to 1. Used in multi-class output layer before CCE. |
| Sigmoid | Converts a single logit to a probability between 0 and 1. Used in binary classification output layer before BCE. |
| Gradient of loss | The derivative of the loss with respect to each weight. Tells the optimizer which direction to move each weight to reduce loss. |

---

## Summary: Which Loss to Use?

```
┌────────────────────────────┬──────────────────────────┬────────────────────────┐
│ Problem                    │ Output Layer             │ Loss Function          │
├────────────────────────────┼──────────────────────────┼────────────────────────┤
│ Regression (continuous)    │ Linear (no activation)   │ MSE / MAE              │
│ Binary classification      │ Sigmoid (1 neuron)       │ Binary Cross-Entropy   │
│ Multi-class classification │ Softmax (N neurons)      │ Categorical CE         │
│ LLM next-token prediction  │ Softmax (vocab size)     │ Cross-Entropy          │
└────────────────────────────┴──────────────────────────┴────────────────────────┘
```

---

## Visualizing Loss Functions

```python
import matplotlib.pyplot as plt

p = np.linspace(0.01, 0.99, 200)   # predicted probability

# Cross-entropy for true label = 1
loss_true_1 = -np.log(p)           # want p → 1, loss → 0
loss_true_0 = -np.log(1 - p)       # want p → 0, loss → 0

plt.figure(figsize=(8, 4))
plt.plot(p, loss_true_1, 'b-', label='True=1: -log(p)')
plt.plot(p, loss_true_0, 'r-', label='True=0: -log(1-p)')
plt.xlabel('Predicted Probability p')
plt.ylabel('Loss')
plt.title('Binary Cross-Entropy')
plt.legend(); plt.grid(True)
plt.ylim(0, 5)
plt.show()
```

---

## Exercise
1. Compute MSE and MAE for `y_true=[2,4,6,8]`, `y_pred=[2.5,3.5,7,7]`. Which is larger?
2. Compute BCE for: true=1, predicted=0.99 vs predicted=0.01. Explain why they differ.
3. What's the CCE loss for true=[1,0,0] with pred=[0.33,0.33,0.34]? And with pred=[0.98,0.01,0.01]?
4. You're building a house price predictor. Which loss do you use? A spam detector?

---

## What to Remember Forever
> Regression → MSE (mean squared error)
> Binary classification → Binary Cross-Entropy
> Multi-class → Categorical Cross-Entropy
> LLMs → Cross-Entropy on next-token prediction
> Lower loss = better model. Training = minimize the loss.
