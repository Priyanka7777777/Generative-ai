# Lesson 3: Overfitting & Regularization

**Date taught:** 2026-09-26
**Folder:** neural-networks/
**Status:** ✅ Taught | Exercise given
**Prev:** [lesson2_optimizers.md](lesson2_optimizers.md)
**Next:** lesson4_pytorch_deepdive.md

---

## Analogy

Two students are preparing for an exam.

**Student A (Overfitter):**
Memorises every question and answer from the practice book word-for-word. On practice tests: 100%. On the real exam with new questions: 40%. They learned the answers, not the subject.

**Student B (Well-Generalised):**
Studies the concepts, understands the reasoning. On practice tests: 92%. On the real exam: 89%. Small gap. They genuinely understand.

> **Overfitting = Student A. Your model memorised the training data including its noise, instead of learning the true pattern.**

The fix is **regularization** — techniques that force the model to learn general patterns, not memorise specific examples.

---

## Diagrams

### The Training Curve — Detecting Overfitting
```
         GOOD                      OVERFITTING               UNDERFITTING
Loss ↑                    Loss ↑                    Loss ↑
     │  ╲ train                │  ╲ train                │
     │   ╲ val                 │   ╲                      │  ─ ─ train ≈ val
     │    ╲                    │    ╲  ← val turns up     │
     │     ╲__                 │     ╲╱                   │  (both stuck high)
     └──────────→ epochs        └──────────→ epochs        └──────────→ epochs

     Train ≈ Val               Train ↓, Val ↑            Train high, Val high
     Both going down           Big gap = memorising       Both high = too simple
```

### The Bias-Variance Tradeoff
```
Total Error = Bias² + Variance + Irreducible Noise

           Error
             ↑
             │           Total Error
             │          ╱
             │   Variance (overfitting)
             │  ╱
             │─╱──────────────────────
             │   ╲  Bias (underfitting)
             │    ╲____________________
             └──────────────────────→ Model Complexity
                ↑            ↑
          Underfitting    Overfitting
          (high bias)     (high variance)

Sweet spot = low bias AND low variance
```

### What Regularization Does to Weights
```
Without regularization:        With L2 regularization:
  w1 = 87.3                      w1 = 2.1
  w2 = -142.6                    w2 = -1.8
  w3 = 0.001                     w3 = 0.4
  ← huge weights = memorising    ← small weights = generalising
```

### Dropout — Visualised
```
Normal forward pass:       With Dropout (50%):
  [N1]─[N2]─[N3]            [N1]─[X ]─[N3]
  [N4]─[N5]─[N6]            [X ]─[N5]─[X ]   ← random neurons zeroed
  [N7]─[N8]─[N9]            [N7]─[X ]─[N9]

Each training step → different random subnetwork.
Inference → all neurons active, scaled by keep probability.
```

### Early Stopping
```
Epoch 10: val_loss=0.35  ← save checkpoint ✓
Epoch 15: val_loss=0.33  ← save checkpoint ✓ (new best)
Epoch 20: val_loss=0.36  ← worse, patience=1
Epoch 25: val_loss=0.40  ← worse, patience=2
...
Epoch 30: val_loss=0.45  ← patience exceeded → STOP
→ Restore epoch 15 weights  ← best generalisation
```

---

## Technical Explanation

### Why Overfitting Happens

A neural network with millions of parameters has enough capacity to memorise your entire training set. Think of fitting a polynomial of degree 9 through 10 data points — it fits them all perfectly but is wildly wrong between points. That's overfitting.

**Symptoms:**
- Train accuracy: 99%, Val accuracy: 60% → big gap = overfitting
- Train loss keeps going down, Val loss starts going up → stop here

**Root causes:**
```
1. Model too complex for the amount of data
2. Not enough training data
3. Trained too many epochs
4. Features contain noise the model memorises
```

---

### Technique 1: L2 Regularization (Weight Decay)

Add a penalty to the loss for having large weights. Forces weights to stay small → can't overfit by assigning huge importance to individual training examples.

```
New Loss = Original Loss + λ × Σ(w²)
                               ↑
                   penalty for large weights

λ (lambda) = regularization strength
  λ = 0      → no regularization
  λ = 0.01   → mild (common starting point)
  λ = 1.0    → heavy (use for tiny datasets)
```

**Effect on gradient update:**
```
Normal:    w = w - lr × gradient
With L2:   w = w - lr × (gradient + 2λw)
                                   ↑
                        extra push toward 0 each step
```

In PyTorch: use `AdamW` with `weight_decay=0.01` — it's L2 built into the optimizer.

---

### Technique 2: L1 Regularization (Lasso)

Same idea but uses absolute value:
```
New Loss = Original Loss + λ × Σ|w|
```

**Key difference from L2:** L1 pushes small weights all the way to exactly zero. Creates sparse models — only important features get non-zero weights.

```
L2 result:  w = [0.01, 0.8, 0.02, 0.7, 0.001]   ← all small but non-zero
L1 result:  w = [0.00, 0.8, 0.00, 0.7, 0.000]   ← unimportant = exactly 0
```

Analogy: L2 = give everyone a pay cut. L1 = fire people who don't contribute.

---

### Technique 3: Dropout

Randomly zero out a fraction of neurons during each training step.

```
keep_probability = 0.8  →  drop 20% of neurons randomly each step
Training: zero out neurons + scale remaining by 1/keep_prob
Testing:  all neurons active (no dropout applied)
```

Why it works: each training step trains a different sub-network. At test time, the full network behaves like an ensemble of all those sub-networks averaged together.

Where to use:
- After fully-connected layers → typical `p=0.3` to `p=0.5`
- After conv layers → `p=0.1` to `p=0.3`

---

### Technique 4: Batch Normalization

Normalize activations within each mini-batch to mean=0, std=1.

```
Input to layer: [25, 100, 0.01, 50, 3]      ← different scales
After BatchNorm: [-0.5, 1.2, -1.1, 0.8, -0.4]  ← stable
Then: y = γ × normalized + β  (learned scale + shift)
```

Why it helps:
- Keeps activations in a healthy range → gradients don't vanish or explode
- Acts as mild regularizer
- Usually allows higher learning rates → faster training

---

### Technique 5: Early Stopping

Monitor validation loss. When it stops improving for N epochs (patience), stop training and restore the best weights.

```python
# Pseudocode
best_val_loss = infinity
for epoch in range(max_epochs):
    train_one_epoch()
    val_loss = evaluate()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_weights()           # ← remember this checkpoint
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            restore_best_weights()
            break
```

---

### Technique 6: Data Augmentation

Artificially expand training data by creating variations:
```
Images:  flip, rotate, crop, zoom, brightness, noise
Text:    synonym replacement, paraphrase, back-translation
Tabular: add small Gaussian noise to numerical features
```

The most effective fix when you have limited data. Try this before regularization.

---

### Decision Guide — Which Technique to Use

```
Problem                               Solution
────────────────────────────────────────────────────────
Val loss turns up early               Early stopping (always try first)
Model too complex for data            L2 / AdamW weight decay
Want automatic feature selection      L1 regularization
Deep network with FC layers           Dropout (p=0.3–0.5)
Unstable training, exploding grads    Batch Normalization
Not enough training data              Data Augmentation
Still overfitting after all above     Reduce model size
```

---

## Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Data ─────────────────────────────────────────────
X, y = make_classification(n_samples=500, n_features=20,
                            n_informative=5, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

# ── Overfit Model: big, no regularization ─────────────
class OverfitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),  nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ── Regularized Model: Dropout + BatchNorm + smaller ─
class RegularizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.BatchNorm1d(128),           # normalize activations
            nn.ReLU(),
            nn.Dropout(0.4),               # drop 40% of neurons randomly

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ── Training with early stopping ──────────────────────
def train(model, optimizer, epochs=300, name="Model"):
    criterion = nn.BCELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    best_weights = None

    for epoch in range(epochs):
        model.train()                      # enable dropout + batchnorm train mode
        pred = model(X_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()                       # disable dropout for evaluation
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"{name}: Early stopped at epoch {epoch}")
                model.load_state_dict(best_weights)
                break

    return train_losses, val_losses

# ── Run both models ───────────────────────────────────
overfit_model = OverfitModel()
optimizer1 = optim.Adam(overfit_model.parameters(), lr=0.001)
train_l1, val_l1 = train(overfit_model, optimizer1, name="Overfit")

reg_model = RegularizedModel()
optimizer2 = optim.AdamW(                  # AdamW = Adam + L2 weight decay
    reg_model.parameters(),
    lr=0.001,
    weight_decay=0.01                      # L2 regularization strength
)
train_l2, val_l2 = train(reg_model, optimizer2, name="Regularized")

# ── Compare ────────────────────────────────────────────
def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return ((model(X) > 0.5).float() == y).float().mean().item()

for name, model in [("Overfit", overfit_model), ("Regularized", reg_model)]:
    tr = accuracy(model, X_train, y_train)
    vl = accuracy(model, X_val,   y_val)
    print(f"{name}: train={tr:.3f}  val={vl:.3f}  gap={tr-vl:.3f}")

# ── Plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_l1, 'b-', label='Train'); axes[0].plot(val_l1, 'r--', label='Val')
axes[0].set_title('Overfit — val diverges'); axes[0].legend(); axes[0].grid(True)
axes[1].plot(train_l2, 'b-', label='Train'); axes[1].plot(val_l2, 'r--', label='Val')
axes[1].set_title('Regularized — val stays close'); axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.show()

# ── Manual L1/L2 (to understand what's happening) ─────
def loss_with_l2(model, base_loss, lambda_=0.01):
    l2 = sum(p.pow(2).sum() for p in model.parameters())
    return base_loss + lambda_ * l2

def loss_with_l1(model, base_loss, lambda_=0.01):
    l1 = sum(p.abs().sum() for p in model.parameters())
    return base_loss + lambda_ * l1
```

---

## Key Terms

| Term | Definition |
|------|-----------|
| Overfitting | Model memorises training data including noise. High train accuracy, low val/test accuracy. Large gap between them. |
| Underfitting | Model too simple to capture patterns. Both train and val accuracy are low. Small gap. |
| Generalisation | Model's ability to perform well on new unseen data. The entire goal of regularization. |
| Bias | Error from wrong assumptions — model too simple. Causes underfitting. High bias = the model consistently misses the true pattern. |
| Variance | Error from sensitivity to training data noise — model too complex. Causes overfitting. High variance = model changes a lot with different training sets. |
| Bias-variance tradeoff | Reducing one tends to increase the other. Regularization finds the sweet spot. |
| L2 regularization | Adds `λ×Σw²` to loss. Pushes all weights toward 0. Also called Ridge or weight decay. The most common regularization. |
| L1 regularization | Adds `λ×Σ\|w\|` to loss. Pushes small weights to exactly 0 (sparse model). Also called Lasso. Good for feature selection. |
| Weight decay | L2 regularization built into the optimizer. Use `AdamW(weight_decay=0.01)`. |
| Dropout | Randomly zeros a fraction of neurons each training step. At inference, all neurons active. Forces robust feature learning. |
| Batch Normalization | Normalises layer activations to mean=0, std=1 within each mini-batch. Stabilises training, mild regularizer. |
| Early stopping | Stop training when val loss stops improving. Restore best-checkpoint weights. Controlled by `patience` hyperparameter. |
| Patience | Early stopping parameter — number of epochs to wait for improvement before stopping. |
| Data augmentation | Creating new training examples by transforming existing ones (flip, rotate, crop). Most effective fix for limited data. |
| `model.train()` | PyTorch mode that enables dropout and batchnorm training behaviour. Call before training loop. |
| `model.eval()` | PyTorch mode that disables dropout, uses running stats for batchnorm. Call before any evaluation. |

---

## Exercise
1. Run the code. What is the train/val accuracy gap for Overfit vs Regularized model?
2. Remove `Dropout` from `RegularizedModel` — does the val accuracy drop?
3. Remove `BatchNorm1d` layers — does training become less stable (val loss noisy)?
4. Set `weight_decay=1.0` (very strong L2) — what happens to train accuracy? Why?
5. Set `patience=3` in early stopping — does it stop too early? Try `patience=50`.

---

## What to Remember Forever
> Overfitting = big gap between train and val accuracy. Model memorised, not learned.
> Always plot training curves — val loss turning up is your early warning signal.
> Default fix order: early stopping → Dropout → AdamW weight decay → more data.
> Always call `model.train()` before training and `model.eval()` before evaluation.
> `AdamW` = Adam + L2 weight decay built-in. Prefer it over plain Adam.
