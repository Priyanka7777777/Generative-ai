# Lesson 2: Gradient Descent & Optimizers

**Date taught:** 2026-09-26
**Status:** ✅ Taught | Exercise given
**Prev:** [lesson1_neural_networks.md](lesson1_neural_networks.md)
**Next:** lesson3_overfitting_regularization.md

---

## The Analogy

You're **blindfolded on a hilly landscape**. Goal: reach the lowest valley.

You can feel the slope under your feet:
- Slopes down to the left → step left
- Slopes down to the right → step right
- Keep stepping downhill until you feel flat ground

**That is gradient descent.**

| Real world | Neural network |
|-----------|---------------|
| Hilly landscape | Loss surface |
| Your position | Current weights |
| Slope under feet | Gradient |
| One step | Weight update |
| The valley | Minimum loss (best weights) |

---

## Diagrams

### Gradient Descent Path
```
Loss
 ↑
 |  ● ← start (random weights, high loss)
 |   ↘
 |    ●
 |     ↘
 |      ●
 |       ↘
 |        ● ← minimum (best weights)
 +--------------------------------→ Weights
```

### Learning Rate Too High (overshooting)
```
Loss
 ↑
 |       ●  start
 |      ↙ ↘
 |     ●    ↘ ← overshoots!
 |      ↗  ●
 |     ●       ← bouncing, never settles
 +--------------------------------→ Weights
```

### Learning Rate Too Low
```
Takes forever — thousands of tiny steps to reach the same valley.
```

---

## Types of Gradient Descent

```
┌─────────────────────┬──────────────────────────────┬─────────────────────┐
│ Type                │ How it works                 │ Problem             │
├─────────────────────┼──────────────────────────────┼─────────────────────┤
│ Batch GD            │ ALL data → 1 step            │ Very slow per step  │
│ Stochastic GD (SGD) │ 1 sample → 1 step            │ Very noisy path     │
│ Mini-batch GD       │ small batch → 1 step         │ Best of both ✅     │
└─────────────────────┴──────────────────────────────┴─────────────────────┘
```
In practice: mini-batch always used. Batch size = 32, 64, 128.

---

## Problems with Basic SGD

```
Problem 1: Gets stuck in local minima
Loss  *         *
       *       *   ← stuck here (local min)
         * * *
                  * * *  ← global min (never reached)

Problem 2: Flat regions → gradient ≈ 0 → tiny steps → very slow

Problem 3: Same learning rate for every weight
           (some need big updates, others need tiny ones)
```

---

## Optimizers

### SGD + Momentum

**Analogy:** Heavy ball rolling downhill — builds speed in consistent directions,
rolls past small bumps and local minima.

```
velocity = momentum × velocity - lr × gradient
weight   = weight + velocity
```

```
Without momentum:  zig-zag path
With momentum:     smooth curved path → faster convergence

    zig-zag                smooth
      ↗↘↗↘↗                ↗
     ↗↘↗↘↗↘→               →→→↘
                                 ↘→ minimum
```

---

### RMSProp

**Analogy:** Different muscles need different workouts. Adapts learning rate per weight.

```
cache  = decay × cache + (1 - decay) × gradient²
weight = weight - (lr / √cache) × gradient
```

Large gradients → smaller effective LR (slowed down)
Small gradients → larger effective LR (sped up)

---

### Adam — Use This by Default

**Analogy:** Momentum + RMSProp combined. Rolling ball that also adapts step size per weight.

```
m = β1 × m + (1 - β1) × gradient          ← momentum
v = β2 × v + (1 - β2) × gradient²         ← per-weight adaptation

weight = weight - lr × m / (√v + ε)
```

Default values (almost never change these):
- β1 = 0.9   (momentum)
- β2 = 0.999 (RMSProp)
- lr = 0.001

```
┌──────────────┬──────────────────────────────────┬──────────────────┐
│ Optimizer    │ Good for                         │ Use when         │
├──────────────┼──────────────────────────────────┼──────────────────┤
│ SGD          │ Simple problems, research        │ Rarely           │
│ SGD+Momentum │ Computer vision (CNNs)           │ Sometimes        │
│ RMSProp      │ RNNs, non-stationary problems    │ Sometimes        │
│ Adam         │ Everything                       │ Almost always ✅ │
└──────────────┴──────────────────────────────────┴──────────────────┘
```

---

## Code — SGD vs Adam from Scratch

```python
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.1;  b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.1;  b2 = np.zeros((1, 1))

def relu(x):    return np.maximum(0, x)
def relu_d(x):  return (x > 0).astype(float)
def sigmoid(x): return 1 / (1 + np.exp(-x))

def forward(X):
    z1 = X @ W1 + b1;  a1 = relu(z1)
    z2 = a1 @ W2 + b2; a2 = sigmoid(z2)
    return a2, a1, z1

def loss(pred, true): return np.mean((pred - true) ** 2)

def reset_weights():
    global W1, b1, W2, b2
    np.random.seed(42)
    W1=np.random.randn(2,4)*0.1; b1=np.zeros((1,4))
    W2=np.random.randn(4,1)*0.1; b2=np.zeros((1,1))

# ── Plain SGD ──────────────────────────────────────────
def train_sgd(epochs=2000, lr=0.1):
    global W1, b1, W2, b2
    reset_weights()
    for epoch in range(epochs):
        pred, a1, z1 = forward(X)
        l = loss(pred, y)
        dout = 2*(pred-y)/len(X) * pred*(1-pred)
        dW2=a1.T@dout; db2=dout.sum(0,keepdims=True)
        dz1=(dout@W2.T)*relu_d(z1)
        dW1=X.T@dz1;   db1=dz1.sum(0,keepdims=True)
        W1-=lr*dW1; b1-=lr*db1; W2-=lr*dW2; b2-=lr*db2
        if epoch%500==0: print(f"SGD  epoch {epoch:4d} loss {l:.4f}")

# ── Adam ───────────────────────────────────────────────
def train_adam(epochs=2000, lr=0.01, b1_=0.9, b2_=0.999, eps=1e-8):
    global W1, b1, W2, b2
    reset_weights()
    mW1=vW1=mW2=vW2=0
    for epoch in range(1, epochs+1):
        pred, a1, z1 = forward(X)
        l = loss(pred, y)
        dout = 2*(pred-y)/len(X) * pred*(1-pred)
        dW2=a1.T@dout; dz1=(dout@W2.T)*relu_d(z1); dW1=X.T@dz1
        # Adam for W1
        mW1 = b1_*mW1 + (1-b1_)*dW1
        vW1 = b2_*vW1 + (1-b2_)*dW1**2
        W1 -= lr*(mW1/(1-b1_**epoch)) / (np.sqrt(vW1/(1-b2_**epoch))+eps)
        # Adam for W2
        mW2 = b1_*mW2 + (1-b1_)*dW2
        vW2 = b2_*vW2 + (1-b2_)*dW2**2
        W2 -= lr*(mW2/(1-b1_**epoch)) / (np.sqrt(vW2/(1-b2_**epoch))+eps)
        if epoch%500==0: print(f"Adam epoch {epoch:4d} loss {l:.4f}")

print("=== SGD ===");   train_sgd()
print("\n=== Adam ==="); train_adam()
```

---

## Key Terms

| Term | Simple meaning |
|------|---------------|
| Loss surface | Landscape of all loss values across all weight combinations |
| Gradient | Slope — direction loss increases (we go opposite) |
| Learning rate | Step size. Too high = overshoot. Too low = too slow. |
| Momentum | Rolling ball effect — builds speed, avoids getting stuck |
| Adaptive LR | Different step size per weight (RMSProp, Adam) |
| Adam | Momentum + adaptive LR. Use by default. |
| Mini-batch | Small subset of data per update step |
| Epoch | One full pass through all training data |

---

## Exercise
1. Run `train_sgd()` and `train_adam()` — compare loss at epoch 500
2. Set `lr=10.0` in SGD — watch loss explode to infinity
3. Set `lr=0.0001` in SGD — watch it crawl
4. Try Adam with `lr=0.001` vs `lr=0.1` — notice Adam is more forgiving

---

## What to Remember Forever
> Gradient descent = step downhill on the loss surface.
> Learning rate controls step size.
> Adam = the default optimizer — combines momentum + per-weight adaptation.
> When in doubt, use Adam with lr=0.001.
