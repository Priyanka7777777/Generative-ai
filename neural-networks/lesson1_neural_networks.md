# Lesson 1: What is a Neural Network?

**Date taught:** 2026-09-26
**Status:** ✅ Taught | Exercise given
**Next:** [lesson2_optimizers.md](lesson2_optimizers.md)

---

## The Analogy

Imagine training a **new bakery employee** to judge if a cake is good or bad.

They look at 3 things: **color**, **texture**, **smell**
They assign importance to each (weights), make a decision, get corrected, improve over time.

That employee = a neural network.

- 3 things they look at = **inputs**
- Their importance scores = **weights**
- "Good / Bad" decision = **output**
- You correcting them = **training**
- Getting better over time = **learning**

Multiple layers of employees (junior → senior → manager) = **deep** neural network.

---

## Technical Explanation

### Architecture
```
INPUT LAYER → HIDDEN LAYER(s) → OUTPUT LAYER
```
Each "employee" = a **neuron**.
A group of neurons = a **layer**.
Connections between neurons carry **weights**.

### How One Neuron Works
```
Step 1: Receive inputs (numbers)
Step 2: Multiply each by a weight, add a bias
Step 3: Pass through activation function

output = activation( w1*x1 + w2*x2 + w3*x3 + bias )
```

### Activation Function — ReLU
```
ReLU(x) = max(0, x)
```
- Negative input → output 0
- Positive input → output as-is
- Why needed: adds non-linearity. Without it, deep networks = useless (all layers collapse to one line)

### Training — 4 Steps (Repeated Thousands of Times)
```
1. FORWARD PASS  → Feed data through, get prediction
2. LOSS          → Measure how wrong prediction is
3. BACKPROP      → Trace error back to find which weights caused it
4. UPDATE        → Nudge weights to reduce error (gradient descent)
```

### Loss (MSE)
```python
loss = mean((prediction - truth) ** 2)
```
"You were 8 points off" → loss = 8 → goal: make this 0.

### Backpropagation
Traces the error backwards through each layer using the **chain rule** of calculus.
Computes a gradient for each weight — the direction and amount to change it.
Analogy: "Smell weight was too high, texture weight was too low — adjust those."

### Gradient Descent
```python
weight = weight - learning_rate * gradient
```
- `learning_rate`: how big a step to take. Too high = overshoot. Too low = too slow.

---

## Code — Neural Network from Scratch (NumPy only)

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Weights — random init (network starts dumb)
np.random.seed(42)
W1 = np.random.randn(2, 3) * 0.1   # input(2) → hidden(3)
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.1   # hidden(3) → output(1)
b2 = np.zeros((1, 1))

def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return z2, a1, z1

def loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def backward(X, y_true, output, a1, z1, lr=0.01):
    global W1, b1, W2, b2
    m = X.shape[0]
    dL_dout = 2 * (output - y_true) / m
    dW2 = a1.T @ dL_dout
    db2 = np.sum(dL_dout, axis=0, keepdims=True)
    dL_da1 = dL_dout @ W2.T
    dL_dz1 = dL_da1 * relu_derivative(z1)
    dW1 = X.T @ dL_dz1
    db1 = np.sum(dL_dz1, axis=0, keepdims=True)
    W1 -= lr * dW1;  b1 -= lr * db1
    W2 -= lr * dW2;  b2 -= lr * db2

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

for epoch in range(1000):
    output, a1, z1 = forward(X)
    l = loss(output, y)
    backward(X, y, output, a1, z1, lr=0.1)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {l:.4f}")

print("\nPredictions:", np.round(forward(X)[0], 2))
```

---

## Key Terms

| Term | Simple Meaning |
|------|---------------|
| Neuron | One unit: takes inputs, weighs them, fires output |
| Layer | Group of neurons |
| Weight | How important an input is (learned during training) |
| Bias | A nudge that shifts the decision boundary |
| Activation (ReLU) | Adds non-linearity — lets network learn complex patterns |
| Loss | How wrong the prediction is (want this → 0) |
| Backprop | Traces error back to find which weights to fix |
| Gradient | Direction + amount to change a weight |
| Learning rate | Step size when updating weights |
| Epoch | One full pass through all training data |

---

## Exercise
1. Run the code — `pip install numpy` if needed
2. Change learning rate `0.1` → `0.001` — what happens to loss?
3. Change `1000` epochs → `100` — does it still learn well?
4. Add `print(a1)` inside `forward()` to see hidden layer activations

---

## What to Remember Forever
> A neural network is just: inputs × weights + bias → activation → repeat through layers → output → measure error → backprop → update weights → repeat.
