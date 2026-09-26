# Lesson: PyTorch — The Deep Learning Framework

**Date taught:** 2026-09-26
**Folder:** pytorch/
**Status:** ✅ Taught

---

## Analogy

Remember our NumPy neural network from Lesson 1? We wrote backprop by hand — 50 lines of math just to train a tiny network.

**PyTorch is a NumPy that**:
1. **Tracks every operation automatically** — so it can compute gradients for you (no manual backprop!)
2. **Runs on GPU** — 100× faster for large models
3. **Has pre-built layers** — you describe the architecture, it handles the math

Think of it like this:
- NumPy = a calculator (you do all the work)
- PyTorch = a smart calculator that remembers every button you pressed and can undo them to compute gradients automatically

> PyTorch = NumPy + automatic differentiation + GPU + pre-built neural network layers

---

## Diagram

```
NumPy (manual)                    PyTorch (automatic)
─────────────────                 ──────────────────────────────
a = np.array([2.0])               a = torch.tensor([2.0], requires_grad=True)
b = a * 3                         b = a * 3
                                       ↑
                                  PyTorch builds a computation graph:
                                  a ──×3──→ b
                                  
To get gradient:                  b.backward()   ← one line!
  manual math: db/da = 3          print(a.grad)  → tensor([3.])

Neural Network Training:
  Input → [Layer1] → [Layer2] → Output → Loss
            ↑           ↑                  |
            └───────────┴──────────────────┘
                    backward() fills all gradients automatically
```

---

## Part 1: Tensors — PyTorch's NumPy Arrays

```python
import torch
import numpy as np

# ── Creating tensors ──────────────────────────────────
t1 = torch.tensor([1.0, 2.0, 3.0])         # from list
t2 = torch.zeros(3, 4)                      # all zeros
t3 = torch.ones(2, 3)                       # all ones
t4 = torch.randn(3, 4)                      # random normal
t5 = torch.arange(0, 10, step=2)            # [0,2,4,6,8]

# ── From/to NumPy ─────────────────────────────────────
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)                   # NumPy → PyTorch
arr_back = t.numpy()                        # PyTorch → NumPy

# ── Shape operations ──────────────────────────────────
t = torch.randn(2, 3)
print(t.shape)          # torch.Size([2, 3])
print(t.dtype)          # torch.float32

t.reshape(3, 2)         # reshape
t.T                     # transpose
t.squeeze()             # remove dims of size 1
t.unsqueeze(0)          # add a dim at position 0: (2,3) → (1,2,3)

# ── GPU (use if available) ────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

t = torch.randn(3, 4).to(device)           # move to GPU
t_cpu = t.cpu()                             # move back to CPU

# ── Autograd — automatic differentiation ─────────────
x = torch.tensor(3.0, requires_grad=True)  # track this tensor
y = x ** 2 + 2 * x + 1                     # y = x² + 2x + 1

y.backward()                                # compute dy/dx automatically
print(x.grad)                               # dy/dx = 2x + 2 = 8.0  ✓
```

---

## Part 2: Building a Neural Network with nn.Module

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ── Define the network ────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)   # fully connected layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define how data flows through layers
        x = self.fc1(x)       # linear: x @ W1 + b1
        x = self.relu(x)      # activation
        x = self.fc2(x)       # linear: x @ W2 + b2
        x = self.sigmoid(x)   # output probability
        return x

# Create model
model = SimpleNet(input_size=2, hidden_size=4, output_size=1)
print(model)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total}")
```

---

## Part 3: The Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Data: XOR problem ─────────────────────────────────
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

dataset    = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ── Model, loss, optimizer ────────────────────────────
model     = SimpleNet(2, 8, 1)
criterion = nn.BCELoss()                    # binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ── Training loop ─────────────────────────────────────
for epoch in range(500):
    model.train()                           # training mode

    for X_batch, y_batch in dataloader:
        # 1. Forward pass
        predictions = model(X_batch)

        # 2. Compute loss
        loss = criterion(predictions, y_batch)

        # 3. Zero gradients (must do before backward!)
        optimizer.zero_grad()

        # 4. Backward pass (autograd computes all gradients)
        loss.backward()

        # 5. Update weights
        optimizer.step()

    if epoch % 100 == 0:
        model.eval()                        # eval mode (disables dropout etc.)
        with torch.no_grad():               # don't track gradients for eval
            preds = model(X)
            l = criterion(preds, y)
        print(f"Epoch {epoch:4d} | Loss: {l.item():.4f}")

# ── Evaluate ──────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds = model(X)
    predicted_classes = (preds > 0.5).float()
    accuracy = (predicted_classes == y).float().mean()
    print(f"\nAccuracy: {accuracy.item():.0%}")
    print(f"Predictions:\n{preds.numpy().round(2)}")
```

---

## Part 4: Common Layers & Loss Functions

```python
# ── Layers ────────────────────────────────────────────
nn.Linear(in, out)          # fully connected / dense layer
nn.ReLU()                   # activation
nn.Sigmoid()                # binary output (0-1)
nn.Softmax(dim=1)           # multi-class output (probabilities)
nn.Dropout(p=0.5)           # regularization (drop 50% of neurons randomly)
nn.BatchNorm1d(features)    # normalize within batch
nn.Embedding(vocab, dim)    # word embeddings (used in NLP/LLMs)
nn.LSTM(input, hidden)      # recurrent layer (sequence data)

# ── Loss functions ─────────────────────────────────────
nn.MSELoss()                # regression
nn.BCELoss()                # binary classification (after sigmoid)
nn.CrossEntropyLoss()       # multi-class (applies softmax internally)

# ── Optimizers ────────────────────────────────────────
optim.SGD(model.parameters(), lr=0.01)
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optim.Adam(model.parameters(), lr=0.001)      # default choice
optim.AdamW(model.parameters(), lr=0.001)     # Adam + weight decay (preferred for LLMs)
```

---

## Part 5: Save & Load Models

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = SimpleNet(2, 8, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

## Diagram: PyTorch Training Loop

```
for each epoch:
    for each batch:
        │
        ├── 1. forward(X_batch)      → predictions
        │
        ├── 2. loss = criterion(predictions, y_batch)
        │
        ├── 3. optimizer.zero_grad() → clear old gradients
        │
        ├── 4. loss.backward()       → compute all gradients (autograd)
        │
        └── 5. optimizer.step()      → update all weights

eval mode:
    with torch.no_grad():            → don't compute gradients (faster)
        predictions = model(X_test)
```

---

## NumPy vs PyTorch — Quick Comparison

```
┌────────────────────┬──────────────────┬─────────────────────────────┐
│ Task               │ NumPy            │ PyTorch                     │
├────────────────────┼──────────────────┼─────────────────────────────┤
│ Create array       │ np.array(...)    │ torch.tensor(...)           │
│ Random normal      │ np.random.randn  │ torch.randn(...)            │
│ Matrix multiply    │ A @ B            │ A @ B  (same!)              │
│ Reshape            │ a.reshape(...)   │ a.reshape(...)  (same!)     │
│ Gradient           │ manual math      │ loss.backward() ← automatic │
│ GPU support        │ ❌               │ .to('cuda')  ✅             │
│ Neural net layers  │ manual code      │ nn.Linear, nn.ReLU etc. ✅  │
└────────────────────┴──────────────────┴─────────────────────────────┘
```

---

## Key Terms

| Term | Meaning |
|------|---------|
| Tensor | PyTorch's array — like NumPy but with autograd + GPU |
| requires_grad | Tell PyTorch to track this tensor for gradient computation |
| backward() | Compute all gradients in the computation graph |
| nn.Module | Base class for all neural networks in PyTorch |
| forward() | Define how data flows through your model |
| DataLoader | Feeds data in batches, shuffles, handles multiprocessing |
| zero_grad() | Clear gradients before each backward pass |
| model.train() | Enable training mode (activates dropout, batchnorm) |
| model.eval() | Evaluation mode (disables dropout, batchnorm) |
| no_grad() | Context manager: skip gradient tracking (faster inference) |
| state_dict | Dictionary of model weights — used for saving/loading |

---

## Exercise
1. Create a `SimpleNet(10, 32, 1)` — print it and count its parameters
2. Train it on random data: `X = torch.randn(100, 10)`, `y = torch.randint(0,2,(100,1)).float()`
3. Save the model to `model.pth`, reload it, verify predictions are identical
4. Move model to GPU if available: `model = model.to(device)`

---

## What to Remember Forever
> PyTorch = NumPy tensors + automatic gradients + GPU + pre-built layers.
> Training loop is always: forward → loss → zero_grad → backward → step.
> Never forget zero_grad() — gradients accumulate by default and will corrupt training.
