# Lesson: Linear Algebra for ML

**Date taught:** 2026-09-26
**Folder:** math-foundations/
**Status:** ✅ Taught

---

## Analogy

Forget the textbook. Think of it this way:

**A vector** = a list of numbers that describes a point in space.
A person described by [age=32, salary=75000, experience=5] is a **vector** in 3D space.

**A matrix** = a table of numbers — a stack of vectors.
A dataset of 1000 people = a matrix of shape (1000, 3).

**Matrix multiplication** = a transformation — rotating, scaling, squishing that space.
In a neural network, every layer applies a transformation to your data using a weight matrix.

> All of deep learning is: data (vectors) → transformations (matrix multiplications) → output (vectors).

---

## Diagrams

### Vector
```
2D vector [3, 4]:
    y
    ↑
  4 │         ● (3, 4)
    │        /
    │       /  ← vector (magnitude = 5)
    │      /
    └────────→ x
         3
```

### Matrix Multiplication
```
A (2×3)  @  B (3×4)  =  C (2×4)

Think: (rows of A) × (cols of B) → element of C

     col →            col →           col →
row  [1 2 3]      [a b c d]      [row·col  ...]
↓    [4 5 6]  @   [e f g h]  =   [...]
               [i j k l]

Rule: A(m×n) @ B(n×p) = C(m×p)
              ↑ ↑
         must match!
```

### In a Neural Network
```
Input X:     shape (batch=32, features=784)
Weights W:   shape (features=784, neurons=128)
Output Z:    shape (batch=32, neurons=128)

Z = X @ W + b

Every "layer" is one matrix multiplication!
```

---

## Technical Explanation

### Vectors — In Depth
A vector is just an ordered list of numbers, but the key insight is that **each number represents one dimension** of the thing you're describing.

- A 2D vector `[3, 4]` = a point (or arrow) in 2D space
- A person described by `[age=32, salary=75000, exp=5]` = a point in 3D space
- A word embedding (like in an LLM) is a vector with 1536 numbers — it lives in 1536-dimensional space

**Why do we care about vectors?** Because similar things have similar vectors. Two sentences with the same meaning will have vectors that point in nearly the same direction. This is why RAG search works.

**The Dot Product** is the most important vector operation:
```
a · b = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]
```
- If a and b point in the **same direction** → dot product is large and positive
- If a and b are **perpendicular** → dot product is exactly 0
- If a and b point in **opposite directions** → dot product is negative

In neural networks, every neuron computes a dot product: it measures "how much does this input match my weights?"

**Magnitude (length)** of a vector: `‖v‖ = √(v[0]² + v[1]² + ... + v[n]²)`
This is just the Pythagorean theorem extended to N dimensions.

**Cosine Similarity** = dot product normalised by lengths:
```
cos_sim(a, b) = (a · b) / (‖a‖ × ‖b‖)
```
Output is always between -1 and 1:
- 1.0 = identical direction (same meaning)
- 0.0 = perpendicular (unrelated)
- -1.0 = opposite direction (opposite meaning)

This is used in **every vector database** — when you query a RAG system, it computes cosine similarity between your query embedding and all stored document embeddings to find the most relevant ones.

---

### Matrices — In Depth
A matrix is a 2D table of numbers — a stack of row vectors.

**Key insight for ML**: A matrix is a **transformation**. When you multiply a vector by a matrix, you're moving that vector to a new position in space — rotating it, scaling it, projecting it.

- Every layer in a neural network applies one matrix transformation: `output = input @ W + b`
- The model **learns** the matrix W during training
- Multiple layers = multiple transformations applied in sequence

**Matrix Multiplication** (`A @ B`):
- A has shape `(m, n)` — m rows, n columns
- B has shape `(n, p)` — n rows, p columns
- Result has shape `(m, p)`
- **The inner dimensions must match**: `(m, n) @ (n, p)` ✅ — the `n` must be the same
- Each element of the result is a dot product of one row of A with one column of B

**Transpose** (`A.T`):
- Flips rows and columns: shape `(m, n)` becomes `(n, m)`
- Used constantly in backprop: when you compute gradients, you need `W.T` to propagate errors backwards through a layer

**Identity Matrix**: A square matrix with 1s on the diagonal. `A @ I = A` — multiplying by I does nothing, like multiplying by 1.

**Inverse** (`A⁻¹`): The matrix that "undoes" a transformation. `A @ A⁻¹ = I`. Used in the closed-form solution to linear regression: `w = (XᵀX)⁻¹ Xᵀ y`.

**Eigenvalues and Eigenvectors** — used in PCA:
- If `A @ v = λ × v`, then v is an eigenvector and λ is its eigenvalue
- Eigenvectors are the "special directions" of a matrix — they don't rotate, they just scale
- PCA finds the eigenvectors of the data's covariance matrix — they are the directions of maximum variance in the data
- The largest eigenvalue = the direction where data is most spread out = the first principal component

---

## Core Concepts + Code

```python
import numpy as np

# ── Vectors ───────────────────────────────────────────
v1 = np.array([3, 4])        # 2D vector
v2 = np.array([1, 2])

# Magnitude (length of vector)
magnitude = np.linalg.norm(v1)    # √(3²+4²) = 5.0

# Dot product (how similar are two vectors?)
dot = np.dot(v1, v2)              # 3×1 + 4×2 = 11
# Positive = pointing same direction
# Zero     = perpendicular (no similarity)
# Negative = pointing opposite directions

# ── Matrices ──────────────────────────────────────────
A = np.array([[1, 2, 3],
              [4, 5, 6]])          # shape (2,3)

# Transpose (flip rows and cols)
A.T                                # shape (3,2)

# Matrix multiplication
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])             # shape (3,2)

C = A @ B                          # (2,3) @ (3,2) = (2,2)
print(C)

# ── Identity Matrix ───────────────────────────────────
I = np.eye(3)     # [[1,0,0],[0,1,0],[0,0,1]]
# A @ I = A  (like multiplying by 1)

# ── Inverse (used in linear regression) ───────────────
M = np.array([[2, 1], [1, 3]])
M_inv = np.linalg.inv(M)
print(M @ M_inv)   # should be identity matrix

# ── Eigenvalues (used in PCA) ─────────────────────────
vals, vecs = np.linalg.eig(M)
# vals = eigenvalues (how much stretch in each direction)
# vecs = eigenvectors (directions of stretch)

# ── Cosine Similarity (used in embeddings & RAG) ──────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Two similar sentences should have high cosine similarity (close to 1)
embedding1 = np.array([0.8, 0.2, 0.5, 0.9])   # "cat runs fast"
embedding2 = np.array([0.7, 0.3, 0.4, 0.8])   # "the cat sprints"
embedding3 = np.array([0.1, 0.9, 0.2, 0.1])   # "quantum physics"

print(cosine_similarity(embedding1, embedding2))   # ~0.99 (similar!)
print(cosine_similarity(embedding1, embedding3))   # ~0.3  (different)

# ── Norms ─────────────────────────────────────────────
x = np.array([3.0, 4.0])
np.linalg.norm(x)         # L2 norm = 5.0  (Euclidean distance)
np.linalg.norm(x, ord=1)  # L1 norm = 7.0  (Manhattan distance)
```

---

## Most Important Concepts for ML

| Concept | Symbol | ML usage |
|---------|--------|----------|
| Dot product | `a · b` | Similarity, attention scores |
| Matrix multiply | `A @ B` | Every layer's forward pass |
| Transpose | `A.T` | Backprop gradient calculation |
| Cosine similarity | `a·b / ‖a‖‖b‖` | Embedding search (RAG) |
| L2 norm | `‖x‖` | Regularization, normalization |
| Inverse | `A⁻¹` | Linear regression closed form |
| Eigenvalues | `Av = λv` | PCA, understanding transformations |

---

## Shapes Rule (Memorize This)

```
(m × n) @ (n × p) = (m × p)
         ↑↑
    inner dims must match

Examples:
(32, 784) @ (784, 128)  = (32, 128)  ✅  batch × features forward pass
(128, 32) @ (32, 10)    = (128, 10)  ✅  layer output
(784, 128) @ (32, 784)  = ERROR      ❌  inner dims don't match → transpose!
```

---

## Exercise
1. Create vectors `a=[1,2,3]` and `b=[4,5,6]`. Compute dot product manually, then verify with `np.dot(a,b)`
2. Create a (3,4) matrix. Multiply by a (4,2) matrix. What's the output shape?
3. Compute cosine similarity between `[1,0,0]` and `[0,1,0]` — what do you expect?
4. Why does `(32,128) @ (784,128)` fail? How do you fix it?

---

## What to Remember Forever
> Matrix multiplication is the entire forward pass of a neural network.
> Cosine similarity is how vector databases find similar items.
> Always check shapes — `(m×n) @ (n×p) = (m×p)`, inner dims must match.
