# Week 1-2 Complete Teaching Summary

**Date:** 2026-09-26
**Status:** ✅ All topics taught in chat session

All topics below were fully taught with analogy → diagram → technical → code.

---

## What Was Covered

### Python Tools (python-ml-basics/)
| Topic | File | Key takeaway |
|-------|------|-------------|
| NumPy | lesson1_numpy.md | `@` = matrix multiply. Shape is everything. |
| Pandas | lesson2_pandas.md | Load → clean → encode → `.values` → model |
| Matplotlib | lesson3_matplotlib.md | Training curve = heartbeat monitor of your model |
| async/await | lesson4_async_await.md | `asyncio.gather()` = parallel LLM calls |
| Pydantic | lesson5_pydantic.md | Validate LLM outputs and tool inputs |

### Math Foundations (math-foundations/)
| Topic | File | Key takeaway |
|-------|------|-------------|
| Linear Algebra | lesson1_linear_algebra.md | `A @ B`: inner dims must match. Cosine sim = RAG search. |
| Probability & Stats | lesson2_probability_statistics.md | Normalize inputs. Softmax → probs. Cross-entropy → loss. |

### ML Core (ml-core/)
| Topic | File | Key takeaway |
|-------|------|-------------|
| Supervised vs Unsupervised | lesson1_supervised_unsupervised.md | LLMs = self-supervised |
| Train/Test Split | lesson2_train_test_split.md | Split first, normalize after. Never peek at test set. |
| Loss Functions | lesson3_loss_functions.md | Regression=MSE, Binary=BCE, Multi-class=CCE |
| scikit-learn | lesson4_sklearn.md | fit() → predict() → score(). Always use Pipeline. |

### Neural Networks (neural-networks/, pytorch/)
| Topic | File | Key takeaway |
|-------|------|-------------|
| Neural Networks | lesson1_neural_networks.md | Forward pass → loss → backprop → update |
| Optimizers | lesson2_optimizers.md | Adam with lr=0.001 is the default |
| PyTorch | pytorch/lesson1_pytorch.md | zero_grad() → backward() → step(). Every time. |

---

## The Golden Rules Across All Topics

```
NumPy    → check .shape when anything breaks
Pandas   → always .head(), .isnull().sum() on new data
Plots    → val loss going up while train goes down = overfitting
async    → asyncio.gather() for parallel LLM calls
Pydantic → validate at every AI system boundary
LinAlg   → (m×n)@(n×p)=(m×p). Cosine similarity for search.
Stats    → normalize inputs. Softmax for output probs.
ML Core  → split first, then normalize. Never leak test data.
Loss     → MSE=regression, BCE=binary, CCE=multi-class
NN       → forward → loss → zero_grad → backward → step
```
