# Lesson: scikit-learn — The ML Swiss Army Knife

**Date taught:** 2026-09-26
**Folder:** ml-core/
**Status:** ✅ Taught

---

## Analogy

Building a neural network from scratch every time is like building a car engine from raw metal every time you want to drive.

**scikit-learn is a garage full of ready-made vehicles:**
- Need to classify emails? There's a model for that.
- Need to scale your data? There's a tool for that.
- Need to tune hyperparameters? There's a tool for that.
- Need to build a pipeline? There's a tool for that.

One consistent interface for ALL of it:
```
model.fit(X_train, y_train)    ← train
model.predict(X_test)          ← predict
model.score(X_test, y_test)    ← evaluate
```
Same three methods. Every single model. That's scikit-learn's superpower.

---

## Diagram: scikit-learn Ecosystem

```
                         scikit-learn
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ESTIMATORS           TRANSFORMERS          UTILITIES
    (models)             (preprocessing)       (evaluation)
          │                   │                   │
   Classification      StandardScaler      train_test_split
   ├── Logistic         MinMaxScaler        cross_val_score
   ├── RandomForest      LabelEncoder        GridSearchCV
   ├── SVM               OneHotEncoder       Pipeline
   └── KNN               PCA                 metrics.*

   Regression
   ├── LinearRegression
   ├── Ridge/Lasso
   └── RandomForest

   Clustering
   ├── KMeans
   └── DBSCAN

Consistent API:
  fit() → predict() → score()      (estimators)
  fit() → transform()              (transformers)
  fit_transform()                   (shortcut for both)
```

---

## Part 1: Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# ── Load data ─────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target             # (569, 30) features, binary labels
print(f"Classes: {data.target_names}")    # ['malignant', 'benign']

# ── Split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale (always for Logistic Regression, SVM) ───────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit on train only!
X_test_s  = scaler.transform(X_test)

# ── Try 3 models ──────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                  SVC(kernel='rbf', probability=True),
}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    acc = model.score(X_test_s, y_test)
    print(f"{name}: {acc:.3f}")

# ── Detailed evaluation ───────────────────────────────
best = LogisticRegression(max_iter=1000)
best.fit(X_train_s, y_train)
y_pred = best.predict(X_test_s)
y_prob = best.predict_proba(X_test_s)[:, 1]   # probability of class 1

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=data.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

## Part 2: Regression

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2 reg)":    Ridge(alpha=1.0),
    "Lasso (L1 reg)":    Lasso(alpha=0.1),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)   # 1.0 = perfect, 0 = as good as predicting mean
    print(f"{name}: MSE={mse:.3f}, R²={r2:.3f}")
```

---

## Part 3: Preprocessing Tools

```python
from sklearn.preprocessing import (
    StandardScaler,      # mean=0, std=1  (most common)
    MinMaxScaler,        # scale to [0, 1]
    LabelEncoder,        # cat → int: ['cat','dog','bird'] → [0,2,1]
    OneHotEncoder,       # cat → binary: ['cat'] → [1,0,0]
)
from sklearn.impute import SimpleImputer
import numpy as np

# ── StandardScaler ────────────────────────────────────
X = np.array([[100, 0.1], [200, 0.5], [150, 0.3]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Mean:", scaler.mean_)     # [150, 0.3]
print("Std:",  scaler.scale_)    # [40.8, 0.16]

# ── LabelEncoder (for target labels) ──────────────────
le = LabelEncoder()
labels = ['cat', 'dog', 'bird', 'cat', 'dog']
encoded = le.fit_transform(labels)   # [1, 2, 0, 1, 2]
decoded = le.inverse_transform([0])  # ['bird']

# ── OneHotEncoder (for input features) ────────────────
enc = OneHotEncoder(sparse_output=False)
dept = np.array([['Eng'], ['Mkt'], ['Sales'], ['Eng']])
encoded = enc.fit_transform(dept)
# [[1,0,0], [0,1,0], [0,0,1], [1,0,0]]

# ── SimpleImputer (handle missing values) ─────────────
imputer = SimpleImputer(strategy='mean')   # fill NaN with column mean
X_with_nan = np.array([[1,2],[np.nan,3],[5,np.nan]])
X_filled = imputer.fit_transform(X_with_nan)
```

---

## Part 4: Pipeline — Chain Everything Together

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Pipelines prevent data leakage automatically:
# fit_transform on train, transform on test

# ── Simple pipeline ───────────────────────────────────
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),    # step 1: fill missing
    ('scaler',  StandardScaler()),                   # step 2: normalize
    ('model',   LogisticRegression())                # step 3: train
])

pipe.fit(X_train, y_train)      # applies all steps in order
pipe.predict(X_test)            # applies same transform steps, then predicts
pipe.score(X_test, y_test)      # end-to-end accuracy

# ── Column transformer (different processing per column) ─
numeric_features     = ['age', 'salary']
categorical_features = ['department']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(),             numeric_features),
    ('cat', OneHotEncoder(drop='first'),  categorical_features),
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])
```

---

## Part 5: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# ── Grid Search (try every combination) ───────────────
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 5, 10],
    'min_samples_split': [2, 5],
}

gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,               # 5-fold CV
    scoring='accuracy',
    n_jobs=-1,          # use all CPU cores
    verbose=1
)
gs.fit(X_train_s, y_train)

print(f"Best params: {gs.best_params_}")
print(f"Best CV score: {gs.best_score_:.3f}")
print(f"Test score: {gs.best_estimator_.score(X_test_s, y_test):.3f}")

# ── Random Search (sample random combinations — faster) ─
from scipy.stats import randint
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth':    randint(3, 20),
}
rs = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=20,   # try 20 random combinations instead of all
    cv=5
)
rs.fit(X_train_s, y_train)
```

---

## Part 6: Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score,           # % correct predictions
    precision_score,          # of predicted positives, how many are real?
    recall_score,             # of real positives, how many did we catch?
    f1_score,                 # harmonic mean of precision and recall
    roc_auc_score,            # area under ROC curve (best single metric)
    classification_report,    # all of the above in one table
    confusion_matrix,
    mean_squared_error,
    r2_score,
)

# ── Classification metrics ────────────────────────────
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1:        {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")

# When to use which:
# Accuracy    → balanced classes
# Precision   → cost of false positive is high (spam filter)
# Recall      → cost of false negative is high (cancer detection)
# F1          → imbalanced classes
# ROC-AUC     → best overall metric, threshold-independent
```

---

## Key Terms

| Term | Meaning |
|------|---------|
| Estimator | Any model: `fit()` + `predict()` |
| Transformer | Preprocessing tool: `fit()` + `transform()` |
| Pipeline | Chain of transformers + estimator |
| Cross-validation | Evaluate model on multiple train/val splits |
| GridSearchCV | Try all hyperparameter combinations to find best |
| fit_transform | `fit()` then `transform()` in one step (train only!) |
| R² score | How well regression model fits (1=perfect, 0=baseline) |
| ROC-AUC | Classification performance across all thresholds |

---

## Exercise
1. Load `load_iris()` — 3-class classification. Train Logistic Regression, print `classification_report`
2. Build a Pipeline: `SimpleImputer → StandardScaler → RandomForest`. Score on test set.
3. Run `GridSearchCV` on `SVC` with `C=[0.1, 1, 10]` and `kernel=['rbf','linear']`. What's best?
4. When would you choose Recall over Precision? Give a real example.

---

## What to Remember Forever
> Every sklearn model: `fit(X_train, y_train)` → `predict(X_test)` → `score(X_test, y_test)`
> Always use Pipeline — it prevents data leakage automatically.
> Use F1 / ROC-AUC for imbalanced classes — accuracy will lie to you.
> GridSearchCV finds best hyperparameters automatically via cross-validation.
