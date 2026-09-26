# Lesson: Pandas — Loading and Preparing Data

**Date taught:** 2026-09-26
**Folder:** python-ml-basics/
**Status:** ✅ Taught

---

## Analogy

Pandas is **Excel in Python** — but programmable, scalable to millions of rows, and designed for data cleaning.

- A **DataFrame** = a spreadsheet table (rows = samples, columns = features)
- A **Series** = one column of that table

In ML, you almost always start with raw data in a CSV file. Pandas is how you load it, clean it, explore it, and prepare it before training.

```
Raw CSV file
    ↓
pandas.read_csv()
    ↓
DataFrame (table in memory)
    ↓
Clean → Filter → Transform
    ↓
NumPy arrays → feed to model
```

---

## Diagram

```
DataFrame structure:

        age   salary   department   churned
row 0:   32    75000    Engineering    0
row 1:   45    92000    Marketing      1
row 2:   28    58000    Engineering    0
row 3:   51   110000    Sales          1
         ↑
      Series (one column)

Rows = samples (one person each)
Cols = features (what you know about them)
Last col = label (what you want to predict)
```

---

## Technical Explanation — How Pandas Works

### DataFrame and Series Under the Hood
A **DataFrame** is essentially a dictionary of **Series** objects, where each key is a column name and each value is a column of data.

A **Series** is a one-dimensional array with a labelled index. Think of it as a NumPy array with named rows.

```
DataFrame = {
    'age':    Series([32, 45, 28, 51]),
    'salary': Series([75000, 92000, 58000, 110000]),
}
```

Each Series has a **dtype** (data type) — `int64`, `float64`, `object` (string), `bool`, `datetime64`. Pandas operations are fast because each column is a contiguous block of the same dtype in memory — just like a NumPy array.

### Indexing — How Pandas Selects Data

Pandas has two indexing systems:
- **`.iloc[]`** — **integer-based** (position): `df.iloc[0]` = first row by position number
- **`.loc[]`** — **label-based** (condition/name): `df.loc[df['age'] > 40]` = filter by condition

This is a source of confusion for beginners — always ask "do I want by position or by condition?"

### Why Pandas Outperforms Plain Python Loops

When you write `df['salary'] * 1.1`, Pandas applies the multiplication to the entire column in one C-level operation — no Python loop. This is called **vectorisation**, same principle as NumPy.

For 1 million rows:
- Python loop: ~0.5 seconds
- Pandas vectorised: ~0.002 seconds (250× faster)

### Missing Data (NaN) — Why It Matters for ML

Real-world datasets almost always have missing values. If you feed `NaN` values into a model, it will produce `NaN` predictions — the entire forward pass becomes NaN and the model breaks.

Strategies:
- **Drop rows**: `df.dropna()` — safe when you have lots of data
- **Fill with mean**: `df.fillna(df.mean())` — most common for numerical columns
- **Fill with mode**: `df.fillna(df.mode().iloc[0])` — for categorical columns
- **Forward fill**: `df.ffill()` — for time series data (fill with previous value)

### Categorical Encoding — Why ML Needs Numbers

Neural networks and most ML algorithms only work with numbers. Text categories like `['Eng', 'Mkt', 'Sales']` must be converted.

**Label encoding** (`[0, 1, 2]`) implies an ordering — `Sales > Mkt > Eng` numerically — which is wrong for most categories. Only use it for ordinal data (like `['low', 'medium', 'high']`).

**One-hot encoding** creates a separate binary column for each category — no false ordering implied. Use this for nominal categories (no natural order). The downside: if a column has 100 unique values, you get 100 new columns (high cardinality problem).

### `groupby` — The SQL GROUP BY Equivalent

```python
df.groupby('department')['salary'].mean()
```
This is equivalent to SQL: `SELECT department, AVG(salary) FROM df GROUP BY department`

Under the hood: split the DataFrame by unique values in 'department', apply `.mean()` to each group, combine results.

---

## Code — Everything You Need

```python
import pandas as pd
import numpy as np

# ── Creating a DataFrame ──────────────────────────────
df = pd.DataFrame({
    'age':        [32, 45, 28, 51, 36],
    'salary':     [75000, 92000, 58000, 110000, 68000],
    'department': ['Eng', 'Mkt', 'Eng', 'Sales', 'Mkt'],
    'churned':    [0, 1, 0, 1, 0]
})

# ── Loading from file ─────────────────────────────────
df = pd.read_csv('data.csv')         # most common
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')

# ── Exploring the data (always do this first!) ────────
df.head()          # first 5 rows
df.tail()          # last 5 rows
df.shape           # (rows, cols) e.g. (1000, 12)
df.columns         # column names
df.dtypes          # data types per column
df.describe()      # count, mean, std, min, max per column
df.info()          # non-null counts, dtypes, memory

# ── Selecting data ────────────────────────────────────
df['age']                    # one column → Series
df[['age', 'salary']]        # multiple columns → DataFrame
df.iloc[0]                   # row by index number
df.iloc[0:3]                 # rows 0,1,2
df.loc[df['age'] > 40]       # filter rows by condition

# ── Common filters ────────────────────────────────────
df[df['churned'] == 1]                         # churned employees only
df[df['salary'].between(60000, 90000)]         # salary range
df[(df['age'] > 30) & (df['churned'] == 0)]   # multiple conditions

# ── Missing data (critical for ML) ───────────────────
df.isnull().sum()             # count NaN per column
df.dropna()                   # drop rows with any NaN
df.fillna(0)                  # fill NaN with 0
df['age'].fillna(df['age'].mean())  # fill with column mean

# ── Adding / transforming columns ─────────────────────
df['salary_k'] = df['salary'] / 1000        # new column
df['senior']   = (df['age'] > 40).astype(int)  # boolean → 0/1

# ── Groupby (like pivot tables) ───────────────────────
df.groupby('department')['salary'].mean()    # avg salary per dept
df.groupby('churned').size()                 # count per class

# ── Encoding categorical columns (ML needs numbers) ───
# Option 1: Label encoding (0, 1, 2...)
df['dept_code'] = df['department'].astype('category').cat.codes

# Option 2: One-hot encoding (new binary column per category)
df = pd.get_dummies(df, columns=['department'])
# creates: department_Eng, department_Mkt, department_Sales

# ── Sorting ───────────────────────────────────────────
df.sort_values('salary', ascending=False)

# ── Dropping columns ──────────────────────────────────
df.drop(columns=['salary_k'])

# ── Renaming columns ──────────────────────────────────
df.rename(columns={'churned': 'label'})

# ── Preparing for ML (the final step) ─────────────────
# Separate features (X) from labels (y)
X = df.drop(columns=['churned']).values    # .values → NumPy array
y = df['churned'].values

print(X.shape)  # (5, 3)
print(y.shape)  # (5,)
```

---

## Typical ML Data Pipeline with Pandas

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load
df = pd.read_csv('employees.csv')

# 2. Explore
print(df.head())
print(df.isnull().sum())
print(df['churned'].value_counts())   # class balance check

# 3. Clean
df = df.dropna()
df = df.drop(columns=['employee_id'])  # drop non-predictive cols

# 4. Encode
df = pd.get_dummies(df, columns=['department'])

# 5. Split features and labels
X = df.drop(columns=['churned']).values
y = df['churned'].values

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

---

## Key Operations for ML

| Task | Pandas code |
|------|------------|
| Load data | `pd.read_csv('file.csv')` |
| Check missing | `df.isnull().sum()` |
| Fill missing | `df.fillna(df.mean())` |
| Encode categories | `pd.get_dummies(df, columns=['col'])` |
| Get features | `df.drop(columns=['label']).values` |
| Get labels | `df['label'].values` |
| Class balance | `df['label'].value_counts()` |

---

## Exercise
1. Create a DataFrame with 5 students: name, score, grade (A/B/C), passed (0/1)
2. Find the average score of students who passed
3. Fill any missing scores with the mean score
4. One-hot encode the grade column
5. Extract X (features) and y (labels) as NumPy arrays

---

## What to Remember Forever
> Pandas = load, explore, clean, transform your data.
> Always call `.head()`, `.shape`, `.isnull().sum()` on any new dataset.
> End result: `.values` to convert to NumPy → feed to model.
