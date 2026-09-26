# Lesson: Pydantic — Data Validation for AI Engineering

**Date taught:** 2026-09-26
**Folder:** python-ml-basics/
**Status:** ✅ Taught

---

## Analogy

Imagine you run a restaurant. A waiter takes an order and passes it to the kitchen.

**Without Pydantic:**
The waiter can pass ANYTHING to the kitchen:
- "I want a pizza" ✓ (valid)
- "I want 42 apples" ← wrong type, kitchen crashes
- "I want -1 burgers" ← impossible quantity, kitchen confused
- No table number ← kitchen doesn't know where to deliver

**With Pydantic:**
Before the order reaches the kitchen, a validation system checks:
- Is the item name a string? ✓
- Is the quantity a positive integer? ✓
- Is the table number provided? ✓
If anything is wrong, it raises a clear error immediately — not deep in the kitchen where it's hard to debug.

> Pydantic = a strict bouncer at the door that validates data before it enters your system.

---

## Why Pydantic Matters for AI Engineering

```
In AI apps, data flows everywhere:
  User input → LLM → Tool call → Agent → API → Database

At each step, wrong data = silent failures, hallucinations, crashes.

Pydantic is used in:
  ✅ FastAPI          → validate all HTTP request/response bodies
  ✅ LangChain        → define tool schemas, structured outputs
  ✅ Claude/OpenAI    → structured output (force model to return valid JSON)
  ✅ Agent tools      → type-safe tool arguments
  ✅ RAG pipelines    → validate document chunks, metadata
```

---

## Diagram

```
Without Pydantic:
  raw_dict = {"name": "John", "age": "not_a_number"}
  process(raw_dict)   ← crashes later, hard to debug

With Pydantic:
  data = UserModel(name="John", age="not_a_number")
                                      ↑
                         ValidationError raised HERE
                         with clear message: "age must be int"

  data = UserModel(name="John", age=25)
  data.name   → "John"   (str, guaranteed)
  data.age    → 25       (int, guaranteed)
  data.dict() → {"name": "John", "age": 25}
  data.json() → '{"name": "John", "age": 25}'
```

---

## Part 1: Basic BaseModel

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

# ── Basic model ───────────────────────────────────────
class User(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool = True          # default value
    score: float = 0.0

# Valid data
user = User(name="Alice", age=30, email="alice@example.com")
print(user.name)           # "Alice"
print(user.age)            # 30
print(user.is_active)      # True  (default)
print(user.model_dump())   # {"name":"Alice", "age":30, ...}
print(user.model_dump_json())  # JSON string

# Invalid data → ValidationError
try:
    bad = User(name="Bob", age="not_a_number", email="bob@b.com")
except Exception as e:
    print(e)  # age: Input should be a valid integer

# Pydantic auto-converts where possible:
user2 = User(name="Carol", age="25", email="carol@c.com")  # "25" → 25 ✓
print(type(user2.age))   # <class 'int'>
```

---

## Part 2: Field Validation — Constraints

```python
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    name:        str   = Field(..., min_length=1, max_length=100)
    price:       float = Field(..., gt=0)           # greater than 0
    quantity:    int   = Field(..., ge=0, le=1000)  # 0 ≤ x ≤ 1000
    category:    str   = Field(default="general")
    description: Optional[str] = None               # nullable field

    # Custom validator
    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v):
        if v.strip() == '':
            raise ValueError('Name cannot be blank')
        return v.strip()   # also transforms the value

# Valid
p = Product(name="Widget", price=9.99, quantity=50)

# Invalid — will raise immediately with clear message
try:
    p2 = Product(name="Bad", price=-1, quantity=50)
except Exception as e:
    print(e)   # price: Input should be greater than 0
```

---

## Part 3: Nested Models

```python
from pydantic import BaseModel
from typing import List, Optional

class Address(BaseModel):
    street: str
    city: str
    country: str = "India"

class Order(BaseModel):
    order_id:   str
    items:      List[str]
    total:      float
    shipping:   Address

# Nested validation works automatically
order = Order(
    order_id="ORD-001",
    items=["laptop", "mouse"],
    total=1299.99,
    shipping={"street": "123 MG Road", "city": "Bangalore"}  # dict auto-converted
)

print(order.shipping.city)           # "Bangalore"
print(order.model_dump())            # full nested dict
```

---

## Part 4: Pydantic for LLM Structured Outputs (the AI use case!)

```python
# Force an LLM to return structured, validated data
import anthropic
from pydantic import BaseModel
from typing import List
import json

# Define the schema you want from the LLM
class MovieReview(BaseModel):
    title:      str
    rating:     float          # must be a number
    pros:       List[str]      # must be a list
    cons:       List[str]
    summary:    str
    recommended: bool

def get_movie_review(movie: str) -> MovieReview:
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        system="""You are a movie critic. Always respond with valid JSON matching this schema:
{
  "title": "string",
  "rating": float between 0-10,
  "pros": ["list", "of", "pros"],
  "cons": ["list", "of", "cons"],
  "summary": "one sentence summary",
  "recommended": true or false
}""",
        messages=[{"role": "user", "content": f"Review the movie: {movie}"}]
    )

    raw_json = response.content[0].text
    data = json.loads(raw_json)

    # Pydantic validates: correct types, required fields, no extra fields
    review = MovieReview(**data)
    return review

review = get_movie_review("Inception")
print(f"Rating: {review.rating}/10")       # guaranteed to be a float
print(f"Recommended: {review.recommended}") # guaranteed to be bool
print(f"Pros: {review.pros}")              # guaranteed to be list
```

---

## Part 5: Pydantic for Agent Tool Schemas

```python
# When building AI agents, tool arguments must be validated
from pydantic import BaseModel, Field
from typing import Literal

# Define tool input schema
class SearchInput(BaseModel):
    query:       str   = Field(..., description="The search query")
    max_results: int   = Field(default=5, ge=1, le=20)
    language:    str   = Field(default="en")

class WeatherInput(BaseModel):
    city:   str
    unit:   Literal["celsius", "fahrenheit"] = "celsius"  # only these two values

class CodeExecutionInput(BaseModel):
    code:     str = Field(..., description="Python code to execute")
    timeout:  int = Field(default=30, le=120, description="Max seconds to run")

# Agent calls tool — Pydantic ensures valid inputs before execution
def search_web(raw_args: dict) -> str:
    args = SearchInput(**raw_args)         # validates here — safe to use
    print(f"Searching: {args.query}, max: {args.max_results}")
    return f"Results for: {args.query}"

# If LLM hallucinates bad args → caught immediately
try:
    search_web({"query": "AI news", "max_results": 999})  # 999 > 20 limit
except Exception as e:
    print(f"Caught bad LLM output: {e}")
```

---

## Part 6: Pydantic with FastAPI (preview)

```python
# FastAPI uses Pydantic automatically for request/response validation
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    text:        str
    max_tokens:  int = 100
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class PredictResponse(BaseModel):
    result:       str
    tokens_used:  int
    model:        str

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # request is automatically validated by Pydantic
    # response is automatically validated and serialized
    return PredictResponse(
        result=f"Result for: {request.text}",
        tokens_used=50,
        model="claude-sonnet-5"
    )
```

---

## Key Terms

| Term | Meaning |
|------|---------|
| BaseModel | Base class — inherit to create a validated data model |
| Field | Add constraints, defaults, and descriptions to fields |
| field_validator | Custom validation logic for a specific field |
| model_dump() | Convert model to Python dict |
| model_dump_json() | Convert model to JSON string |
| ValidationError | Raised when data doesn't match the schema |
| Optional[T] | Field can be T or None |
| Literal[...] | Field can only be one of the listed values |

---

## Pydantic vs Plain Dict

```
┌──────────────────────┬──────────────────┬──────────────────────────────┐
│ Feature              │ Plain dict       │ Pydantic model               │
├──────────────────────┼──────────────────┼──────────────────────────────┤
│ Type safety          │ ❌ None          │ ✅ Guaranteed types          │
│ Validation           │ ❌ Manual        │ ✅ Automatic on creation     │
│ Default values       │ Manual           │ ✅ Declarative               │
│ Nested validation    │ ❌ Manual        │ ✅ Recursive                 │
│ Error messages       │ Crash later      │ ✅ Clear, immediate          │
│ IDE autocomplete     │ ❌ No            │ ✅ Full autocomplete          │
│ JSON serialization   │ json.dumps(d)    │ model.model_dump_json()      │
└──────────────────────┴──────────────────┴──────────────────────────────┘
```

---

## Exercise
1. Create a `ChatMessage` model: `role` (only "user" or "assistant"), `content` (str, min 1 char), `timestamp` (datetime, optional)
2. Create a `LLMConfig` model: `model` (str), `temperature` (float, 0-2), `max_tokens` (int, 1-4096), `stream` (bool, default False)
3. Try creating an `LLMConfig` with `temperature=5.0` — see the error message
4. Create a `RAGDocument` model: `content` (str), `source` (str), `score` (float, 0-1), `metadata` (dict, default empty)

---

## What to Remember Forever
> Pydantic = define the shape of your data, get validation for free.
> Every AI app boundary needs Pydantic: API inputs, LLM outputs, tool args.
> `BaseModel` → inherit it. `Field(...)` → add constraints. `.model_dump()` → get dict.
> LLM outputs are strings — use Pydantic to parse and validate them into typed objects.
