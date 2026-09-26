# Lesson: async/await — Concurrent Programming in Python

**Date taught:** 2026-09-26
**Folder:** python-ml-basics/
**Status:** ✅ Taught

---

## Analogy

You go to a restaurant. You give the waiter your order.

**Blocking (normal code):** The waiter stands at your table staring at you until your food is ready. Nobody else gets served.

**Non-blocking (async):** The waiter takes your order, walks away, takes orders from 5 other tables, and comes back when YOUR food is ready.

> async/await = don't wait around doing nothing. While waiting for one thing, do other things.

---

## Why This Matters for AI Engineering

In AI applications you constantly:
- Call LLM APIs (takes 1–5 seconds each)
- Make 10+ simultaneous API calls
- Wait for database/vector DB queries
- Stream responses

Without async: call 10 APIs → wait 5s each = **50 seconds total**
With async: call 10 APIs simultaneously → wait 5s = **5 seconds total**

```
Sequential (blocking):
  API call 1 ──────5s──────→
                            API call 2 ──────5s──────→
                                                      API call 3 ...
  Total: 15s+

Async (concurrent):
  API call 1 ──────5s──────→
  API call 2 ──────5s──────→   ← all running at the same time
  API call 3 ──────5s──────→
  Total: ~5s
```

---

## Core Concepts

```
async def    →  defines a coroutine (a function that can pause)
await        →  "pause here and let others run while I wait"
asyncio      →  the event loop that manages all paused tasks
```

### The Event Loop

```
Event Loop
    │
    ├── Task 1: calling Claude API  (paused, waiting for response)
    ├── Task 2: calling OpenAI API  (paused, waiting for response)
    ├── Task 3: querying vector DB  (paused, waiting)
    │
    → when any task gets a response, resume it
```

---

## Code

```python
import asyncio
import time

# ── Basic async function ──────────────────────────────
async def fetch_data(name, delay):
    print(f"{name}: starting...")
    await asyncio.sleep(delay)      # simulates API call (non-blocking wait)
    print(f"{name}: done after {delay}s")
    return f"{name} result"

# ── Running one async function ────────────────────────
async def main_single():
    result = await fetch_data("API call", 2)
    print(result)

asyncio.run(main_single())

# ── Running multiple concurrently (the real power) ────
async def main_concurrent():
    start = time.time()

    # Run all 3 at the same time
    results = await asyncio.gather(
        fetch_data("Claude API",   2),
        fetch_data("OpenAI API",   2),
        fetch_data("Vector DB",    1),
    )

    elapsed = time.time() - start
    print(f"\nAll done in {elapsed:.1f}s")   # ~2s, not 5s!
    print(results)

asyncio.run(main_concurrent())

# ── Real pattern: calling LLM API async ───────────────
import anthropic

async def ask_claude(prompt: str, client) -> str:
    message = await client.messages.create(
        model="claude-sonnet-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

async def answer_many_questions():
    client = anthropic.AsyncAnthropic()    # async client

    questions = [
        "What is machine learning?",
        "What is a neural network?",
        "What is gradient descent?",
    ]

    # All 3 questions go to Claude simultaneously
    answers = await asyncio.gather(
        *[ask_claude(q, client) for q in questions]
    )

    for q, a in zip(questions, answers):
        print(f"Q: {q}\nA: {a[:100]}...\n")

asyncio.run(answer_many_questions())

# ── Streaming responses (important for chatbots) ──────
async def stream_claude():
    client = anthropic.AsyncAnthropic()

    async with client.messages.stream(
        model="claude-sonnet-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Explain AI in 3 sentences"}]
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)   # prints word by word as it arrives

asyncio.run(stream_claude())

# ── Error handling in async ───────────────────────────
async def safe_fetch(name, delay, fail=False):
    try:
        await asyncio.sleep(delay)
        if fail:
            raise ValueError(f"{name} failed!")
        return f"{name} ok"
    except Exception as e:
        return f"Error: {e}"

async def main_safe():
    results = await asyncio.gather(
        safe_fetch("API 1", 1),
        safe_fetch("API 2", 1, fail=True),   # this one fails
        safe_fetch("API 3", 1),
        return_exceptions=True               # don't crash on one failure
    )
    print(results)

asyncio.run(main_safe())
```

---

## Key Rules

```
Rule 1: async def defines a coroutine — calling it doesn't run it yet
Rule 2: await runs it and pauses until it's done
Rule 3: asyncio.run() starts the event loop (call once at the top level)
Rule 4: asyncio.gather() runs multiple coroutines concurrently
Rule 5: use AsyncAnthropic / AsyncOpenAI clients for LLM calls
```

---

## When You'll Use This

| Scenario | Pattern |
|----------|---------|
| Call LLM once | `await client.messages.create(...)` |
| Call LLM 10x in parallel | `await asyncio.gather(*[call(q) for q in questions])` |
| Stream response to user | `async for text in stream.text_stream` |
| Multiple tool calls in agent | `await asyncio.gather(tool1(), tool2(), tool3())` |

---

## Exercise
1. Write an async function `fetch(url, delay)` that prints "fetching url..." then sleeps `delay` seconds, then prints "done url"
2. Use `asyncio.gather()` to fetch 4 URLs with delays [1,2,1,3] simultaneously. Time it — should finish in ~3s not 7s
3. Add error handling: one URL should raise an exception, use `return_exceptions=True` to catch it

---

## What to Remember Forever
> `asyncio.gather()` = run multiple API calls simultaneously.
> Without it, calling 10 LLMs sequentially takes 10× longer.
> Always use `AsyncAnthropic` / `AsyncOpenAI` clients in production AI apps.
