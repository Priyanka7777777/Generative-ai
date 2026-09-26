# LangChain Explained

---

## Simple Analogy First

Imagine you're building a **smart restaurant**.

- **The chef** = the LLM (GPT, Claude, etc.) — does the actual cooking (thinking)
- **The recipe book** = prompts — instructions for what to cook
- **The waiter** = the chain — takes your order, goes to kitchen, brings back food
- **The pantry** = memory — stores what ingredients were used before
- **The supplier** = tools/APIs — external sources the chef can call for fresh ingredients
- **The menu system** = agents — decides *which* chef to use, *which* recipe, in *what* order

LangChain is the **restaurant management system** that wires all of this together so you don't have to manually coordinate every interaction.

---

## Technically

LangChain is an **open-source framework** (Python/JS) for building applications powered by LLMs. It provides abstractions over:

1. **LLM calls** — unified interface across providers (OpenAI, Anthropic, etc.)
2. **Prompt management** — templating, versioning, few-shot examples
3. **Chaining** — composing multiple LLM calls or steps into a pipeline
4. **Memory** — persisting conversation state across turns
5. **Tools/Agents** — letting LLMs decide what actions to take and execute them
6. **Retrieval** — connecting LLMs to external data (RAG)

The core design pattern is: **Input → Transform → LLM → Transform → Output**, composable and repeatable.

---

## All Concepts

### 1. Models / LLMs
The actual language model. LangChain wraps them with a unified interface.

```
LLM("Tell me a joke")  →  "Why did the chicken..."
ChatModel([HumanMessage("Hi")])  →  AIMessage("Hello!")
```

Types:
- `LLM` — raw text in, text out
- `ChatModel` — message list in, message out
- `Embeddings` — text in, vector out (for search)

---

### 2. Prompts
Templates that structure what you send to the LLM.

```
PromptTemplate("Translate {text} to {language}")
ChatPromptTemplate([SystemMessage, HumanMessage])
```

- `PromptTemplate` — string with variables
- `ChatPromptTemplate` — multi-role conversation template
- `FewShotPromptTemplate` — includes examples to guide the model

---

### 3. Chains
The backbone. A chain connects components in a sequence.

```
chain = prompt | llm | output_parser
result = chain.invoke({"topic": "AI"})
```

- `LLMChain` — prompt + LLM
- `SequentialChain` — chain of chains (output of one feeds next)
- `LCEL (LangChain Expression Language)` — modern `|` pipe syntax for composing chains

---

### 4. Memory
Lets your app "remember" previous messages in a conversation.

```
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
```

Types:
- `ConversationBufferMemory` — stores the full history
- `ConversationSummaryMemory` — summarizes old turns to save tokens
- `ConversationWindowMemory` — keeps last N messages only
- `VectorStoreMemory` — retrieves relevant past messages by similarity

---

### 5. Document Loaders
Load data from various sources into a standard format.

```
loader = PDFLoader("doc.pdf")
docs = loader.load()
```

Sources: PDFs, URLs, CSVs, Notion, S3, YouTube transcripts, databases, etc.

---

### 6. Text Splitters
Chunks large documents into pieces the LLM can handle.

```
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chunks = splitter.split_documents(docs)
```

- `CharacterTextSplitter`
- `RecursiveCharacterTextSplitter` — smarter, respects sentence/paragraph boundaries
- `TokenTextSplitter` — splits by token count

---

### 7. Embeddings + Vector Stores (RAG)
The core of Retrieval-Augmented Generation.

```
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
```

Flow: `Document → Embed → Store → Query → Retrieve similar chunks → Feed to LLM`

Vector stores: FAISS, Pinecone, Chroma, Weaviate, pgvector, etc.

---

### 8. Retrievers
Abstract interface over vector stores for fetching relevant documents.

```
docs = retriever.get_relevant_documents("What is RAG?")
```

Types:
- `VectorStoreRetriever` — similarity search
- `MultiQueryRetriever` — generates multiple queries for better recall
- `ContextualCompressionRetriever` — filters/compresses retrieved docs
- `SelfQueryRetriever` — LLM builds the query itself

---

### 9. Tools
Functions the LLM can call to interact with the outside world.

```python
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return fetch_weather_api(city)
```

Built-in tools: Google Search, Wikipedia, Python REPL, SQL DB, calculator, etc.

---

### 10. Agents
The LLM acts as a **reasoning engine** that decides which tools to call and in what order.

```
agent = initialize_agent(tools, llm, agent=AgentType.REACT)
agent.run("What's the weather in Paris and convert it to Kelvin?")
```

Reasoning patterns:
- **ReAct** — Reason + Act loop (most common)
- **OpenAI Functions** — uses function-calling API
- **Plan-and-Execute** — plans all steps first, then executes
- **Self-Ask** — breaks question into sub-questions

---

### 11. Output Parsers
Converts raw LLM text output into structured data.

```
parser = PydanticOutputParser(pydantic_object=MySchema)
chain = prompt | llm | parser
```

- `StrOutputParser` — plain string
- `PydanticOutputParser` — validated Python object
- `JsonOutputParser` — JSON dict
- `CommaSeparatedListOutputParser`

---

### 12. Callbacks
Hooks into every step of a chain/agent for logging, monitoring, streaming.

```
chain.invoke(input, config={"callbacks": [MyCallback()]})
```

Use cases: streaming tokens to UI, logging to LangSmith, cost tracking.

---

### 13. LangSmith
LangChain's **observability platform** — traces every LLM call, chain step, tool use. Useful for debugging, evaluation, and monitoring in production.

---

### 14. LangGraph
Extension for building **stateful, multi-actor workflows** as a graph (nodes = steps, edges = transitions). Built for complex agentic systems with loops, branching, and human-in-the-loop.

```
graph = StateGraph(AgentState)
graph.add_node("agent", agent_fn)
graph.add_node("tools", tool_fn)
graph.add_edge("agent", "tools")
```

---

## Mental Model Summary

```
Data Sources
    ↓
Document Loaders → Text Splitters → Embeddings → Vector Store
                                                       ↓
User Input → Prompt Template → LLM ← Retriever (RAG) ←┘
                                ↓
                         Output Parser → Structured Output
                                ↑
                    Memory (conversation history)
                                ↑
                    Tools + Agent (if autonomous)
```

LangChain's value is that each box in that diagram is a **swappable, standard interface** — you can change the LLM, the vector store, or the memory strategy without rewriting the whole pipeline.