# FastAPI LangGraph Agent — Production-Ready Template

A full-stack backend template for shipping AI agents to production. Built on **FastAPI** and **LangGraph**, it solves the problems that come _after_ you get a prototype working: persistent memory across sessions, resilient LLM calls, auth, observability, rate limiting, and deployability — all wired together from day one.

> **Built for AI engineers** who want a serious foundation, not a tutorial project.

---

## The Problem This Solves

Most AI agent demos share the same fatal flaw: they work beautifully in a Jupyter notebook and fall apart the moment you try to ship them. The gap between a working prototype and a production-grade service involves dozens of non-trivial decisions:

- How do you persist conversation state so the agent can resume after an interrupt?
- How do you give the agent per-user long-term memory without paying a vector search cost on every message?
- What happens when your primary LLM is rate-limited or times out mid-request?
- How do you trace what the LLM actually did when something goes wrong in production?
- How do you secure and rate-limit your endpoints without writing that infrastructure from scratch?

This template answers all of those questions with production-tested patterns, so you can fork it and build your agent logic on top of a solid foundation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Application                    │
│  Middleware (rate-limit · metrics · logging · profiling) │
│  Auth (JWT)   →   API Routes (/chat · /chat/stream · …) │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     LangGraph Agent      │
              │  StateGraph              │
              │  chat → tool_call → chat │
              │  AsyncPostgresSaver      │  ← checkpoint per thread_id
              └──┬──────────┬───────────┘
                 │          │
        ┌────────▼──┐  ┌────▼──────────┐
        │ LLM Service│  │ Memory Service │
        │ fallback + │  │ mem0 + pgvector│
        │ retry +    │  │ + Valkey cache │
        │ timeout    │  └───────┬────────┘
        └────────────┘          │
                          ┌─────▼──────────────────────────┐
                          │  PostgreSQL + pgvector           │
                          │  (checkpoints + vector memory)   │
                          └─────────────────────────────────┘
```

### Request Lifecycle

Every chat request flows through the following path:

1. **Middleware** stamps a correlation ID, starts Prometheus timers, and logs structured context.
2. **JWT auth** validates the Bearer token and loads the session (no extra DB query for username — it's copied at session creation time).
3. **LangGraph `aget_state` and `memory.search` run concurrently** via `asyncio.gather`, saving 200–500ms per request.
4. The `chat` node builds the system prompt, injects retrieved memories, and calls the LLM.
5. If the LLM returns tool calls, they all execute **concurrently** in the `tool_call` node.
6. The final response is returned; memory update runs as a **background task** so it doesn't block the client.

### The Agent Graph

The agent is a two-node `StateGraph` that is deliberately minimal — easy to extend:

```
START → [chat] ──(tool calls?)──► [tool_call] → back to [chat]
              └──(no tool calls)──► END
```

State is checkpointed to PostgreSQL via `AsyncPostgresSaver` after every step. This means:
- Multi-turn conversations survive server restarts.
- Human-in-the-loop interrupts (`GraphInterrupt`) can resume exactly where they left off on the next request.
- You get full conversation replay for debugging.

---

## Key Technical Decisions

### Resilient LLM Calls

The `LLMService` wraps every LLM call with a three-layer protection strategy:

1. **Exponential backoff retries** on `RateLimitError`, `APITimeoutError`, and transient `APIError` (up to `MAX_LLM_CALL_RETRIES`, default 3).
2. **Circular model fallback** — if all retries on the current model fail, it automatically switches to the next model in the registry and tries again, cycling through all available models before giving up.
3. **Hard total timeout** — the entire fallback loop is wrapped in `asyncio.wait_for(timeout=LLM_TOTAL_TIMEOUT)` so a stuck LLM call can never hang your server indefinitely.

### Long-Term Memory with Caching

User memory is powered by **mem0** backed by **pgvector**. The `MemoryService` adds a Valkey (Redis-compatible) cache layer in front of every `memory.search()` call. Cache keys are hashed from `(user_id, query)`, so semantically identical questions hit the cache instead of paying the vector search cost. Memory writes happen asynchronously after the response is sent so they never add latency to the user.

The memory service is pre-warmed at startup to absorb the ~130ms pgvector cold-init cost on the first request.

### Conversation State vs. Long-Term Memory

This template maintains a deliberate distinction between two kinds of memory:

| | Short-Term (Checkpointer) | Long-Term (mem0) |
|---|---|---|
| Scope | Per session / thread | Per user, cross-session |
| Storage | PostgreSQL (checkpoint tables) | pgvector embeddings |
| Retrieval | Full message list on every turn | Semantic top-k search |
| Use case | Resume interrupted graphs, message history | Personalization, facts about the user |

### Performance Optimizations

- **Concurrent state + memory lookup** on every non-resumed request.
- **Concurrent tool execution** when the LLM emits multiple tool calls in a single response.
- **System prompt loaded once at startup** — per-request cost is only a `.format()` call with username, datetime, and retrieved memories.
- **LangGraph graph pre-warmed at startup** — the connection pool is opened and the graph compiled once, not on the first user request.
- **Username copied to session at creation** — no extra DB round-trip to look up the user's display name during chat.

---

## What's Included

### Core
- **LangGraph stateful agent** with PostgreSQL checkpointing, tool calling, streaming, and human-in-the-loop
- **LLM service** with model registry, circular fallback, retry with exponential backoff, and total timeout budget
- **Long-term semantic memory** via mem0 + pgvector, with optional Valkey/Redis cache layer
- **JWT authentication** — user registration, login, session management, token verification
- **Alembic database migrations** — schema versioning from day one

### Observability
- **Langfuse tracing** on all LLM calls — every prompt, response, and tool call is traced
- **Prometheus metrics** exposed at `/metrics` — request counts, latencies, LLM inference duration, stream duration
- **Grafana dashboards** — pre-configured LLM latency dashboard, provisioned automatically
- **Structured JSON logging** via `structlog` with request ID, session ID, and user ID bound to every log line
- **Request profiling** in DEBUG mode via `pyinstrument` — HTML profiles saved for slow requests

### Production Hardening
- **Rate limiting** via `slowapi` — configurable per-endpoint limits with environment-specific defaults
- **CORS, input sanitization, password strength validation**
- **Correlation ID middleware** for distributed tracing
- **Docker Compose stack** — API, PostgreSQL + pgvector, Valkey, Prometheus, Grafana, cAdvisor
- **GitHub Actions CI/CD** workflow included

### Evaluation Framework

An offline LLM evaluation system that fetches recent traces from Langfuse and scores them using an LLM-as-judge pattern across five metrics:

| Metric | What it measures |
|---|---|
| Relevancy | Does the response address the user's actual question? |
| Helpfulness | Does it provide actionable, useful information? |
| Conciseness | Is it appropriately brief without losing substance? |
| Hallucination | Does it make unsupported factual claims? |
| Toxicity | Is the language safe and appropriate? |

Scores are written back to Langfuse for visualization and produce a local JSON report.

---

## Project Structure

```
app/
  api/v1/
    auth.py          # Register, login, session CRUD endpoints
    chatbot.py       # /chat, /chat/stream, /messages endpoints
  core/
    langgraph/
      graph.py       # LangGraphAgent — the main agent orchestrator
      tools/         # Tool definitions (DuckDuckGo search, ask_human)
    prompts/
      system.md      # System prompt template (loaded once at startup)
    cache.py         # Valkey/Redis with in-memory fallback
    config.py        # Settings — all config in one place
    metrics.py       # Prometheus counter/histogram definitions
    middleware.py    # Metrics, logging context, profiling middlewares
    observability.py # Langfuse initialization
  models/            # SQLModel ORM — User, Session, Thread
  schemas/           # Pydantic request/response models
  services/
    llm.py           # LLMService — registry, retries, circular fallback
    memory.py        # MemoryService — mem0 + pgvector + cache
    database.py      # DatabaseService — user/session CRUD
alembic/             # Database migrations
evals/               # Offline LLM evaluation framework
docs/                # Deep-dive documentation per subsystem
```

---

## Quickstart

**Prerequisites:** Python 3.13+, Docker, an OpenAI API key.

```bash
# 1. Clone and enter the project
git clone <repo-url> my-agent && cd my-agent

# 2. Set up environment
cp .env.example .env.development
# Fill in OPENAI_API_KEY, JWT_SECRET_KEY, and optionally LANGFUSE_* keys

# 3. Install dependencies
make install

# 4. Start the full stack (API + PostgreSQL + Valkey + Prometheus + Grafana)
make docker-up
```

The API is now at [http://localhost:8000](http://localhost:8000).  
Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)  
Grafana: [http://localhost:3000](http://localhost:3000) (admin / admin)

> For local development without Docker, see [docs/getting-started.md](docs/getting-started.md).

### First API Call

```bash
# Register a user
curl -X POST http://localhost:8000/api/v1/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "StrongPass1!", "username": "you"}'

# Login and get a token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/login \
  -d "email=you@example.com&password=StrongPass1!&grant_type=password" \
  | jq -r .access_token)

# Create a session
SESSION_TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/session \
  -H "Authorization: Bearer $TOKEN" | jq -r .token.access_token)

# Chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What can you help me with?"}]}'
```

---

## Extending the Template

**Add a tool:** Create a file in `app/core/langgraph/tools/`, define a `@tool`-decorated async function, and add it to the `tools` list in `app/core/langgraph/tools/__init__.py`. The agent picks it up automatically.

**Add an LLM model:** Add an entry to `LLMRegistry.LLMS` in `app/services/llm.py`. It becomes part of the automatic fallback chain.

**Customize the system prompt:** Edit `app/core/prompts/system.md`. The `{username}`, `{datetime}`, and `{long_term_memory}` placeholders are injected at runtime.

**Change the graph:** Modify `LangGraphAgent.create_graph()` in `app/core/langgraph/graph.py` to add nodes, edges, or conditions.

---

## Documentation

| Guide | What it covers |
|---|---|
| [Getting Started](docs/getting-started.md) | Prerequisites, local setup, first API call |
| [Architecture](docs/architecture.md) | System design, request lifecycle, Mermaid diagrams |
| [Configuration](docs/configuration.md) | All environment variables with defaults and descriptions |
| [Authentication](docs/authentication.md) | JWT flow, session lifecycle, endpoint reference |
| [Database & Migrations](docs/database.md) | Schema, Alembic workflow, pgvector setup |
| [LLM Service](docs/llm-service.md) | Model registry, retry logic, circular fallback, timeout budget |
| [Memory](docs/memory.md) | mem0 long-term memory, pgvector, cache layer |
| [Observability](docs/observability.md) | Langfuse tracing, structured logging, Prometheus, profiling |
| [Evaluation](docs/evaluation.md) | Eval framework, custom metrics, running evaluations |
| [Docker](docs/docker.md) | Docker setup, Compose services, full monitoring stack |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn + uvloop |
| Agent Orchestration | LangGraph |
| LLM Provider | OpenAI (configurable via model registry) |
| Long-Term Memory | mem0 + pgvector |
| Short-Term Memory | LangGraph AsyncPostgresSaver |
| Cache | Valkey (Redis-compatible) |
| Database | PostgreSQL (SQLModel + psycopg3) |
| Migrations | Alembic |
| Auth | JWT (python-jose) + bcrypt |
| Observability | Langfuse · structlog · Prometheus · Grafana |
| Rate Limiting | slowapi |
| Evaluation | LLM-as-judge via Langfuse traces |
| Python Version | 3.13+ |

---

## Contributing

PRs welcome. See [AGENTS.md](AGENTS.md) for coding conventions and [docs/getting-started.md](docs/getting-started.md) for setup. Report security issues privately — see [SECURITY.md](SECURITY.md).

## License

See [LICENSE](LICENSE).
