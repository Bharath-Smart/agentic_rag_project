# Production-Oriented Agentic Knowledge Intelligence Platform

An engineering project for building and evaluating an agentic Retrieval-Augmented Generation (RAG) application over PDF documents.

The repository began as an Agentic RAG foundation and is being extended toward a reliable, observable, and evaluation-driven knowledge intelligence platform. This README describes the **current implementation**, not every capability planned for the finished flagship project.

> **Current maturity:** working application foundation with PDF ingestion, vector/hybrid retrieval, agentic tool use, persisted chat sessions, and streaming. Evaluation, evidence/citation handling, failure semantics, authorization, and several production-readiness areas remain active engineering work. Do not interpret the project title as a claim that all production requirements are already satisfied.

## Contents

- [What the application currently does](#what-the-application-currently-does)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Quick start with Docker Compose](#quick-start-with-docker-compose)
- [Ingest PDF documents](#ingest-pdf-documents)
- [Run locally with uv](#run-locally-with-uv)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Known limitations and engineering roadmap](#known-limitations-and-engineering-roadmap)
- [Security and operational notes](#security-and-operational-notes)
- [License](#license)

## What the application currently does

The current implementation provides:

- **PDF ingestion:** extracts text from PDF files with PyMuPDF.
- **Chunking:** splits extracted text with LangChain's `RecursiveCharacterTextSplitter`.
- **Embeddings:** generates document and query embeddings using the configured OpenAI embedding model.
- **Persistence:** stores documents, chunks, metadata, and embeddings in PostgreSQL with pgvector.
- **Retrieval:** exposes vector search and hybrid vector/full-text search.
- **Agentic chat:** a LangChain v1 agent can select from vector search, hybrid search, document lookup, and document-listing tools.
- **Conversation context:** persists user and final assistant messages and supplies up to 10 recent messages to the agent in chronological order with their roles preserved.
- **Streaming:** exposes chat responses through Server-Sent Events (SSE).
- **Web interface:** provides a Streamlit chat UI.
- **Containerized runtime:** runs PostgreSQL/pgvector, the FastAPI service, and Streamlit using Docker Compose.

The current prompt and corpus configuration originated from a sustainability-report use case. General enterprise/technical knowledge is the intended product direction, but the system prompt still needs to be aligned with that broader domain.

## Architecture

### High-level components

```text
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         │      :8501           │
                         └──────────┬───────────┘
                                    │ HTTP / SSE
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │      :8058           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  LangChain v1 Agent  │
                         │     create_agent     │
                         └──────────┬───────────┘
                                    │ tool calls
                  ┌─────────────────┼──────────────────┐
                  ▼                 ▼                  ▼
          ┌──────────────┐  ┌──────────────┐  ┌────────────────┐
          │ Vector Search│  │ Hybrid Search│  │ Document Tools │
          └──────┬───────┘  └──────┬───────┘  └───────┬────────┘
                 └─────────────────┼──────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ PostgreSQL + pgvector│
                         │       :5432          │
                         └──────────────────────┘

PDF files ──► PyMuPDF ──► Recursive chunking ──► Embeddings
                                                   │
                                                   ▼
                                          PostgreSQL + pgvector
```

### Chat request flow

1. The Streamlit UI sends a chat request to FastAPI.
2. FastAPI creates or resolves a session.
3. The API retrieves up to 10 recent stored messages and orders them chronologically.
4. The current user message is appended once, preserving the historical `user` and `assistant` roles.
5. The LangChain agent receives the messages and its registered tools.
6. The model may call one or more retrieval/document tools.
7. Tool results are returned to the agent, which generates the final response.
8. The API persists the user message and final assistant text. The streaming route emits text/tool/end or error events over SSE.

The agent is built with LangChain v1's `create_agent()`. This repository does not define a custom LangGraph `StateGraph`.

### Document ingestion flow

```text
PDF files
   │
   ▼
PyMuPDF text extraction
   │
   ▼
RecursiveCharacterTextSplitter
   │
   ▼
OpenAI document embeddings
   │
   ▼
PostgreSQL documents + chunks + pgvector
```

The current chunker uses character-based chunk size and overlap settings. The default CLI values are 850 characters and 150 characters of overlap; these are not tokenizer-based token counts.

## Technology stack

| Area | Technology |
|---|---|
| Language/runtime | Python 3.12 |
| API | FastAPI, Uvicorn |
| Agent orchestration | LangChain v1 `create_agent()` |
| LLM integration | LangChain OpenAI integration |
| Embeddings | OpenAI embeddings |
| PDF extraction | PyMuPDF |
| Chunking | LangChain text splitters |
| Database | PostgreSQL 17 |
| Vector search | pgvector |
| Database client | asyncpg |
| Frontend | Streamlit |
| HTTP/SSE client | requests, aiohttp |
| Dependency management | uv with `uv.lock` |
| Containers | Docker, Docker Compose |
| Validation | Pydantic |

The default embedding model is `text-embedding-3-small`, and the current database schema uses `vector(1536)`. If the embedding model or dimensions are changed, ingestion, query embedding, and database schema compatibility must be checked together.

## Quick start with Docker Compose

### Prerequisites

- Git
- Docker Engine/Desktop with Docker Compose
- An OpenAI API key with access to the configured chat and embedding models

### 1. Clone the repository

```bash
git clone https://github.com/Bharath-Smart/agentic_rag_project.git
cd agentic_rag_project
```

### 2. Create the environment file

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set your own key:

```dotenv
OPENAI_API_KEY=your_own_openai_api_key
```

The example file uses Docker Compose service names (`postgres` and `api`) for container-to-container networking. The example database password is for local development only; replace it with a strong value for any non-local environment. Never commit `.env` or a real API key.

### 3. Build and start the services

```bash
docker compose up --build -d
```

Check service status and logs:

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f ui
docker compose logs -f postgres
```

### 4. Open the application

| Service | Local address |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8058 |
| FastAPI Swagger docs | http://localhost:8058/docs |
| FastAPI health endpoint | http://localhost:8058/health |
| PostgreSQL host port | localhost:6543 |

The PostgreSQL container listens on port 5432 internally and is mapped to port 6543 on the host.

The database uses a named Docker volume (`postgres_data`) for persistence. Stopping containers does not normally remove this volume. Do not run `docker compose down -v` unless you intentionally want to delete the persisted database volume.

## Ingest PDF documents

The application does not currently provide a document-upload UI or upload API. Ingestion is performed through the CLI.

### 1. Add PDF files

Place PDF files under the repository's `documents/` directory. The ingestion command searches recursively for PDF files.

### 2. Run ingestion from the host

The Compose environment configures database host `postgres` and port `5432`, which are appropriate inside the Docker network. When running the ingestion CLI directly on your host, override these values to use the published host port:

**Linux / macOS shell:**

```bash
DB_HOST=localhost DB_PORT=6543 uv run python -m ingestion.ingest --documents documents/
```

**Windows PowerShell:**

```powershell
$env:DB_HOST = "localhost"
$env:DB_PORT = "6543"
uv run python -m ingestion.ingest --documents documents/
```

The CLI uses the configured OpenAI key and embedding model. It initializes the database/schema if required, extracts PDF text, chunks it, generates embeddings, and stores document/chunk records.

### Supported CLI options

```bash
uv run python -m ingestion.ingest --help
```

| Option | Purpose | Default |
|---|---|---|
| `--documents`, `-d` | Directory to scan for PDFs | `documents` |
| `--clean`, `-c` | Delete existing chat and document data before ingestion | Disabled |
| `--chunk-size` | Character chunk size | `850` |
| `--chunk-overlap` | Character overlap | `150` |
| `--verbose`, `-v` | Enable debug logging | Disabled |
| `--sql-schema-path`, `-sql` | Schema SQL file path | `sql/schema.sql` |

> **Destructive operation:** Do not use `--clean` casually. The current implementation deletes rows from `messages`, `sessions`, `chunks`, and `documents`; it therefore removes persisted chat history as well as indexed knowledge.

Re-ingestion does not currently have a verified content-hash/versioning policy. Re-running ingestion for the same PDF may create additional document/chunk records.

## Run locally with uv

Use this path when developing or running the API/UI outside Docker.

### 1. Install Python and uv

Use Python 3.12 (the repository's `.python-version` specifies 3.12.6) and install uv using its official installation instructions.

### 2. Install locked dependencies

```bash
uv sync --locked
```

### 3. Configure local database connectivity

Start PostgreSQL using Docker Compose, then configure the host-run application to connect through the published host port:

```dotenv
DB_HOST=localhost
DB_PORT=6543
API_URL=http://localhost:8058
```

Keep the remaining values and your own `OPENAI_API_KEY` in `.env`.

### 4. Start the API

```bash
uv run uvicorn agent.api:app --host 0.0.0.0 --port 8058
```

### 5. Start the Streamlit UI in a second terminal

```bash
uv run streamlit run ui/app.py --server.port 8501
```

Open http://localhost:8501.

## API reference

Interactive API documentation is available at http://localhost:8058/docs when the API is running.

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Database-oriented health response |
| POST | `/chat` | Non-streaming agent chat |
| POST | `/chat/stream` | Agent chat using SSE |
| POST | `/search/vector` | Direct vector search |
| POST | `/search/hybrid` | Direct hybrid search |
| GET | `/documents` | List indexed documents (supports `limit` and `offset`) |
| GET | `/sessions/{session_id}` | Retrieve session metadata |

### Chat request

```json
{
  "message": "What does the document say about its main findings?",
  "session_id": null,
  "user_id": "local-user",
  "search_type": "hybrid",
  "metadata": {}
}
```

`session_id` and `user_id` are optional. The API creates a session when no valid session is supplied.

**Important contract limitation:** `search_type` is accepted by the chat request, but it does not deterministically route the agent to vector-only or hybrid retrieval. Both retrieval tools are registered with the agent, and the model selects tools. The non-streaming response metadata echoes the request value; that echo is not proof that the requested mode was applied. The UI currently sends `hybrid`, but this does not enforce hybrid-only retrieval.

### Direct search request

```json
{
  "query": "renewable energy strategy",
  "search_type": "hybrid",
  "limit": 10,
  "filters": {}
}
```

Use the route to select the retrieval method:

- `POST /search/vector` invokes vector search.
- `POST /search/hybrid` invokes hybrid search.

The route determines the search method; the request's `search_type` field does not switch the route. The current direct-search handlers also do not apply `filters`. Hybrid search currently uses a fixed text weight of `0.3`.

### Streaming events

The `POST /chat/stream` endpoint emits SSE data events. Current event types include:

| Event type | Meaning |
|---|---|
| `session` | Provides the session ID |
| `text` | Contains a streamed text fragment |
| `tools` | Contains tool-call information |
| `error` | Indicates a stream-side error |
| `end` | Indicates successful completion |

A client should treat a stream as successfully completed only after receiving the `end` event. A connection closing without that event is an incomplete response.

The current Streamlit client handles explicit `error` events and missing completion events. It also appends tool-call details to the streamed response text; separating user-facing answers from tool diagnostics remains an identified UI improvement.

### Response and evidence behavior

The chat response model includes a `sources` field, but the current chat endpoint does not consistently populate structured source evidence, and the UI does not render a verified citation section. Do not assume that every generated answer has machine-verified citations. Structured evidence propagation is tracked as future engineering work.

## Configuration

The application loads configuration from environment variables and, when available, a local `.env` file.

| Variable | Purpose | Current default / example |
|---|---|---|
| `APP_ENV` | Application environment | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `APP_HOST` | API bind address | `0.0.0.0` |
| `APP_PORT` | API port | `8058` |
| `API_URL` | UI-to-API base URL | `http://api:8058` in Compose |
| `SERVER_HOST` | Streamlit bind address | `0.0.0.0` |
| `SERVER_PORT` | Streamlit port | `8501` |
| `DB_USER` | PostgreSQL user | `postgres` |
| `DB_PASSWORD` | PostgreSQL password | Development example only |
| `DB_HOST` | Database hostname | `postgres` in Compose; `localhost` from host |
| `DB_PORT` | Database port | `5432` in Compose; `6543` from host |
| `DB_NAME` | Database name | `vector_db` |
| `OPENAI_API_KEY` | OpenAI API credential | Required for model/embedding calls |
| `LLM_CHOICE` | Chat model | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |

The settings object reads environment values during application initialization. Restart the relevant process after changing its environment.

## Project structure

```text
agentic_rag_project/
├── agent/
│   ├── agent.py          # LangChain agent construction and tool registration
│   ├── api.py            # FastAPI routes, chat orchestration, SSE
│   ├── config.py         # Environment-backed settings
│   ├── db_utils.py       # PostgreSQL connection pool and data access
│   ├── models.py         # Pydantic request/response and domain models
│   ├── prompts.py        # Agent system prompt
│   ├── providers.py      # LLM and embedding client factories
│   └── tools.py          # Retrieval and document tools
├── ingestion/
│   ├── extract_files.py  # PDF text extraction
│   ├── chunker.py        # Recursive character chunking
│   └── ingest.py         # CLI ingestion and database persistence
├── ui/
│   └── app.py            # Streamlit chat interface
├── sql/
│   └── schema.sql        # PostgreSQL/pgvector schema and search functions
├── tests/
│   ├── agent/            # API, DB utility, model, and tool tests
│   ├── ingestion/        # Chunker tests
│   └── test_ui.py        # Streamlit UI/streaming tests
├── documents/            # Local PDF ingestion input
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── .env.example
```

## Tests

Install dependencies first:

```bash
uv sync --locked
```

Run the full test suite:

```bash
uv run pytest
```

Run focused tests:

```bash
uv run pytest tests/agent/test_models.py
uv run pytest tests/agent/test_tools.py
uv run pytest tests/agent/test_api.py
uv run pytest tests/ingestion/test_chunker.py
uv run pytest tests/test_ui.py
```

The current test tree includes model, tool, API, database utility, UI/streaming, and chunker tests. Dedicated PDF extractor tests and full extraction-to-database ingestion pipeline tests were not present in the inspected test tree.

A test command in this README is an instruction, not a claim that the tests have been run successfully on every commit. Record actual test output when validating a change.

## Known limitations and engineering roadmap

The following are known gaps from the codebase investigation and finalized backlogs. They are not described as completed features.

### Retrieval and answer quality
- Chat `search_type` does not enforce a retrieval mode; direct-search request fields `search_type` and `filters` do not control route behavior/filtering.
- Structured source evidence is not consistently propagated from retrieval through chat responses and the UI.
- Retrieval tool exceptions can be converted into empty results, making service failures difficult to distinguish from legitimate no-match outcomes.
- Hybrid retrieval score calibration and vector-index/query-plan behavior have not yet been established through a repeatable evaluation.
- No versioned, repeatable RAG evaluation baseline is currently documented.

### Ingestion and provenance
- The current extractor reads PDF text; OCR for scanned/image-only PDFs is not implemented.
- Chunk size and overlap are character-based; token counts are estimates rather than tokenizer-derived counts.
- The configured `max_chunk_size` field is not currently enforced by the splitter.
- Original-document character offsets and page-level citation provenance are not established as reliable.
- Embedding batch cardinality and vector dimensions need explicit validation before persistence.
- Re-ingestion identity/update semantics are not defined; duplicate records may be created.
- The `--clean` option deletes chat messages and sessions as well as indexed documents/chunks.
- Dedicated extractor and end-to-end ingestion regression tests remain to be added.

### Agent, sessions, and streaming
- Only user messages and final assistant text are persisted; tool-call and tool-result messages are not replayed.
- A fixed 10-message history is not a token-budget guarantee.
- Session identifiers are not a substitute for authentication or authorization; multi-user ownership enforcement is not established.
- Backend stream cancellation, disconnect, retry, and incomplete-turn persistence semantics need further testing.
- Tool activity is currently mixed into the displayed streamed response rather than presented in a separate diagnostics surface.

### Production readiness
- The health endpoint checks database connectivity; the `llm_connection` response field is currently hard-coded and should not be interpreted as a verified provider health check.
- The current CORS configuration is permissive and must be restricted before public deployment.
- Authentication, authorization, rate limiting, and comprehensive request/error hardening are not established.
- Schema initialization is not a versioned migration system; the SQL schema file contains destructive `DROP TABLE` statements and must not be used as a routine production migration.
- Structured request tracing, tool-level metrics, token/cost tracking, and production observability remain to be implemented.
- The system prompt still needs alignment with the intended general enterprise/technical knowledge domain.

The engineering backlog prioritizes a repeatable evaluation baseline, reliable evidence and failure semantics, ingestion correctness, and proportionate production hardening. More infrastructure is not automatically better; proposed changes should be justified by evidence and project needs.

## Security and operational notes

- Use only API keys and infrastructure that you own or are authorized to use.
- Keep real credentials in local environment configuration or an appropriate secret manager; never commit them.
- The example database credentials are not suitable for production.
- The current API does not establish a complete authentication/authorization boundary.
- Do not expose the development service publicly with permissive CORS and no authentication.
- Treat ingested document text as untrusted input. Retrieved content must not be treated as trusted instructions for the agent.
- Back up the PostgreSQL volume before any destructive maintenance.
- Do not manually execute `sql/schema.sql` against an existing database without inspecting its destructive statements and understanding the consequences.

## License

MIT License. See [LICENCE](LICENCE).
