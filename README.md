# Nour — Multilingual Real-Time Voice Receptionist (LiveKit)

A production-style voice AI agent that answers calls for a fictional Algerian clinic
in **Arabic (incl. Darja), French and English**, answers questions from clinic documents,
and books appointments with validated structured outputs.

> Status: voice pipeline, clinic knowledge base (15 docs, FR/AR/EN) and pgvector schema done. RAG over gRPC + LangGraph in progress.

## Architecture (target)

```
Caller (web / phone)
   │  WebRTC
LiveKit Agent ── VAD + multilingual turn detection + noise cancellation
   │  STT → LLM → TTS (streaming, interruptible)
   ▼
LangGraph orchestrator ──► gRPC ► RAG service (pgvector, hybrid search, rerank)
                       ──► book_appointment (Pydantic-validated)
                       ──► human callback / handoff
Metrics: per-turn latency (EOU, TTFT, TTFB) → latency_log.jsonl
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env.local                            # add LiveKit keys
docker compose up -d                                  # Postgres + pgvector, loads db/schema.sql
python -m livekit.agents download-files               # VAD + turn detector models
python agent/agent.py console                         # talk in your terminal
python agent/agent.py dev                             # connect to LiveKit Cloud / Playground
pytest && ruff check .                                # tests + lint
```

## Repo layout

```
agent/            LiveKit voice agent (STT -> LLM -> TTS, tools, latency logging)
orchestrator/     LangGraph graph: intent routing -> RAG answer / booking / callback
rag_service/      gRPC RAG service: ingestion, embeddings, hybrid retrieval
proto/            rag.proto + generated stubs
data/clinic_docs/ fictional clinic documents (FR / AR / EN)
db/               schema.sql (pgvector, HNSW, full-text)
evals/            eval set, scoring + latency report scripts
tests/            unit tests
```

## Results

| Metric | Value |
|---|---|
| p50 latency to first audio | _tbd_ |
| p95 latency to first audio | _tbd_ |
| RAG answer accuracy (eval set) | _tbd_ |

## Design decisions
See [DESIGN.md](DESIGN.md).
