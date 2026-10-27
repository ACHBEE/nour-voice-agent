# Nour — Multilingual Real-Time Voice Receptionist (LiveKit)

A production-style voice AI agent that answers calls for a fictional Algerian clinic
in **Arabic (incl. Darja), French and English**, answers questions from clinic documents,
and books appointments with validated structured outputs.

> Status: voice pipeline, knowledge base, and hybrid RAG service over gRPC done. LangGraph orchestrator in progress.

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
python -m rag_service.ingest                          # chunk + embed clinic docs into pgvector
python -m rag_service.server                          # RAG gRPC service on :50051
python -m rag_service.client "Combien coûte une consultation ?"   # try a search
python -m livekit.agents download-files               # VAD + turn detector models
python agent/agent.py console                         # talk in your terminal
python agent/agent.py dev                             # connect to LiveKit Cloud / Playground
pytest && ruff check .                                # tests + lint
```

## Repo layout

```
agent/            LiveKit voice agent (STT -> LLM -> TTS, tools, latency logging)
orchestrator/     LangGraph graph: intent routing -> RAG answer / booking / callback
rag_service/      gRPC RAG service: ingestion, embeddings, hybrid retrieval (pb/ = generated stubs)
proto/            rag.proto (stubs: python scripts/gen_proto.py)
data/clinic_docs/ fictional clinic documents (FR / AR / EN)
db/               schema.sql (pgvector, HNSW, full-text)
evals/            eval set, scoring + latency report scripts
tests/            unit tests
```

## Results

### Speech-to-text for Darja / French / English (measured)

8 recorded phrases, one speaker, laptop mic; lower is better. Full table and method:
[evals/stt/results.md](evals/stt/results.md), [DESIGN.md](DESIGN.md#stt-for-darja-measure-dont-guess).

| STT model (LiveKit Inference) | WER Darja | WER French | WER all | p50 latency |
|---|---|---|---|---|
| **openai/gpt-live-transcribe:ar** (chosen) | 0.88 | 0.17 | 0.78 | 1.50 s |
| speechmatics/linden-1:ar | 0.67 | 1.00 | 0.78 | 0.85 s |
| deepgram/nova-3:multi (previous default) | 1.00 | 0.33 | 0.87 | 0.57 s |

### Voice pipeline and RAG

| Metric | Value |
|---|---|
| p50 / p95 end-to-end turn latency | not measured yet (Phase 5) |
| RAG search round trip over gRPC (laptop CPU) | ~40-65 ms (embed ~18 ms, search ~10-16 ms) |
| Retrieval hit rate@k | not measured yet (Phase 5) |
| Answer correctness (LLM-as-judge) | not measured yet (Phase 5) |

## Design decisions
See [DESIGN.md](DESIGN.md).
