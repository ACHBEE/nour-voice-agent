# Nour — Multilingual Real-Time Voice Receptionist (LiveKit)

A production-style voice AI agent that answers calls for a fictional Algerian clinic
in **Arabic (incl. Darja), French and English**, answers questions from clinic documents,
and books appointments with validated structured outputs.

> Status: voice agent wired to the LangGraph orchestrator (grounded RAG answers over gRPC, bookings and callbacks in Postgres, safety phrases, filler on slow turns). Evals done (results below); Docker + CI next.

## Architecture (target)

```
Caller (web / phone)
   │  WebRTC
LiveKit Agent ── VAD + multilingual turn detection + noise cancellation
   │  STT → LLM → TTS (streaming, interruptible)
   ▼
LangGraph orchestrator ──► gRPC ► RAG service (pgvector, hybrid search + RRF)
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
python -m orchestrator.chat                           # text chat with the LangGraph orchestrator
python -m livekit.agents download-files               # VAD + turn detector models
python agent/agent.py console                         # talk in your terminal (RAG server must be running)
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

### Assistant quality: 38-case eval set (measured)

French, English, Darja and Modern Standard Arabic questions, many cross-lingual (an English
question whose answer is only in an Arabic document). Answers graded by an LLM judge
(`openai/gpt-4.1`, rubric in [evals/judge_rubric.md](evals/judge_rubric.md)) against a
reference; answer and router model `openai/gpt-4.1-mini`. Run: `python evals/run_evals.py`.

| Metric | Single search query | Multi-query retrieval + grounding rule |
|---|---|---|
| Retrieval hit@5 / MRR (22 answerable) | 77% / 0.54 | **100% / 0.91** |
| Answer correctness, LLM-as-judge (22 answerable) | 73% | **95%** |
| Answers grounded in retrieved chunks | 82% | **95%** |
| "Don't have that info" + callback offer (4 not in docs) | 75% | **100%** |
| Medical advice refused (4) | 100% | 100% |
| Emergencies routed to the fixed safety phrase (4) | 100% | 100% |
| Replies containing medical advice (all 38) | 0 | 0 |
| Intent accuracy (all 38) | 100% | 100% |
| Orchestrator time to first text, p50 / p95 | 1.09 / 1.49 s | 1.28 / 1.80 s |

One run each, so one case moves a rate by 4.5 points (22 cases) or 25 points (4 cases). Full
runs with every reply and judge reason: [evals/results/](evals/results/). How these choices
were made: [DESIGN.md, Phase 5](DESIGN.md#phase-5-evals-and-what-they-changed).

Retrieval alone, on the raw caller question (no LLM, runs in CI):

| Search mode | hit@1 | hit@5 | MRR |
|---|---|---|---|
| Hybrid (vector + full-text, RRF) | 55% | 77% | 0.60 |
| Vector only (multilingual-e5-small) | 64% | 82% | 0.70 |
| Full-text only | 50% | 50% | 0.50 |

### Latency

| Stage | Value |
|---|---|
| RAG search over gRPC (laptop CPU) | ~40-65 ms per query (embed ~18 ms, search ~10-16 ms) |
| Orchestrator, text (router + 4 parallel searches + answer TTFT) | p50 1.28 s, p95 1.80 s to first text |
| Voice end to end (caller stops -> Nour speaks) | not measured yet on the current pipeline: needs a call, then `python evals/latency_report.py --since <date>` |

## Design decisions
See [DESIGN.md](DESIGN.md).
