# CLAUDE.md — Nour Voice Agent

## What this project is
Portfolio project for a Senior AI Engineer / AI Lead application (Algeria). Deadline: 3 days.
"Nour" is a real-time multilingual voice receptionist for a FICTIONAL clinic, Clinique El Amal (El Oued, Algeria).
Callers speak Arabic (incl. Algerian Darja), French or English. Nour answers from clinic docs via RAG,
books appointments (Pydantic-validated), and requests a human callback when it can't help.

Every component must visibly demonstrate a skill from the job post: LiveKit voice agents (hard requirement),
Python, LangGraph, RAG + embeddings + semantic search, pgvector, microservices, gRPC/Protobuf,
structured outputs, prompt engineering, model evaluation, MLOps basics (Docker, CI).

## Architecture
Caller → LiveKit Agent (VAD, multilingual turn detection, noise cancellation, STT → LLM → TTS, interruptible)
→ LangGraph orchestrator (answer_from_docs / book_appointment / human_callback)
→ gRPC → RAG service (Postgres + pgvector, hybrid vector + full-text search, RRF fusion)
Bookings → Postgres. Metrics → latency_log.jsonl. Evals → evals/, run in GitHub Actions.

## Layout
- agent/ — LiveKit agent
- orchestrator/ — LangGraph graph, state, nodes
- rag_service/ — gRPC server, ingestion, retrieval
- proto/ — .proto files (generated stubs are produced by a script, not hand-edited)
- data/clinic_docs/ — fictional clinic docs (FR / AR / EN)
- db/ — schema.sql
- evals/ — questions.jsonl, run_evals.py, latency_report.py
- tests/
- docker-compose.yml, .github/workflows/ci.yml, README.md, DESIGN.md

## Rules
- Verify before you code: LiveKit Agents and LangGraph change fast. Check the current docs
  (docs.livekit.io, livekit/agents repo, langchain-ai/langgraph) or the installed package source
  before using an API. Never invent APIs, model IDs or parameters.
- Build on existing code; don't rewrite working parts from scratch.
- Simple and working beats impressive and broken. No Kubernetes, no custom frontend, no fine-tuning.
- Work phase by phase. After each phase: run it, run tests, show the output, then commit.
  Don't start the next phase until the current one works.
- Secrets live only in .env.local (gitignored). Never print or commit keys. If a key is missing, stop and ask.
- I (Houcine) test anything involving voice/audio myself. Tell me exactly what to say and what to listen for.
- Record every meaningful tradeoff in DESIGN.md as you make it (2-4 lines each: decision, alternatives, why).
- README numbers must be real measured results, never placeholders presented as results.

## Voice agent behavior
- Replies are 1-2 sentences, spoken style: no lists, no markdown, no emojis.
- Reply in the caller's language; handle code-switching between Darja, French and English.
- Only state clinic facts from retrieved documents. If not found: say so and offer a human callback.
- Never give medical advice or diagnoses. Emergencies → go to the emergency unit or call 14.
- Confirm booking details back to the caller before calling the booking tool.
- Keep perceived latency low: short filler phrase if retrieval is slow; log retrieval time.

## Code style
- Python 3.11+, type hints, Pydantic v2 for all structured data.
- Small modules, clear names, docstrings on tools (LLMs read them).
- pytest for tests. ruff for lint.
- Logging via the standard logging module; no print() in services.

## Commands (update as they're created)
- Run agent locally: `python agent/agent.py console` (needs Postgres + RAG server running)
- Run agent with LiveKit Cloud: `python agent/agent.py dev`
- Download agent models: `python -m livekit.agents download-files`
- Start infra only: `docker compose up -d postgres` (Postgres + pgvector on localhost:5433)
- Whole stack in Docker: `docker compose up -d --build` (postgres -> ingest -> rag -> agent; agent needs .env.local)
- Regenerate gRPC stubs: `python scripts/gen_proto.py`
- Ingest docs into pgvector: `python -m rag_service.ingest` (`--force` after changing the embedding model)
- RAG server / client: `python -m rag_service.server`, `python -m rag_service.client "query"`
- Orchestrator text chat (needs Postgres + RAG server): `python -m orchestrator.chat` (`-v` for node logs, or pass turns as args)
- Tests: `pytest` (DB-backed tests skip if Postgres isn't up)
- Evals: `python evals/run_evals.py` (`--retrieval-only` needs no LLM; `--router-model`, `--keyword-weight`, `--tag`)
- Latency report: `python evals/latency_report.py --since <ISO date>` (reads latency_log.jsonl)

## Progress
- [x] Phase 0 — Setup, restructure, voice agent working (live Darja call verified)
- [x] Phase 1 — Clinic docs + DB schema + docker-compose
- [x] Phase 2 — RAG service over gRPC
- [x] Phase 3 — LangGraph orchestrator + bookings
- [x] Phase 4 — Agent ↔ orchestrator integration (verified in LiveKit text mode; live voice check by Houcine pending)
- [x] Phase 5 — Evals + latency metrics (voice e2e latency on the current pipeline still needs a call)
- [x] Phase 6 — Docker, CI, README, DESIGN.md, demo script (CI runs once pushed to GitHub; demo video + voice latency numbers pending a recorded call)
Update this checklist at the end of each phase.
