# Design decisions

Short notes on each tradeoff, written as the project is built.

## Phase 0: Foundations

**LiveKit Agents pinned to 1.8.9.** The SDK moves fast (several APIs used in early-2026 tutorials are
already deprecated), so the exact version is pinned and the code was checked against the installed
package source rather than blog posts. Upgrades are deliberate, not accidental.

**Current APIs only.** The starter used three APIs that are now deprecated:

| Old | Current | Why it matters |
|---|---|---|
| `AgentSession(turn_detection=...)` | `turn_handling=TurnHandlingOptions(turn_detection=...)` | One place for endpointing, interruption and preemptive-generation config |
| `livekit.plugins.turn_detector.MultilingualModel` | `livekit.agents.inference.TurnDetector()` | Audio-based end-of-turn model, cloud `v1` in dev with automatic fallback to local `v1-mini`; thresholds ship for `ar`, `fr`, `en` |
| `metrics_collected` + `UsageCollector` | `conversation_item_added` -> `ChatMessage.metrics`, `session.usage` | Latency is recorded **per conversational turn** (EOT delay, LLM TTFT, TTS TTFB, end-to-end) instead of per component event, which is exactly what p50/p95 reporting needs |

**LiveKit Inference for STT/LLM/TTS.** One set of LiveKit credentials covers every model, and model
IDs are env-configurable (`STT_MODEL`, `LLM_MODEL`, `TTS_MODEL`), so providers can be A/B tested
for Arabic quality and latency without code changes.

**Open question: STT for Darja.** Deepgram `nova-3:multi` code-switching is not documented to
include Arabic. Candidates are `openai/gpt-live-transcribe:multi` (Arabic + multi listed) and
`deepgram/nova-3:ar`. This will be decided by measured WER/latency on real Darja/French audio,
not by assumption.

**Paths resolved from the repo root**, so the agent behaves the same from the CLI, Docker and CI.
Blocking file writes inside async tools are pushed to a thread so they never stall the audio loop.

## Phase 1: Knowledge base and database

**15 short docs, deliberately mixed-language** (5 FR, 5 AR including 1 Darja FAQ, 5 EN). Real
Algerian clinic info is scattered across French and Arabic, and callers ask in Darja, so the
retriever must work **cross-lingually** (a Darja question hitting a French price list). Each fact
lives in one "home" doc, with a few overlaps (prices in FR + Darja) to test consistency. Some
things are deliberately absent (MRI, ophthalmology, cesarean prices) to evaluate "I don't know".

**Markdown + tiny front matter** (`doc_id`, `lang`, `dialect`, `category`). Headings give natural
chunk boundaries, and the metadata enables language filters and per-language full-text configs.
No YAML dependency: the loader is 20 lines and validated by Pydantic.

**Embedding model: `intfloat/multilingual-e5-small` (384 dims).** Alternatives: `bge-m3` (1024 d,
stronger on Arabic but ~2 GB, slower CPU queries), OpenAI `text-embedding-3-small` (paid API,
breaks the "evals run free in CI" goal). e5-small runs on CPU in CI, is fast enough for a voice
turn budget, and covers Arabic and French. If the Phase 5 hit-rate on Darja queries is weak,
`bge-m3` is the upgrade path (a dimension change + re-ingest).

**Hybrid storage in one table.** Each chunk has `embedding vector(384)` and a generated
`tsvector`. Vectors handle paraphrase and cross-lingual matches; full-text handles exact tokens
that embeddings blur (doctor names, "CNAS", "Edahabia", prices). Both are fused with RRF in
Phase 2. Keeping them in Postgres means one system to run, transactional with bookings.

**Per-language text-search config.** `ts_config` is stored per chunk (`french`, `english`,
`arabic` when the server has it, otherwise `simple`), and `tsv` is a generated column. The
alternative of one `simple` config everywhere loses French/English stemming.

**HNSW over IVFFlat.** No training step, so it can be built on an empty table at init, and it has
better recall at this scale. `m=16, ef_construction=64` are the pgvector defaults; tuning is
pointless for ~100 chunks.

**Bookings enforced at two layers.** Pydantic validates in the app (Phase 3), and the DB keeps
`CHECK` constraints (department enum, phone format) plus a partial unique index that blocks
double-booking a slot. LLM-generated structured output should never be the last line of defense.

**`pgvector/pgvector:0.8.7-pg17`, host port 5433.** Pinned image; pg17 rather than pg18 because
pg18 changed the data directory layout for volumes. Port 5433 avoids clashing with a local
Postgres.

## STT for Darja: measure, don't guess

**Observed problem.** With `deepgram/nova-3:multi`, «واش تخدمو نهار الجمعة؟» was transcribed as
«ويستخدمون هال الجمعه»: "Friday" survived, the Darja verbs didn't. The LLM still recovered the
meaning, but that's luck unless we design for it.

**Two-layer fix.**
1. *Prompt for noisy input:* the system prompt tells the LLM that transcripts come from ASR that
   is often wrong for Darja (dialect words turned into similar-sounding MSA, French written in
   Arabic letters), to interpret by sound and context, and to ask one short clarifying question
   rather than guess. It also asks for Darja replies to Darja callers.
2. *Pick the STT model with data:* `evals/stt/benchmark.py` streams the same recorded phrases
   (Darja, Darja+French, French, English) through every Arabic-capable LiveKit Inference STT
   model in real time and reports WER, CER and end-of-speech-to-final-transcript latency.

**WER over CER as the headline metric.** On the example above CER is only 0.17 (most letters
survive) while WER is 0.75 (3 of 4 words wrong), and WER is much closer to whether the meaning
got through. Both are computed after normalizing spelling variants that Darja has no standard
for (alef/hamza forms, taa marbuta, diacritics, Latin accents).

**Limits.** One speaker, 11 phrases: enough to rank providers for this demo, not a general claim.
Recordings stay out of git (it's a person's voice).

**Result (2026-10-27, 8 phrases, one speaker, laptop mic).** Full table:
[evals/stt/results.md](evals/stt/results.md).

| Model | WER Darja | WER French | WER English | WER all | p50 latency |
|---|---|---|---|---|---|
| openai/gpt-live-transcribe:ar | 0.88 | **0.17** | 0.71 | **0.78** | 1.50 s |
| speechmatics/linden-1:ar | **0.67** | 1.00 | 1.00 | **0.78** | 0.85 s |
| deepgram/nova-3:multi (old default) | 1.00 | 0.33 | 0.71 | 0.87 | 0.57 s |

**Decision: `openai/gpt-live-transcribe:ar`.** The top two tie on average WER, so the call is
made on *coverage*: a receptionist line can't fail a whole language. Speechmatics is best on
pure Darja but scores 1.00 on French; Deepgram `multi` returned Turkish/Persian for Darja;
Deepgram `ar-DZ` fails French and English entirely. gpt-live-transcribe is the only model that is
usable in all three, with the best CER (0.57) and zero failed requests.
**Cost:** ~0.9 s more end-of-speech-to-transcript latency than Deepgram. Phase 5 measures the
real per-turn effect; if it's too slow, preemptive generation and a filler phrase are the levers.

**Honest caveats.** (1) 8 phrases, one speaker: the differences between the top models are within
noise; the language-failure pattern is the robust signal. (2) The laptop mic delivers almost no
energy above 4 kHz (telephone-band audio), which hurts every model on Arabic fricatives, but is
also what real phone calls sound like. (3) Absolute Darja WER is high for every provider:
Darja ASR is an unsolved problem off the shelf, which is why the prompt is designed to tolerate
bad transcripts and ask for clarification.

## Phase 2: RAG service over gRPC

**Separate service, gRPC between agent and RAG.** Retrieval has its own lifecycle and resource
profile: a ~470 MB embedding model and a DB pool, warm-loaded once. The voice agent should stay
lean and restart fast. gRPC over REST because the contract is typed and versioned
(`nour.rag.v1`), stubs are generated (`scripts/gen_proto.py`, never hand-edited), binary
framing is cheap, and the standard gRPC health service gives Docker a real readiness probe.
Measured round trip: **~40-65 ms** (embed ~18 ms, Postgres + fusion ~10-16 ms).

**fastembed (ONNX Runtime) instead of sentence-transformers.** Same e5 weights, but no PyTorch:
~470 MB instead of 2+ GB of dependencies, faster cold start, smaller Docker image, cheaper CI.
`multilingual-e5-small` isn't in fastembed's catalog, so it's registered as a custom model from
the official repo's ONNX export. e5's `query:` / `passage:` prefixes are applied. Query embedding
takes ~18 ms on a laptop CPU.

**Chunking: one chunk per markdown section, ~150 chars median, contextual header.** The docs are
written as short sections that each answer one question, so headings are natural boundaries;
fixed-size windows would cut facts in half. Each chunk is prefixed with
`<doc title> › <section>` so "2000 DZD" carries "Tarifs › Consultations" with it, for both the
embedding and the LLM. 15 docs → 88 chunks. Long sections (>900 chars) split on paragraphs.

**Keyword search details.** (1) Each chunk is matched with its own language's stemmer, chosen at
ingest. (2) Query words that are a stopword in *any* language are dropped first; otherwise English
"the" survives as a junk term against French/Arabic chunks. (3) Terms are OR-ed: `plainto_tsquery`
ANDs every word, which almost never matches a spoken question. `ts_rank_cd` still rewards chunks
matching more terms.

**RRF with k=60.** It fuses ranks, not scores, so cosine similarity and `ts_rank_cd` (incomparable
scales) need no normalization. **Known weakness, measured in Phase 5:** keyword search is
language-bound, so for a cross-lingual query its list fills with same-language chunks, and RRF's
reward for "in both lists" can push the right foreign-language chunk down. Examples: "Who is the
pediatrician?" (vector #1, hybrid #4); "Do you accept the Chifa card?" (the Arabic insurance doc
is outside the top 3). Levers to test with data: weighted RRF, cross-lingual aliases in docs,
or a bigger embedding model.

**Incremental, idempotent ingestion.** Content hash per document: unchanged docs are skipped,
changed docs are re-chunked and re-embedded in one transaction, deleted files are removed.
`--force` re-embeds everything (needed after changing the embedding model).

**Windows note.** psycopg's async mode can't use Windows' default Proactor event loop; the pool
then retries silently forever. `rag_service/aio.run()` selects the selector loop on Windows, and
pool opening has a timeout so a bad DB fails in seconds instead of hanging.

## Phase 3: LangGraph orchestrator

**One graph run per caller turn: `route -> one branch -> END`.** Branches: retrieve + grounded
answer, smalltalk answer, booking form, callback form, safety. Alternative: a ReAct-style agent
looping over tools. A fixed graph has a bounded number of LLM calls per turn (2), is testable
node by node, and the safety branch can't be talked out of its rules. Multi-turn behaviour
lives in state, not in a loop.

**Per-call memory in a checkpointer (`thread_id` = call / room id).** The transcript and any
half-filled booking or callback form survive across turns, including a clinic question asked
in the middle of a booking. `InMemorySaver` because a call lives in one agent process and dies
with it; durable state (bookings, callbacks) goes to Postgres. `langgraph-checkpoint-postgres`
would only matter if calls could move between workers.

**LLM extracts, code decides.** The booking node asks the LLM for a strict-JSON-schema
extraction (`BookingExtraction`, Pydantic) and nothing else. Phone normalisation (+213, Arabic
digits), department days and hours, date range and Friday closure are deterministic checks in
`orchestrator/booking.py`, run on partial forms so Nour objects to "pediatrics on Tuesday" as
soon as both are known. The final `Booking` Pydantic model is the gate before the INSERT, and a
partial unique index in Postgres is the gate against concurrent double-booking (`SlotTaken`).

**Read-backs and safety replies are templates, not generated.** What the caller confirms is
exactly what gets saved, and a medical-advice or emergency reply can't drift. Cost: less natural
phrasing, and 3 languages to maintain by hand (`phrases.py`, Darja for Arabic). Free-text
answers to clinic questions stay LLM-generated, streamed, and constrained to retrieved chunks.

**Refused values are remembered.** The extractor re-reads the conversation each turn, so it
would return the refused "Tuesday" again forever. Refused values are stored per field and
ignored on re-extraction; Nour then simply asks for that field again.

**Router rewrites the search query.** The routing call also produces a standalone,
ASR-corrected question ("et pour le cardiologue ?" -> "prix consultation cardiologue"), which
is what goes to retrieval, at no extra LLM call.

**Graceful degradation instead of errors.** Router failure: treat as a clinic question and guess
the language from the script. Retrieval failure: answer from no documents, which makes the
prompt say "I don't have that information" and offer a callback. Form or DB failure: keep the
form, apologise, offer a callback. All covered by offline tests with a scripted fake LLM.

**LLM via LiveKit Inference, behind a 2-method interface.** `structured()` and `stream()` in
`orchestrator/llm.py`; same gateway and keys as the voice agent, so no extra provider key.
LangChain chat models aren't needed for a LangGraph graph; nodes are plain async functions.

**Measured cost (text chat, laptop, gpt-4.1-mini):** router 0.5-1.1 s warm (first call of a
process 1.5-4 s), retrieval 30-90 ms, answer TTFT 0.4-1.0 s. The router call sits before the
first spoken word, which matters on a phone line. Levers to measure in Phase 5: a smaller router
model (gpt-4.1-nano), retrieving in parallel with routing, prewarming, and the filler phrase.
Known grounding gap: "Do you do MRI?" was answered correctly, but the only doc that says so
(Arabic `departments`) wasn't in the top 5, the cross-lingual retrieval weakness from Phase 2.
