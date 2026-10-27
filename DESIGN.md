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
