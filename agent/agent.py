"""
Nour — multilingual voice receptionist for a fictional Algerian clinic (Clinique El Amal).

LiveKit Agents handles the real-time audio: VAD, multilingual turn detection, noise
cancellation, streaming STT -> (reply) -> TTS, and interruptions. The reply itself comes from
the LangGraph orchestrator (orchestrator/), which routes each caller turn to a grounded RAG
answer (RAG service over gRPC), a booking or callback form (Postgres), or a fixed safety phrase.

Per-turn latency goes to latency_log.jsonl: LiveKit's STT/LLM/TTS timings per message, plus one
"orchestrator" row per turn with route, retrieval and answer timings.
"""

import json
import logging
import os
import sys
from collections.abc import AsyncIterable
from datetime import UTC, datetime
from pathlib import Path

# Resolve paths from the repo root so the agent works from any cwd (CLI, Docker, CI), and make
# the repo packages importable when run as `python agent/agent.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from livekit import agents  # noqa: E402
from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    FlushSentinel,
    JobContext,
    ModelSettings,
    TurnHandlingOptions,
    inference,
    llm,
    room_io,
)
from livekit.plugins import noise_cancellation, silero  # noqa: E402

from agent.turn_stream import TurnStats, filler_for, with_filler  # noqa: E402
from orchestrator.booking import Booking, CallbackRequest  # noqa: E402
from orchestrator.graph import Orchestrator  # noqa: E402
from orchestrator.llm import LiveKitLLM  # noqa: E402
from orchestrator.store import BookingStore  # noqa: E402
from rag_service.client import RagClient  # noqa: E402
from rag_service.config import RAG_GRPC_ADDR  # noqa: E402

load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

logger = logging.getLogger("nour")
logger.setLevel(logging.INFO)

LATENCY_LOG = Path(os.getenv("LATENCY_LOG", ROOT / "latency_log.jsonl"))

# Per-turn latency fields from ChatMessage.metrics (livekit-agents >= 1.5).
# User turns carry STT/endpointing delays; assistant turns carry LLM/TTS timings.
# With the orchestrator, "llm_node_ttft" is the time to Nour's first spoken text (filler included).
TURN_METRIC_KEYS = (
    "transcription_delay",
    "end_of_turn_delay",
    "on_user_turn_completed_delay",
    "llm_node_ttft",
    "tts_node_ttfb",
    "e2e_latency",
)
ORCH_METRIC_KEYS = ("route_ms", "retrieval_ms", "rag_server_ms", "answer_ttft_ms")

# Models are configurable so you can swap providers without touching code.
# Defaults use LiveKit Inference (only your LiveKit keys needed).
# Chosen by evals/stt/benchmark.py: the only model usable in Darja, French and English.
STT_MODEL = os.getenv("STT_MODEL", "openai/gpt-live-transcribe:ar")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")  # used by the orchestrator's nodes
TTS_MODEL = os.getenv("TTS_MODEL", "cartesia/sonic-3")

GREETING = "Bonjour, السلام عليكم، ici Nour de la Clinique El Amal, comment puis-je vous aider ?"

# Not sent to any LLM: llm_node is replaced by the orchestrator, whose prompts live in
# orchestrator/prompts.py. Kept because Agent requires instructions.
INSTRUCTIONS = "Nour, voice receptionist of Clinique El Amal. Replies come from the orchestrator."


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def latest_user_text(chat_ctx: llm.ChatContext) -> str:
    for item in reversed(chat_ctx.items):
        if getattr(item, "type", None) == "message" and item.role == "user":
            return (item.text_content or "").strip()
    return ""


class UnavailableStore:
    """Used when Postgres is down at call start: forms then apologise and offer a callback
    instead of the whole call failing (the nodes catch these errors)."""

    async def slot_free(self, *_: object) -> bool:
        raise ConnectionError("booking database unavailable")

    async def create_booking(self, booking: Booking, call_id: str) -> int:
        raise ConnectionError("booking database unavailable")

    async def create_callback(self, request: CallbackRequest, call_id: str) -> int:
        raise ConnectionError("booking database unavailable")


class Receptionist(Agent):
    def __init__(self, orchestrator: Orchestrator, call_id: str) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self.orchestrator = orchestrator
        self.call_id = call_id

    async def on_enter(self) -> None:
        # Fixed text: no LLM round trip before the first word of the call.
        self.session.say(GREETING)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str | FlushSentinel]:
        """One caller turn through the LangGraph orchestrator, streamed to TTS."""
        text = latest_user_text(chat_ctx)
        if not text:
            return
        stats = TurnStats()
        chunks = self.orchestrator.turn(self.call_id, text)
        async for piece in with_filler(chunks, filler_for(text), stats=stats):
            yield piece
        await self._log_turn(stats)

    async def _log_turn(self, stats: TurnStats) -> None:
        s = await self.orchestrator.state(self.call_id)
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "room": self.call_id,
            "role": "orchestrator",
            "intent": s.get("intent"),
            "language": s.get("language"),
            **{k: s.get(k) for k in ORCH_METRIC_KEYS},
            "first_text_ms": stats.first_text_ms,
            "total_ms": stats.total_ms,
            "filler": stats.filler,
            "docs": [c["doc_id"] for c in s.get("chunks") or []],
        }
        _append_line(LATENCY_LOG, json.dumps(row, ensure_ascii=False))
        logger.info(
            "turn intent=%s route=%.0fms retrieval=%.0fms first_text=%sms filler=%s",
            row["intent"],
            row["route_ms"] or 0,
            row["retrieval_ms"] or 0,
            row["first_text_ms"],
            row["filler"],
        )


def prewarm(proc: agents.JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer()
server.setup_fnc = prewarm


async def _open_store() -> BookingStore | UnavailableStore:
    try:
        return await BookingStore.open(connect_timeout=5)
    except Exception:  # noqa: BLE001 - keep answering questions without bookings
        logger.exception("Postgres unavailable: bookings and callbacks disabled for this call")
        return UnavailableStore()


@server.rtc_session(agent_name="nour")
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    call_id = ctx.room.name

    store = await _open_store()
    rag = RagClient(RAG_GRPC_ADDR)
    orch_llm = LiveKitLLM(LLM_MODEL)
    orchestrator = Orchestrator(orch_llm, rag, store)

    session = AgentSession(
        stt=STT_MODEL,
        llm=LLM_MODEL,  # required for replies to run at all; llm_node never calls it
        tts=TTS_MODEL,
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            # Speculative replies before the turn ends would run the stateful graph (transcript,
            # booking form, DB writes) on text that may still change. See DESIGN.md.
            preemptive_generation={"enabled": False},
        ),
    )

    # ---- Per-turn latency (the numbers for your README) ----
    # metrics_collected is deprecated; per-turn timings now live on ChatMessage.metrics.
    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if getattr(item, "type", None) != "message":
            return
        m = item.metrics or {}
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "room": ctx.room.name,
            "role": item.role,
            "interrupted": item.interrupted,
        }
        row.update({k: m[k] for k in TURN_METRIC_KEYS if k in m})
        if len(row) > 4:  # only log turns that actually carry timings
            _append_line(LATENCY_LOG, json.dumps(row))

    async def _shutdown() -> None:
        logger.info("usage summary: %s", session.usage)
        await orchestrator.end_call(call_id)
        await rag.close()
        await orch_llm.aclose()
        if isinstance(store, BookingStore):
            await store.close()

    ctx.add_shutdown_callback(_shutdown)

    await session.start(
        room=ctx.room,
        agent=Receptionist(orchestrator, call_id),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
