"""
Nour — multilingual voice receptionist for a fictional Algerian clinic (Clinique El Amal).

Day 1: STT -> LLM -> TTS voice pipeline on LiveKit Agents, with
  - multilingual turn detection + noise cancellation
  - interruption handling (on by default in AgentSession)
  - a first function tool (appointment booking, structured with Pydantic)
  - per-turn latency metrics logged to latency_log.jsonl

Day 2 replaces the stub knowledge + booking with LangGraph + pgvector RAG over gRPC.
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from pydantic import BaseModel, Field, ValidationError

# Resolve paths from the repo root so the agent works from any cwd (CLI, Docker, CI).
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

logger = logging.getLogger("nour")
logger.setLevel(logging.INFO)

LATENCY_LOG = Path(os.getenv("LATENCY_LOG", ROOT / "latency_log.jsonl"))
BOOKINGS_FILE = Path(os.getenv("BOOKINGS_FILE", ROOT / "bookings.jsonl"))

# Per-turn latency fields from ChatMessage.metrics (livekit-agents >= 1.5).
# User turns carry STT/endpointing delays; assistant turns carry LLM/TTS timings.
TURN_METRIC_KEYS = (
    "transcription_delay",
    "end_of_turn_delay",
    "on_user_turn_completed_delay",
    "llm_node_ttft",
    "tts_node_ttfb",
    "e2e_latency",
)

# Models are configurable so you can swap providers without touching code.
# Defaults use LiveKit Inference (only your LiveKit keys needed).
# Check the exact model IDs available in your LiveKit Cloud dashboard.
# Chosen by evals/stt/benchmark.py: the only model usable in Darja, French and English.
STT_MODEL = os.getenv("STT_MODEL", "openai/gpt-live-transcribe:ar")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "cartesia/sonic-3")


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------- Structured output for bookings ----------
class Booking(BaseModel):
    patient_name: str = Field(..., min_length=2)
    phone: str = Field(..., min_length=8)
    department: str = Field(..., description="e.g. general, pediatrics, cardiology, dentistry")
    preferred_date: str = Field(..., description="ISO date YYYY-MM-DD")
    preferred_time: str = Field(..., description="HH:MM, 24h")
    language: str = Field("fr", description="Language the caller used: ar, fr, en")


# ---------- Temporary knowledge (replaced by RAG on Day 2) ----------
CLINIC_FACTS = """
Clinique El Amal (fictional), Rue des Palmiers, El Oued.
Hours: Saturday–Thursday 08:00–18:00. Closed Friday. Emergencies 24/7.
Departments: general medicine, pediatrics, cardiology, dentistry.
General consultation: 2000 DZD. Specialist consultation: 3500 DZD.
CNAS cards accepted for general medicine.
"""

INSTRUCTIONS = f"""
You are Nour, the voice receptionist of Clinique El Amal.
- Reply in the caller's language and register: if they speak Algerian Darja, answer in simple Darja; Modern Standard Arabic gets Arabic; French gets French; English gets English. Callers often mix Darja and French; that's normal.
- The caller's words reach you through automatic speech recognition, which is often wrong for Darja: dialect words may come out as similar-sounding standard Arabic words, and French words may be written in Arabic letters. Interpret the transcript by how it sounds and by context. If you genuinely can't tell what they asked, ask one short clarifying question instead of guessing.
- This is a phone call: keep answers short (1–2 sentences), no lists, no markdown, no emojis.
- Only state clinic facts from the knowledge below. If you don't know, say so and offer a human callback.
- To book an appointment, collect name, phone, department, date and time, confirm them back, then call book_appointment.
- Never give medical advice or diagnoses. For emergencies, tell the caller to come to the emergency unit or call 14.

Knowledge:
{CLINIC_FACTS}
"""


class Receptionist(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the caller briefly in French and Arabic, introduce yourself as Nour from Clinique El Amal, and ask how you can help."
        )

    @function_tool()
    async def book_appointment(
        self,
        context: RunContext,
        patient_name: str,
        phone: str,
        department: str,
        preferred_date: str,
        preferred_time: str,
        language: str = "fr",
    ) -> str:
        """Book a clinic appointment after the caller has confirmed all details.

        Args:
            patient_name: Full name of the patient.
            phone: Caller phone number.
            department: general, pediatrics, cardiology or dentistry.
            preferred_date: Date in YYYY-MM-DD format.
            preferred_time: Time in HH:MM 24h format.
            language: Language the caller is using (ar, fr or en).
        """
        try:
            booking = Booking(
                patient_name=patient_name,
                phone=phone,
                department=department,
                preferred_date=preferred_date,
                preferred_time=preferred_time,
                language=language,
            )
        except ValidationError as e:  # let the LLM ask the caller to correct it
            return f"Booking invalid, ask the caller to correct: {e}"

        await asyncio.to_thread(_append_line, BOOKINGS_FILE, booking.model_dump_json())
        logger.info("booking saved: %s", booking.model_dump())
        return (
            f"Booked {booking.department} on {booking.preferred_date} at {booking.preferred_time}."
        )

    @function_tool()
    async def request_human_callback(self, context: RunContext, phone: str, reason: str) -> str:
        """Use when the caller asks for a human or the question can't be answered.

        Args:
            phone: Caller phone number.
            reason: Short reason for the callback.
        """
        logger.info("human callback requested: %s / %s", phone, reason)
        return "Callback request registered; a staff member will call back today."


def prewarm(proc: agents.JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session(agent_name="nour")
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=STT_MODEL,
        llm=LLM_MODEL,
        tts=TTS_MODEL,
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
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

    async def _log_usage() -> None:
        logger.info("usage summary: %s", session.usage)

    ctx.add_shutdown_callback(_log_usage)

    await session.start(
        room=ctx.room,
        agent=Receptionist(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
