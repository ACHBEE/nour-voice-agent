"""Graph state and the structured-output schemas the LLM fills.

State values are plain JSON types (str / dict / list) so the checkpointer stores them as-is.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from orchestrator.schedule import Department

Lang = Literal["ar", "fr", "en"]
Intent = Literal[
    "clinic_info", "book_appointment", "human_callback", "emergency", "medical_advice", "smalltalk"
]


class Msg(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Form(TypedDict):
    """A booking or callback being collected over several turns."""

    fields: dict[str, str | None]
    rejected: dict[str, list[str]]  # values already refused, so they aren't re-extracted
    awaiting_confirmation: bool


class CallState(TypedDict, total=False):
    messages: Annotated[list[Msg], operator.add]  # whole call transcript, appended per turn
    user_text: str  # this turn's caller utterance
    # set by the router, every turn
    intent: Intent
    language: Lang
    search_query: str
    # set by the branch that handled the turn
    chunks: list[dict]
    reply: str
    booking: Form | None
    callback: Form | None
    booking_id: int | None
    callback_id: int | None
    # per-turn timings (ms), for latency logs
    route_ms: float
    retrieval_ms: float
    rag_server_ms: float
    answer_ttft_ms: float


# ---------- LLM structured outputs ----------


class RouteDecision(BaseModel):
    intent: Intent
    language: Lang = Field(description="Language of the caller's latest message")
    search_query: str = Field(
        description="clinic_info only: standalone search question in the caller's language; else empty"
    )


class BookingExtraction(BaseModel):
    department: Department | None
    appointment_date: str | None = Field(description="YYYY-MM-DD")
    appointment_time: str | None = Field(description="HH:MM, 24-hour clock")
    patient_name: str | None
    phone: str | None = Field(description="Digits as said by the caller")
    confirmation: Literal["yes", "no", "none"] = Field(
        description="Does the caller's latest message confirm the details read back to them?"
    )


class CallbackExtraction(BaseModel):
    phone: str | None = Field(description="Digits as said by the caller")
    reason: str | None = Field(description="What the caller needs, one short English sentence")
    confirmation: Literal["yes", "no", "none"]
