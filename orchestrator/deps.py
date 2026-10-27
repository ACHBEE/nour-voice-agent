"""Run-scoped dependencies injected into every node via LangGraph's Runtime context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, time
from typing import Protocol

from orchestrator.booking import Booking, CallbackRequest
from orchestrator.llm import LLMClient
from orchestrator.schedule import clinic_today
from rag_service.retrieval import SearchResult


class RagSearch(Protocol):
    """Implemented by rag_service.client.RagClient (gRPC)."""

    async def search(self, query: str, top_k: int = 5, language: str = "") -> SearchResult: ...


class Store(Protocol):
    """Implemented by orchestrator.store.BookingStore (Postgres)."""

    async def slot_free(self, department: str, day: date, at: time) -> bool: ...
    async def create_booking(self, booking: Booking, call_id: str) -> int: ...
    async def create_callback(self, request: CallbackRequest, call_id: str) -> int: ...


@dataclass
class Deps:
    llm: LLMClient
    rag: RagSearch
    store: Store
    call_id: str = "local"
    today: Callable[[], date] = field(default=clinic_today)
