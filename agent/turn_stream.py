"""Streams one orchestrator turn to TTS, with a filler phrase if the first words are slow.

Kept free of LiveKit session objects so it can be unit-tested with a fake orchestrator.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from livekit.agents import FlushSentinel

from orchestrator.nodes import guess_language
from orchestrator.state import Lang

logger = logging.getLogger("nour.turn")

# Say a filler if the orchestrator hasn't produced any text after this long. Typical turn:
# router ~0.6-1 s, retrieval ~50 ms, answer TTFT ~0.5 s. Templates (forms, safety) come in < 1 s.
FILLER_AFTER_S = float(os.getenv("FILLER_AFTER_MS", "1200")) / 1000

FILLERS: dict[Lang, str] = {
    "fr": "Un instant, je vérifie.",
    "en": "One moment, let me check.",
    "ar": "لحظة، نشوف.",
}


@dataclass
class TurnStats:
    first_text_ms: float | None = None  # first orchestrator text (excluding the filler)
    total_ms: float | None = None
    filler: bool = False


async def with_filler(
    chunks: AsyncIterator[str],
    filler: str,
    after_s: float = FILLER_AFTER_S,
    stats: TurnStats | None = None,
) -> AsyncIterator[str | FlushSentinel]:
    """Yield `chunks`; if the first one takes longer than `after_s`, yield `filler` first.

    The filler is followed by a FlushSentinel so TTS speaks it immediately instead of waiting
    for the rest of the sentence. If the caller interrupts, LiveKit closes this generator and
    the pending orchestrator step is cancelled with it.
    """
    stats = stats if stats is not None else TurnStats()
    t0 = time.perf_counter()
    it = aiter(chunks)
    first = asyncio.ensure_future(anext(it))
    try:
        done, _ = await asyncio.wait({first}, timeout=after_s)
        if not done:
            stats.filler = True
            yield filler + " "  # keeps the transcript readable: "je vérifie. L'échographie"
            yield FlushSentinel()
        try:
            chunk = await first
        except StopAsyncIteration:
            return
        stats.first_text_ms = round((time.perf_counter() - t0) * 1000, 1)
        yield chunk
        async for chunk in it:
            yield chunk
        stats.total_ms = round((time.perf_counter() - t0) * 1000, 1)
    finally:
        if not first.done():
            first.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await first
        if hasattr(it, "aclose"):
            await it.aclose()


def filler_for(text: str) -> str:
    """Filler in the caller's language, guessed from the script (no LLM call to wait for)."""
    return FILLERS[guess_language(text)]
