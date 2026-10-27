"""Filler phrase logic, alone and wrapped around the real LangGraph graph (no network)."""

import asyncio

from livekit.agents import FlushSentinel

from agent.turn_stream import FILLERS, TurnStats, filler_for, with_filler
from orchestrator.graph import Orchestrator
from orchestrator.state import RouteDecision
from tests.test_orchestrator_graph import TODAY, FakeRag, FakeStore


async def _chunks(delay_s, parts=("Bonjour", " !")):
    await asyncio.sleep(delay_s)
    for p in parts:
        yield p


def _run(agen):
    async def go():
        return [x async for x in agen]

    return asyncio.run(go())


def test_fast_turn_has_no_filler():
    stats = TurnStats()
    out = _run(with_filler(_chunks(0.0), "Un instant.", after_s=0.2, stats=stats))
    assert out == ["Bonjour", " !"]
    assert not stats.filler and stats.first_text_ms is not None and stats.total_ms is not None


def test_slow_turn_says_filler_first_and_flushes_it():
    stats = TurnStats()
    out = _run(with_filler(_chunks(0.3), "Un instant.", after_s=0.05, stats=stats))
    assert out[0].strip() == "Un instant." and isinstance(out[1], FlushSentinel)
    assert out[2:] == ["Bonjour", " !"]
    assert stats.filler and stats.first_text_ms >= 250


def test_empty_turn_yields_nothing():
    assert _run(with_filler(_chunks(0.0, parts=()), "x", after_s=0.2)) == []


def test_interruption_cancels_the_pending_orchestrator_step():
    cancelled = asyncio.Event()

    async def slow():
        try:
            await asyncio.sleep(10)
            yield "never"
        finally:
            cancelled.set()

    async def go():
        agen = with_filler(slow(), "Un instant.", after_s=0.01)
        assert (await anext(agen)).strip() == "Un instant."
        await agen.aclose()  # what LiveKit does when the caller barges in
        return cancelled.is_set()

    assert asyncio.run(go())


def test_filler_language_follows_the_caller():
    assert filler_for("واش تخدمو نهار الجمعة؟") == FILLERS["ar"]
    assert filler_for("Combien coûte une consultation ?") == FILLERS["fr"]
    assert filler_for("How much is a consultation?") == FILLERS["en"]


class SlowRouterLLM:
    """Router takes 0.3 s, like a real LLM call; then a streamed answer."""

    async def structured(self, system, history, schema):
        await asyncio.sleep(0.3)
        return RouteDecision(
            intent="clinic_info", language="fr", search_fr="prix", search_en="", search_ar=""
        )

    async def stream(self, system, history):
        for w in ("La", " consultation", " coûte", " 2000", " dinars."):
            yield w


def test_filler_around_the_real_graph():
    orch = Orchestrator(SlowRouterLLM(), FakeRag(), FakeStore(), today=lambda: TODAY)
    stats = TurnStats()

    async def go():
        out = [x async for x in with_filler(orch.turn("c1", "prix ?"), "Un instant.", 0.1, stats)]
        return out, await orch.state("c1")

    out, state = asyncio.run(go())
    assert out[0].strip() == "Un instant." and isinstance(out[1], FlushSentinel)
    assert "".join(out[2:]) == "La consultation coûte 2000 dinars."
    assert state["reply"] == "La consultation coûte 2000 dinars."  # checkpoint written
    assert state["route_ms"] >= 300
