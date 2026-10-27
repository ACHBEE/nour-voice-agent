"""Smoke tests: the agent module imports against the pinned LiveKit API and is wired to the
orchestrator. No network or API keys required."""

import asyncio

from livekit.agents import llm

from agent import agent as nour


def test_server_and_entrypoint_registered():
    assert isinstance(nour.server, nour.AgentServer)
    assert nour.server.setup_fnc is nour.prewarm


def test_receptionist_replaces_llm_node_and_has_no_tools():
    # Clinic facts, bookings and callbacks all live in the orchestrator now.
    assert nour.Receptionist.llm_node is not nour.Agent.llm_node
    assert not hasattr(nour, "CLINIC_FACTS")


def test_latest_user_text_picks_the_last_user_message():
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content="sys")
    ctx.add_message(role="user", content="first")
    ctx.add_message(role="assistant", content="reply")
    ctx.add_message(role="user", content="  واش تخدمو نهار الجمعة؟ ")
    assert nour.latest_user_text(ctx) == "واش تخدمو نهار الجمعة؟"
    assert nour.latest_user_text(llm.ChatContext.empty()) == ""


def test_unavailable_store_raises_so_forms_degrade():
    store = nour.UnavailableStore()
    try:
        asyncio.run(store.slot_free("dentistry", None, None))
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError")
