"""Smoke tests: the agent module imports against the pinned LiveKit API and
the Booking schema validates. No network or API keys required."""

import pytest
from pydantic import ValidationError

from agent import agent as nour


def test_server_and_entrypoint_registered():
    assert isinstance(nour.server, nour.AgentServer)
    assert nour.server.setup_fnc is nour.prewarm


def test_receptionist_exposes_tools():
    tool_names = {t.info.name for t in nour.Receptionist().tools}
    assert {"book_appointment", "request_human_callback"} <= tool_names


def test_booking_validation():
    ok = nour.Booking(
        patient_name="Amina B.",
        phone="0661234567",
        department="pediatrics",
        preferred_date="2026-11-02",
        preferred_time="10:30",
    )
    assert ok.language == "fr"
    with pytest.raises(ValidationError):
        nour.Booking(
            patient_name="A",
            phone="1",
            department="x",
            preferred_date="d",
            preferred_time="t",
        )
