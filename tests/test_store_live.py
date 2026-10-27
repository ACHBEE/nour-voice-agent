"""BookingStore against live Postgres: inserts, the double-booking guard, callbacks.

Skipped when the database isn't reachable. Rows are tagged with a unique room id and deleted.
"""

import uuid
from datetime import date, time, timedelta

import pytest

psycopg = pytest.importorskip("psycopg")

from orchestrator.booking import Booking, CallbackRequest  # noqa: E402
from orchestrator.schedule import SAT, clinic_today  # noqa: E402
from orchestrator.store import BookingStore, SlotTaken  # noqa: E402
from rag_service.aio import run  # noqa: E402
from rag_service.config import DATABASE_URL  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def db_up():
    try:
        psycopg.connect(DATABASE_URL, connect_timeout=3).close()
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")


def _far_saturday() -> date:
    d = clinic_today() + timedelta(days=50)  # inside the 60-day window, away from demo data
    return d + timedelta(days=(SAT - d.weekday()) % 7)


def test_booking_insert_and_double_booking_guard():
    room = f"pytest-{uuid.uuid4().hex[:8]}"
    booking = Booking(
        patient_name="Test Patient",
        phone="0555 00 00 01",
        department="general_medicine",
        appointment_date=_far_saturday(),
        appointment_time=time(17, 30),
        language="en",
    )

    async def go():
        store = await BookingStore.open()
        try:
            slot = (booking.department, booking.appointment_date, booking.appointment_time)
            assert await store.slot_free(*slot)
            booking_id = await store.create_booking(booking, room)
            assert booking_id > 0
            assert not await store.slot_free(*slot)
            with pytest.raises(SlotTaken):
                await store.create_booking(
                    booking.model_copy(update={"patient_name": "Other"}), room
                )
            cb_id = await store.create_callback(
                CallbackRequest(phone="0770112233", reason="pytest", language="fr"), room
            )
            assert cb_id > 0
        finally:
            with store.pool.connection() as conn:
                conn.execute("DELETE FROM bookings WHERE room = %s", (room,))
                conn.execute("DELETE FROM callbacks WHERE room = %s", (room,))
            await store.close()

    run(go())
