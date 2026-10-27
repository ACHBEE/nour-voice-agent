"""Postgres persistence for bookings and callback requests (tables in db/schema.sql).

Async interface over psycopg's *sync* pool, run in a worker thread. psycopg's async mode needs
a selector event loop, but on Windows LiveKit Agents runs jobs on the default Proactor loop;
a thread works on any loop. Writes are rare (a few per call), so the thread hop costs nothing.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, time

from psycopg import errors
from psycopg_pool import ConnectionPool

from orchestrator.booking import Booking, CallbackRequest
from rag_service.config import DATABASE_URL

logger = logging.getLogger("orchestrator.store")


class SlotTaken(Exception):
    """Another booking already holds this department / date / time."""


class BookingStore:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    @classmethod
    async def open(cls, url: str = DATABASE_URL, connect_timeout: float = 10) -> BookingStore:
        pool = ConnectionPool(url, min_size=1, max_size=4, open=False)
        # fail fast if Postgres is down
        await asyncio.to_thread(pool.open, wait=True, timeout=connect_timeout)
        return cls(pool)

    async def close(self) -> None:
        await asyncio.to_thread(self.pool.close)

    def _one(self, sql: str, params: tuple) -> tuple | None:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchone()

    async def slot_free(self, department: str, day: date, at: time) -> bool:
        """Early check so Nour doesn't read back a slot that's already taken.

        The unique index is still what prevents double-booking under concurrency.
        """
        row = await asyncio.to_thread(
            self._one,
            "SELECT 1 FROM bookings WHERE department = %s AND appointment_date = %s"
            " AND appointment_time = %s AND status <> 'cancelled'",
            (department, day, at),
        )
        return row is None

    async def create_booking(self, booking: Booking, call_id: str) -> int:
        try:
            (booking_id,) = await asyncio.to_thread(
                self._one,
                "INSERT INTO bookings (patient_name, phone, department, appointment_date,"
                " appointment_time, language, room) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                " RETURNING id",
                (
                    booking.patient_name,
                    booking.phone,
                    booking.department,
                    booking.appointment_date,
                    booking.appointment_time,
                    booking.language,
                    call_id,
                ),
            )
        except errors.UniqueViolation as e:
            raise SlotTaken(booking.department) from e
        logger.info(
            "booking %d saved (%s %s)", booking_id, booking.department, booking.appointment_date
        )
        return booking_id

    async def create_callback(self, request: CallbackRequest, call_id: str) -> int:
        (callback_id,) = await asyncio.to_thread(
            self._one,
            "INSERT INTO callbacks (phone, reason, language, room) VALUES (%s, %s, %s, %s)"
            " RETURNING id",
            (request.phone, request.reason, request.language, call_id),
        )
        logger.info("callback %d saved", callback_id)
        return callback_id
