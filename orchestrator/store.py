"""Postgres persistence for bookings and callback requests (tables in db/schema.sql)."""

from __future__ import annotations

import logging
from datetime import date, time

from psycopg import errors
from psycopg_pool import AsyncConnectionPool

from orchestrator.booking import Booking, CallbackRequest
from rag_service.config import DATABASE_URL

logger = logging.getLogger("orchestrator.store")


class SlotTaken(Exception):
    """Another booking already holds this department / date / time."""


class BookingStore:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    @classmethod
    async def open(cls, url: str = DATABASE_URL, connect_timeout: float = 10) -> BookingStore:
        pool = AsyncConnectionPool(url, min_size=1, max_size=4, open=False)
        await pool.open(wait=True, timeout=connect_timeout)  # fail fast if Postgres is down
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def slot_free(self, department: str, day: date, at: time) -> bool:
        """Early check so Nour doesn't read back a slot that's already taken.

        The unique index is still what prevents double-booking under concurrency.
        """
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM bookings WHERE department = %s AND appointment_date = %s"
                " AND appointment_time = %s AND status <> 'cancelled'",
                (department, day, at),
            )
            return await cur.fetchone() is None

    async def create_booking(self, booking: Booking, call_id: str) -> int:
        try:
            async with self.pool.connection() as conn:
                cur = await conn.execute(
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
                (booking_id,) = await cur.fetchone()
        except errors.UniqueViolation as e:
            raise SlotTaken(booking.department) from e
        logger.info(
            "booking %d saved (%s %s)", booking_id, booking.department, booking.appointment_date
        )
        return booking_id

    async def create_callback(self, request: CallbackRequest, call_id: str) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO callbacks (phone, reason, language, room) VALUES (%s, %s, %s, %s)"
                " RETURNING id",
                (request.phone, request.reason, request.language, call_id),
            )
            (callback_id,) = await cur.fetchone()
        logger.info("callback %d saved", callback_id)
        return callback_id
