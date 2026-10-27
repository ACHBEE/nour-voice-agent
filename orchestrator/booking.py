"""Structured, validated bookings and callback requests.

The LLM only *extracts* what the caller said into loose strings. Everything that decides whether
a booking is valid is deterministic code here: phone normalisation, clinic days and hours,
date range. The final Pydantic models are the gate in front of the database.

Field checks work on partial forms (so Nour can say "pediatrics is not on Tuesdays" before it
has asked for the name), and the same rules back the model validators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_core import PydanticCustomError

from orchestrator.schedule import (
    DEPARTMENTS,
    FRI,
    MAX_DAYS_AHEAD,
    SCHEDULE,
    Department,
    clinic_today,
)

Lang = Literal["ar", "fr", "en"]

# Order in which Nour asks for missing booking details.
BOOKING_FIELDS = ("department", "appointment_date", "appointment_time", "patient_name", "phone")
CALLBACK_FIELDS = ("phone", "reason")

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_MOBILE = re.compile(r"0[5-7]\d{8}")  # 05xx / 06xx / 07xx + 6 digits
_LANDLINE = re.compile(r"0[2-4]\d{7}")  # e.g. 032 11 22 33 (El Oued)


@dataclass(frozen=True)
class Problem:
    """Why a form field can't be accepted yet; `code` selects what Nour says (phrases.py)."""

    field: str
    code: str
    ctx: dict[str, str] = field(default_factory=dict)


def normalize_phone(raw: str) -> str | None:
    """Algerian number in canonical national form (0XXXXXXXXX), or None if it isn't one.

    Accepts +213 / 00213 prefixes, spaces and dashes, and Arabic-Indic digits from ASR.
    """
    s = re.sub(r"[\s\-.()/]", "", raw.translate(_ARABIC_DIGITS))
    if s.startswith("+213"):
        s = "0" + s[4:]
    elif s.startswith("00213"):
        s = "0" + s[5:]
    elif s.startswith("213") and len(s) == 12:
        s = "0" + s[3:]
    elif len(s) == 9 and s[0] in "567":  # mobile said without the leading 0
        s = "0" + s
    return s if _MOBILE.fullmatch(s) or _LANDLINE.fullmatch(s) else None


def date_problem(d: date, today: date) -> str | None:
    if d < today:
        return "past_date"
    if d > today + timedelta(days=MAX_DAYS_AHEAD):
        return "too_far"
    if d.weekday() == FRI:
        return "friday_closed"
    return None


def schedule_problem(dept: Department, d: date) -> str | None:
    return None if d.weekday() in SCHEDULE[dept].days else "department_closed_that_day"


def time_problem(dept: Department, t: time) -> str | None:
    hours = SCHEDULE[dept]
    return None if hours.start <= t <= hours.last_start else "outside_hours"


def _parse(kind: type[date] | type[time], value: str) -> date | time | None:
    try:
        return kind.fromisoformat(value)
    except ValueError:
        return None


def check_booking_fields(fields: dict[str, str | None], today: date) -> list[Problem]:
    """Problems with a (possibly partial) booking form, in the order Nour should ask."""
    found: dict[str, Problem] = {}
    dept = fields.get("department")
    d = _parse(date, fields["appointment_date"]) if fields.get("appointment_date") else None
    t = _parse(time, fields["appointment_time"]) if fields.get("appointment_time") else None

    if dept is not None and dept not in DEPARTMENTS:
        found["department"] = Problem("department", "unknown_department")
    if fields.get("appointment_date"):
        code = "invalid_date" if d is None else date_problem(d, today)
        if code is None and dept in DEPARTMENTS:
            code = schedule_problem(dept, d)
        if code:
            found["appointment_date"] = Problem(
                "appointment_date", code, {"department": dept or ""}
            )
    if fields.get("appointment_time"):
        code = "invalid_time" if t is None else None
        if code is None and dept in DEPARTMENTS:
            code = time_problem(dept, t)
        if code:
            found["appointment_time"] = Problem(
                "appointment_time", code, {"department": dept or ""}
            )
    name = fields.get("patient_name")
    if name is not None and len(name.strip()) < 2:
        found["patient_name"] = Problem("patient_name", "invalid_name")
    if fields.get("phone") and normalize_phone(fields["phone"]) is None:
        found["phone"] = Problem("phone", "invalid_phone")

    return [
        found.get(f) or Problem(f, "missing")
        for f in BOOKING_FIELDS
        if f in found or not fields.get(f)
    ]


def check_callback_fields(fields: dict[str, str | None]) -> list[Problem]:
    problems = []
    if not fields.get("phone"):
        problems.append(Problem("phone", "missing"))
    elif normalize_phone(fields["phone"]) is None:
        problems.append(Problem("phone", "invalid_phone"))
    if not (fields.get("reason") or "").strip():
        problems.append(Problem("reason", "missing"))
    return problems


def _phone(v: str) -> str:
    phone = normalize_phone(v)
    if phone is None:
        raise PydanticCustomError("invalid_phone", "not an Algerian phone number: {v}", {"v": v})
    return phone


class Booking(BaseModel):
    """A bookable appointment. Validates against the clinic schedule.

    Pass `context={"today": date}` to validate relative to a fixed day (tests, evals).
    """

    patient_name: str = Field(min_length=2, max_length=80)
    phone: str
    department: Department
    appointment_date: date
    appointment_time: time
    language: Lang = "fr"

    @field_validator("patient_name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return " ".join(v.split())

    @field_validator("phone")
    @classmethod
    def _normalize_phone(cls, v: str) -> str:
        return _phone(v)

    @field_validator("appointment_date")
    @classmethod
    def _bookable_day(cls, v: date, info: ValidationInfo) -> date:
        today = (info.context or {}).get("today") or clinic_today()
        if code := date_problem(v, today):
            raise PydanticCustomError(code, "date {d} is not bookable", {"d": v.isoformat()})
        return v

    @model_validator(mode="after")
    def _within_department_schedule(self) -> Booking:
        if code := schedule_problem(self.department, self.appointment_date):
            raise PydanticCustomError(code, "{dept} is closed that day", {"dept": self.department})
        if code := time_problem(self.department, self.appointment_time):
            raise PydanticCustomError(
                code, "{dept} is closed at that time", {"dept": self.department}
            )
        return self


class CallbackRequest(BaseModel):
    """A request for a human to call the caller back."""

    phone: str
    reason: str = Field(min_length=3, max_length=300)
    language: Lang = "fr"

    @field_validator("phone")
    @classmethod
    def _normalize_phone(cls, v: str) -> str:
        return _phone(v)
