"""Deterministic booking rules: phone normalisation, clinic schedule, Pydantic gate, read-backs."""

from datetime import date, time

import pytest
from pydantic import ValidationError

from orchestrator.booking import Booking, CallbackRequest, check_booking_fields, normalize_phone
from orchestrator.phrases import phrase, say_phone, say_when, schedule_ctx

TODAY = date(2026, 10, 27)  # a Tuesday
SATURDAY = "2026-10-31"
FULL = {
    "department": "pediatrics",
    "appointment_date": SATURDAY,
    "appointment_time": "10:30",
    "patient_name": "Yanis Benamar",
    "phone": "0555 12 34 56",
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0555 12 34 56", "0555123456"),
        ("+213 770 11 22 33", "0770112233"),
        ("00213-661-22-33-44", "0661223344"),
        ("555123456", "0555123456"),  # mobile said without the leading 0
        ("٠٥٥٥١٢٣٤٥٦", "0555123456"),  # Arabic-Indic digits from ASR
        ("032 11 22 33", "032112233"),  # El Oued landline
        ("12345", None),
        ("0855123456", None),
    ],
)
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def _codes(fields):
    return [(p.field, p.code) for p in check_booking_fields(fields, TODAY)]


def test_complete_valid_form_has_no_problems():
    assert _codes(FULL) == []


def test_missing_fields_are_asked_in_order():
    assert _codes({"department": "cardiology"}) == [
        ("appointment_date", "missing"),
        ("appointment_time", "missing"),
        ("patient_name", "missing"),
        ("phone", "missing"),
    ]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"appointment_date": "2026-10-20"}, ("appointment_date", "past_date")),
        ({"appointment_date": "2027-03-01"}, ("appointment_date", "too_far")),
        ({"appointment_date": "2026-10-30"}, ("appointment_date", "friday_closed")),
        ({"appointment_date": "2026-11-03"}, ("appointment_date", "department_closed_that_day")),
        ({"appointment_date": "next week"}, ("appointment_date", "invalid_date")),
        ({"appointment_time": "15:00"}, ("appointment_time", "outside_hours")),
        ({"appointment_time": "13:00"}, ("appointment_time", "outside_hours")),  # closes 13:00
        ({"phone": "12 34"}, ("phone", "invalid_phone")),
        ({"department": "ophthalmology"}, ("department", "unknown_department")),
    ],
)
def test_rule_violations(changes, expected):
    assert _codes({**FULL, **changes})[0] == expected


def test_schedule_is_checked_before_all_fields_are_known():
    # Nour should object to pediatrics-on-Tuesday before asking for the name.
    partial = {"department": "pediatrics", "appointment_date": "2026-10-27"}
    assert _codes(partial)[0] == ("appointment_date", "department_closed_that_day")


def test_booking_model_normalises_and_validates():
    b = Booking.model_validate({**FULL, "language": "fr"}, context={"today": TODAY})
    assert b.phone == "0555123456"
    assert b.appointment_date == date(2026, 10, 31) and b.appointment_time == time(10, 30)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"appointment_date": "2026-10-30"}, "friday_closed"),
        ({"appointment_time": "15:00"}, "outside_hours"),
        ({"phone": "123"}, "invalid_phone"),
    ],
)
def test_booking_model_rejects_with_rule_codes(changes, code):
    with pytest.raises(ValidationError) as exc:
        Booking.model_validate({**FULL, **changes}, context={"today": TODAY})
    assert exc.value.errors()[0]["type"] == code


def test_callback_request_validates_phone():
    assert CallbackRequest(phone="+213 770 11 22 33", reason="invoice").phone == "0770112233"
    with pytest.raises(ValidationError):
        CallbackRequest(phone="77", reason="invoice")


def test_read_back_formatting():
    d, t = date(2026, 10, 31), time(10, 30)
    assert say_when(d, t, "fr") == "samedi 31 octobre à 10h30"
    assert say_when(d, t, "en") == "Saturday, October 31 at 10:30"
    assert say_when(d, t, "ar") == "السبت 31 أكتوبر على 10:30"
    assert say_phone("0555123456") == "0555 12 34 56"
    assert say_phone("032112233") == "032 11 22 33"


def test_schedule_phrase_uses_real_days_and_hours():
    text = phrase("department_closed_that_day", "fr", **schedule_ctx("pediatrics", "fr"))
    assert "samedi, lundi et mercredi" in text
    text = phrase("outside_hours", "en", **schedule_ctx("cardiology", "en"))
    assert text.startswith("Cardiology appointments are between 8:30 and 12:00")
