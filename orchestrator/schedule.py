"""Booking rules: which departments take appointments, on which days, at which hours.

Mirrors data/clinic_docs/04_doctors_fr.md. The documents tell callers the schedule; this module
enforces it when a booking is made (in a real clinic both would come from the scheduling system).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

CLINIC_TZ = timezone(timedelta(hours=1), "Africa/Algiers")  # Algeria: UTC+1 all year, no DST

Department = Literal["general_medicine", "pediatrics", "cardiology", "gynecology", "dentistry"]
DEPARTMENTS: tuple[Department, ...] = (
    "general_medicine",
    "pediatrics",
    "cardiology",
    "gynecology",
    "dentistry",
)

# datetime.weekday(): Monday = 0 ... Sunday = 6. The Algerian week runs Saturday to Thursday.
MON, TUE, WED, THU, FRI, SAT, SUN = range(7)
WEEK_ORDER = (SAT, SUN, MON, TUE, WED, THU, FRI)
SAT_TO_THU = frozenset({SAT, SUN, MON, TUE, WED, THU})

SLOT = timedelta(minutes=30)  # the last appointment starts one slot before closing
MAX_DAYS_AHEAD = 60


@dataclass(frozen=True)
class Hours:
    days: frozenset[int]
    start: time
    end: time

    @property
    def last_start(self) -> time:
        return (datetime.combine(date.min, self.end) - SLOT).time()


SCHEDULE: dict[Department, Hours] = {
    "general_medicine": Hours(SAT_TO_THU, time(8, 0), time(18, 0)),
    "pediatrics": Hours(frozenset({SAT, MON, WED}), time(8, 30), time(13, 0)),
    "cardiology": Hours(frozenset({SUN, TUE, THU}), time(8, 30), time(12, 30)),
    "gynecology": Hours(frozenset({SAT, SUN, MON, TUE, WED}), time(9, 0), time(15, 0)),
    "dentistry": Hours(SAT_TO_THU, time(9, 0), time(17, 0)),
}


def clinic_today() -> date:
    return datetime.now(CLINIC_TZ).date()
