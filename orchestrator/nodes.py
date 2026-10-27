"""Graph nodes. Each caller turn runs: route -> one branch -> END.

Every node that speaks pushes its text through the stream writer (so the voice agent can start
TTS on the first tokens) and appends it to the transcript.

Failures degrade instead of crashing the call: a router error falls back to a document answer,
a retrieval error answers from no documents (which yields "I don't know + callback"), and form
or database errors keep the form and apologise.
"""

from __future__ import annotations

import logging
import re
import time as clock
from datetime import date, time
from typing import Any

from langgraph.runtime import Runtime
from pydantic import ValidationError

from orchestrator.booking import (
    BOOKING_FIELDS,
    CALLBACK_FIELDS,
    Booking,
    CallbackRequest,
    Problem,
    check_booking_fields,
    check_callback_fields,
    normalize_phone,
)
from orchestrator.config import HISTORY_TURNS, RAG_TOP_K
from orchestrator.deps import Deps
from orchestrator.phrases import phrase, say_department, say_phone, say_when, schedule_ctx
from orchestrator.prompts import answer_prompt, booking_prompt, callback_prompt, router_prompt
from orchestrator.state import (
    BookingExtraction,
    CallbackExtraction,
    CallState,
    Form,
    Lang,
    Msg,
    RouteDecision,
)
from orchestrator.store import SlotTaken

logger = logging.getLogger("orchestrator")

_ARABIC = re.compile(r"[؀-ۿ]")
_FRENCH = re.compile(r"[éèêàçù]|\b(je|vous|est|les|quel|quelle|combien|bonjour|rendez)\b", re.I)


def guess_language(text: str) -> Lang:
    """Cheap fallback when the router LLM is unavailable."""
    if _ARABIC.search(text):
        return "ar"
    return "fr" if _FRENCH.search(text) else "en"


def _ms(t0: float) -> float:
    return round((clock.perf_counter() - t0) * 1000, 1)


def _history(state: CallState) -> list[Msg]:
    return state.get("messages", [])[-HISTORY_TURNS:]


def _lang(state: CallState) -> Lang:
    return state.get("language") or "fr"


def _say(runtime: Runtime[Deps], text: str, **updates: Any) -> dict[str, Any]:
    runtime.stream_writer({"text": text})
    return {"reply": text, "messages": [{"role": "assistant", "content": text}], **updates}


# ---------- router ----------


async def route(state: CallState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    t0 = clock.perf_counter()
    try:
        d = await deps.llm.structured(
            router_prompt(state, deps.today()), _history(state), RouteDecision
        )
    except Exception:  # noqa: BLE001 - any LLM failure: answer from documents instead
        logger.exception("router failed, falling back to clinic_info")
        text = state["user_text"]
        d = RouteDecision(intent="clinic_info", language=guess_language(text), search_query=text)
    if d.intent == "clinic_info" and not d.search_query.strip():
        d.search_query = state["user_text"]
    route_ms = _ms(t0)
    logger.info("route %s lang=%s %.0fms q=%r", d.intent, d.language, route_ms, d.search_query)
    return {  # reset per-turn fields so nothing leaks from the previous turn
        "intent": d.intent,
        "language": d.language,
        "search_query": d.search_query,
        "chunks": [],
        "reply": "",
        "booking_id": None,
        "callback_id": None,
        "route_ms": route_ms,
        "retrieval_ms": 0.0,
        "rag_server_ms": 0.0,
        "answer_ttft_ms": 0.0,
    }


def next_step(state: CallState) -> str:
    return {
        "clinic_info": "retrieve",
        "smalltalk": "answer",
        "book_appointment": "booking",
        "human_callback": "callback",
        "emergency": "safety",
        "medical_advice": "safety",
    }[state["intent"]]


# ---------- RAG branch ----------


async def retrieve(state: CallState, runtime: Runtime[Deps]) -> dict[str, Any]:
    t0 = clock.perf_counter()
    try:
        res = await runtime.context.rag.search(
            state["search_query"], top_k=RAG_TOP_K, language=_lang(state)
        )
    except Exception:  # noqa: BLE001 - answer without documents rather than drop the call
        logger.exception("retrieval failed")
        return {"chunks": [], "retrieval_ms": _ms(t0)}
    chunks = [c.model_dump(include={"doc_id", "section", "content", "score"}) for c in res.chunks]
    retrieval_ms = _ms(t0)
    logger.info(
        "retrieval %.0fms (server %.0fms) docs=%s",
        retrieval_ms,
        res.embed_ms + res.search_ms,
        [c["doc_id"] for c in chunks],
    )
    return {
        "chunks": chunks,
        "retrieval_ms": retrieval_ms,
        "rag_server_ms": round(res.embed_ms + res.search_ms, 1),
    }


async def answer(state: CallState, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Grounded answer: the prompt restricts clinic facts to the retrieved chunks."""
    t0 = clock.perf_counter()
    ttft = 0.0
    parts: list[str] = []
    try:
        async for delta in runtime.context.llm.stream(answer_prompt(state), _history(state)):
            if not parts:
                ttft = _ms(t0)
            parts.append(delta)
            runtime.stream_writer({"text": delta})
    except Exception:  # noqa: BLE001
        logger.exception("answer generation failed")
        if not parts:  # nothing spoken yet: apologise instead
            return _say(runtime, phrase("tech_error", _lang(state)))
    text = "".join(parts).strip()
    logger.info("answer ttft=%.0fms total=%.0fms", ttft, _ms(t0))
    return {
        "reply": text,
        "messages": [{"role": "assistant", "content": text}],
        "answer_ttft_ms": ttft,
    }


# ---------- safety branch (fixed phrases, never the LLM) ----------


async def safety(state: CallState, runtime: Runtime[Deps]) -> dict[str, Any]:
    return _say(runtime, phrase(state["intent"], _lang(state)))


# ---------- forms (booking, callback) ----------


def _new_form(names: tuple[str, ...]) -> Form:
    return {"fields": dict.fromkeys(names), "rejected": {}, "awaiting_confirmation": False}


def _merge(form: Form, extracted: dict[str, str | None]) -> tuple[Form, bool]:
    """Apply newly extracted values; ignore nulls and values already refused this call."""
    fields = dict(form["fields"])
    changed = False
    for name, value in extracted.items():
        if value is None or value in form["rejected"].get(name, []):
            continue
        if fields.get(name) != value:
            fields[name] = value
            changed = True
    return {**form, "fields": fields}, changed


def _reject(form: Form, field: str) -> Form:
    value = form["fields"].get(field)
    rejected = {**form["rejected"]}
    if value:
        rejected[field] = [*rejected.get(field, []), value]
    return {
        "fields": {**form["fields"], field: None},
        "rejected": rejected,
        "awaiting_confirmation": False,
    }


def _problem_text(p: Problem, lang: Lang, ask_prefix: str = "ask") -> str:
    if p.code == "missing":
        return phrase(f"{ask_prefix}.{p.field}", lang)
    if p.code in ("department_closed_that_day", "outside_hours"):
        return phrase(p.code, lang, **schedule_ctx(p.ctx["department"], lang))
    return phrase(p.code, lang)


def _booking_slot(fields: dict[str, str | None]) -> tuple[str, date, time]:
    return (
        fields["department"],
        date.fromisoformat(fields["appointment_date"]),
        time.fromisoformat(fields["appointment_time"]),
    )


async def booking(state: CallState, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Collect, validate, read back, then save an appointment over several turns."""
    deps = runtime.context
    lang = _lang(state)
    today = deps.today()
    form = state.get("booking") or _new_form(BOOKING_FIELDS)
    try:
        ext = await deps.llm.structured(
            booking_prompt(form, today), _history(state), BookingExtraction
        )
    except Exception:  # noqa: BLE001
        logger.exception("booking extraction failed")
        return _say(runtime, phrase("tech_error", lang), booking=form)
    form, changed = _merge(form, ext.model_dump(include=set(BOOKING_FIELDS)))
    logger.info("booking form=%s changed=%s confirm=%s", form["fields"], changed, ext.confirmation)

    if form["awaiting_confirmation"] and not changed:
        if ext.confirmation == "yes":
            return await _save_booking(runtime, form, lang, today)
        if ext.confirmation == "no":
            return _say(
                runtime,
                phrase("ask_change", lang),
                booking={**form, "awaiting_confirmation": False},
            )

    if problems := check_booking_fields(form["fields"], today):
        p = problems[0]
        if p.code != "missing":
            form = _reject(form, p.field)
        return _say(
            runtime, _problem_text(p, lang), booking={**form, "awaiting_confirmation": False}
        )

    try:
        free = await deps.store.slot_free(*_booking_slot(form["fields"]))
    except Exception:  # noqa: BLE001
        logger.exception("slot check failed")
        return _say(runtime, phrase("tech_error", lang), booking=form)
    if not free:
        return _say(runtime, phrase("slot_taken", lang), booking=_reject(form, "appointment_time"))

    f = form["fields"]
    dept, day, at = _booking_slot(f)
    text = phrase(
        "confirm_booking",
        lang,
        department=say_department(dept, lang),
        when=say_when(day, at, lang),
        name=f["patient_name"],
        phone=say_phone(normalize_phone(f["phone"])),
    )
    return _say(runtime, text, booking={**form, "awaiting_confirmation": True})


async def _save_booking(
    runtime: Runtime[Deps], form: Form, lang: Lang, today: date
) -> dict[str, Any]:
    try:
        b = Booking.model_validate({**form["fields"], "language": lang}, context={"today": today})
    except ValidationError:  # rules are checked before read-back, so this is a bug if it fires
        logger.exception("confirmed booking failed validation")
        return _say(
            runtime, phrase("ask_change", lang), booking={**form, "awaiting_confirmation": False}
        )
    try:
        booking_id = await runtime.context.store.create_booking(b, runtime.context.call_id)
    except SlotTaken:  # someone booked it between the read-back and the "yes"
        return _say(runtime, phrase("slot_taken", lang), booking=_reject(form, "appointment_time"))
    except Exception:  # noqa: BLE001
        logger.exception("saving booking failed")
        return _say(runtime, phrase("tech_error", lang), booking=form)
    text = phrase(
        "booked",
        lang,
        department=say_department(b.department, lang),
        when=say_when(b.appointment_date, b.appointment_time, lang),
    )
    return _say(runtime, text, booking=None, booking_id=booking_id)


async def callback(state: CallState, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Collect a phone number and reason, read the number back, then log the callback."""
    deps = runtime.context
    lang = _lang(state)
    form = state.get("callback") or _new_form(CALLBACK_FIELDS)
    try:
        ext = await deps.llm.structured(callback_prompt(form), _history(state), CallbackExtraction)
    except Exception:  # noqa: BLE001
        logger.exception("callback extraction failed")
        return _say(runtime, phrase("tech_error", lang), callback=form)
    form, changed = _merge(form, ext.model_dump(include=set(CALLBACK_FIELDS)))
    logger.info("callback form=%s changed=%s confirm=%s", form["fields"], changed, ext.confirmation)

    if form["awaiting_confirmation"] and not changed:
        if ext.confirmation == "yes":
            return await _save_callback(runtime, form, lang)
        if ext.confirmation == "no":
            return _say(
                runtime, phrase("callback.ask.phone", lang), callback=_reject(form, "phone")
            )

    if problems := check_callback_fields(form["fields"]):
        p = problems[0]
        if p.code != "missing":
            form = _reject(form, p.field)
        return _say(
            runtime,
            _problem_text(p, lang, ask_prefix="callback.ask"),
            callback={**form, "awaiting_confirmation": False},
        )
    phone = say_phone(normalize_phone(form["fields"]["phone"]))
    return _say(
        runtime,
        phrase("confirm_callback", lang, phone=phone),
        callback={**form, "awaiting_confirmation": True},
    )


async def _save_callback(runtime: Runtime[Deps], form: Form, lang: Lang) -> dict[str, Any]:
    try:
        req = CallbackRequest.model_validate({**form["fields"], "language": lang})
        callback_id = await runtime.context.store.create_callback(req, runtime.context.call_id)
    except Exception:  # noqa: BLE001
        logger.exception("saving callback failed")
        return _say(runtime, phrase("tech_error", lang), callback=form)
    return _say(runtime, phrase("callback_saved", lang), callback=None, callback_id=callback_id)
