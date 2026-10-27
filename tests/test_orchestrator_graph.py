"""The LangGraph orchestrator with a scripted LLM, fake RAG and in-memory store (no network).

Each structured LLM call pops the next scripted object, so a test reads like the call:
what the router decides, what the extractor pulls out, and what Nour says back.
"""

import asyncio
from datetime import date, time

import pytest

from orchestrator.graph import Orchestrator
from orchestrator.phrases import phrase
from orchestrator.state import BookingExtraction, CallbackExtraction, RouteDecision
from orchestrator.store import SlotTaken
from rag_service.retrieval import RetrievedChunk, SearchResult

TODAY = date(2026, 10, 27)  # Tuesday
CALL = "test-call"


class FakeLLM:
    def __init__(self, *scripted, answer="La consultation coûte 2000 dinars."):
        self.queue = list(scripted)
        self.answer = answer
        self.calls: list[tuple[str, str]] = []  # (kind, system prompt)

    async def structured(self, system, history, schema):
        self.calls.append((schema.__name__, system))
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, schema), f"expected {schema.__name__}, scripted {item!r}"
        return item

    async def stream(self, system, history):
        self.calls.append(("stream", system))
        for word in self.answer.split(" "):
            yield word + " "


class FakeRag:
    def __init__(self, fail=False):
        self.fail = fail
        self.queries: list[str] = []

    async def search(self, query, top_k=5, language=""):
        self.queries.append(query)
        if self.fail:
            raise ConnectionError("rag down")
        chunk = RetrievedChunk(
            chunk_id=1,
            doc_id="prices",
            title="Tarifs",
            section="Consultations",
            content="Tarifs › Consultations\nMédecine générale : 2000 DZD.",
            lang="fr",
            source_path="x",
            score=0.03,
        )
        return SearchResult(chunks=[chunk], embed_ms=15.0, search_ms=10.0)


class FakeStore:
    def __init__(self, taken=(), race=False):
        self.taken = set(taken)
        self.race = race  # slot gets taken between read-back and "yes"
        self.bookings = []
        self.callbacks = []

    async def slot_free(self, department, day, at):
        return (department, day, at) not in self.taken

    async def create_booking(self, booking, call_id):
        if self.race:
            raise SlotTaken(booking.department)
        self.bookings.append(booking)
        return len(self.bookings)

    async def create_callback(self, request, call_id):
        self.callbacks.append(request)
        return len(self.callbacks)


def route(intent, lang="fr", q=""):
    return RouteDecision(intent=intent, language=lang, search_query=q)


def extract(confirmation="none", **fields):
    keys = ("department", "appointment_date", "appointment_time", "patient_name", "phone")
    return BookingExtraction(confirmation=confirmation, **{k: fields.get(k) for k in keys})


FULL = {
    "department": "pediatrics",
    "appointment_date": "2026-10-31",
    "appointment_time": "10:30",
    "patient_name": "Yanis Benamar",
    "phone": "0555 12 34 56",
}


def make(llm, rag=None, store=None):
    return Orchestrator(llm, rag or FakeRag(), store or FakeStore(), today=lambda: TODAY)


def say(orch, text):
    return asyncio.run(orch.reply(CALL, text))


def state(orch):
    return asyncio.run(orch.state(CALL))


def conversation(orch, *turns):
    """Run several turns in ONE event loop (the checkpointer outlives each turn anyway)."""

    async def go():
        return [await orch.reply(CALL, t) for t in turns]

    return asyncio.run(go())


# ---------- routing ----------


def test_clinic_info_retrieves_then_streams_a_grounded_answer():
    llm = FakeLLM(route("clinic_info", q="prix consultation généraliste"))
    rag = FakeRag()
    orch = make(llm, rag)
    chunks = asyncio.run(_collect(orch.turn(CALL, "C'est combien la consultation ?")))
    assert len(chunks) > 1, "answer should stream in several chunks"
    assert "".join(chunks).strip() == "La consultation coûte 2000 dinars."
    assert rag.queries == ["prix consultation généraliste"]
    answer_system = llm.calls[-1][1]
    assert "Médecine générale : 2000 DZD." in answer_system  # chunk text reached the prompt
    s = state(orch)
    assert s["intent"] == "clinic_info" and s["chunks"][0]["doc_id"] == "prices"
    assert s["retrieval_ms"] >= 0 and s["rag_server_ms"] == 25.0  # server-side embed + search
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]


async def _collect(stream):
    return [c async for c in stream]


@pytest.mark.parametrize("intent", ["emergency", "medical_advice"])
def test_safety_intents_use_fixed_phrases_never_the_llm(intent):
    llm = FakeLLM(route(intent, lang="ar"))
    orch = make(llm)
    assert say(orch, "عندي وجع في صدري") == phrase(intent, "ar")
    assert [kind for kind, _ in llm.calls] == ["RouteDecision"]  # no generation call at all


def test_smalltalk_skips_retrieval():
    rag = FakeRag()
    orch = make(FakeLLM(route("smalltalk", lang="en"), answer="You're welcome!"), rag)
    assert say(orch, "thanks!").strip() == "You're welcome!"
    assert rag.queries == []


def test_router_failure_falls_back_to_documents():
    rag = FakeRag()
    orch = make(FakeLLM(TimeoutError("llm down")), rag)
    say(orch, "واش تخدمو نهار الجمعة؟")
    s = state(orch)
    assert s["intent"] == "clinic_info" and s["language"] == "ar"
    assert rag.queries == ["واش تخدمو نهار الجمعة؟"]


def test_retrieval_failure_answers_without_documents():
    llm = FakeLLM(route("clinic_info", q="horaires"))
    orch = make(llm, FakeRag(fail=True))
    say(orch, "Vous ouvrez à quelle heure ?")
    assert "(none retrieved for this turn)" in llm.calls[-1][1]
    assert state(orch)["chunks"] == []


# ---------- booking ----------


def test_booking_over_several_turns_with_a_rejected_day():
    store = FakeStore()
    llm = FakeLLM(
        route("book_appointment"),
        extract(department="pediatrics", appointment_date="2026-10-27"),  # Tuesday: no pediatrics
        route("book_appointment"),
        # the extractor re-reads the history and returns the refused Tuesday again
        extract(department="pediatrics", appointment_date="2026-10-27"),
        route("book_appointment"),
        extract(**FULL),
        route("book_appointment"),
        extract(confirmation="yes", **FULL),
    )
    orch = make(llm, store=store)
    r1, r2, r3, r4 = conversation(
        orch,
        "Un rendez-vous chez le pédiatre aujourd'hui",
        "Pour Yanis",
        "Samedi 10h30, Yanis Benamar, 0555 12 34 56",
        "Oui",
    )
    assert "samedi, lundi et mercredi" in r1
    assert r2 == phrase("ask.appointment_date", "fr")  # refused value not re-accepted
    assert r3 == (
        "Je récapitule : pédiatrie le samedi 31 octobre à 10h30, au nom de Yanis Benamar,"
        " numéro 0555 12 34 56. C'est bien ça ?"
    )
    assert r4.startswith("C'est noté")
    assert len(store.bookings) == 1 and store.bookings[0].phone == "0555123456"
    s = state(orch)
    assert s["booking"] is None and s["booking_id"] == 1


def test_nothing_is_saved_without_a_yes():
    store = FakeStore()
    llm = FakeLLM(
        route("book_appointment"),
        extract(**FULL),
        route("book_appointment"),
        extract(confirmation="no", **FULL),
    )
    orch = make(llm, store=store)
    _, r2 = conversation(orch, "RDV pédiatre samedi 10h30, Yanis Benamar, 0555123456", "Non")
    assert r2 == phrase("ask_change", "fr")
    assert store.bookings == []
    assert state(orch)["booking"]["awaiting_confirmation"] is False


def test_correction_after_read_back_reads_back_again():
    store = FakeStore()
    llm = FakeLLM(
        route("book_appointment"),
        extract(**FULL),
        route("book_appointment"),
        extract(confirmation="no", **{**FULL, "appointment_time": "11:00"}),
    )
    orch = make(llm, store=store)
    _, r2 = conversation(orch, "RDV ...", "Non, plutôt 11h")
    assert "à 11h," in r2 and r2.endswith("C'est bien ça ?")
    assert store.bookings == []


def test_taken_slot_is_refused_before_read_back():
    store = FakeStore(taken={("pediatrics", date(2026, 10, 31), time(10, 30))})
    orch = make(FakeLLM(route("book_appointment"), extract(**FULL)), store=store)
    assert say(orch, "RDV ...") == phrase("slot_taken", "fr")
    assert state(orch)["booking"]["fields"]["appointment_time"] is None


def test_slot_taken_between_read_back_and_yes():
    store = FakeStore(race=True)
    llm = FakeLLM(
        route("book_appointment"),
        extract(**FULL),
        route("book_appointment"),
        extract(confirmation="yes", **FULL),
    )
    orch = make(llm, store=store)
    _, r2 = conversation(orch, "RDV ...", "Oui")
    assert r2 == phrase("slot_taken", "fr")
    assert state(orch)["booking"]["fields"]["appointment_time"] is None


def test_question_in_the_middle_of_a_booking_keeps_the_form():
    llm = FakeLLM(
        route("book_appointment"),
        extract(department="cardiology"),
        route("clinic_info", q="adresse de la clinique"),
    )
    orch = make(llm)
    conversation(orch, "RDV cardiologie", "C'est où la clinique ?")
    s = state(orch)
    assert s["intent"] == "clinic_info"
    assert s["booking"]["fields"]["department"] == "cardiology"


# ---------- callback ----------


def test_callback_reads_number_back_then_saves():
    store = FakeStore()
    llm = FakeLLM(
        route("human_callback", lang="en"),
        CallbackExtraction(phone=None, reason="question about an invoice", confirmation="none"),
        route("human_callback", lang="en"),
        CallbackExtraction(
            phone="+213 770 11 22 33", reason="question about an invoice", confirmation="none"
        ),
        route("human_callback", lang="en"),
        CallbackExtraction(
            phone="+213 770 11 22 33", reason="question about an invoice", confirmation="yes"
        ),
    )
    orch = make(llm, store=store)
    r1, r2, r3 = conversation(orch, "Can someone call me?", "+213 770 11 22 33", "Yes")
    assert r1 == phrase("callback.ask.phone", "en")
    assert r2 == phrase("confirm_callback", "en", phone="0770 11 22 33")
    assert r3 == phrase("callback_saved", "en")
    assert store.callbacks[0].phone == "0770112233"
