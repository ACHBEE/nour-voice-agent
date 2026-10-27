"""Eval tooling: the dataset is well-formed and the metrics compute what they claim."""

import json
from typing import get_args

import pytest

from evals import latency_report
from evals.run_evals import load_cases, summarize
from evals.stats import first_relevant_rank, percentile, ranking_metrics
from orchestrator.state import Intent
from rag_service.docs import load_docs

CASES = load_cases()
DOC_IDS = {d.doc_id for d in load_docs()}


def test_eval_set_is_well_formed():
    assert len(CASES) >= 25
    assert len({c.id for c in CASES}) == len(CASES), "duplicate case ids"
    assert {c.lang for c in CASES} == {"ar", "fr", "en"}
    for c in CASES:
        assert c.expected_intent in get_args(Intent), c.id
        assert (c.category == "answerable") == bool(c.expected_docs), c.id


def test_expected_docs_exist():
    # A typo here silently counts as a retrieval miss (it happened once: "policies_payment").
    for c in CASES:
        assert set(c.expected_docs) <= DOC_IDS, (c.id, set(c.expected_docs) - DOC_IDS)


def test_eval_set_covers_the_safety_cases():
    cats = {c.category for c in CASES}
    assert {"medical_advice", "emergency", "unanswerable"} <= cats
    assert sum(c.category == "medical_advice" for c in CASES) >= 3


def test_percentile_interpolates_like_numpy():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == pytest.approx(9.55)


def test_ranking_metrics():
    assert first_relevant_rank(["a", "b", "c"], {"c", "x"}) == 3
    assert first_relevant_rank(["a"], {"x"}) is None
    m = ranking_metrics([1, 3, None, 6])
    assert m["hit@1"] == 0.25 and m["hit@3"] == 0.5 and m["hit@5"] == 0.5
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 6) / 4)


def _rec(category, intent, judge=None, **kw):
    r = {
        "category": category,
        "intent": intent,
        "intent_ok": True,
        "route_ms": 500.0,
        "retrieval_ms": 30.0,
        "answer_ttft_ms": 400.0,
        "first_text_ms": 1000.0,
    }
    if judge is not None:
        base = dict(correct=True, grounded=True, offers_callback=False, language_ok=True,
                    concise=True, medical_advice=False)  # fmt: skip
        r["judge"] = {**base, **judge}
    return r | kw


def test_summary_counts_safety_correctly():
    records = [
        _rec("answerable", "clinic_info", {}, rewritten_rank=1),
        _rec("answerable", "clinic_info", {"correct": False}, rewritten_rank=None),
        _rec("unanswerable", "clinic_info", {"offers_callback": True}),
        _rec("medical_advice", "medical_advice"),  # fixed phrase: refused
        _rec("medical_advice", "clinic_info", {"medical_advice": True}),  # leaked advice
        _rec("emergency", "emergency"),
    ]
    s = summarize(records, None)
    assert s["answer_correct"] == 0.5
    assert s["unanswerable_ok"] == 1.0
    assert s["refusal_accuracy"] == 0.5
    assert s["medical_advice_leaks"] == 1
    assert s["emergency_routing"] == 1.0
    assert s["retrieval_rewritten_query"]["hit@5"] == 0.5


def test_latency_report_reads_all_row_kinds(tmp_path):
    log = tmp_path / "latency_log.jsonl"
    rows = [
        {"ts": "2026-10-28T09:00", "room": "r1", "role": "user", "end_of_turn_delay": 0.6},
        {"ts": "2026-10-28T09:00", "room": "r1", "role": "orchestrator", "route_ms": 700,
         "first_text_ms": 1200, "filler": True, "intent": "clinic_info"},
        {"ts": "2026-10-28T09:00", "room": "r1", "role": "assistant", "e2e_latency": 2.1},
        {"ts": "2026-10-20T09:00", "room": "old", "role": "assistant", "e2e_latency": 9.0},
    ]  # fmt: skip
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    kept = latency_report.load_rows(log, since="2026-10-28")
    stats = {s["field"]: s for s in latency_report.stage_stats(kept)}
    assert stats["end_of_turn_delay"]["p50"] == 600
    assert stats["route_ms"]["p50"] == 700
    assert stats["e2e_latency"]["n"] == 1 and stats["e2e_latency"]["p50"] == 2100
    assert "Filler phrase spoken on 1/1 turns" in latency_report.to_markdown(
        kept, list(stats.values()), "x"
    )
