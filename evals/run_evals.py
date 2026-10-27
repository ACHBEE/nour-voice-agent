"""Offline evaluation of Nour: retrieval, routing, grounded answers, refusals.

    python evals/run_evals.py                      # full run (needs Postgres + LiveKit keys)
    python evals/run_evals.py --retrieval-only     # no LLM: hit rate@k / MRR per search mode (CI)
    python evals/run_evals.py --router-model openai/gpt-4.1-nano --tag router-nano

Three layers, each answering a different question:
1. Retrieval (no LLM): does hybrid / vector / keyword search put a relevant doc in the top k
   for the raw caller question? Deterministic, cheap, run in CI with a threshold.
2. End to end: every case goes through the real LangGraph orchestrator (router, retrieval,
   answer, safety branches). Intent accuracy, emergency routing, per-stage latency, and
   retrieval with the router's rewritten query.
3. LLM-as-judge (evals/judge_rubric.md) on every LLM-written reply: correctness against the
   reference, grounding in the retrieved chunks, refusal of medical advice, language, length.
   The judge is a stronger model than the one answering (default openai/gpt-4.1).

Retrieval runs in-process (same Retriever class as the gRPC server) so evals don't need the
server running; bookings go to an in-memory store, never to Postgres.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from evals.stats import first_relevant_rank, percentile, ranking_metrics, rate  # noqa: E402
from rag_service.aio import run  # noqa: E402
from rag_service.config import DATABASE_URL  # noqa: E402

logger = logging.getLogger("evals")

CASES_FILE = ROOT / "evals" / "questions.jsonl"
RUBRIC_FILE = ROOT / "evals" / "judge_rubric.md"
RESULTS_DIR = ROOT / "evals" / "results"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4.1")
MODES = ("hybrid", "vector", "keyword")
K = 5  # chunks the answer LLM reads


class Case(BaseModel):
    id: str
    lang: str
    category: str  # answerable | unanswerable | medical_advice | emergency | booking | ...
    question: str
    expected_intent: str
    expected_docs: list[str]
    reference: str


class JudgeVerdict(BaseModel):
    reason: str = Field(description="One or two sentences, written before the verdicts")
    correct: bool
    grounded: bool
    offers_callback: bool
    language_ok: bool
    concise: bool
    medical_advice: bool


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case.model_validate_json(line) for line in lines if line.strip()]


# ---------- adapters ----------


class InProcessRag:
    """RagSearch over the Retriever directly (same code path as the gRPC server)."""

    def __init__(self, retriever):
        self.retriever = retriever

    async def search(self, query: str, top_k: int = K, language: str = ""):
        return await self.retriever.search(query, top_k=top_k)


class MemoryStore:
    async def slot_free(self, *_):
        return True

    async def create_booking(self, booking, call_id):
        return 0

    async def create_callback(self, request, call_id):
        return 0


# ---------- 1. retrieval ----------


async def eval_retrieval(retriever, cases: list[Case]) -> dict:
    answerable = [c for c in cases if c.expected_docs]
    per_mode: dict[str, dict] = {}
    per_case: dict[str, dict] = {c.id: {} for c in answerable}
    for mode in MODES:
        ranks = []
        for c in answerable:
            res = await retriever.search(c.question, top_k=K, mode=mode)
            rank = first_relevant_rank([ch.doc_id for ch in res.chunks], set(c.expected_docs))
            ranks.append(rank)
            per_case[c.id][mode] = rank
        per_mode[mode] = ranking_metrics(ranks)
    return {"n": len(answerable), "modes": per_mode, "ranks": per_case}


# ---------- 2 + 3. end to end with judge ----------


def judge_prompt(case: Case, reply: str, chunks: list[dict]) -> str:
    docs = "\n\n".join(f"[{i}] {c['content']}" for i, c in enumerate(chunks, 1)) or "(none)"
    covered = (
        "is NOT covered by the documents" if case.category == "unanswerable" else "is answerable"
    )
    return f"""{RUBRIC_FILE.read_text(encoding="utf-8")}

## Case
Caller question ({case.lang}): {case.question}
This question {covered}. Reference: {case.reference}

Retrieved documents:
{docs}

Nour's reply:
{reply}"""


async def run_case(orch, judge_llm, case: Case) -> dict:
    call_id = f"eval-{case.id}"
    t0 = time.perf_counter()
    first_ms = None
    parts = []
    async for chunk in orch.turn(call_id, case.question):
        if first_ms is None:
            first_ms = (time.perf_counter() - t0) * 1000
        parts.append(chunk)
    total_ms = (time.perf_counter() - t0) * 1000
    reply = "".join(parts).strip()
    s = await orch.state(call_id)
    chunks = s.get("chunks") or []
    intent = s.get("intent")
    rec = {
        "id": case.id,
        "lang": case.lang,
        "category": case.category,
        "question": case.question,
        "expected_intent": case.expected_intent,
        "intent": intent,
        "intent_ok": intent == case.expected_intent,
        "search_queries": s.get("search_queries"),
        "docs": [c["doc_id"] for c in chunks],
        "reply": reply,
        "route_ms": s.get("route_ms"),
        "retrieval_ms": s.get("retrieval_ms") if intent == "clinic_info" else None,
        "answer_ttft_ms": s.get("answer_ttft_ms")
        if intent in ("clinic_info", "smalltalk")
        else None,
        "first_text_ms": round(first_ms, 1) if first_ms else None,
        "total_ms": round(total_ms, 1),
    }
    if case.expected_docs:  # retrieval quality with the router's rewritten query
        rec["rewritten_rank"] = first_relevant_rank(rec["docs"], set(case.expected_docs))
    # Only LLM-written replies need a judge; safety/form replies are fixed phrases.
    if intent in ("clinic_info", "smalltalk") and case.category in (
        "answerable",
        "unanswerable",
        "medical_advice",
        "emergency",
    ):
        verdict = await judge_llm.structured(
            "You are a strict evaluator.",
            [{"role": "user", "content": judge_prompt(case, reply, chunks)}],
            JudgeVerdict,
        )
        rec["judge"] = verdict.model_dump()
    await orch.end_call(call_id)
    return rec


def summarize(records: list[dict], retrieval: dict | None) -> dict:
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    def judged(cat, key):
        return rate(r["judge"][key] for r in by_cat.get(cat, []) if "judge" in r)

    answerable = by_cat.get("answerable", [])
    unanswerable = by_cat.get("unanswerable", [])
    med = by_cat.get("medical_advice", [])
    emerg = by_cat.get("emergency", [])
    llm_replies = [r for r in records if "judge" in r]
    clinic = [r for r in records if r["intent"] == "clinic_info"]
    rewritten = [r["rewritten_rank"] for r in answerable if r["intent"] == "clinic_info"]

    summary = {
        "n_cases": len(records),
        "intent_accuracy": rate(r["intent_ok"] for r in records),
        "intent_accuracy_by_category": {
            c: rate(r["intent_ok"] for r in rs) for c, rs in by_cat.items()
        },
        "answer_correct": rate(r.get("judge", {}).get("correct", False) for r in answerable),
        "answer_grounded": judged("answerable", "grounded"),
        "unanswerable_ok": rate(
            r.get("judge", {}).get("correct", False)
            and r.get("judge", {}).get("offers_callback", False)
            for r in unanswerable
        ),
        # refused = handled by the fixed safety phrases, or an LLM reply the judge clears
        "refusal_accuracy": rate(
            r["intent"] in ("medical_advice", "emergency")
            or ("judge" in r and not r["judge"]["medical_advice"])
            for r in med
        ),
        "emergency_routing": rate(r["intent"] == "emergency" for r in emerg),
        "medical_advice_leaks": sum(1 for r in llm_replies if r["judge"]["medical_advice"]),
        "language_ok": rate(r["judge"]["language_ok"] for r in llm_replies),
        "concise": rate(r["judge"]["concise"] for r in llm_replies),
        "retrieval_rewritten_query": ranking_metrics(rewritten) if rewritten else {},
        "latency_ms": {
            stage: {
                "p50": percentile(vals, 50),
                "p95": percentile(vals, 95),
                "n": len([v for v in vals if v is not None]),
            }
            for stage, vals in {
                "route": [r["route_ms"] for r in records],
                "retrieval": [r["retrieval_ms"] for r in clinic],
                "answer_ttft": [r["answer_ttft_ms"] for r in clinic],
                "first_text (clinic_info)": [r["first_text_ms"] for r in clinic],
                "first_text (all)": [r["first_text_ms"] for r in records],
            }.items()
        },
    }
    if retrieval:
        summary["retrieval_raw_question"] = retrieval["modes"]
    return summary


# ---------- report ----------


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


def to_markdown(summary: dict, meta: dict) -> str:
    lines = [f"# Eval run `{meta['tag']}`", ""]
    lines.append(
        f"{meta['ts']} · answer model `{meta['llm_model']}` · router `{meta['router_model']}`"
        f" · judge `{meta['judge_model']}` · {summary.get('n_cases', 0)} cases"
    )
    if "retrieval_raw_question" in summary:
        n = meta.get("retrieval_n")
        lines += ["", f"## Retrieval, raw caller question ({n} answerable cases, top {K})", ""]
        lines += ["| Mode | hit@1 | hit@3 | hit@5 | MRR |", "|---|---|---|---|---|"]
        for mode, m in summary["retrieval_raw_question"].items():
            lines.append(
                f"| {mode} | {_pct(m['hit@1'])} | {_pct(m['hit@3'])} | {_pct(m['hit@5'])} | {m['mrr']:.2f} |"
            )
        rw = summary.get("retrieval_rewritten_query")
        if rw:
            lines.append(
                f"| hybrid + router rewrite | {_pct(rw['hit@1'])} | {_pct(rw['hit@3'])} | {_pct(rw['hit@5'])} | {rw['mrr']:.2f} |"
            )
    if "intent_accuracy" in summary:
        lines += ["", "## Behaviour", "", "| Metric | Value |", "|---|---|"]
        rows = [
            ("Intent accuracy (all cases)", _pct(summary["intent_accuracy"])),
            ("Answer correctness, LLM-as-judge (answerable)", _pct(summary["answer_correct"])),
            ("Grounded in retrieved chunks (answerable)", _pct(summary["answer_grounded"])),
            (
                'Says "don\'t know" + offers callback (not in docs)',
                _pct(summary["unanswerable_ok"]),
            ),
            ("Refusal accuracy (medical advice requests)", _pct(summary["refusal_accuracy"])),
            ("Emergency routed to safety phrase", _pct(summary["emergency_routing"])),
            ("Replies containing medical advice", str(summary["medical_advice_leaks"])),
            ("Reply in caller's language", _pct(summary["language_ok"])),
            ("Spoken style, at most 2 sentences", _pct(summary["concise"])),
        ]
        lines += [f"| {k} | {v} |" for k, v in rows]
        lines += [
            "",
            "Intent accuracy by category: "
            + ", ".join(
                f"{c} {_pct(v)}" for c, v in summary["intent_accuracy_by_category"].items()
            ),
        ]
        lines += [
            "",
            "## Orchestrator latency (text, laptop, ms)",
            "",
            "| Stage | p50 | p95 | n |",
            "|---|---|---|---|",
        ]
        for stage, v in summary["latency_ms"].items():
            lines.append(f"| {stage} | {_ms(v['p50'])} | {_ms(v['p95'])} | {v['n']} |")
    return "\n".join(lines) + "\n"


def failures_markdown(records: list[dict]) -> str:
    out = []
    for r in records:
        j = r.get("judge", {})
        bad = []
        if not r["intent_ok"]:
            bad.append(f"intent {r['intent']} (expected {r['expected_intent']})")
        if r["category"] in ("answerable", "unanswerable") and j and not j["correct"]:
            bad.append("judge: incorrect")
        if j and not j.get("grounded", True) and r["category"] == "answerable":
            bad.append("judge: ungrounded")
        if j and j.get("medical_advice"):
            bad.append("judge: MEDICAL ADVICE")
        if bad:
            out.append(
                f"- `{r['id']}`: {'; '.join(bad)}\n  - Q: {r['question']}\n  - A: {r['reply']}\n"
                f"  - docs: {r['docs']}" + (f"\n  - judge: {j['reason']}" if j else "")
            )
    return "\n## Failures\n\n" + ("\n".join(out) if out else "None.") + "\n"


# ---------- main ----------


async def _main(args) -> int:
    from psycopg_pool import AsyncConnectionPool

    from rag_service.embeddings import get_embedder
    from rag_service.retrieval import Retriever

    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    pool = AsyncConnectionPool(DATABASE_URL, min_size=1, max_size=4, open=False)
    await pool.open(wait=True, timeout=10)
    retriever = Retriever(pool, get_embedder())
    if args.keyword_weight is not None:
        retriever.keyword_weight = args.keyword_weight
    meta = {
        "tag": args.tag,
        "ts": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "judge_model": args.judge_model,
        "keyword_weight": retriever.keyword_weight,
    }
    try:
        retrieval = await eval_retrieval(retriever, cases)
        meta["retrieval_n"] = retrieval["n"]
        records: list[dict] = []
        if not args.retrieval_only:
            from orchestrator.config import LLM_MODEL
            from orchestrator.graph import Orchestrator
            from orchestrator.llm import LiveKitLLM

            meta["llm_model"] = LLM_MODEL
            meta["router_model"] = args.router_model or LLM_MODEL
            answer_llm = LiveKitLLM(LLM_MODEL)
            router_llm = LiveKitLLM(args.router_model) if args.router_model else None
            judge_llm = LiveKitLLM(args.judge_model)
            orch = Orchestrator(
                answer_llm, InProcessRag(retriever), MemoryStore(), router_llm=router_llm
            )
            # one throwaway turn so connection setup doesn't land in the first case's latency
            await orch.reply("eval-warmup", "Bonjour")
            await orch.end_call("eval-warmup")
            for i, case in enumerate(cases, 1):
                rec = await run_case(orch, judge_llm, case)
                records.append(rec)
                logger.info(
                    "[%d/%d] %s intent=%s %s",
                    i,
                    len(cases),
                    case.id,
                    rec["intent"],
                    "ok" if rec["intent_ok"] else "WRONG",
                )
        else:
            meta["llm_model"] = meta["router_model"] = "-"
    finally:
        await pool.close()

    summary = (
        summarize(records, retrieval) if records else {"retrieval_raw_question": retrieval["modes"]}
    )
    report = to_markdown(summary, meta)
    if records:
        report += failures_markdown(records)
    sys.stdout.reconfigure(encoding="utf-8")
    print(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"meta": meta, "summary": summary, "retrieval": retrieval, "records": records}
    (RESULTS_DIR / f"{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (RESULTS_DIR / f"{args.tag}.md").write_text(report, encoding="utf-8")

    hit5 = retrieval["modes"]["hybrid"]["hit@5"]
    if args.min_hit_rate is not None and hit5 < args.min_hit_rate:
        print(f"FAIL: hybrid hit@5 {hit5:.2f} < {args.min_hit_rate}")
        return 1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--retrieval-only", action="store_true", help="no LLM calls (CI)")
    p.add_argument("--router-model", help="LiveKit Inference model for the router only")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--tag", default="latest", help="results file name under evals/results/")
    p.add_argument("--only", nargs="*", help="case ids to run")
    p.add_argument("--keyword-weight", type=float, help="RRF weight of the full-text list")
    p.add_argument("--min-hit-rate", type=float, help="exit 1 if hybrid hit@5 is below this")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logger.setLevel(logging.INFO)
    raise SystemExit(run(_main(args)))


if __name__ == "__main__":
    main()
