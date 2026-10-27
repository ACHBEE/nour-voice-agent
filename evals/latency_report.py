"""p50 / p95 per pipeline stage from real calls (latency_log.jsonl written by agent/agent.py).

    python evals/latency_report.py                       # whole log
    python evals/latency_report.py --since 2026-10-28    # only calls after a date (ISO prefix)
    python evals/latency_report.py --room console        # one room / call

Stages, in the order a turn goes through them:
  user rows (LiveKit):        end_of_turn_delay, transcription_delay
  orchestrator rows (ours):   route_ms, retrieval_ms, rag_server_ms, answer_ttft_ms, first_text_ms
  assistant rows (LiveKit):   llm_node_ttft (= first text out of our llm_node), tts_node_ttfb,
                              e2e_latency (end of user speech -> first agent audio)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.stats import percentile  # noqa: E402

LOG = ROOT / "latency_log.jsonl"
OUT = ROOT / "evals" / "results" / "latency.md"

# (row role, field, label, unit multiplier to ms)
STAGES = [
    ("user", "end_of_turn_delay", "End of turn detected (after speech ends)", 1000),
    ("user", "transcription_delay", "Final transcript ready (STT)", 1000),
    ("orchestrator", "route_ms", "Orchestrator: router LLM", 1),
    ("orchestrator", "retrieval_ms", "Orchestrator: multi-query retrieval", 1),
    ("orchestrator", "answer_ttft_ms", "Orchestrator: answer LLM first token", 1),
    ("orchestrator", "first_text_ms", "Orchestrator: first reply text (no filler)", 1),
    ("assistant", "llm_node_ttft", "First text out of llm_node (filler counts)", 1000),
    ("assistant", "tts_node_ttfb", "TTS first audio byte", 1000),
    ("assistant", "e2e_latency", "End to end: caller stops -> Nour starts", 1000),
]


def load_rows(path: Path, since: str | None = None, room: str | None = None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if since and row.get("ts", "") < since:
            continue
        if room and row.get("room") != room:
            continue
        rows.append(row)
    return rows


def stage_stats(rows: list[dict]) -> list[dict]:
    out = []
    for role, field, label, mult in STAGES:
        vals = [r[field] * mult for r in rows if r.get("role") == role and r.get(field)]
        out.append(
            {
                "stage": label,
                "field": field,
                "n": len(vals),
                "p50": percentile(vals, 50),
                "p95": percentile(vals, 95),
            }
        )
    return out


def to_markdown(rows: list[dict], stats: list[dict], source: str) -> str:
    rooms = {r.get("room") for r in rows}
    ts = sorted(r["ts"] for r in rows if "ts" in r)
    span = f"{ts[0][:16]} to {ts[-1][:16]} UTC" if ts else "no rows"
    lines = [
        "# Latency report",
        "",
        f"Source `{source}`: {len(rows)} rows, {len(rooms)} call(s), {span}.",
        "",
        "| Stage | p50 (ms) | p95 (ms) | n |",
        "|---|---|---|---|",
    ]
    for s in stats:
        p50 = "-" if s["p50"] is None else f"{s['p50']:.0f}"
        p95 = "-" if s["p95"] is None else f"{s['p95']:.0f}"
        lines.append(f"| {s['stage']} | {p50} | {p95} | {s['n']} |")
    orch = [r for r in rows if r.get("role") == "orchestrator"]
    if orch:
        fillers = sum(1 for r in orch if r.get("filler"))
        intents = Counter(r.get("intent") for r in orch)
        lines += [
            "",
            f"Filler phrase spoken on {fillers}/{len(orch)} turns. Intents: "
            + ", ".join(f"{k} {v}" for k, v in intents.most_common())
            + ".",
        ]
    interrupted = sum(1 for r in rows if r.get("role") == "assistant" and r.get("interrupted"))
    lines.append(f"Assistant turns interrupted by the caller: {interrupted}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--file", type=Path, default=LOG)
    p.add_argument("--since", help="ISO timestamp prefix, e.g. 2026-10-28 or 2026-10-28T09:00")
    p.add_argument("--room", help="only this room / call id")
    p.add_argument("--out", type=Path, default=OUT, help="markdown copy of the report")
    args = p.parse_args()
    if not args.file.exists():
        raise SystemExit(f"{args.file} not found: make a call with `python agent/agent.py console`")
    rows = load_rows(args.file, args.since, args.room)
    report = to_markdown(rows, stage_stats(rows), args.file.name)
    sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
