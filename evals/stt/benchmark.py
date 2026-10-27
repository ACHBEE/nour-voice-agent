"""Benchmark LiveKit Inference STT models on recorded Darja / French / English phrases.

    python evals/stt/record.py --input-device 20     # once: record evals/stt/recordings/*.wav
    python evals/stt/benchmark.py                    # all candidates
    python evals/stt/benchmark.py --models "deepgram/nova-3:ar-DZ"

Each recording is streamed in real time (20 ms frames, then 1.5 s of silence) exactly like a live
call, and we measure:
  - WER / CER: word / character error rate after Arabic/Latin normalization (metrics.py)
  - latency: time from the end of speech to the last final transcript, lower is better
Results: evals/stt/results.md (table) and evals/stt/results.json (every transcript).
Needs LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET in .env.local.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import inference, stt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from metrics import cer, wer  # noqa: E402

SR = 16_000
FRAME = SR // 50  # 20 ms
TRAILING_SILENCE_S = 1.5
TIMEOUT_S = 30

# Clinic + Darja vocabulary for keyterm-biasing variants (Deepgram "keyterm").
KEYTERMS = [
    "عيادة الأمل",
    "كارط الشفاء",
    "الكناص",
    "رونديفو",
    "الكونسيلتاسيون",
    "Chifa",
    "CNAS",
    "échographie",
    "cardiologue",
    "pédiatre",
    "El Oued",
]


@dataclass
class Candidate:
    label: str
    model: str
    language: str
    extra: dict = field(default_factory=dict)


CANDIDATES = [
    Candidate("deepgram/nova-3:multi", "deepgram/nova-3", "multi"),
    Candidate("deepgram/nova-3:ar-DZ", "deepgram/nova-3", "ar-DZ"),
    Candidate("deepgram/nova-3:ar-DZ +keyterms", "deepgram/nova-3", "ar-DZ", {"keyterm": KEYTERMS}),
    Candidate("openai/gpt-live-transcribe:ar", "openai/gpt-live-transcribe", "ar"),
    Candidate("google/gemini-3.5-transcribe-live:ar", "google/gemini-3.5-transcribe-live", "ar"),
    Candidate("speechmatics/linden-1:ar", "speechmatics/linden-1", "ar"),
    Candidate("assemblyai/universal-3-6-pro:ar", "assemblyai/universal-3-6-pro", "ar"),
]


def load_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


def frames(pcm: bytes) -> list[rtc.AudioFrame]:
    step = FRAME * 2
    out = []
    for i in range(0, len(pcm), step):
        chunk = pcm[i : i + step].ljust(step, b"\x00")
        out.append(rtc.AudioFrame(chunk, SR, 1, FRAME))
    return out


def last_voiced_frame(pcm: bytes) -> int:
    """Index of the last 20 ms frame containing speech (energy above 10% of the loudest frame),
    so latency is measured from the end of speech, not from the end of the file's silence."""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = len(a) // FRAME
    if n == 0:
        return 0
    rms = np.sqrt((a[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))
    voiced = np.nonzero(rms > max(rms.max() * 0.1, 1.0))[0]
    return int(voiced[-1]) if len(voiced) else n - 1


async def transcribe(engine: stt.STT, pcm: bytes) -> tuple[str, float | None]:
    stream = engine.stream()
    speech_end = 0.0
    last_final: float | None = None
    texts: list[str] = []
    end_frame = last_voiced_frame(pcm)

    async def push() -> None:
        nonlocal speech_end
        for i, f in enumerate(frames(pcm)):
            stream.push_frame(f)
            await asyncio.sleep(0.02)
            if i == end_frame:
                speech_end = time.perf_counter()
        for f in frames(b"\x00" * int(SR * TRAILING_SILENCE_S) * 2):
            stream.push_frame(f)
            await asyncio.sleep(0.02)
        stream.end_input()

    pusher = asyncio.create_task(push())
    try:
        async for ev in stream:
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                text = ev.alternatives[0].text.strip()
                if text:
                    texts.append(text)
                    last_final = time.perf_counter()
        await pusher
    finally:
        pusher.cancel()
        await stream.aclose()
    latency = (last_final - speech_end) if last_final and speech_end else None
    return " ".join(texts), latency


async def run_candidate(
    c: Candidate, phrases: list[dict], session: aiohttp.ClientSession, rec_dir: Path
) -> dict:
    engine = inference.STT(
        model=c.model,
        language=c.language,
        http_session=session,
        **({"extra_kwargs": c.extra} if c.extra else {}),
    )
    rows = []
    for p in phrases:
        pcm = load_pcm(rec_dir / f"{p['id']}.wav")
        try:
            hyp, lat = await asyncio.wait_for(transcribe(engine, pcm), TIMEOUT_S)
            scores = {"wer": wer(p["text"], hyp), "cer": cer(p["text"], hyp)}
            rows.append({**p, "hyp": hyp, **scores, "latency": lat, "error": None})
        except Exception as e:  # noqa: BLE001 - one failing provider must not kill the benchmark
            rows.append(
                {
                    **p,
                    "hyp": "",
                    "wer": None,
                    "cer": None,
                    "latency": None,
                    "error": f"{type(e).__name__}: {e}"[:200],
                }
            )
        print(
            f"  {c.label:<40} {p['id']:<22} {rows[-1]['hyp'] or rows[-1]['error'] or '(empty)'}",
            flush=True,
        )
    return {"label": c.label, "rows": rows}


def _mean(xs: list[float]) -> str:
    return f"{statistics.mean(xs):.2f}" if xs else "–"


def summarize(results: list[dict]) -> str:
    """Markdown table ranked by overall WER (lower is better)."""
    groups = [("Darja", "darja"), ("Darja+FR", "darja+fr"), ("French", "fr"), ("English", "en")]
    head = "| Model | " + " | ".join(f"WER {g}" for g, _ in groups)
    head += " | WER all | CER all | p50 latency (s) | errors |"
    lines = [head, "|" + "---|" * (len(groups) + 5)]

    def ok_rows(r: dict) -> list[dict]:
        return [x for x in r["rows"] if x["error"] is None]

    def overall(r: dict) -> float:
        return statistics.mean([x["wer"] for x in ok_rows(r)] or [99.0])

    for r in sorted(results, key=overall):
        ok = ok_rows(r)
        cells = [_mean([x["wer"] for x in ok if x["lang"] == lang]) for _, lang in groups]
        lats = [x["latency"] for x in ok if x["latency"] is not None]
        p50 = f"{statistics.median(lats):.2f}" if lats else "–"
        errs = len(r["rows"]) - len(ok)
        lines.append(
            f"| {r['label']} | "
            + " | ".join(cells)
            + f" | {_mean([x['wer'] for x in ok])} | {_mean([x['cer'] for x in ok])} | {p50} | {errs} |"
        )
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", help="comma-separated candidate labels (default: all)")
    ap.add_argument(
        "--recordings", type=Path, default=HERE / "recordings", help="folder of <id>.wav"
    )
    ap.add_argument("--only", help="comma-separated phrase ids (default: all)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252: no Arabic
    load_dotenv(ROOT / ".env.local")
    phrases = [
        json.loads(line)
        for line in (HERE / "phrases.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        phrases = [p for p in phrases if p["id"] in keep]
    missing = [p["id"] for p in phrases if not (args.recordings / f"{p['id']}.wav").exists()]
    if missing:
        sys.exit(
            f"Missing recordings: {missing}. Run: python evals/stt/record.py --input-device <id>"
        )

    wanted = {m.strip() for m in args.models.split(",")} if args.models else None
    candidates = [c for c in CANDIDATES if not wanted or c.label in wanted]

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *(run_candidate(c, phrases, session, args.recordings) for c in candidates)
        )

    table = summarize(results)
    (HERE / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (HERE / "results.md").write_text(
        f"# STT benchmark ({len(phrases)} phrases, one speaker)\n\n{table}\n", encoding="utf-8"
    )
    print("\n" + table)


if __name__ == "__main__":
    asyncio.run(main())
