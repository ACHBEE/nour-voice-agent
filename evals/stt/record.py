"""Record the STT benchmark phrases from your microphone (16 kHz mono WAV, one file per phrase).

    python evals/stt/record.py --input-device 20          # all phrases
    python evals/stt/record.py --input-device 20 --only darja_01_friday
    python evals/stt/record.py --list-devices

For each phrase: press Enter, say it naturally (like on the phone), press Enter again.
Recordings go to evals/stt/recordings/ (gitignored: it's your voice).
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

HERE = Path(__file__).resolve().parent
PHRASES = HERE / "phrases.jsonl"
OUT_DIR = HERE / "recordings"
TARGET_SR = 16_000


def load_phrases() -> list[dict]:
    return [json.loads(line) for line in PHRASES.read_text(encoding="utf-8").splitlines() if line]


def record_until_enter(device: int | str | None) -> tuple[np.ndarray, int]:
    sr = int(sd.query_devices(device, "input")["default_samplerate"])
    chunks: list[np.ndarray] = []
    with sd.InputStream(
        device=device,
        channels=1,
        samplerate=sr,
        dtype="float32",
        callback=lambda data, *_: chunks.append(data.copy()),
    ):
        input("   recording... press Enter to stop ")
    audio = np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype=np.float32)
    return audio, sr


def to_16k_int16(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != TARGET_SR and len(audio):
        n_out = int(len(audio) * TARGET_SR / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def save_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input-device", help="device id or name substring (see --list-devices)")
    ap.add_argument("--only", help="record a single phrase id")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252: no Arabic

    if args.list_devices:
        print(sd.query_devices())
        return

    device = (
        int(args.input_device)
        if args.input_device and args.input_device.isdigit()
        else args.input_device
    )
    OUT_DIR.mkdir(exist_ok=True)
    phrases = [p for p in load_phrases() if not args.only or p["id"] == args.only]

    for i, p in enumerate(phrases, 1):
        print(f"\n[{i}/{len(phrases)}] {p['id']}  ({p['lang']})\n   say:  {p['text']}")
        input("   press Enter to start ")
        audio, sr = record_until_enter(device)
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if peak < 0.01:
            print(
                f"   WARNING: almost silent (peak {peak:.3f}). Check the mic, then re-run with --only {p['id']}"
            )
        pcm = to_16k_int16(audio, sr)
        save_wav(OUT_DIR / f"{p['id']}.wav", pcm)
        print(f"   saved {len(pcm) / TARGET_SR:.1f}s  (peak {peak:.2f})")

    print(f"\nDone. Recordings in {OUT_DIR}")


if __name__ == "__main__":
    main()
