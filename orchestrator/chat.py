"""Text chat with the orchestrator, without voice. Needs Postgres and the RAG gRPC server running.

python -m orchestrator.chat                         # interactive, empty line to quit
python -m orchestrator.chat "msg 1" "msg 2" ...     # scripted turns in one call
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid

from orchestrator.graph import Orchestrator
from orchestrator.llm import LiveKitLLM
from orchestrator.store import BookingStore
from rag_service.aio import run
from rag_service.client import RagClient
from rag_service.config import RAG_GRPC_ADDR


def _debug_line(s: dict) -> str:
    parts = [
        f"intent={s.get('intent')}",
        f"lang={s.get('language')}",
        f"route {s.get('route_ms', 0):.0f}ms",
    ]
    if s.get("intent") == "clinic_info":
        docs = ",".join(dict.fromkeys(c["doc_id"] for c in s.get("chunks", []))) or "-"
        parts += [
            f"rag {s.get('retrieval_ms', 0):.0f}ms",
            f"ttft {s.get('answer_ttft_ms', 0):.0f}ms",
            f"docs={docs}",
        ]
    for kind in ("booking", "callback"):
        if s.get(f"{kind}_id"):
            parts.append(f"{kind} #{s[f'{kind}_id']} saved")
        elif s.get(kind):
            filled = {k: v for k, v in s[kind]["fields"].items() if v}
            parts.append(f"{kind}={filled}")
    return "   [" + " | ".join(parts) + "]"


async def _main(turns: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252: no Arabic
    store = await BookingStore.open()
    llm = LiveKitLLM()
    call_id = f"chat-{uuid.uuid4().hex[:8]}"
    try:
        async with RagClient(RAG_GRPC_ADDR) as rag:
            orch = Orchestrator(llm, rag, store)
            interactive = not turns
            while True:
                if interactive:
                    text = (await asyncio.to_thread(input, "you> ")).strip()
                elif turns:
                    text = turns.pop(0)
                    print(f"you> {text}")
                else:
                    break
                if not text:
                    break
                print("nour> ", end="", flush=True)
                async for chunk in orch.turn(call_id, text):
                    print(chunk, end="", flush=True)
                print()
                print(_debug_line(await orch.state(call_id)))
    finally:
        await llm.aclose()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("turns", nargs="*", help="caller messages to send in order")
    parser.add_argument("-v", "--verbose", action="store_true", help="show node logs")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        run(_main(args.turns))
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
