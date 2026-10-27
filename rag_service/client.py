"""Async gRPC client for the RAG service (used by the orchestrator and the voice agent).

python -m rag_service.client "Combien coûte une consultation ?"
python -m rag_service.client "واش تخدمو نهار الجمعة؟" --lang ar --k 3 --mode vector
"""

from __future__ import annotations

import argparse
import sys
import time

import grpc

from rag_service.aio import run
from rag_service.config import RAG_GRPC_ADDR
from rag_service.pb import rag_pb2, rag_pb2_grpc
from rag_service.retrieval import Mode, RetrievedChunk, SearchResult

_MODES = {
    "hybrid": rag_pb2.SEARCH_MODE_HYBRID,
    "vector": rag_pb2.SEARCH_MODE_VECTOR,
    "keyword": rag_pb2.SEARCH_MODE_KEYWORD,
}


class RagClient:
    """Thin typed wrapper: proto messages in, Pydantic models out."""

    def __init__(self, addr: str = RAG_GRPC_ADDR, timeout_s: float = 3.0):
        self.addr = addr
        self.timeout_s = timeout_s
        self._channel: grpc.aio.Channel | None = None
        self._stub: rag_pb2_grpc.RagServiceStub | None = None

    async def __aenter__(self) -> RagClient:
        self._channel = grpc.aio.insecure_channel(self.addr)
        self._stub = rag_pb2_grpc.RagServiceStub(self._channel)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None

    async def search(
        self, query: str, top_k: int = 5, language: str = "", mode: Mode = "hybrid"
    ) -> SearchResult:
        if self._stub is None:
            await self.__aenter__()
        resp = await self._stub.Search(
            rag_pb2.SearchRequest(query=query, top_k=top_k, language=language, mode=_MODES[mode]),
            timeout=self.timeout_s,
        )
        return SearchResult(
            chunks=[
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    title=c.title,
                    section=c.section,
                    content=c.content,
                    lang=c.lang,
                    source_path=c.source_path,
                    score=c.score,
                    vector_rank=c.vector_rank,
                    keyword_rank=c.keyword_rank,
                )
                for c in resp.chunks
            ],
            embed_ms=resp.embed_ms,
            search_ms=resp.search_ms,
        )


async def _cli() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252: no Arabic
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lang", default="")
    ap.add_argument("--mode", choices=list(_MODES), default="hybrid")
    ap.add_argument("--addr", default=RAG_GRPC_ADDR)
    args = ap.parse_args()

    t0 = time.perf_counter()
    async with RagClient(args.addr) as client:
        res = await client.search(args.query, top_k=args.k, language=args.lang, mode=args.mode)
    rtt = (time.perf_counter() - t0) * 1000
    print(f"round trip {rtt:.0f} ms (embed {res.embed_ms:.0f} ms, search {res.search_ms:.0f} ms)")
    for i, c in enumerate(res.chunks, 1):
        snippet = c.content.replace("\n", " / ")[:110]
        print(f"{i}. [{c.score:.4f} v{c.vector_rank} k{c.keyword_rank}] {c.doc_id} › {c.section}")
        print(f"   {snippet}")


if __name__ == "__main__":
    run(_cli())
