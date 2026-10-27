"""gRPC contract tests: a real server and client over a local port, with a fake retriever.

No database or embedding model needed; this checks the proto mapping, validation and health.
"""

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from rag_service.aio import run
from rag_service.client import RagClient
from rag_service.retrieval import RetrievedChunk, SearchResult
from rag_service.server import build_server


class FakeRetriever:
    def __init__(self):
        self.calls = []

    async def search(self, query, top_k=5, mode="hybrid"):
        self.calls.append((query, top_k, mode))
        chunk = RetrievedChunk(
            chunk_id=7,
            doc_id="prices",
            title="Tarifs",
            section="Consultations",
            content="Consultation de médecine générale : 2000 DZD.",
            lang="fr",
            source_path="05_prices_fr.md",
            score=0.0328,
            vector_rank=1,
            keyword_rank=2,
        )
        return SearchResult(chunks=[chunk] * min(top_k, 2), embed_ms=12.5, search_ms=3.0)


async def _with_server(fn):
    retriever = FakeRetriever()
    server, port = await build_server(retriever, 0)  # 0 = any free port
    await server.start()
    try:
        return await fn(f"localhost:{port}", retriever)
    finally:
        await server.stop(None)


def test_search_round_trip_maps_every_field():
    async def body(addr, retriever):
        async with RagClient(addr) as client:
            return await client.search(
                "Combien ?", top_k=3, language="fr", mode="vector"
            ), retriever

    res, retriever = run(_with_server(body))
    assert retriever.calls == [("Combien ?", 3, "vector")]
    assert len(res.chunks) == 2
    c = res.chunks[0]
    assert (c.doc_id, c.section, c.lang, c.vector_rank, c.keyword_rank) == (
        "prices",
        "Consultations",
        "fr",
        1,
        2,
    )
    assert "2000 DZD" in c.content
    assert c.score == pytest.approx(0.0328, rel=1e-5)
    assert res.embed_ms == pytest.approx(12.5)


@pytest.mark.parametrize(("query", "top_k"), [("   ", 3), ("ok", 99)])
def test_invalid_requests_are_rejected(query, top_k):
    async def body(addr, _):
        async with RagClient(addr) as client:
            await client.search(query, top_k=top_k)

    with pytest.raises(grpc.aio.AioRpcError) as e:
        run(_with_server(body))
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_health_service_reports_serving():
    async def body(addr, _):
        async with grpc.aio.insecure_channel(addr) as ch:
            stub = health_pb2_grpc.HealthStub(ch)
            return await stub.Check(health_pb2.HealthCheckRequest(service="nour.rag.v1.RagService"))

    resp = run(_with_server(body))
    assert resp.status == health_pb2.HealthCheckResponse.SERVING
