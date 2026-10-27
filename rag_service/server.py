"""gRPC server for the RAG service.

    python -m rag_service.server          # listens on RAG_GRPC_PORT (default 50051)

Loads the embedding model once at startup (warm), keeps a Postgres connection pool, and exposes
the standard gRPC health service so Docker / orchestrators can probe readiness.
"""

from __future__ import annotations

import logging

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from psycopg_pool import AsyncConnectionPool

from rag_service.aio import run
from rag_service.config import DATABASE_URL, DEFAULT_TOP_K, RAG_GRPC_PORT
from rag_service.embeddings import get_embedder
from rag_service.pb import rag_pb2, rag_pb2_grpc
from rag_service.retrieval import Mode, Retriever, SearchResult

logger = logging.getLogger("rag.server")

MODES: dict[int, Mode] = {
    rag_pb2.SEARCH_MODE_HYBRID: "hybrid",
    rag_pb2.SEARCH_MODE_VECTOR: "vector",
    rag_pb2.SEARCH_MODE_KEYWORD: "keyword",
}
MAX_TOP_K = 20


def to_proto(result: SearchResult) -> rag_pb2.SearchResponse:
    return rag_pb2.SearchResponse(
        chunks=[rag_pb2.Chunk(**c.model_dump()) for c in result.chunks],
        embed_ms=result.embed_ms,
        search_ms=result.search_ms,
    )


class RagServicer(rag_pb2_grpc.RagServiceServicer):
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    async def Search(  # noqa: N802 - name fixed by the proto
        self, request: rag_pb2.SearchRequest, context: grpc.aio.ServicerContext
    ) -> rag_pb2.SearchResponse:
        query = request.query.strip()
        if not query:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query must not be empty")
        top_k = request.top_k or DEFAULT_TOP_K
        if not 1 <= top_k <= MAX_TOP_K:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"top_k must be 1..{MAX_TOP_K}")
        mode = MODES.get(request.mode)
        if mode is None:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"unknown mode {request.mode}")
        logger.info("Search lang=%s mode=%s k=%d q=%r", request.language or "?", mode, top_k, query)
        result = await self.retriever.search(query, top_k=top_k, mode=mode)
        return to_proto(result)


async def build_server(retriever: Retriever, port: int) -> tuple[grpc.aio.Server, int]:
    server = grpc.aio.server()
    rag_pb2_grpc.add_RagServiceServicer_to_server(RagServicer(retriever), server)
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    bound = server.add_insecure_port(f"[::]:{port}")
    await health_servicer.set("nour.rag.v1.RagService", health_pb2.HealthCheckResponse.SERVING)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    return server, bound


async def serve() -> None:
    embedder = get_embedder()
    embedder.embed_query("warmup")  # load weights before accepting traffic
    pool = AsyncConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=False)
    await pool.open(wait=True, timeout=15)  # fail fast instead of retrying forever
    try:
        server, port = await build_server(Retriever(pool, embedder), RAG_GRPC_PORT)
        await server.start()
        logger.info("RAG gRPC server listening on :%d", port)
        await server.wait_for_termination()
    finally:
        await pool.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run(serve())


if __name__ == "__main__":
    main()
