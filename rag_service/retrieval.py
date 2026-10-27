"""Hybrid retrieval over pgvector: dense vector search + Postgres full-text, fused with RRF."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal, Protocol

from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from rag_service.fusion import rrf

logger = logging.getLogger("rag.retrieval")

Mode = Literal["hybrid", "vector", "keyword"]
CANDIDATES = 20  # per-list depth before fusion; well under HNSW's default ef_search (40)


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class RetrievedChunk(BaseModel):
    chunk_id: int
    doc_id: str
    title: str
    section: str
    content: str
    lang: str
    source_path: str
    score: float
    vector_rank: int = 0  # 1-based, 0 = not in that list
    keyword_rank: int = 0


class SearchResult(BaseModel):
    chunks: list[RetrievedChunk]
    embed_ms: float
    search_ms: float


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


_VECTOR_SQL = """
SELECT id, 1 - (embedding <=> %(q)s::vector) AS score
FROM chunks
ORDER BY embedding <=> %(q)s::vector
LIMIT %(n)s
"""

# 1. Drop words that are a stopword in ANY of our languages: each chunk parses the query with its
#    own config, so English "the"/"is" would otherwise survive as junk terms against French and
#    Arabic chunks (and vice versa). to_tsvector is used for the test because it doesn't emit
#    NOTICEs on stopwords the way plainto_tsquery does.
# 2. Parse the remaining words with each chunk's own config (french/english/arabic stemming).
# 3. plainto_tsquery ANDs every word, which almost never matches a spoken question, so terms are
#    rewritten to OR; ts_rank_cd then rewards chunks matching more (and closer) terms.
_KEYWORD_SQL = """
WITH terms AS (
    SELECT coalesce(string_agg(w, ' '), '') AS q
    FROM regexp_split_to_table(%(q)s, '\\s+') AS w
    WHERE length(to_tsvector('english', w)) > 0
      AND length(to_tsvector('french', w)) > 0
      AND length(to_tsvector(nour_ts_config('ar'), w)) > 0
)
SELECT c.id, ts_rank_cd(c.tsv, q.query) AS score
FROM chunks c
CROSS JOIN terms t
CROSS JOIN LATERAL (
    SELECT to_tsquery(c.ts_config,
                      replace(plainto_tsquery(c.ts_config, t.q)::text, ' & ', ' | ')) AS query
) q
WHERE numnode(q.query) > 0 AND c.tsv @@ q.query
ORDER BY score DESC, c.id
LIMIT %(n)s
"""

_DETAILS_SQL = """
SELECT c.id, d.doc_id, d.title, coalesce(c.section, ''), c.content, c.lang, d.source_path
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.id = ANY(%(ids)s)
"""


class Retriever:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder, candidates: int = CANDIDATES):
        self.pool = pool
        self.embedder = embedder
        self.candidates = candidates

    async def search(self, query: str, top_k: int = 5, mode: Mode = "hybrid") -> SearchResult:
        t0 = time.perf_counter()
        qvec = None
        if mode in ("hybrid", "vector"):
            # embedding is CPU-bound: keep it off the event loop
            qvec = to_pgvector(await asyncio.to_thread(self.embedder.embed_query, query))
        t1 = time.perf_counter()

        async with self.pool.connection() as conn:
            vec_rows = kw_rows = []
            if qvec is not None:
                cur = await conn.execute(_VECTOR_SQL, {"q": qvec, "n": self.candidates})
                vec_rows = await cur.fetchall()
            if mode in ("hybrid", "keyword"):
                cur = await conn.execute(_KEYWORD_SQL, {"q": query, "n": self.candidates})
                kw_rows = await cur.fetchall()

            if mode == "hybrid":
                fused = rrf(
                    {"vector": [r[0] for r in vec_rows], "keyword": [r[0] for r in kw_rows]}
                )
                ranked = [(f.id, f.score) for f in fused[:top_k]]
            else:
                ranked = [(r[0], float(r[1])) for r in (vec_rows or kw_rows)[:top_k]]

            vec_rank = {r[0]: i for i, r in enumerate(vec_rows, 1)}
            kw_rank = {r[0]: i for i, r in enumerate(kw_rows, 1)}
            cur = await conn.execute(_DETAILS_SQL, {"ids": [cid for cid, _ in ranked]})
            details = {row[0]: row for row in await cur.fetchall()}
        t2 = time.perf_counter()

        chunks = [
            RetrievedChunk(
                chunk_id=cid,
                doc_id=details[cid][1],
                title=details[cid][2],
                section=details[cid][3],
                content=details[cid][4],
                lang=details[cid][5],
                source_path=details[cid][6],
                score=score,
                vector_rank=vec_rank.get(cid, 0),
                keyword_rank=kw_rank.get(cid, 0),
            )
            for cid, score in ranked
        ]
        result = SearchResult(chunks=chunks, embed_ms=(t1 - t0) * 1000, search_ms=(t2 - t1) * 1000)
        logger.info(
            "search mode=%s k=%d embed=%.0fms search=%.0fms top=%s",
            mode,
            top_k,
            result.embed_ms,
            result.search_ms,
            [c.doc_id for c in chunks[:3]],
        )
        return result
