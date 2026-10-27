"""End-to-end retrieval against live Postgres + the real embedding model.

Skipped when the database isn't reachable. Ingestion runs first (idempotent: unchanged docs are
skipped), so this also tests the ingest pipeline. The cases are cross-lingual on purpose:
a Darja or English question whose answer lives in a French or Arabic document.
"""

import pytest

psycopg = pytest.importorskip("psycopg")

from rag_service.aio import run  # noqa: E402
from rag_service.config import DATABASE_URL  # noqa: E402

CASES = [
    # (query, acceptable doc_ids in the top 5 = the context the LLM will read)
    ("Combien coûte une consultation chez le généraliste ?", {"prices", "faq_darja"}),
    ("واش تخدمو نهار الجمعة؟", {"opening_hours", "faq_darja"}),
    ("Do you accept the CNAS Chifa card?", {"insurance_cnas", "faq_darja"}),
    ("Quels sont les horaires du laboratoire ?", {"opening_hours", "lab_preparation"}),
    ("عندي وجع في صدري واش ندير؟", {"emergencies", "cardiology", "faq_darja"}),
    # Known hard case: keyword search only matches English chunks, so hybrid ranks
    # emergencies/Children first and the French doctors chunk 4th (vector alone ranks it 1st).
    ("Who is the pediatrician and which days?", {"doctors", "pediatrics_vaccination"}),
    ("Do I need to fast before a blood test?", {"lab_preparation", "faq_darja"}),
    ("وين راهي العيادة؟", {"about_location", "faq_darja"}),
]


@pytest.fixture(scope="module")
def ingested():
    try:
        conn = psycopg.connect(DATABASE_URL, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")
    from rag_service.docs import load_docs
    from rag_service.ingest import ingest

    with conn:
        stats = ingest(conn, load_docs())
        n_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return stats, n_chunks


def _search(query, mode="hybrid", top_k=5):
    from psycopg_pool import AsyncConnectionPool

    from rag_service.embeddings import get_embedder
    from rag_service.retrieval import Retriever

    async def go():
        pool = AsyncConnectionPool(DATABASE_URL, min_size=1, max_size=2, open=False)
        await pool.open(wait=True, timeout=10)
        try:
            return await Retriever(pool, get_embedder()).search(query, top_k=top_k, mode=mode)
        finally:
            await pool.close()

    return run(go())


def test_ingest_populates_all_docs(ingested):
    stats, n_chunks = ingested
    assert stats["docs"] == 15
    assert n_chunks >= 60


@pytest.mark.parametrize(("query", "expected"), CASES)
def test_hybrid_top5_contains_answer_doc(ingested, query, expected):
    res = _search(query)
    got = [c.doc_id for c in res.chunks]
    assert expected & set(got), f"{query!r} -> {got}"


def test_modes_report_their_own_ranks(ingested):
    vec = _search("Combien coûte une échographie ?", mode="vector")
    kw = _search("Combien coûte une échographie ?", mode="keyword")
    hyb = _search("Combien coûte une échographie ?", mode="hybrid")
    assert all(c.vector_rank > 0 and c.keyword_rank == 0 for c in vec.chunks)
    assert all(c.keyword_rank > 0 and c.vector_rank == 0 for c in kw.chunks)
    assert all(c.vector_rank > 0 or c.keyword_rank > 0 for c in hyb.chunks)
    assert hyb.embed_ms > 0 and kw.embed_ms < vec.embed_ms  # keyword mode skips embedding


def test_keyword_only_query_with_no_lexemes_returns_nothing(ingested):
    assert _search("the of a", mode="keyword").chunks == []
