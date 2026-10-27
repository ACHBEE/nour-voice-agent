"""Integration tests for db/schema.sql against a live Postgres + pgvector.

Skipped automatically when the database is not reachable (start it with `docker compose up -d`).
Every test runs inside a transaction that is rolled back, so the database is left untouched.
"""

import os

import pytest

psycopg = pytest.importorskip("psycopg")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://nour:nour_dev@localhost:5433/nour")


@pytest.fixture(scope="module")
def db_available():
    try:
        psycopg.connect(DATABASE_URL, connect_timeout=3).close()
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")


@pytest.fixture
def conn(db_available):
    c = psycopg.connect(DATABASE_URL)
    try:
        yield c
    finally:
        c.rollback()
        c.close()


def _vec(*head: float) -> str:
    values = list(head) + [0.0] * (384 - len(head))
    return "[" + ",".join(str(v) for v in values) + "]"


def test_pgvector_extension_and_hnsw_index(conn):
    assert conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
    indexdef = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'"
    ).fetchone()[0]
    assert "hnsw" in indexdef and "vector_cosine_ops" in indexdef


def test_chunk_insert_vector_and_fulltext_search(conn):
    doc = conn.execute(
        "INSERT INTO documents (doc_id, title, lang, category, source_path, content_hash) "
        "VALUES ('t_doc', 'Test', 'fr', 'test', 't.md', 'h') RETURNING id"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO chunks (document_id, chunk_index, content, lang, embedding, ts_config) "
        "VALUES (%s, 0, 'Les consultations coûtent 2000 DZD', 'fr', %s, nour_ts_config('fr'))",
        (doc, _vec(1.0)),
    )
    # nearest neighbour by cosine distance
    dist = conn.execute(
        "SELECT embedding <=> %s::vector FROM chunks WHERE document_id = %s", (_vec(1.0), doc)
    ).fetchone()[0]
    assert dist == pytest.approx(0.0, abs=1e-6)
    # french stemming: "consultation" matches "consultations"
    hit = conn.execute(
        "SELECT count(*) FROM chunks WHERE document_id = %s "
        "AND tsv @@ websearch_to_tsquery(nour_ts_config('fr'), 'consultation')",
        (doc,),
    ).fetchone()[0]
    assert hit == 1


def test_booking_constraints(conn):
    insert = (
        "INSERT INTO bookings (patient_name, phone, department, appointment_date, appointment_time) "
        "VALUES (%s, %s, %s, '2026-11-02', '10:30')"
    )
    conn.execute(insert, ("Amina B.", "0661234567", "pediatrics"))
    with pytest.raises(psycopg.errors.UniqueViolation):  # same slot twice
        with conn.transaction():
            conn.execute(insert, ("Omar K.", "0551234567", "pediatrics"))
    with pytest.raises(psycopg.errors.CheckViolation):  # unknown department
        with conn.transaction():
            conn.execute(insert, ("Omar K.", "0551234567", "ophthalmology"))
    with pytest.raises(psycopg.errors.CheckViolation):  # malformed phone
        with conn.transaction():
            conn.execute(insert, ("Omar K.", "12ab", "dentistry"))
