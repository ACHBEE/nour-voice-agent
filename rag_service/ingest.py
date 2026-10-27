"""Ingest data/clinic_docs/ into Postgres: chunk, embed, upsert.

    python -m rag_service.ingest            # incremental: only changed docs are re-embedded
    python -m rag_service.ingest --force    # re-embed everything (e.g. after changing model)

Idempotent: a document whose content hash is unchanged is skipped; deleted files are removed.
"""

from __future__ import annotations

import argparse
import logging
import time

import psycopg

from rag_service.chunking import chunk_doc
from rag_service.config import DATABASE_URL
from rag_service.docs import ClinicDoc, load_docs
from rag_service.embeddings import get_embedder
from rag_service.retrieval import to_pgvector

logger = logging.getLogger("rag.ingest")


def ingest(conn: psycopg.Connection, docs: list[ClinicDoc], force: bool = False) -> dict:
    embedder = get_embedder()
    stats = {"docs": len(docs), "embedded_docs": 0, "chunks": 0, "skipped": 0, "removed": 0}

    existing = dict(conn.execute("SELECT doc_id, content_hash FROM documents").fetchall())
    for doc in docs:
        if not force and existing.get(doc.doc_id) == doc.content_hash:
            stats["skipped"] += 1
            continue
        chunks = chunk_doc(doc)
        vectors = embedder.embed_passages([c.content for c in chunks])
        with conn.transaction():
            doc_pk = conn.execute(
                """
                INSERT INTO documents (doc_id, title, lang, dialect, category, source_path, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title, lang = EXCLUDED.lang, dialect = EXCLUDED.dialect,
                    category = EXCLUDED.category, source_path = EXCLUDED.source_path,
                    content_hash = EXCLUDED.content_hash, ingested_at = now()
                RETURNING id
                """,
                (
                    doc.doc_id,
                    doc.title,
                    doc.lang,
                    doc.dialect,
                    doc.category,
                    doc.source_path,
                    doc.content_hash,
                ),
            ).fetchone()[0]
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_pk,))
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_index, section, content, lang, token_count,
                         embedding, ts_config)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, nour_ts_config(%s))
                    """,
                    [
                        (
                            doc_pk,
                            c.chunk_index,
                            c.section,
                            c.content,
                            c.lang,
                            c.token_count,
                            to_pgvector(v),
                            c.lang,
                        )
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
        stats["embedded_docs"] += 1
        stats["chunks"] += len(chunks)
        logger.info("ingested %-24s %2d chunks", doc.doc_id, len(chunks))

    gone = set(existing) - {d.doc_id for d in docs}
    if gone:
        conn.execute("DELETE FROM documents WHERE doc_id = ANY(%s)", (list(gone),))
        stats["removed"] = len(gone)
    conn.commit()
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-embed every document")
    args = ap.parse_args()
    t0 = time.perf_counter()
    with psycopg.connect(DATABASE_URL) as conn:
        stats = ingest(conn, load_docs(), force=args.force)
    logger.info("done in %.1fs: %s", time.perf_counter() - t0, stats)


if __name__ == "__main__":
    main()
