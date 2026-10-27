"""Pure-Python parts of the RAG service: chunking and reciprocal rank fusion."""

import pytest

from rag_service.chunking import MAX_CHARS, chunk_doc
from rag_service.docs import ClinicDoc, load_docs
from rag_service.fusion import RRF_K, rrf

DOCS = load_docs()


def _doc(body: str) -> ClinicDoc:
    return ClinicDoc(
        doc_id="t", title="Tarifs", lang="fr", category="prices", body=body, source_path="t.md"
    )


def test_every_doc_chunks_within_size_limit():
    for doc in DOCS:
        chunks = chunk_doc(doc)
        assert chunks, doc.doc_id
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(len(c.content) <= MAX_CHARS + 200 for c in chunks)  # + header
        assert all(c.lang == doc.lang for c in chunks)


def test_chunks_carry_title_and_section_header():
    chunks = chunk_doc(
        _doc("# Tarifs\n\nIntro text here.\n\n## Consultations\n\n- Générale : 2000 DZD.")
    )
    assert [c.section for c in chunks] == ["Tarifs", "Consultations"]
    assert chunks[1].content.startswith("Tarifs › Consultations\n")
    assert "2000 DZD" in chunks[1].content


def test_long_sections_split_on_paragraphs():
    para = "Phrase de test assez longue pour remplir. " * 6
    body = "## Long\n\n" + "\n\n".join([para] * 8)
    chunks = chunk_doc(_doc(body), max_chars=600)
    assert len(chunks) > 1
    assert all(c.section == "Long" for c in chunks)


def test_rrf_rewards_agreement_between_lists():
    fused = rrf({"vector": [1, 2, 3], "keyword": [3, 4, 1]})
    assert [f.id for f in fused][:2] == [1, 3]  # in both lists beats #2 in one list
    assert fused[0].ranks == {"vector": 1, "keyword": 3}
    assert fused[0].score == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 3))


def test_rrf_single_list_preserves_order_and_handles_empty():
    assert [f.id for f in rrf({"vector": [5, 7, 6]})] == [5, 7, 6]
    assert [f.id for f in rrf({"vector": [9], "keyword": []})] == [9]
    assert rrf({}) == []
