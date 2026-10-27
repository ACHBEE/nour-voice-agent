"""The knowledge base is well-formed and its key facts stay consistent across languages."""

from collections import Counter

import pytest

from rag_service.docs import load_docs

DOCS = load_docs()
BY_ID = {d.doc_id: d for d in DOCS}


def test_doc_count_and_unique_ids():
    assert 12 <= len(DOCS) <= 15
    assert len(BY_ID) == len(DOCS), "duplicate doc_id"


def test_all_three_languages_and_darja_present():
    langs = Counter(d.lang for d in DOCS)
    assert {"ar", "fr", "en"} <= set(langs)
    assert all(n >= 3 for n in langs.values()), langs
    assert any(d.dialect == "darja" for d in DOCS)


def test_arabic_docs_are_written_in_arabic_script():
    for d in DOCS:
        if d.lang == "ar":
            arabic = sum("؀" <= ch <= "ۿ" for ch in d.body)
            assert arabic / len(d.body) > 0.5, d.doc_id


@pytest.mark.parametrize(
    ("doc_id", "fact"),
    [
        ("prices", "2000 DZD"),  # general consultation
        ("prices", "3500 DZD"),  # specialist consultation
        ("faq_darja", "2000"),
        ("opening_hours", "closed on Friday"),
        ("emergencies", "call 14"),
        ("insurance_cnas", "20 بالمائة"),
        ("about_location", "Rue des Palmiers"),
    ],
)
def test_key_facts_present(doc_id, fact):
    assert fact in BY_ID[doc_id].body
