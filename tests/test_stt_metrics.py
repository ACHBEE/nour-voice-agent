"""Normalization should ignore spelling noise that doesn't change pronunciation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals" / "stt"))
from metrics import cer, normalize, wer  # noqa: E402


def test_arabic_normalization_folds_variants_and_diacritics():
    assert normalize("الجمعة؟") == normalize("الجمعه")
    assert normalize("أَحْمَد") == "احمد"
    assert normalize("على") == normalize("علي")


def test_latin_normalization_ignores_case_accents_punctuation():
    assert normalize("L'Échographie !") == "l echographie"


def test_cer_perfect_and_partial():
    assert cer("واش تخدمو نهار الجمعة؟", "واش تخدمو نهار الجمعه") == 0.0
    # the real nova-3:multi transcript of this phrase: letters mostly survive, words don't
    ref, hyp = "واش تخدمو نهار الجمعة؟", "ويستخدمون هال الجمعه"
    assert 0.1 < cer(ref, hyp) < 0.3
    assert wer(ref, hyp) == 0.75


@pytest.mark.parametrize("hyp", ["", "something else entirely"])
def test_cer_bad_hypothesis_is_high(hyp):
    assert cer("Bonjour", hyp) >= 0.5
