from __future__ import annotations

from datetime import date

import pytest

from agent.stages.s2_classify import MARKERS, _classify_text
from agent.stages.s4a_covenants import (
    _extract_period_candidates,
    _missing_punkt_slots,
    _punkt_marker_present,
)
from agent.stages.s4c_adjustments import NOTE_HEADING_PATTERN, _covenant_section


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EXECUTION COPY\nAccount ACC-1001", "LOAN"),
        ("SENIOR SECURED LOAN AGREEMENT", "LOAN"),
        ("CONFIDENTIAL · EXECUTION", "LOAN"),
        ("NOT APPLICABLE — superseded loan", "LOAN_SUPERSEDED"),
        ("PRIOR VERSION — archive copy", "LOAN_SUPERSEDED"),
        ("NOTES TO THE FINANCIAL STATEMENTS", "AUDIT_NOTES"),
        ("AUDIT FILE — working papers", "AUDIT_NOTES"),
        ("KNOW YOUR CUSTOMER dossier", "KYC"),
        ("CUSTOMER DUE DILIGENCE review", "KYC"),
        ("EXTERNAL AUDIT — PLANNING memo", "AUDIT_PLANNING"),
        ("PLANNING MEMORANDUM", "AUDIT_PLANNING"),
        ("AGREED-UPON PROCEDURES report", "ADJUSTMENT_SOURCE"),
        ("TREASURY MEMORANDUM", "ADJUSTMENT_SOURCE"),
        ("DRAFT — not to be relied upon", "SUPERSEDED_DRAFT"),
        ("SUPERSEDED BY FINAL version", "SUPERSEDED_DRAFT"),
        ("random bank brochure", "NOISE"),
    ],
)
def test_classify_text_english_markers(text: str, expected: str) -> None:
    assert _classify_text(text) == expected


def test_marker_patterns_are_case_insensitive() -> None:
    assert _classify_text("execution copy") == "LOAN"
    assert _classify_text("know your customer") == "KYC"


def test_article6_clause_markers_bilingual() -> None:
    russian = "Пункт 6.1 ratio test\nПункт 6.2 revenue\nПункт 6.3 payments"
    english = "Clause 6.1 ratio test\nClause 6.2 revenue\nSection 6.3 payments"
    assert _missing_punkt_slots(russian) == []
    assert _missing_punkt_slots(english) == []
    assert _punkt_marker_present(english, "2") is True


def test_covenant_period_candidates_bilingual() -> None:
    russian = "за период с 2025-01-01 по 2025-12-31"
    english = "for the period from 2025-01-01 to 2025-12-31"
    ru = _extract_period_candidates(russian)
    en = _extract_period_candidates(english)
    assert ru == [(date(2025, 1, 1), date(2025, 12, 31), russian)]
    assert en == [(date(2025, 1, 1), date(2025, 12, 31), english)]


def test_adjustment_note_heading_bilingual() -> None:
    ru = "Примечание 12 — Корректировки EBITDA"
    en = "Note 12 — EBITDA Adjustments"
    assert NOTE_HEADING_PATTERN.search(ru) is not None
    assert NOTE_HEADING_PATTERN.search(en) is not None


def test_covenant_supplement_heading_bilingual() -> None:
    pages = ["COVENANT COMPLIANCE SUPPLEMENT\n(1.1) adjustment"]
    assert _covenant_section(pages).startswith("COVENANT COMPLIANCE SUPPLEMENT")


def test_marker_list_covers_all_doc_types() -> None:
    doc_types = {name for name, _ in MARKERS}
    assert doc_types == {
        "SUPERSEDED_DRAFT",
        "LOAN_SUPERSEDED",
        "LOAN",
        "AUDIT_NOTES",
        "ADJUSTMENT_SOURCE",
        "KYC",
        "AUDIT_PLANNING",
    }
