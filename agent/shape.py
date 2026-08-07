"""Open-dataset shape constants and departure checks."""

from __future__ import annotations

from typing import Any

from pathlib import Path

from agent.config import DEFAULT_INPUT

OPEN_SCENARIO_COUNT = 12
OPEN_SLOTS = ("6.1", "6.2", "6.3")
OPEN_OCR_PAGE_COUNT = 7

OPEN_PDF_COUNTS: dict[str, int] = {
    "NOISE": 137,
    "SUPERSEDED_DRAFT": 5,
    "ADJUSTMENT_SOURCE": 2,
    "LOAN": 12,
    "LOAN_SUPERSEDED": 12,
    "AUDIT_NOTES": 12,
    "KYC": 12,
    "AUDIT_PLANNING": 8,
}
OPEN_MISSING_KYC: list[str] = []
OPEN_MISSING_LOAN: list[str] = []
OPEN_MISSING_AUDIT: list[str] = []


def is_canonical_open_dataset(work_dir: Path) -> bool:
    try:
        return Path(work_dir).resolve() == DEFAULT_INPUT.resolve()
    except OSError:
        return False


def expects_golden_calibration(
    *,
    scenario_count: int,
    slots: list[str],
) -> bool:
    return scenario_count == OPEN_SCENARIO_COUNT and list(slots) == list(OPEN_SLOTS)


def _format_conflict(conflict: dict[str, Any]) -> str:
    kind = str(conflict.get("kind", "UNKNOWN"))
    parts = [f"conflict {kind}"]
    scenario_id = conflict.get("scenario_id")
    if isinstance(scenario_id, str) and scenario_id:
        parts.append(f"scenario {scenario_id}")
    doc_id = conflict.get("doc_id")
    if isinstance(doc_id, str) and doc_id:
        parts.append(f"doc {doc_id}")
    page = conflict.get("page")
    if page is not None:
        parts.append(f"page {page}")
    slot = conflict.get("slot")
    if isinstance(slot, str) and slot:
        parts.append(f"slot {slot}")
    return ": ".join(parts)


def shape_warnings(
    *,
    scenario_count: int,
    slots: list[str],
    ocr_page_count: int,
    pdf_counts: dict[str, int],
    missing_loans: list[str],
    missing_kyc: list[str],
    missing_audit: list[str],
    conflicts: list[dict[str, Any]],
    multiple_active_loans: list[str],
) -> list[str]:
    warnings: list[str] = []
    if scenario_count != OPEN_SCENARIO_COUNT:
        warnings.append(
            f"scenario count {scenario_count} differs from open baseline {OPEN_SCENARIO_COUNT}",
        )
    if list(slots) != list(OPEN_SLOTS):
        warnings.append(
            f"template slots {slots} differ from open baseline {list(OPEN_SLOTS)}",
        )
    if ocr_page_count != OPEN_OCR_PAGE_COUNT:
        warnings.append(
            f"ocr page count {ocr_page_count} differs from open baseline {OPEN_OCR_PAGE_COUNT}",
        )
    for doc_type, expected in OPEN_PDF_COUNTS.items():
        actual = int(pdf_counts.get(doc_type, 0))
        if actual != expected:
            warnings.append(
                f"document type {doc_type}: {actual} PDFs, open baseline {expected}",
            )
    for scenario_id in missing_loans:
        warnings.append(f"scenario {scenario_id} missing loan document")
    for scenario_id in missing_kyc:
        warnings.append(f"scenario {scenario_id} missing KYC document")
    for scenario_id in missing_audit:
        warnings.append(f"scenario {scenario_id} missing audit notes")
    for scenario_id in multiple_active_loans:
        warnings.append(f"scenario {scenario_id} has multiple active loans")
    for conflict in conflicts:
        warnings.append(_format_conflict(conflict))
    return warnings
