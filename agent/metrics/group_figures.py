"""Resolve GROUP-scoped covenant figures from audit disclosures."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

ZERO = Decimal("0")


def _d(value: Any) -> Decimal:
    if value is None:
        return ZERO
    if isinstance(value, Decimal):
        return value
    text = str(value).strip()
    if not text:
        return ZERO
    return Decimal(text)

PPE_ADDITIONS_PATTERN = re.compile(
    r"Net book value at the beginning of the year\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)"
    r".*?"
    r"Depreciation charge for the year\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)"
    r".*?"
    r"Net book value at the end of the year\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE | re.DOTALL,
)
BORROWER_HEADER_PATTERN = re.compile(
    r"Примечания к финансовой отчётности\s*\n\s*([^\n·]+?)\s*·",
    re.IGNORECASE,
)
CONSOLIDATED_DOC_MARKERS = (
    "consolidated financial statements",
    "consolidated statement of financial position",
    "консолидированн",
)


def _parse_money(text: str) -> Decimal:
    return _d(text.replace(",", "").replace("$", "").strip())


def _normalized_text(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _text_contains_borrower(text: str, borrower_name: str) -> bool:
    return _normalized_text(borrower_name) in _normalized_text(text)


def _borrower_name_from_audit_notes(text: str) -> str | None:
    match = BORROWER_HEADER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def _capex_keywords_match(include_keywords: list[str]) -> bool:
    keys = [keyword.casefold() for keyword in include_keywords]
    return any("капитал" in key for key in keys)


def _ppe_additions_from_text(text: str) -> Decimal | None:
    match = PPE_ADDITIONS_PATTERN.search(text)
    if not match:
        return None
    beginning, depreciation, ending = (_parse_money(part) for part in match.groups())
    additions = ending + depreciation - beginning
    if additions <= ZERO:
        return None
    return additions


def _inventory_documents(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "01_inventory.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("documents") or {}


def _scenario_bindings(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "03_bound.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("scenarios") or {}


def resolve_group_figure(
    *,
    scenario_id: str,
    include_keywords: list[str],
    work_dir: Path | None,
    audit_notes_text: str | None = None,
    inventory_documents: dict[str, Any] | None = None,
) -> tuple[Decimal | None, str | None]:
    """Return a consolidated group figure and a short source label."""
    if not _capex_keywords_match(include_keywords):
        return None, None

    texts: list[tuple[str, str]] = []
    if audit_notes_text:
        texts.append(("audit_notes", audit_notes_text))

    if work_dir is not None:
        inventory = inventory_documents or _inventory_documents(work_dir)
        bindings = _scenario_bindings(work_dir)
        audit_doc_id = (bindings.get(scenario_id) or {}).get("audit_notes")
        if audit_doc_id and audit_doc_id in inventory:
            audit_text = "\n".join(inventory[audit_doc_id].get("pages") or [])
            if audit_text and ("audit_notes", audit_text) not in texts:
                texts.append(("audit_notes", audit_text))
            borrower_name = _borrower_name_from_audit_notes(audit_text)
        else:
            borrower_name = None

        if borrower_name:
            for doc_id, doc in inventory.items():
                if doc_id == audit_doc_id:
                    continue
                full_text = "\n".join(doc.get("pages") or [])
                lowered = full_text.casefold()
                if not _text_contains_borrower(full_text, borrower_name):
                    continue
                if not any(marker in lowered for marker in CONSOLIDATED_DOC_MARKERS):
                    continue
                texts.append((f"consolidated:{doc_id}", full_text))

    for source, text in texts:
        figure = _ppe_additions_from_text(text)
        if figure is not None:
            return figure, source
    return None, None
