"""Cross-artifact validation invariants."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import duckdb

from agent.evidence.quotes import verify_quote
from agent.parsing.numbers import round_half_up


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _keys_match(left: Any, right: Any, *, path: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            missing = sorted(right_keys - left_keys)
            extra = sorted(left_keys - right_keys)
            if missing:
                errors.append(f"{path}: missing keys {missing}")
            if extra:
                errors.append(f"{path}: extra keys {extra}")
        for key in sorted(left_keys & right_keys):
            child_path = f"{path}.{key}" if path else key
            errors.extend(_keys_match(left[key], right[key], path=child_path))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{path}: list length {len(left)} != {len(right)}")
    return errors


def _template_cell_count(template: dict[str, Any]) -> int:
    count = 0
    for scenario in template.get("answers", {}).values():
        count += len(scenario)
    return count


def _submission_cell_count(submission: dict[str, Any]) -> int:
    count = 0
    for scenario in submission.get("answers", {}).values():
        count += len(scenario)
    return count


def _is_two_decimal_places(value: float | int) -> bool:
    quantized = Decimal(str(value)).quantize(Decimal("0.01"))
    return quantized == Decimal(str(value))


def _ledger_txn_ids(work_dir: Path) -> set[str]:
    parquet = work_dir / "05_ledger.parquet"
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT txn_id FROM read_parquet(?)",
            [str(parquet)],
        ).fetchall()
    finally:
        con.close()
    return {str(row[0]) for row in rows if row[0] is not None}


def _ledger_null_amounts(work_dir: Path) -> list[str]:
    parquet = work_dir / "05_ledger.parquet"
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT txn_id, amount_usd FROM read_parquet(?)",
            [str(parquet)],
        ).fetchall()
    finally:
        con.close()
    nulls: list[str] = []
    for txn_id, amount in rows:
        if amount is None or str(amount).strip() == "":
            nulls.append(str(txn_id))
    return nulls


def _page_text(inventory: dict[str, Any], doc_id: str, page: int) -> str:
    document = (inventory.get("documents") or {}).get(doc_id)
    if not document:
        return ""
    pages = document.get("pages") or []
    if page < 1 or page > len(pages):
        return ""
    return pages[page - 1]


def validate(
    work_dir: Path,
    submission: dict[str, Any] | None = None,
    *,
    template: dict[str, Any] | None = None,
) -> None:
    """Validate submission and pipeline artifacts; raise on any invariant breach."""
    work_dir = Path(work_dir)
    template_path = work_dir / "submission_template.json"
    if template is None:
        template = _load_json(template_path)
    if submission is None:
        submission_path = work_dir / "submission.json"
        submission = _load_json(submission_path)

    errors: list[str] = []
    errors.extend(_keys_match(submission, template))

    template_cells = _template_cell_count(template)
    submission_cells = _submission_cell_count(submission)
    if submission_cells != template_cells:
        errors.append(
            f"cell count {submission_cells} != template cell count {template_cells}",
        )

    ledger_ids = _ledger_txn_ids(work_dir)

    for scenario_id, slots in submission.get("answers", {}).items():
        for slot, cell in slots.items():
            status = cell.get("status")
            if status not in {"COMPLIANT", "BREACH"}:
                errors.append(f"{scenario_id}/{slot}: invalid status {status!r}")

            actual = cell.get("actual")
            if actual is None:
                errors.append(f"{scenario_id}/{slot}: actual is missing")
                continue
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                errors.append(f"{scenario_id}/{slot}: actual is not numeric")
                continue
            if actual < 0:
                errors.append(f"{scenario_id}/{slot}: actual is negative")
            try:
                Decimal(str(actual))
            except InvalidOperation:
                errors.append(f"{scenario_id}/{slot}: actual is not numeric")
            if not _is_two_decimal_places(actual):
                errors.append(f"{scenario_id}/{slot}: actual not rounded to 2 decimals")

            evidence = cell.get("evidence_txn_id")
            if evidence is not None and str(evidence) not in ledger_ids:
                errors.append(f"{scenario_id}/{slot}: evidence_txn_id not in ledger")
            if evidence and status == "COMPLIANT":
                errors.append(f"{scenario_id}/{slot}: evidence on COMPLIANT cell")

    covenants_path = work_dir / "04a_covenants.json"
    inventory_path = work_dir / "01_inventory.json"
    if covenants_path.exists() and inventory_path.exists():
        covenants_payload = _load_json(covenants_path)
        inventory = _load_json(inventory_path)
        for covenant in covenants_payload.get("covenants") or []:
            period = covenant.get("period") or []
            if not all(str(item).startswith("2025") for item in period):
                errors.append(
                    f"{covenant['scenario_id']}/{covenant['slot']}: period not in 2025",
                )
            source = covenant.get("source") or {}
            quote = str(source.get("quote", ""))
            page_text = _page_text(inventory, str(source.get("doc_id", "")), int(source.get("page", 1)))
            if quote and not verify_quote(quote, page_text):
                errors.append(
                    f"{covenant['scenario_id']}/{covenant['slot']}: threshold quote unverified",
                )

    null_amounts = _ledger_null_amounts(work_dir)
    if null_amounts:
        errors.append(f"ledger rows with null amount after adjustments: {null_amounts[:5]}")

    adjustments_path = work_dir / "04c_adjustments.json"
    if adjustments_path.exists():
        adjustments_payload = _load_json(adjustments_path)
        if "unrecognised" not in adjustments_payload:
            errors.append("04c_adjustments.json missing unrecognised list")
    else:
        errors.append("04c_adjustments.json not found")

    if errors:
        message = "validation failed:\n" + "\n".join(f"  - {item}" for item in errors)
        raise ValueError(message)
