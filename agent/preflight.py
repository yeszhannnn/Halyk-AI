"""Preflight shape report — stages 1–3 without LLM spend."""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from agent.shape import (
    OPEN_OCR_PAGE_COUNT,
    OPEN_SCENARIO_COUNT,
    OPEN_SLOTS,
    _format_conflict,
    shape_warnings,
)
from agent.stages import s1_ingest, s2_classify, s3_bind
from agent.stages.s2_classify import first_nonempty_line
from agent.template import load_template, template_scenarios, template_slots

KYC_THRESHOLD_PATTERN = re.compile(
    r"(?:"
    r"владеет\s+(\d{1,3}[.,]\d+)\s*%\s+и\s+более"
    r"|"
    r"holds\s+(\d{1,3}[.,]\d+)\s*%\s+or\s+more"
    r")",
    re.IGNORECASE,
)
PERIMETER_THRESHOLD_PATTERN = re.compile(
    r"(?:"
    r"(\d{1,3}[.,]\d+)\s*%\s+.*(?:ниже|менее)"
    r"|"
    r"(\d{1,3}[.,]\d+)\s*%\s+.*(?:below|less than)"
    r")",
    re.IGNORECASE,
)
DOC_TYPE_ORDER = (
    "SUPERSEDED_DRAFT",
    "LOAN_SUPERSEDED",
    "LOAN",
    "AUDIT_NOTES",
    "ADJUSTMENT_SOURCE",
    "KYC",
    "AUDIT_PLANNING",
    "NOISE",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ocr_documents(inventory: dict[str, Any]) -> dict[str, list[int]]:
    documents: dict[str, list[int]] = {}
    for doc_id, doc in sorted(inventory.get("documents", {}).items()):
        page_numbers = [
            int(entry["page_number"])
            for entry in doc.get("ocr_pages") or []
            if isinstance(entry, dict) and "page_number" in entry
        ]
        if page_numbers:
            documents[doc_id] = sorted(page_numbers)
    return documents


def _ledger_row_counts(
    work_dir: Path,
    scenario_ids: set[str],
) -> tuple[dict[str, int], int]:
    ledger_path = work_dir / "master_ledger_2025.csv"
    if not ledger_path.is_file():
        return {}, 0

    ledger = pd.read_csv(ledger_path)
    per_scenario: dict[str, int] = {scenario_id: 0 for scenario_id in sorted(scenario_ids)}
    unassigned = 0
    for txn_id in ledger["txn_id"].astype(str):
        parts = txn_id.split("-")
        scenario_id = parts[1] if len(parts) >= 2 else ""
        if scenario_id in scenario_ids:
            per_scenario[scenario_id] += 1
        else:
            unassigned += 1
    return per_scenario, unassigned


def _missing_bindings(bound: dict[str, Any], field: str) -> list[str]:
    missing: list[str] = []
    for scenario_id, record in sorted((bound.get("scenarios") or {}).items()):
        if not record.get(field):
            missing.append(scenario_id)
    return missing


def _multiple_active_loans(bound: dict[str, Any]) -> list[str]:
    scenarios: list[str] = []
    for conflict in bound.get("conflicts") or []:
        if conflict.get("kind") == "MULTIPLE_ACTIVE_LOANS":
            scenario_id = conflict.get("scenario_id")
            if isinstance(scenario_id, str) and scenario_id not in scenarios:
                scenarios.append(scenario_id)
    return sorted(scenarios)


def _parse_threshold(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", "."))
    except Exception:
        return None


def _distinct_kyc_thresholds(
    inventory: dict[str, Any],
    bound: dict[str, Any],
) -> tuple[list[str], int]:
    thresholds: set[Decimal] = set()
    unreadable_dossiers = 0
    for _scenario_id, record in sorted((bound.get("scenarios") or {}).items()):
        doc_id = record.get("kyc")
        if not doc_id:
            continue
        doc = inventory["documents"].get(doc_id)
        if not doc:
            continue
        text = "\n".join(doc.get("pages") or [])
        found = False
        for pattern in (KYC_THRESHOLD_PATTERN, PERIMETER_THRESHOLD_PATTERN):
            for match in pattern.finditer(text):
                threshold_raw = next(group for group in match.groups() if group)
                threshold = _parse_threshold(threshold_raw)
                if threshold is not None:
                    thresholds.add(threshold)
                    found = True
        if not found:
            unreadable_dossiers += 1
    return [format(value.normalize(), "f") for value in sorted(thresholds)], unreadable_dossiers


def _collect_unmarked_pdfs(
    inventory: dict[str, Any],
    classified: dict[str, Any],
) -> list[dict[str, str]]:
    unmarked: list[dict[str, str]] = []
    for doc_id, record in sorted((classified.get("documents") or {}).items()):
        if record.get("doc_type") != "NOISE":
            continue
        doc = inventory.get("documents", {}).get(doc_id)
        if not doc or doc.get("file_type") != "pdf":
            continue
        unmarked.append(
            {
                "doc_id": doc_id,
                "first_line": first_nonempty_line(doc.get("pages") or []),
            },
        )
    return unmarked


def _collect_conflicts(
    classified: dict[str, Any],
    bound: dict[str, Any],
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (classified.get("conflicts") or [], bound.get("conflicts") or []):
        for conflict in source:
            key = json.dumps(conflict, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            conflicts.append(conflict)
    return conflicts


def run_preflight(input_dir: Path) -> dict[str, Any]:
    """Run ingest, classify, and bind; return a shape report dict."""
    input_dir = Path(input_dir)
    work_dir = input_dir

    s1_ingest.run(input_dir=input_dir, work_dir=work_dir)
    s2_classify.run(work_dir=work_dir)
    s3_bind.run(work_dir=work_dir)

    template = load_template(work_dir)
    inventory = _load_json(work_dir / "01_inventory.json")
    classified = _load_json(work_dir / "02_classified.json")
    bound = _load_json(work_dir / "03_bound.json")

    scenarios = template_scenarios(template)
    scenario_ids = set(scenarios)
    slots = template_slots(template)
    pdf_counts = Counter()
    for doc_id, record in classified.get("documents", {}).items():
        if inventory["documents"][doc_id]["file_type"] != "pdf":
            continue
        pdf_counts[record["doc_type"]] += 1

    ocr_documents = _ocr_documents(inventory)
    ocr_page_count = sum(len(pages) for pages in ocr_documents.values())
    ledger_rows_per_scenario, unassigned_ledger_rows = _ledger_row_counts(work_dir, scenario_ids)
    missing_loan = _missing_bindings(bound, "loan")
    missing_kyc = _missing_bindings(bound, "kyc")
    missing_audit = _missing_bindings(bound, "audit_notes")
    multiple_loans = _multiple_active_loans(bound)
    kyc_thresholds, unreadable_kyc_thresholds = _distinct_kyc_thresholds(inventory, bound)
    unmarked_pdfs = _collect_unmarked_pdfs(inventory, classified)
    conflicts = _collect_conflicts(classified, bound)

    report = {
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "slots": slots,
        "document_type_counts": {
            doc_type: int(pdf_counts.get(doc_type, 0)) for doc_type in DOC_TYPE_ORDER
        },
        "ocr_page_count": ocr_page_count,
        "ocr_documents": ocr_documents,
        "ledger_rows_per_scenario": ledger_rows_per_scenario,
        "unassigned_ledger_rows": unassigned_ledger_rows,
        "missing_loan": missing_loan,
        "missing_kyc": missing_kyc,
        "missing_audit": missing_audit,
        "multiple_active_loans": multiple_loans,
        "kyc_thresholds": kyc_thresholds,
        "kyc_thresholds_text_layer_only": True,
        "unreadable_kyc_thresholds": unreadable_kyc_thresholds,
        "unmarked_pdfs": unmarked_pdfs,
        "conflicts": conflicts,
        "warnings": shape_warnings(
            scenario_count=len(scenarios),
            slots=slots,
            ocr_page_count=ocr_page_count,
            pdf_counts=dict(pdf_counts),
            missing_loans=missing_loan,
            missing_kyc=missing_kyc,
            missing_audit=missing_audit,
            conflicts=conflicts,
            multiple_active_loans=multiple_loans,
        ),
    }
    return report


def print_preflight_report(report: dict[str, Any]) -> None:
    scenarios = ", ".join(report["scenarios"])
    print(f"scenarios: {report['scenario_count']} ({scenarios})")
    print(f"slots: {', '.join(report['slots'])}")
    print("document types:")
    for doc_type, count in report["document_type_counts"].items():
        print(f"  {doc_type}: {count}")

    unmarked_pdfs = report.get("unmarked_pdfs") or []
    print(f"unmarked pdfs (no classification marker): {len(unmarked_pdfs)}")
    for entry in unmarked_pdfs:
        doc_id = entry.get("doc_id", "")
        first_line = entry.get("first_line", "")
        preview = first_line if first_line else "(empty)"
        print(f"  {doc_id}: {preview}")

    ocr_documents = report.get("ocr_documents") or {}
    ocr_page_count = report["ocr_page_count"]
    print(f"ocr_pages: {ocr_page_count} ({len(ocr_documents)} documents)")
    for doc_id, pages in sorted(ocr_documents.items()):
        page_list = ", ".join(str(page) for page in pages)
        print(f"  {doc_id}: {page_list}")

    print("ledger rows per scenario:")
    for scenario_id, count in report["ledger_rows_per_scenario"].items():
        print(f"  {scenario_id}: {count}")
    print(f"unassigned ledger rows: {report['unassigned_ledger_rows']}")

    if report["missing_loan"]:
        print(f"missing loan: {', '.join(report['missing_loan'])}")
    else:
        print("missing loan: none")

    if report["missing_kyc"]:
        print(f"missing kyc: {', '.join(report['missing_kyc'])}")
    else:
        print("missing kyc: none")

    if report["missing_audit"]:
        print(f"missing audit: {', '.join(report['missing_audit'])}")
    else:
        print("missing audit: none")

    if report["multiple_active_loans"]:
        print(f"multiple active loans: {', '.join(report['multiple_active_loans'])}")
    else:
        print("multiple active loans: none")

    conflicts = report.get("conflicts") or []
    if conflicts:
        print("conflicts:")
        for conflict in conflicts:
            print(f"  - {_format_conflict(conflict)}")
    else:
        print("conflicts: none")

    thresholds = report.get("kyc_thresholds") or []
    unreadable = int(report.get("unreadable_kyc_thresholds") or 0)
    if thresholds:
        threshold_list = ", ".join(thresholds)
        print(
            "kyc thresholds (text-layer only): "
            f"{threshold_list} ({unreadable} dossiers unreadable at this stage)",
        )
    else:
        print(
            "kyc thresholds (text-layer only): none found in text "
            f"({unreadable} dossiers unreadable at this stage)",
        )

    warnings = report.get("warnings") or []
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("warnings: none")
