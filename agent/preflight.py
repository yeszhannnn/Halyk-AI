"""Preflight shape report — stages 1–3 without LLM spend."""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from agent.shape import OPEN_OCR_PAGE_COUNT, OPEN_SCENARIO_COUNT, OPEN_SLOTS, shape_warnings
from agent.stages import s1_ingest, s2_classify, s3_bind
from agent.template import load_template, template_scenarios, template_slots

KYC_THRESHOLD_PATTERN = re.compile(
    r"владеет\s+(\d{1,3}[.,]\d+)\s*%\s+и\s+более",
    re.IGNORECASE,
)
PERIMETER_THRESHOLD_PATTERN = re.compile(
    r"(\d{1,3}[.,]\d+)\s*%\s+.*(?:ниже|менее)",
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
) -> list[str]:
    thresholds: set[Decimal] = set()
    for scenario_id, record in sorted((bound.get("scenarios") or {}).items()):
        doc_id = record.get("kyc")
        if not doc_id:
            continue
        doc = inventory["documents"].get(doc_id)
        if not doc:
            continue
        text = "\n".join(doc.get("pages") or [])
        for pattern in (KYC_THRESHOLD_PATTERN, PERIMETER_THRESHOLD_PATTERN):
            for match in pattern.finditer(text):
                threshold = _parse_threshold(match.group(1))
                if threshold is not None:
                    thresholds.add(threshold)
    return [format(value.normalize(), "f") for value in sorted(thresholds)]


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
    kyc_thresholds = _distinct_kyc_thresholds(inventory, bound)

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
        "warnings": shape_warnings(
            scenario_count=len(scenarios),
            slots=slots,
            ocr_page_count=ocr_page_count,
            pdf_counts=dict(pdf_counts),
            missing_loans=missing_loan,
            missing_kyc=missing_kyc,
            missing_audit=missing_audit,
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

    thresholds = report.get("kyc_thresholds") or []
    if thresholds:
        print(f"kyc thresholds: {', '.join(thresholds)}")
    else:
        print("kyc thresholds: none found in text")

    warnings = report.get("warnings") or []
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("warnings: none")
