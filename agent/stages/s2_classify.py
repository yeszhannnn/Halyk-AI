from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from agent.stages import StageResult

MARKERS: list[tuple[str, re.Pattern[str]]] = [
    (
        "SUPERSEDED_DRAFT",
        re.compile(r"ЗАМЕНЕНА ОКОНЧАТЕЛЬНЫМ|ПРОЕКТ|не может служить основанием"),
    ),
    ("LOAN_SUPERSEDED", re.compile(r"НЕ ПРИМЕНЯЕТСЯ|НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ")),
    ("LOAN", re.compile(r"ИСПОЛНИТЕЛЬНЫЙ ЭКЗЕМПЛЯР|Старший обеспеченный заём")),
    ("AUDIT_NOTES", re.compile(r"АУДИТОРСКОЕ ДЕЛО|Примечания к финансовой отчётности")),
    (
        "ADJUSTMENT_SOURCE",
        re.compile(r"Отчёт о выполнении согласованных процедур|Служебная записка казначейства"),
    ),
    ("KYC", re.compile(r"Знай своего клиента|НАДЛЕЖАЩАЯ ПРОВЕРКА КЛИЕНТА")),
    ("AUDIT_PLANNING", re.compile(r"Внешний аудит — Записка о планировании")),
]

ACC_PATTERN = re.compile(r"ACC-\d+")

EXPECTED_PDF_COUNTS: dict[str, int] = {
    "NOISE": 138,
    "SUPERSEDED_DRAFT": 5,
    "ADJUSTMENT_SOURCE": 2,
    "LOAN": 12,
    "LOAN_SUPERSEDED": 12,
    "AUDIT_NOTES": 12,
    "KYC": 11,
    "AUDIT_PLANNING": 8,
}


def _document_text(pages: list[str]) -> str:
    return "\n".join(pages)


def _classify_text(text: str) -> str:
    for doc_type, pattern in MARKERS:
        if pattern.search(text):
            return doc_type
    return "NOISE"


def _extract_acc_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in ACC_PATTERN.finditer(text):
        acc_id = match.group(0)
        if acc_id not in seen:
            seen.add(acc_id)
            ordered.append(acc_id)
    return ordered


def _print_summary_table(counts: Counter[str], *, pdf_total: int) -> None:
    order = [
        "SUPERSEDED_DRAFT",
        "LOAN_SUPERSEDED",
        "LOAN",
        "AUDIT_NOTES",
        "ADJUSTMENT_SOURCE",
        "KYC",
        "AUDIT_PLANNING",
        "NOISE",
    ]
    print("\nclassification summary (PDFs):")
    print(f"{'type':<18} {'count':>6}")
    print("-" * 26)
    for doc_type in order:
        print(f"{doc_type:<18} {counts.get(doc_type, 0):>6}")
    print("-" * 26)
    print(f"{'TOTAL':<18} {pdf_total:>6}")


def run(*, work_dir: Path) -> StageResult:
    inventory_path = work_dir / "01_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    documents_in = inventory["documents"]

    classified: dict[str, dict] = {}
    pdf_counts: Counter[str] = Counter()

    for doc_id, doc in documents_in.items():
        text = _document_text(doc["pages"])
        doc_type = _classify_text(text)
        acc_ids = _extract_acc_ids(text)

        record: dict = {
            "doc_type": doc_type,
            "acc_ids": acc_ids,
            "unbound": len(acc_ids) == 0,
            "file_type": doc.get("file_type", "pdf"),
        }
        if doc_type == "AUDIT_PLANNING":
            record["exclude_from_extraction"] = True

        classified[doc_id] = record
        if record["file_type"] == "pdf":
            pdf_counts[doc_type] += 1

    pdf_total = sum(pdf_counts.values())
    _print_summary_table(pdf_counts, pdf_total=pdf_total)

    for doc_type, expected in EXPECTED_PDF_COUNTS.items():
        actual = pdf_counts.get(doc_type, 0)
        if actual != expected:
            print(
                f"warning: PDF count for {doc_type} is {actual}, expected {expected}",
            )

    active_loans = pdf_counts.get("LOAN", 0)
    if active_loans != 12:
        print(
            f"warning: active LOAN count is {active_loans}, expected 12 scenarios",
        )

    output = {
        "documents": classified,
        "summary": {
            "pdf_counts": dict(pdf_counts),
            "pdf_total": pdf_total,
        },
    }
    output_path = work_dir / "02_classified.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return StageResult(item_count=len(classified), row_count=pdf_total)
