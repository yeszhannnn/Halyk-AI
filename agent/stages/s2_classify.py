from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agent.llm.client import LLMClient
from agent.llm.schemas.classify import DocumentClassifyExtract
from agent.llm.vision_guard import complete_vision_dual
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

VALID_DOC_TYPES = {name for name, _ in MARKERS} | {"NOISE"}

ACC_PATTERN = re.compile(r"ACC-\d+")

EXPECTED_PDF_COUNTS: dict[str, int] = {
    "NOISE": 137,
    "SUPERSEDED_DRAFT": 5,
    "ADJUSTMENT_SOURCE": 2,
    "LOAN": 12,
    "LOAN_SUPERSEDED": 12,
    "AUDIT_NOTES": 12,
    "KYC": 12,
    "AUDIT_PLANNING": 8,
}

CLASSIFY_SYSTEM_PROMPT = """You classify bank PDF documents from scanned page images.

Assign exactly one doc_type by checking markers in this strict order (first match wins):
1. SUPERSEDED_DRAFT — markers: "ЗАМЕНЕНА ОКОНЧАТЕЛЬНЫМ", "ПРОЕКТ", "не может служить основанием"
2. LOAN_SUPERSEDED — markers: "НЕ ПРИМЕНЯЕТСЯ", "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ"
3. LOAN — markers: "ИСПОЛНИТЕЛЬНЫЙ ЭКЗЕМПЛЯР", "Старший обеспеченный заём"
4. AUDIT_NOTES — markers: "АУДИТОРСКОЕ ДЕЛО", "Примечания к финансовой отчётности"
5. ADJUSTMENT_SOURCE — markers: "Отчёт о выполнении согласованных процедур", "Служебная записка казначейства"
6. KYC — markers: "Знай своего клиента", "НАДЛЕЖАЩАЯ ПРОВЕРКА КЛИЕНТА"
7. AUDIT_PLANNING — markers: "Внешний аудит — Записка о планировании"
8. NOISE — when none of the above markers appear on the page

Also list every ACC- account id visible on the page (for example from a header line "Счёт ACC-....").
Return marker_quote as verbatim text from the image that supports the chosen doc_type.
"""


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


def _first_ocr_image_path(work_dir: Path, ocr_pages: list[Any]) -> Path | None:
    numbered: list[tuple[int, Path]] = []
    for entry in ocr_pages:
        if not isinstance(entry, dict):
            continue
        page_number = entry.get("page_number")
        image_path = entry.get("image_path")
        if page_number is None or not image_path:
            continue
        numbered.append((int(page_number), work_dir / str(image_path)))
    if not numbered:
        return None
    return min(numbered, key=lambda item: item[0])[1]


def _normalize_acc_ids(acc_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for acc_id in acc_ids:
        normalized = acc_id.strip().upper()
        if not ACC_PATTERN.fullmatch(normalized):
            continue
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


async def _classify_vision_page(
    client: LLMClient,
    *,
    doc_id: str,
    image_path: Path,
) -> tuple[DocumentClassifyExtract, list[dict[str, Any]]]:
    prompt = (
        f"Document id: {doc_id}\n"
        "Classify this scanned PDF page using the marker list."
    )
    extracted, digit_mismatches = await complete_vision_dual(
        client,
        response_model=DocumentClassifyExtract,
        system_prompt=CLASSIFY_SYSTEM_PROMPT,
        prompt=prompt,
        image_paths=[image_path],
        context={"doc_id": doc_id, "stage": "s2_classify"},
    )
    return extracted, digit_mismatches


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


OCR_TEXT_THRESHOLD = 100


async def _run_async(*, work_dir: Path) -> StageResult:
    inventory_path = work_dir / "01_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    documents_in = inventory["documents"]

    classified: dict[str, dict] = {}
    pdf_counts: Counter[str] = Counter()
    conflicts: list[dict] = []
    vision_candidates: list[tuple[str, Path]] = []

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

        if record["file_type"] == "pdf":
            if doc_type == "NOISE" and doc.get("ocr_pages"):
                image_path = _first_ocr_image_path(work_dir, doc["ocr_pages"])
                if image_path is not None and image_path.is_file():
                    vision_candidates.append((doc_id, image_path))

            pdf_counts[doc_type] += 1
            if doc_type == "NOISE" and acc_ids:
                conflicts.append({"kind": "UNMARKED_PDF", "doc_id": doc_id})
            ocr_page_nums = {
                int(entry["page_number"])
                for entry in doc.get("ocr_pages") or []
                if isinstance(entry, dict) and "page_number" in entry
            }
            for page_number, page_text in enumerate(doc["pages"], start=1):
                if len(page_text.strip()) < OCR_TEXT_THRESHOLD and page_number not in ocr_page_nums:
                    conflicts.append(
                        {
                            "kind": "NO_OCR",
                            "doc_id": doc_id,
                            "page": page_number,
                        },
                    )

        classified[doc_id] = record

    if vision_candidates:
        async with LLMClient() as client:
            for doc_id, image_path in vision_candidates:
                extracted, digit_mismatches = await _classify_vision_page(
                    client,
                    doc_id=doc_id,
                    image_path=image_path,
                )
                conflicts.extend(digit_mismatches)
                doc_type = extracted.doc_type if extracted.doc_type in VALID_DOC_TYPES else "NOISE"
            record = classified[doc_id]
            previous_type = record["doc_type"]
            if previous_type != doc_type:
                pdf_counts[previous_type] -= 1
                pdf_counts[doc_type] += 1
                record["doc_type"] = doc_type
                if doc_type == "AUDIT_PLANNING":
                    record["exclude_from_extraction"] = True
                elif "exclude_from_extraction" in record:
                    del record["exclude_from_extraction"]
            vision_acc_ids = _normalize_acc_ids(extracted.acc_ids)
            if vision_acc_ids:
                record["acc_ids"] = vision_acc_ids
                record["unbound"] = False
            record["vision_classified"] = True
            record["vision_marker_quote"] = extracted.marker_quote

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
        "conflicts": conflicts,
        "review": [entry for entry in conflicts if entry.get("kind") == "DIGIT_MISMATCH"],
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


def run(*, work_dir: Path) -> StageResult:
    return asyncio.run(_run_async(work_dir=work_dir))
