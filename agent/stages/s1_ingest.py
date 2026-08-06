from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pdfplumber

from agent.stages import StageResult

logger = logging.getLogger(__name__)

SUPPORTED_TEXT_SUFFIXES = {".txt", ".csv"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_pdf(path: Path) -> tuple[list[str], str | None]:
    pages: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
    except Exception as exc:  # noqa: BLE001 — corrupt PDFs must not abort ingest
        return [], str(exc)
    return pages, None


def _read_text_file(path: Path) -> tuple[str | None, str | None]:
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return path.read_text(encoding=encoding), None
        except UnicodeDecodeError:
            continue
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except OSError as exc:
        return None, str(exc)


def _ingest_file(path: Path, *, rel_path: str) -> tuple[dict | None, dict | None]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        sha256 = _sha256_file(path)
        pages, error = _extract_pdf(path)
        if error is not None:
            return None, {"path": rel_path, "reason": error}
        first_page = pages[0].strip() if pages else ""
        return {
            "sha256": sha256,
            "page_count": len(pages),
            "pages": pages,
            "needs_ocr": len(first_page) < 20,
            "file_type": "pdf",
            "source_path": rel_path,
        }, None

    if suffix in SUPPORTED_TEXT_SUFFIXES:
        sha256 = _sha256_file(path)
        content, error = _read_text_file(path)
        if error is not None or content is None:
            return None, {"path": rel_path, "reason": error or "unreadable text file"}
        stripped = content.strip()
        return {
            "sha256": sha256,
            "page_count": 1,
            "pages": [content],
            "needs_ocr": len(stripped) < 20,
            "file_type": suffix.lstrip("."),
            "source_path": rel_path,
        }, None

    return None, {"path": rel_path, "reason": f"unsupported file type: {suffix or '(none)'}"}


def run(*, input_dir: Path, work_dir: Path) -> StageResult:
    documents_dir = input_dir / "documents"
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"documents directory not found: {documents_dir}")

    documents: dict[str, dict] = {}
    unreadable: list[dict[str, str]] = []

    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file():
            continue

        rel_path = path.relative_to(input_dir).as_posix()
        doc_id = path.stem
        record, failure = _ingest_file(path, rel_path=rel_path)

        if failure is not None:
            unreadable.append(failure)
            logger.warning("skipped unreadable file %s: %s", rel_path, failure["reason"])
            continue

        documents[doc_id] = record

    inventory = {"documents": documents, "unreadable": unreadable}
    output_path = work_dir / "01_inventory.json"
    output_path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    pdf_count = sum(1 for doc in documents.values() if doc["file_type"] == "pdf")
    pdf_pages = sum(doc["page_count"] for doc in documents.values() if doc["file_type"] == "pdf")
    total_pages = sum(doc["page_count"] for doc in documents.values())
    needs_ocr = sum(1 for doc in documents.values() if doc.get("needs_ocr"))

    print(
        f"ingest: pdfs={pdf_count} pdf_pages={pdf_pages} total_pages={total_pages} "
        f"needs_ocr={needs_ocr} unreadable={len(unreadable)}"
    )

    return StageResult(item_count=len(documents), row_count=pdf_pages)
