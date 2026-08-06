from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import fitz
import pdfplumber

from agent.stages import StageResult

logger = logging.getLogger(__name__)

SUPPORTED_TEXT_SUFFIXES = {".txt", ".csv"}
OCR_TEXT_THRESHOLD = 100
OCR_DPI = 150


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


def _detect_image_only_pages(path: Path, pages: list[str]) -> list[int]:
    """Pages with sparse text and at least one embedded image."""
    candidates: list[int] = []
    try:
        with fitz.open(path) as document:
            for page_index, page_text in enumerate(pages):
                if len(page_text.strip()) >= OCR_TEXT_THRESHOLD:
                    continue
                if page_index >= document.page_count:
                    continue
                if document[page_index].get_images():
                    candidates.append(page_index)
    except Exception as exc:  # noqa: BLE001 — keep ingest alive, surface reason
        logger.warning("image page detection failed for %s: %s", path, exc)
    return candidates


def _render_pdf_pages(
    path: Path,
    *,
    doc_id: str,
    page_indices: list[int],
    ocr_root: Path,
) -> tuple[list[dict[str, int | str]], str | None]:
    ocr_dir = ocr_root / doc_id
    ocr_dir.mkdir(parents=True, exist_ok=True)
    zoom = OCR_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    rendered: list[dict[str, int | str]] = []

    try:
        with fitz.open(path) as document:
            for page_index in page_indices:
                if page_index >= document.page_count:
                    continue
                page = document[page_index]
                image_name = f"page_{page_index + 1:04d}.png"
                image_path = ocr_dir / image_name
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(image_path)
                rendered.append(
                    {
                        "page_number": page_index + 1,
                        "image_path": f"ocr/{doc_id}/{image_name}",
                    },
                )
    except Exception as exc:  # noqa: BLE001 — keep ingest alive, surface reason
        return rendered, str(exc)

    if len(rendered) != len(page_indices):
        return rendered, f"rendered {len(rendered)} of {len(page_indices)} pages"
    return rendered, None


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


def _ingest_file(
    path: Path,
    *,
    rel_path: str,
    ocr_root: Path,
) -> tuple[dict | None, dict | None]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        sha256 = _sha256_file(path)
        pages, error = _extract_pdf(path)
        if error is not None:
            return None, {"path": rel_path, "reason": error}

        doc_id = path.stem
        record: dict = {
            "sha256": sha256,
            "page_count": len(pages),
            "pages": pages,
            "ocr_pages": [],
            "file_type": "pdf",
            "source_path": rel_path,
        }

        image_page_indices = _detect_image_only_pages(path, pages)
        if image_page_indices:
            ocr_pages, render_error = _render_pdf_pages(
                path,
                doc_id=doc_id,
                page_indices=image_page_indices,
                ocr_root=ocr_root,
            )
            record["ocr_pages"] = ocr_pages
            if render_error is not None:
                logger.warning(
                    "partial OCR render for %s: %s",
                    rel_path,
                    render_error,
                )

        return record, None

    if suffix in SUPPORTED_TEXT_SUFFIXES:
        sha256 = _sha256_file(path)
        content, error = _read_text_file(path)
        if error is not None or content is None:
            return None, {"path": rel_path, "reason": error or "unreadable text file"}
        return {
            "sha256": sha256,
            "page_count": 1,
            "pages": [content],
            "ocr_pages": [],
            "file_type": suffix.lstrip("."),
            "source_path": rel_path,
        }, None

    return None, {"path": rel_path, "reason": f"unsupported file type: {suffix or '(none)'}"}


def run(*, input_dir: Path, work_dir: Path) -> StageResult:
    documents_dir = input_dir / "documents"
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"documents directory not found: {documents_dir}")

    ocr_root = work_dir / "ocr"
    ocr_root.mkdir(parents=True, exist_ok=True)

    documents: dict[str, dict] = {}
    unreadable: list[dict[str, str]] = []

    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file():
            continue

        rel_path = path.relative_to(input_dir).as_posix()
        doc_id = path.stem
        record, failure = _ingest_file(path, rel_path=rel_path, ocr_root=ocr_root)

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
    ocr_docs = sum(1 for doc in documents.values() if doc.get("ocr_pages"))
    ocr_page_count = sum(len(doc.get("ocr_pages", [])) for doc in documents.values())

    print(
        f"ingest: pdfs={pdf_count} pdf_pages={pdf_pages} total_pages={total_pages} "
        f"ocr_docs={ocr_docs} ocr_pages={ocr_page_count} unreadable={len(unreadable)}",
    )

    return StageResult(item_count=len(documents), row_count=pdf_pages)
