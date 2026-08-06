from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz


def normalize_quote_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def verify_quote(quote: str, page_text: str) -> bool:
    if not quote or not page_text:
        return False
    return normalize_quote_text(quote) in normalize_quote_text(page_text)


def locate_bbox(pdf_path: Path, page: int, quote: str) -> list[float] | None:
    if not quote:
        return None

    normalized_quote = normalize_quote_text(quote)
    with fitz.open(pdf_path) as document:
        if page < 1 or page > document.page_count:
            return None
        page_obj = document[page - 1]
        page_text = page_obj.get_text("text")
        if normalized_quote not in normalize_quote_text(page_text):
            return None

        for needle in (quote, " ".join(quote.split())):
            rects = page_obj.search_for(needle)
            if rects:
                rect = rects[0]
                return [rect.x0, rect.y0, rect.x1, rect.y1]

    return None


def mark_field_unverified(payload: dict[str, Any], field_name: str) -> None:
    payload[f"{field_name}_verified"] = False


def verify_extracted_fields(
    payload: dict[str, Any],
    *,
    fields: list[tuple[str, str, str]],
) -> list[str]:
    """Return field names whose quotes failed substring verification."""
    failed: list[str] = []
    for field_name, quote, page_text in fields:
        verified = verify_quote(quote, page_text)
        payload[f"{field_name}_verified"] = verified
        if not verified:
            failed.append(field_name)
    return failed


def apply_quote_verification_with_retry(
    payload: dict[str, Any],
    *,
    fields: list[tuple[str, str, str]],
    retried: bool,
) -> tuple[list[str], bool]:
    """Check quotes once; caller retries extraction once before marking unverified."""
    failed = verify_extracted_fields(payload, fields=fields)
    if failed and not retried:
        return failed, True
    if failed and retried:
        for field_name in failed:
            mark_field_unverified(payload, field_name)
    return failed, False
