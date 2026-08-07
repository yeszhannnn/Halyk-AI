from __future__ import annotations

import asyncio
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import fitz
import pandas as pd

from agent.evidence.quotes import verify_extracted_fields, verify_quote
from agent.llm.client import LLMClient
from agent.llm.schemas.adjustments import AdjustmentExtract, VisionAdjustmentsExtract
from agent.models import Provenance
from agent.shape import is_canonical_open_dataset
from agent.stages import StageResult
from agent.stages.s4b_parties import normalize_counterparty

COVENANT_SUPPLEMENT_HEADING = "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ"
MARKER_PATTERN = re.compile(r"\((\d+(?:\.\d+)?)\)")
NOTE_HEADING_PATTERN = re.compile(
    r"Примечание\s+(\d+)\s*[—–-]\s*([^\n]+)",
    re.IGNORECASE,
)
TXN_ID_PATTERN = re.compile(r"TXN-[A-Z]\d+-\d+", re.IGNORECASE)
EXTRACTOR = "s4c_adjustments"
TEXT_CONFIDENCE = Decimal("0.95")
TEXT_UNVERIFIED_CONFIDENCE = Decimal("0.70")
OCR_CONFIDENCE = Decimal("0.60")
OCR_RENDER_DPI = 150

SYSTEM_PROMPT = """You classify covenant audit adjustments from Russian/English bank documents.

Return exactly one adjustment per provided segment or image batch.

Kinds (choose exactly one):
- RECLASS: amount moves from one category to another
- CUTOFF: transaction covers services rendered outside the covenant period
- EXCLUDE: transaction excluded from the covenant period entirely
- OFF_LEDGER: amount disclosed but has no separate ledger entry
- AMOUNT_FILL: ledger row exists but amount column is empty and document supplies the figure
- FX: foreign-currency settlement disclosure (invoice in foreign currency, paid in USD)
- EBITDA_ADDBACK: table of one-off items with a materiality threshold sentence
- NONE: explicitly states no adjustment was made or required
- UNRECOGNISED: cannot map to any kind above

Rules:
- Use short English slugs for categories (consulting, opex, interest, insurance, personnel, etc.).
- For NONE, look for phrases like "корректировка ... не производилась/не требуется".
- For FX, extract both foreign invoice amount+currency and USD settlement amount; no rate table is provided.
- For EBITDA_ADDBACK, extract every table row (item, counterparty, amount) and the materiality floor sentence.
- For AMOUNT_FILL and OFF_LEDGER about missing registry amounts, distinguish: AMOUNT_FILL names a txn_id;
  OFF_LEDGER is a disclosed obligation with no txn_id.
- Every populated scalar field needs a matching *_quote copied verbatim from the source text.
- Do not invent txn_id, amounts, or counterparties not present in the source.
"""

VISION_PROMPT = """Extract every covenant adjustment visible on these scanned audit-note page images.

Look especially for EBITDA add-back tables under "Корректировки EBITDA" with a materiality threshold.
Return each distinct adjustment as a separate item using the same kind taxonomy as text segments.
"""


def _decimal_to_str(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _parse_amount(value: str | None) -> Decimal | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    cleaned = str(value).strip().replace("$", "").replace(",", "").replace(" ", "")
    if not cleaned:
        return None
    return Decimal(cleaned)


def _ocr_page_image_paths(
    work_dir: Path,
    doc_id: str,
    doc: dict[str, Any],
    ocr_pages: list[Any],
) -> list[tuple[int, Path]]:
    paths: list[tuple[int, Path]] = []
    to_render: list[tuple[int, Path]] = []

    for entry in ocr_pages:
        if isinstance(entry, dict):
            page = int(entry.get("page_number") or entry.get("page") or 0)
            image_path = work_dir / str(entry["image_path"])
            if image_path.is_file():
                paths.append((page, image_path))
            elif page:
                to_render.append((page, image_path))
        else:
            image_path = work_dir / str(entry)
            if image_path.is_file():
                paths.append((0, image_path))

    if not to_render:
        return paths

    source_path = work_dir / doc["source_path"]
    if not source_path.is_file():
        return paths

    out_dir = work_dir / "ocr" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = OCR_RENDER_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(source_path) as pdf:
        for page_number, image_path in to_render:
            page_index = page_number - 1
            if page_index < 0 or page_index >= pdf.page_count:
                continue
            if not image_path.is_file():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                pdf[page_index].get_pixmap(matrix=matrix, alpha=False).save(image_path)
            paths.append((page_number, image_path))

    return sorted(paths, key=lambda item: item[0])


def _covenant_section(pages: list[str]) -> str:
    text = "\n".join(pages)
    if COVENANT_SUPPLEMENT_HEADING in text:
        return text[text.index(COVENANT_SUPPLEMENT_HEADING) :]
    return text


def _segment_numbered_items(section: str) -> list[dict[str, Any]]:
    note_positions: list[tuple[int, str, str]] = []
    for match in NOTE_HEADING_PATTERN.finditer(section):
        note_positions.append((match.start(), match.group(1), match.group(2).strip()))

    matches = list(MARKER_PATTERN.finditer(section))
    segments: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        marker = match.group(1)
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        text = section[start:end].strip()
        note_number = marker.split(".", maxsplit=1)[0]
        note_heading = ""
        for pos, number, heading in reversed(note_positions):
            if pos <= start and number == note_number:
                note_heading = heading
                break
        segments.append(
            {
                "marker": marker,
                "text": text,
                "note_heading": note_heading,
            },
        )
    return segments


def _source_page_for_quote(pages: list[str], quote: str, fallback: int = 1) -> int:
    if not quote:
        return fallback
    for page_number, page_text in enumerate(pages, start=1):
        if verify_quote(quote, page_text):
            return page_number
    needle = " ".join(quote.split())[:80]
    for page_number, page_text in enumerate(pages, start=1):
        if needle and needle.casefold() in " ".join(page_text.split()).casefold():
            return page_number
    return fallback


def _quote_checks(result: AdjustmentExtract, verification_text: str) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = [("kind", result.kind_quote, verification_text)]
    optional = (
        ("txn_id", result.txn_id_quote),
        ("amount", result.amount_quote),
        ("counterparty", result.counterparty_quote),
        ("from_category", result.from_category_quote),
        ("to_category", result.to_category_quote),
        ("category", result.category_quote),
        ("fx", result.fx_quote),
        ("materiality_floor", result.materiality_floor_quote),
    )
    for field_name, quote in optional:
        if quote.strip():
            checks.append((field_name, quote, verification_text))
    for index, row in enumerate(result.ebitda_rows):
        checks.append((f"ebitda_{index}_item", row.item_quote, verification_text))
        checks.append((f"ebitda_{index}_counterparty", row.counterparty_quote, verification_text))
        checks.append((f"ebitda_{index}_amount", row.amount_quote, verification_text))
    return checks


def _collect_verification_flags(payload: dict[str, Any]) -> dict[str, bool]:
    return {
        key.removesuffix("_verified"): bool(value)
        for key, value in payload.items()
        if key.endswith("_verified")
    }


def _confidence_from_verification(verification: dict[str, bool], *, source_kind: str) -> Decimal:
    if source_kind == "ocr":
        return OCR_CONFIDENCE
    if verification and all(verification.values()):
        return TEXT_CONFIDENCE
    return TEXT_UNVERIFIED_CONFIDENCE


def _serialize_provenance(
    provenance: Provenance,
    *,
    source_kind: str,
) -> dict[str, Any]:
    return {
        "doc_id": provenance.doc_id,
        "page": provenance.page,
        "quote": provenance.quote,
        "extractor": provenance.extractor,
        "source_kind": source_kind,
    }


def _adjustment_id(scenario_id: str, marker: str) -> str:
    marker_slug = marker.replace(".", "_")
    return f"adj_{scenario_id.casefold()}_{marker_slug}"


def _vision_adjustment_id(scenario_id: str, page: int, index: int) -> str:
    return f"adj_{scenario_id.casefold()}_ocr_p{page}_{index}"


def _match_txn_in_ledger(
    ledger: pd.DataFrame,
    *,
    scenario_id: str,
    txn_id: str | None,
    amount: Decimal | None,
    counterparty: str | None,
) -> tuple[str | None, str, list[str]]:
    scenario_rows = ledger[ledger["txn_id"].str.startswith(f"TXN-{scenario_id}-", na=False)]

    if txn_id:
        normalized = txn_id.strip().upper()
        if normalized in set(scenario_rows["txn_id"].astype(str)):
            return normalized, "txn_id", [normalized]
        return None, "txn_id", []

    if amount is None or not counterparty:
        return None, "none", []

    target_amount = abs(amount)
    target_key = normalize_counterparty(counterparty)
    candidates: list[str] = []
    for row in scenario_rows.itertuples():
        row_amount = _parse_amount(getattr(row, "amount", None))
        if row_amount is None:
            continue
        if abs(row_amount) != target_amount:
            continue
        if normalize_counterparty(str(row.counterparty)) != target_key:
            continue
        candidates.append(str(row.txn_id))

    if len(candidates) == 1:
        return candidates[0], "amount+counterparty", candidates
    return None, "amount+counterparty", candidates


def _fx_rate(settlement_usd: Decimal, source_amount: Decimal) -> Decimal:
    if source_amount == 0:
        return Decimal("0")
    return (settlement_usd / source_amount).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _serialize_adjustment(
    *,
    adjustment_id: str,
    scenario_id: str,
    extracted: AdjustmentExtract,
    doc_id: str,
    pages: list[str],
    marker: str,
    source_kind: str,
    verification: dict[str, bool],
    ledger: pd.DataFrame,
    conflicts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    kind = extracted.kind.strip().upper()
    if kind == "UNRECOGNISED":
        return None

    page = _source_page_for_quote(pages, extracted.kind_quote)
    provenance = Provenance(
        doc_id=doc_id,
        page=page,
        quote=extracted.kind_quote,
        extractor=EXTRACTOR,
    )

    matched_txn, match_method, candidates = _match_txn_in_ledger(
        ledger,
        scenario_id=scenario_id,
        txn_id=extracted.txn_id,
        amount=extracted.amount,
        counterparty=extracted.counterparty,
    )

    if match_method == "amount+counterparty" and len(candidates) != 1:
        conflicts.append(
            {
                "kind": "AMBIGUOUS_RECLASS_MATCH",
                "scenario_id": scenario_id,
                "adjustment": adjustment_id,
                "candidates": candidates,
            },
        )

    payload: dict[str, Any] = {
        "id": adjustment_id,
        "kind": kind,
        "scenario_id": scenario_id,
        "marker": marker,
        "txn_id": extracted.txn_id.strip().upper() if extracted.txn_id else None,
        "amount": _decimal_to_str(extracted.amount) if extracted.amount is not None else None,
        "counterparty": extracted.counterparty,
        "from_category": extracted.from_category,
        "to_category": extracted.to_category,
        "category": extracted.category,
        "matched_txn": matched_txn,
        "match_method": match_method,
        "source": _serialize_provenance(provenance, source_kind=source_kind),
        "confidence": _decimal_to_str(_confidence_from_verification(verification, source_kind=source_kind)),
        "verification": verification,
    }

    if kind == "FX" and extracted.fx_source_amount and extracted.fx_settlement_usd:
        payload["fx_source_amount"] = _decimal_to_str(extracted.fx_source_amount)
        payload["fx_source_currency"] = extracted.fx_source_currency
        payload["fx_settlement_usd"] = _decimal_to_str(extracted.fx_settlement_usd)
        payload["rate"] = _decimal_to_str(_fx_rate(extracted.fx_settlement_usd, extracted.fx_source_amount))

    if kind == "EBITDA_ADDBACK":
        floor = extracted.materiality_floor or Decimal("0")
        payload["materiality_floor"] = _decimal_to_str(floor)
        serialized_rows: list[dict[str, Any]] = []
        for row in extracted.ebitda_rows:
            row_matched, _, row_candidates = _match_txn_in_ledger(
                ledger,
                scenario_id=scenario_id,
                txn_id=None,
                amount=row.amount,
                counterparty=row.counterparty,
            )
            if not row_matched and len(row_candidates) > 1:
                conflicts.append(
                    {
                        "kind": "AMBIGUOUS_RECLASS_MATCH",
                        "scenario_id": scenario_id,
                        "adjustment": adjustment_id,
                        "row_counterparty": row.counterparty,
                        "row_amount": _decimal_to_str(row.amount),
                        "candidates": row_candidates,
                    },
                )
            serialized_rows.append(
                {
                    "item": row.item,
                    "counterparty": row.counterparty,
                    "amount": _decimal_to_str(row.amount),
                    "above_floor": row.amount >= floor,
                    "matched_txn": row_matched,
                },
            )
        payload["rows"] = serialized_rows
        payload["matched_txn"] = None
        payload["match_method"] = "none"

    if kind in {"NONE", "OFF_LEDGER"}:
        payload["matched_txn"] = None
        payload["match_method"] = "none"

    if kind == "AMOUNT_FILL" and extracted.txn_id:
        payload["matched_txn"] = extracted.txn_id.strip().upper()
        payload["match_method"] = "txn_id"

    return payload


def _extract_txn_id_from_text(text: str) -> str | None:
    match = TXN_ID_PATTERN.search(text)
    return match.group(0).upper() if match else None


def _apply_kind_overrides(extracted: AdjustmentExtract, segment_text: str) -> AdjustmentExtract:
    text_cf = segment_text.casefold()
    txn_id = extracted.txn_id.strip().upper() if extracted.txn_id else _extract_txn_id_from_text(segment_text)
    kind = extracted.kind.strip().upper()
    updates: dict[str, Any] = {}

    registry_gap_phrases = (
        "не отражена в выгрузке реестра",
        "сумма не отражена",
        "amount column is empty",
        "empty amount",
    )
    if txn_id and any(phrase in text_cf for phrase in registry_gap_phrases):
        updates["kind"] = "AMOUNT_FILL"
        updates["txn_id"] = txn_id

    if kind == "OFF_LEDGER" and txn_id:
        updates["kind"] = "AMOUNT_FILL"
        updates["txn_id"] = txn_id

    if (
        "не отражается отдельной операцией" in text_cf
        or "не отражается отдельной операцией в бухгалтерской книге" in text_cf
    ) and not txn_id:
        updates["kind"] = "OFF_LEDGER"

    if kind == "UNRECOGNISED" and (
        "не повторяется" in text_cf
        or "не производилась" in text_cf
        or "не требуется" in text_cf
    ):
        updates["kind"] = "NONE"

    if updates:
        return extracted.model_copy(update=updates)
    return extracted


async def _classify_segment(
    client: LLMClient,
    *,
    scenario_id: str,
    doc_id: str,
    marker: str,
    segment_text: str,
    pages: list[str],
) -> tuple[AdjustmentExtract, dict[str, bool], str]:
    verification_text = segment_text
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Scenario: {scenario_id}\n"
                f"Document: {doc_id}\n"
                f"Marker: ({marker})\n\n"
                "Classify this numbered adjustment segment:\n\n"
                f"{segment_text}"
            ),
        },
    ]
    extracted = await client.complete_verified(
        response_model=AdjustmentExtract,
        messages=messages,
        quote_checks=lambda result: _quote_checks(result, verification_text),
    )
    payload = extracted.model_dump(mode="python")
    verify_extracted_fields(payload, fields=_quote_checks(extracted, verification_text))
    verification = _collect_verification_flags(payload)
    extracted = _apply_kind_overrides(extracted, segment_text)
    return extracted, verification, "text"


async def _classify_vision_pages(
    client: LLMClient,
    *,
    scenario_id: str,
    doc_id: str,
    page_numbers: list[int],
    image_paths: list[Path],
    pages: list[str],
) -> tuple[list[AdjustmentExtract], str]:
    page_label = ", ".join(str(page) for page in page_numbers)
    extracted = await client.complete_vision(
        response_model=VisionAdjustmentsExtract,
        system_prompt=SYSTEM_PROMPT,
        prompt=(
            f"Scenario: {scenario_id}\n"
            f"Document: {doc_id}\n"
            f"Pages: {page_label}\n\n"
            f"{VISION_PROMPT}"
        ),
        image_paths=image_paths,
    )
    return [_apply_kind_overrides(item, "\n".join(pages)) for item in extracted.items], "ocr"


def _discover_sources(
    classified: dict[str, Any],
    bound: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Return (scenario_id, doc_id, source_kind) tuples."""
    audit_to_scenario = {
        record["audit_notes"]: scenario_id
        for scenario_id, record in bound["scenarios"].items()
        if record.get("audit_notes")
    }
    acc_to_scenario = bound["account_to_scenario"]
    sources: list[tuple[str, str, str]] = []

    for doc_id, record in classified["documents"].items():
        doc_type = record["doc_type"]
        if doc_type == "AUDIT_NOTES":
            scenario_id = audit_to_scenario.get(doc_id)
            if scenario_id:
                sources.append((scenario_id, doc_id, "audit_notes"))
        elif doc_type == "ADJUSTMENT_SOURCE":
            acc_ids = record.get("acc_ids") or []
            if not acc_ids:
                continue
            scenario_id = acc_to_scenario.get(acc_ids[0])
            if scenario_id:
                sources.append((scenario_id, doc_id, "adjustment_source"))

    return sources


def _dedupe_key(adjustment: dict[str, Any]) -> tuple[Any, ...]:
    return (
        adjustment["scenario_id"],
        adjustment["kind"],
        adjustment.get("txn_id"),
        adjustment.get("amount"),
        adjustment.get("counterparty"),
        adjustment.get("marker"),
    )


def _verify_payload(payload: dict[str, Any]) -> None:
    adjustments = payload["adjustments"]
    scenarios_with_adjustments = {
        adj["scenario_id"]
        for adj in adjustments.values()
        if adj["kind"] != "NONE"
    }
    if len(scenarios_with_adjustments) < 6:
        raise AssertionError(
            f"expected adjustments in >=6 scenarios, got {len(scenarios_with_adjustments)}: "
            f"{sorted(scenarios_with_adjustments)}",
        )

    off_ledger = [
        adj for adj in adjustments.values() if adj["kind"] == "OFF_LEDGER"
    ]
    if not off_ledger:
        raise AssertionError("expected one OFF_LEDGER adjustment")
    off_amount = Decimal(str(off_ledger[0]["amount"]))
    if abs(off_amount - Decimal("918447.52")) > Decimal("0.01"):
        raise AssertionError(f"OFF_LEDGER amount expected ~918447.52, got {off_amount}")

    amount_fill = [adj for adj in adjustments.values() if adj["kind"] == "AMOUNT_FILL"]
    if len(amount_fill) != 2:
        raise AssertionError(f"expected 2 AMOUNT_FILL adjustments, got {len(amount_fill)}")

    ebitda = [adj for adj in adjustments.values() if adj["kind"] == "EBITDA_ADDBACK"]
    if len(ebitda) != 1:
        raise AssertionError(f"expected 1 EBITDA_ADDBACK adjustment, got {len(ebitda)}")
    above_floor_total = sum(
        Decimal(str(row["amount"]))
        for row in ebitda[0]["rows"]
        if row.get("above_floor")
    )
    expected_total = Decimal("824152.91")
    if above_floor_total != expected_total:
        raise AssertionError(
            f"EBITDA above-floor total expected {expected_total}, got {above_floor_total}",
        )

    if "unrecognised" not in payload:
        raise AssertionError("unrecognised list missing from output")


async def _run_async(work_dir: Path) -> StageResult:
    inventory = json.loads((work_dir / "01_inventory.json").read_text(encoding="utf-8"))
    classified = json.loads((work_dir / "02_classified.json").read_text(encoding="utf-8"))
    bound = json.loads((work_dir / "03_bound.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(work_dir / "master_ledger_2025.csv")

    client = LLMClient()
    conflicts: list[dict[str, Any]] = []
    unrecognised: list[dict[str, Any]] = []
    adjustments: dict[str, dict[str, Any]] = {}

    for scenario_id, doc_id, source_kind in _discover_sources(classified, bound):
        doc = inventory["documents"][doc_id]
        pages = doc["pages"]
        if source_kind == "audit_notes":
            section = _covenant_section(pages)
        else:
            section = "\n".join(pages)

        segments = _segment_numbered_items(section)
        for segment in segments:
            marker = segment["marker"]
            adjustment_id = _adjustment_id(scenario_id, marker)
            extracted, verification, segment_source_kind = await _classify_segment(
                client,
                scenario_id=scenario_id,
                doc_id=doc_id,
                marker=marker,
                segment_text=segment["text"],
                pages=pages,
            )
            if extracted.kind.strip().upper() == "UNRECOGNISED":
                unrecognised.append(
                    {
                        "scenario_id": scenario_id,
                        "doc_id": doc_id,
                        "marker": marker,
                        "text": segment["text"],
                        "source_kind": segment_source_kind,
                    },
                )
                continue

            serialized = _serialize_adjustment(
                adjustment_id=adjustment_id,
                scenario_id=scenario_id,
                extracted=extracted,
                doc_id=doc_id,
                pages=pages,
                marker=marker,
                source_kind=segment_source_kind,
                verification=verification,
                ledger=ledger,
                conflicts=conflicts,
            )
            if serialized is not None:
                adjustments[adjustment_id] = serialized

        ocr_pages = doc.get("ocr_pages") or []
        if ocr_pages:
            page_paths = _ocr_page_image_paths(work_dir, doc_id, doc, ocr_pages)
            if page_paths:
                page_numbers = [page for page, _ in page_paths]
                image_paths = [path for _, path in page_paths]
                vision_items, vision_source_kind = await _classify_vision_pages(
                    client,
                    scenario_id=scenario_id,
                    doc_id=doc_id,
                    page_numbers=page_numbers,
                    image_paths=image_paths,
                    pages=pages,
                )
                for index, item in enumerate(vision_items, start=1):
                    page = page_numbers[min(index - 1, len(page_numbers) - 1)]
                    adjustment_id = _vision_adjustment_id(scenario_id, page, index)
                    if item.kind.strip().upper() == "UNRECOGNISED":
                        unrecognised.append(
                            {
                                "scenario_id": scenario_id,
                                "doc_id": doc_id,
                                "page": page,
                                "source_kind": vision_source_kind,
                                "kind_quote": item.kind_quote,
                            },
                        )
                        continue

                    verification = {field: False for field in ("kind",)}
                    serialized = _serialize_adjustment(
                        adjustment_id=adjustment_id,
                        scenario_id=scenario_id,
                        extracted=item,
                        doc_id=doc_id,
                        pages=pages,
                        marker=f"ocr_p{page}_{index}",
                        source_kind=vision_source_kind,
                        verification=verification,
                        ledger=ledger,
                        conflicts=conflicts,
                    )
                    if serialized is None:
                        continue
                    existing = adjustments.get(adjustment_id)
                    if existing is None:
                        adjustments[adjustment_id] = serialized

    deduped: dict[str, dict[str, Any]] = {}
    seen_keys: set[tuple[Any, ...]] = set()
    for adjustment_id, adjustment in adjustments.items():
        key = _dedupe_key(adjustment)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped[adjustment_id] = adjustment

    payload = {
        "adjustments": deduped,
        "conflicts": conflicts,
        "unrecognised": unrecognised,
        "summary": {
            "adjustment_count": len(deduped),
            "scenario_count": len({item["scenario_id"] for item in deduped.values()}),
            "actionable_count": sum(1 for item in deduped.values() if item["kind"] != "NONE"),
            "unrecognised_count": len(unrecognised),
            "conflict_count": len(conflicts),
        },
    }

    if is_canonical_open_dataset(work_dir):
        _verify_payload(payload)

    output_path = work_dir / "04c_adjustments.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"s4c_adjustments: adjustments={len(deduped)} "
        f"unrecognised={len(unrecognised)} conflicts={len(conflicts)}",
    )

    return StageResult(item_count=len(deduped), row_count=len(deduped))


def run(*, work_dir: Path) -> StageResult:
    return asyncio.run(_run_async(work_dir))
