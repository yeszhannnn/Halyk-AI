from __future__ import annotations

import asyncio
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import fitz
import pandas as pd

from agent.evidence.quotes import verify_extracted_fields, verify_quote
from agent.llm.client import LLMClient
from agent.llm.schemas.parties import KycPartiesExtract
from agent.llm.vision_guard import complete_vision_dual
from agent.models import Provenance, RelatedParty
from agent.parsing.numbers import capture_absent_values
from agent.shape import is_canonical_open_dataset
from agent.stages import StageResult

HEADER_ACCOUNT_PATTERN = re.compile(r"Сч[её]т\s+(ACC-\d+)", re.IGNORECASE)
LEGAL_SUFFIX_PATTERN = re.compile(
    r"[\s,.\-]*(?:l(?:\.[\s]*){2}p|l\.?\s*l\.?\s*p\.?|j\.?\s*s\.?\s*c\.?|gmbh)\.?\s*$",
    re.IGNORECASE,
)
SPARSE_PAGE_TEXT_THRESHOLD = 100
OCR_RENDER_DPI = 150
EXTRACTOR = "s4b_parties"
TEXT_CONFIDENCE = Decimal("0.95")
TEXT_UNVERIFIED_CONFIDENCE = Decimal("0.70")
OCR_CONFIDENCE = Decimal("0.60")
DIRECTION_BY_SEMANTICS = {
    "RELATED_PARTY": "AT_OR_ABOVE",
    "UNRESTRICTED_SUBSIDIARY": "BELOW",
}

SYSTEM_PROMPT = """You extract KYC ownership/perimeter data from Russian/English bank dossiers.

Rules:
- Return only information explicitly visible in the provided dossier text or images.
- header_account must be the account id from the dossier header line beginning with "Счёт ACC-...".
- ownership_rows must list every row from the table beneath the header with counterparty name and percentage.
- table_semantics must be one of:
  - RELATED_PARTY: the rule sentence defines related parties; organisations at or above threshold_pct
    are related parties (e.g. Group holds 35.0% or more of voting rights).
  - UNRESTRICTED_SUBSIDIARY: the rule sentence defines a security/perimeter threshold; subsidiaries
    below threshold_pct are unrestricted (e.g. pledged assets below 50.0% of voting rights).
- threshold_pct is the numeric percentage from that rule sentence — read it from each dossier, never hardcode.
- Every scalar field must have a matching *_quote field copied verbatim from the source.
- Percentages are numeric values without the % sign (41.2 not 0.412).
- Do not infer semantics from the "РАБОЧИЙ ДОКУМЕНТ" label; use the rule sentence beneath the table.
"""

VISION_PROMPT = """Extract the KYC dossier header account, ownership/perimeter table, rule sentence beneath it,
and threshold from these scanned dossier page images.

Determine table_semantics from the rule sentence:
- RELATED_PARTY when the threshold marks related-party status at or above the percentage.
- UNRESTRICTED_SUBSIDIARY when subsidiaries below the percentage are unrestricted.
"""


def normalize_counterparty(name: str) -> str:
    stripped = name.strip()
    for char in "\"'«»""''":
        stripped = stripped.replace(char, "")
    stripped = stripped.strip("\"'«»""''")
    collapsed = " ".join(stripped.split()).casefold()
    collapsed = re.sub(r",\s*(?=(?:l\.?\s*){0,3}p\.?\s*$)", "", collapsed, flags=re.IGNORECASE)
    collapsed = LEGAL_SUFFIX_PATTERN.sub("", collapsed).strip()
    return collapsed.rstrip(".,;:-").strip()


def _ocr_page_image_paths(work_dir: Path, ocr_pages: list[Any]) -> list[Path]:
    paths: list[Path] = []
    for entry in ocr_pages:
        if isinstance(entry, dict):
            paths.append(work_dir / str(entry["image_path"]))
        else:
            paths.append(work_dir / str(entry))
    return paths


def _decimal_to_str(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _extract_header_account(text: str) -> str | None:
    match = HEADER_ACCOUNT_PATTERN.search(text)
    return match.group(1) if match else None


def _quote_checks(result: KycPartiesExtract, verification_text: str) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = [
        ("header_account", result.header_account_quote, verification_text),
        ("table_semantics", result.table_semantics_quote, verification_text),
        ("threshold", result.threshold_quote, verification_text),
    ]
    for index, row in enumerate(result.ownership_rows):
        checks.append((f"ownership_{index}_name", row.counterparty_quote, verification_text))
        checks.append((f"ownership_{index}_pct", row.ownership_pct_quote, verification_text))
    return checks


def _source_page_for_quote(pages: list[str], quote: str, fallback: int = 1) -> int:
    if not quote:
        return fallback
    for page_number, page_text in enumerate(pages, start=1):
        if verify_quote(quote, page_text):
            return page_number
    return fallback


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


def _review_fields_for_source(source_kind: str, verification: dict[str, bool]) -> list[str]:
    if source_kind == "ocr":
        return ["header_account", "table_semantics", "threshold", "ownership_rows"]
    return [field for field, verified in verification.items() if not verified]


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


def _direction_for_semantics(table_semantics: str) -> str:
    direction = DIRECTION_BY_SEMANTICS.get(table_semantics)
    if direction is None:
        raise ValueError(f"unsupported table_semantics: {table_semantics}")
    return direction


def _row_is_related(
    ownership_pct: Decimal,
    threshold: Decimal,
    direction: str,
) -> bool:
    if direction == "BELOW":
        return ownership_pct >= threshold
    if direction == "AT_OR_ABOVE":
        return ownership_pct >= threshold
    raise ValueError(f"unsupported direction: {direction}")


def _related_parties_from_extract(
    extracted: KycPartiesExtract,
    *,
    doc_id: str,
    pages: list[str],
    threshold: Decimal,
    table_semantics: str,
    source_kind: str,
) -> tuple[list[RelatedParty], list[dict[str, Any]]]:
    direction = _direction_for_semantics(table_semantics)
    parties: list[RelatedParty] = []
    serialized_rows: list[dict[str, Any]] = []

    for row in extracted.ownership_rows:
        is_related = _row_is_related(row.ownership_pct, threshold, direction)
        page = _source_page_for_quote(pages, row.counterparty_quote)
        provenance = Provenance(
            doc_id=doc_id,
            page=page,
            quote=row.counterparty_quote,
            extractor=EXTRACTOR,
        )
        party = RelatedParty(
            counterparty=row.counterparty,
            ownership_pct=row.ownership_pct,
            is_related=is_related,
            source=provenance,
        )
        parties.append(party)
        serialized_rows.append(
            {
                "counterparty": party.counterparty,
                "ownership_pct": _decimal_to_str(party.ownership_pct),
                "is_related": party.is_related,
                "source": _serialize_provenance(party.source, source_kind=source_kind),
            },
        )

    return parties, serialized_rows


def _build_ledger_map(
    ledger_names: list[str],
    related_counterparties: list[str],
) -> dict[str, list[str]]:
    related_keys = {normalize_counterparty(name): name for name in related_counterparties}
    ledger_map: dict[str, list[str]] = {name: [] for name in related_counterparties}

    for ledger_name in ledger_names:
        key = normalize_counterparty(ledger_name)
        dossier_name = related_keys.get(key)
        if dossier_name is not None:
            ledger_map[dossier_name].append(ledger_name)

    return ledger_map


def _is_scanned_dossier(doc: dict[str, Any]) -> bool:
    return bool(doc.get("ocr_pages")) and all(
        len(page.strip()) < SPARSE_PAGE_TEXT_THRESHOLD for page in doc["pages"]
    )


def _discover_kyc_candidates(
    inventory: dict[str, Any],
    classified: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for doc_id, doc in inventory["documents"].items():
        record = classified["documents"][doc_id]
        if record["doc_type"] == "KYC" or _is_scanned_dossier(doc):
            candidates.append((doc_id, doc))
    return candidates


def _render_sparse_pages(
    *,
    work_dir: Path,
    doc_id: str,
    doc: dict[str, Any],
) -> list[Path]:
    source_path = work_dir / doc["source_path"]
    if not source_path.is_file():
        return []

    sparse_indices: list[int] = []
    with fitz.open(source_path) as pdf:
        for index, page_text in enumerate(doc["pages"]):
            if len(page_text.strip()) >= SPARSE_PAGE_TEXT_THRESHOLD:
                continue
            if index >= pdf.page_count:
                continue
            if pdf[index].get_images():
                sparse_indices.append(index)

    if not sparse_indices:
        return []

    out_dir = work_dir / "ocr" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = OCR_RENDER_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    rendered: list[Path] = []

    with fitz.open(source_path) as pdf:
        for page_index in sparse_indices:
            image_path = out_dir / f"sparse_page_{page_index + 1:04d}.png"
            if not image_path.is_file():
                pdf[page_index].get_pixmap(matrix=matrix, alpha=False).save(image_path)
            rendered.append(image_path)

    return rendered


def _vision_image_paths(
    *,
    work_dir: Path,
    doc_id: str,
    doc: dict[str, Any],
) -> list[Path]:
    if doc.get("ocr_pages"):
        return _ocr_page_image_paths(work_dir, doc["ocr_pages"])
    return _render_sparse_pages(work_dir=work_dir, doc_id=doc_id, doc=doc)


async def _extract_from_text(
    client: LLMClient,
    *,
    scenario_id: str,
    doc_id: str,
    pages: list[str],
) -> tuple[KycPartiesExtract, dict[str, bool], str]:
    verification_text = "\n".join(pages)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Scenario: {scenario_id}\n"
                f"Document: {doc_id}\n\n"
                "Extract the ownership/perimeter table, rule sentence, and threshold from this dossier text:\n\n"
                f"{verification_text}"
            ),
        },
    ]
    extracted = await client.complete_verified(
        response_model=KycPartiesExtract,
        messages=messages,
        quote_checks=lambda result: _quote_checks(result, verification_text),
    )
    payload = extracted.model_dump(mode="python")
    verify_extracted_fields(payload, fields=_quote_checks(extracted, verification_text))
    verification = _collect_verification_flags(payload)
    return extracted, verification, "text"


async def _extract_from_images(
    client: LLMClient,
    *,
    scenario_id: str,
    doc_id: str,
    image_paths: list[Path],
) -> tuple[KycPartiesExtract, dict[str, bool], str, list[dict[str, Any]]]:
    prompt = (
        f"Scenario: {scenario_id}\n"
        f"Document: {doc_id}\n\n"
        f"{VISION_PROMPT}"
    )
    extracted, digit_mismatches = await complete_vision_dual(
        client,
        response_model=KycPartiesExtract,
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        image_paths=image_paths,
        context={
            "scenario_id": scenario_id,
            "doc_id": doc_id,
            "stage": EXTRACTOR,
        },
    )
    verification = {
        "header_account": False,
        "table_semantics": False,
        "threshold": False,
    }
    for index in range(len(extracted.ownership_rows)):
        verification[f"ownership_{index}_name"] = False
        verification[f"ownership_{index}_pct"] = False
    return extracted, verification, "ocr", digit_mismatches


def _needs_vision_fallback(extracted: KycPartiesExtract, verification: dict[str, bool]) -> bool:
    if not extracted.ownership_rows:
        return True
    if not extracted.threshold_quote.strip():
        return True
    if not extracted.table_semantics_quote.strip():
        return True
    if not all(verification.values()):
        return True
    return False


def _merge_text_and_vision_extract(
    text_extracted: KycPartiesExtract,
    vision_extracted: KycPartiesExtract,
    *,
    text_header: str | None,
) -> KycPartiesExtract:
    updates: dict[str, Any] = {
        "ownership_rows": vision_extracted.ownership_rows,
        "table_semantics": vision_extracted.table_semantics,
        "table_semantics_quote": vision_extracted.table_semantics_quote,
        "threshold_pct": vision_extracted.threshold_pct,
        "threshold_quote": vision_extracted.threshold_quote,
    }
    if text_header is None:
        updates["header_account"] = vision_extracted.header_account
        updates["header_account_quote"] = vision_extracted.header_account_quote
    return text_extracted.model_copy(update=updates)


async def _extract_dossier(
    client: LLMClient,
    *,
    work_dir: Path,
    scenario_id: str,
    doc_id: str,
    doc: dict[str, Any],
) -> tuple[KycPartiesExtract, dict[str, bool], str, list[str], KycPartiesExtract | None, list[dict[str, Any]]]:
    pages = doc["pages"]
    text_header = _extract_header_account("\n".join(pages))
    fully_scanned = text_header is None and _is_scanned_dossier(doc)
    perimeter: KycPartiesExtract | None = None

    if fully_scanned:
        image_paths = _vision_image_paths(work_dir=work_dir, doc_id=doc_id, doc=doc)
        extracted, verification, source_kind, digit_mismatches = await _extract_from_images(
            client,
            scenario_id=scenario_id,
            doc_id=doc_id,
            image_paths=image_paths,
        )
        return (
            extracted,
            verification,
            source_kind,
            _review_fields_for_source(source_kind, verification),
            None,
            digit_mismatches,
        )

    extracted, verification, source_kind = await _extract_from_text(
        client,
        scenario_id=scenario_id,
        doc_id=doc_id,
        pages=pages,
    )

    if doc.get("ocr_pages"):
        image_paths = _vision_image_paths(work_dir=work_dir, doc_id=doc_id, doc=doc)
        vision_extracted, vision_verification, _, vision_mismatches = await _extract_from_images(
            client,
            scenario_id=scenario_id,
            doc_id=doc_id,
            image_paths=image_paths,
        )
        if _needs_vision_fallback(extracted, verification):
            extracted = _merge_text_and_vision_extract(
                extracted,
                vision_extracted,
                text_header=text_header,
            )
            verification = vision_verification
            if text_header is not None:
                verification["header_account"] = True
            source_kind = "ocr"
        elif (
            vision_extracted.table_semantics == "UNRESTRICTED_SUBSIDIARY"
            and extracted.ownership_rows
            and vision_extracted.ownership_rows
        ):
            perimeter = vision_extracted
        digit_mismatches = vision_mismatches
    elif _needs_vision_fallback(extracted, verification):
        image_paths = _vision_image_paths(work_dir=work_dir, doc_id=doc_id, doc=doc)
        digit_mismatches = []
        if image_paths:
            extracted, verification, source_kind, digit_mismatches = await _extract_from_images(
                client,
                scenario_id=scenario_id,
                doc_id=doc_id,
                image_paths=image_paths,
            )
    else:
        digit_mismatches = []

    review_fields = _review_fields_for_source(source_kind, verification)
    return extracted, verification, source_kind, review_fields, perimeter, digit_mismatches


async def _resolve_scenario_for_dossier(
    client: LLMClient,
    *,
    work_dir: Path,
    doc_id: str,
    doc: dict[str, Any],
    account_to_scenario: dict[str, str],
) -> tuple[str | None, KycPartiesExtract | None]:
    header_account = _extract_header_account("\n".join(doc["pages"]))
    if header_account is not None:
        return account_to_scenario.get(header_account), None

    if doc.get("ocr_pages"):
        image_paths = _vision_image_paths(work_dir=work_dir, doc_id=doc_id, doc=doc)
        extracted, _, _, _ = await _extract_from_images(
            client,
            scenario_id="bind",
            doc_id=doc_id,
            image_paths=image_paths,
        )
        header_account = extracted.header_account.strip().upper()
        return account_to_scenario.get(header_account), extracted

    return None, None


def _sum_related_outflows(
    ledger: pd.DataFrame,
    *,
    scenario_id: str,
    ledger_map: dict[str, list[str]],
) -> Decimal:
    matched_names = {
        ledger_name
        for names in ledger_map.values()
        for ledger_name in names
    }
    if not matched_names:
        return Decimal("0")

    rows = ledger[
        (ledger["txn_id"].str.startswith(f"TXN-{scenario_id}-"))
        & (ledger["counterparty"].isin(matched_names))
        & (ledger["amount"] < 0)
    ]
    return abs(Decimal(str(rows["amount"].sum())))


def _related_outflow_rows(
    ledger: pd.DataFrame,
    *,
    scenario_id: str,
    ledger_map: dict[str, list[str]],
) -> pd.DataFrame:
    matched_names = {
        ledger_name
        for names in ledger_map.values()
        for ledger_name in names
    }
    return ledger[
        (ledger["txn_id"].str.startswith(f"TXN-{scenario_id}-"))
        & (ledger["counterparty"].isin(matched_names))
        & (ledger["amount"] < 0)
    ]


def _check_payload_structure(
    payload: dict[str, Any],
    *,
    conflicts: list[dict[str, Any]],
) -> None:
    scenarios = payload.get("scenarios") or {}
    for scenario_id, record in scenarios.items():
        threshold = record.get("threshold_pct")
        if threshold is None:
            continue
        threshold_value = Decimal(str(threshold))
        for row in record.get("ownership") or []:
            ownership_pct = Decimal(str(row["ownership_pct"]))
            if ownership_pct < 0 or ownership_pct > 100:
                conflicts.append(
                    {
                        "kind": "OWNERSHIP_PCT_OUT_OF_RANGE",
                        "scenario_id": scenario_id,
                        "name": row.get("name"),
                        "ownership_pct": str(ownership_pct),
                    },
                )
            expected_related = _row_is_related(
                ownership_pct,
                threshold_value,
                record.get("direction", "AT_OR_ABOVE"),
            )
            if row.get("is_related") != expected_related:
                conflicts.append(
                    {
                        "kind": "RELATED_FLAG_MISMATCH",
                        "scenario_id": scenario_id,
                        "name": row.get("name"),
                        "expected": expected_related,
                        "actual": row.get("is_related"),
                    },
                )


async def _run_async(work_dir: Path) -> StageResult:
    inventory = json.loads((work_dir / "01_inventory.json").read_text(encoding="utf-8"))
    classified = json.loads((work_dir / "02_classified.json").read_text(encoding="utf-8"))
    bound = json.loads((work_dir / "03_bound.json").read_text(encoding="utf-8"))

    ledger = pd.read_csv(work_dir / "master_ledger_2025.csv")
    account_to_scenario = bound["account_to_scenario"]
    scenario_ids = sorted(set(account_to_scenario.values()))

    conflicts: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    candidates = _discover_kyc_candidates(inventory, classified)
    scenario_bindings: dict[str, str] = {}
    pre_extracted: dict[str, KycPartiesExtract] = {}

    async with LLMClient() as client:
        for doc_id, doc in candidates:
            scenario_id, ocr_extract = await _resolve_scenario_for_dossier(
                client,
                work_dir=work_dir,
                doc_id=doc_id,
                doc=doc,
                account_to_scenario=account_to_scenario,
            )
            if scenario_id is None:
                if doc.get("ocr_pages") and ocr_extract is not None:
                    conflicts.append(
                        {
                            "kind": "UNBOUND_DOCUMENT",
                            "doc_id": doc_id,
                            "header_account": ocr_extract.header_account,
                        },
                    )
                elif not doc.get("ocr_pages"):
                    header_account = _extract_header_account("\n".join(doc["pages"]))
                    if header_account is None:
                        conflicts.append({"kind": "KYC_HEADER_ACCOUNT_NOT_FOUND", "doc_id": doc_id})
                    else:
                        conflicts.append(
                            {
                                "kind": "UNBOUND_DOCUMENT",
                                "doc_id": doc_id,
                                "header_account": header_account,
                            },
                        )
                continue

            previous = scenario_bindings.get(scenario_id)
            if previous is not None and previous != doc_id:
                conflicts.append(
                    {
                        "kind": "DUPLICATE_KYC",
                        "scenario_id": scenario_id,
                        "doc_ids": [previous, doc_id],
                    },
                )
                continue

            scenario_bindings[scenario_id] = doc_id
            if ocr_extract is not None:
                pre_extracted[doc_id] = ocr_extract

        scenario_payloads: dict[str, dict[str, Any]] = {}

        for scenario_id in scenario_ids:
            doc_id = scenario_bindings.get(scenario_id)
            if doc_id is None:
                conflicts.append({"kind": "MISSING_KYC", "scenario_id": scenario_id})
                continue

            doc = inventory["documents"][doc_id]
            if doc_id in pre_extracted:
                extracted = pre_extracted[doc_id]
                verification = {
                    "header_account": False,
                    "table_semantics": False,
                    "threshold": False,
                }
                for index in range(len(extracted.ownership_rows)):
                    verification[f"ownership_{index}_name"] = False
                    verification[f"ownership_{index}_pct"] = False
                source_kind = "ocr"
                review_fields = _review_fields_for_source(source_kind, verification)
                perimeter = None
                digit_mismatches: list[dict[str, Any]] = []
            else:
                with capture_absent_values() as absent_fields:
                    extracted, verification, source_kind, review_fields, perimeter, digit_mismatches = (
                        await _extract_dossier(
                            client,
                            work_dir=work_dir,
                            scenario_id=scenario_id,
                            doc_id=doc_id,
                            doc=doc,
                        )
                    )
                for field_name in absent_fields:
                    conflicts.append(
                        {
                            "kind": "ABSENT_VALUE",
                            "field": field_name,
                            "scenario_id": scenario_id,
                            "doc_id": doc_id,
                        },
                    )

            conflicts.extend(digit_mismatches)
            review.extend(digit_mismatches)

            if not extracted.threshold_quote.strip():
                conflicts.append(
                    {
                        "kind": "KYC_THRESHOLD_NOT_FOUND",
                        "scenario_id": scenario_id,
                        "doc_id": doc_id,
                    },
                )
                continue

            threshold = extracted.threshold_pct
            if threshold is None:
                conflicts.append(
                    {
                        "kind": "KYC_THRESHOLD_ABSENT",
                        "scenario_id": scenario_id,
                        "doc_id": doc_id,
                    },
                )
                continue
            table_semantics = extracted.table_semantics
            pages = doc["pages"]
            parties, ownership_rows = _related_parties_from_extract(
                extracted,
                doc_id=doc_id,
                pages=pages,
                threshold=threshold,
                table_semantics=table_semantics,
                source_kind=source_kind,
            )
            related_counterparties = [party.counterparty for party in parties if party.is_related]

            scenario_ledger = ledger[ledger["txn_id"].str.startswith(f"TXN-{scenario_id}-")]
            ledger_map = _build_ledger_map(
                scenario_ledger["counterparty"].tolist(),
                related_counterparties,
            )

            confidence = _confidence_from_verification(verification, source_kind=source_kind)
            threshold_page = _source_page_for_quote(pages, extracted.threshold_quote)
            semantics_page = _source_page_for_quote(pages, extracted.table_semantics_quote)

            scenario_payloads[scenario_id] = {
                "scenario_id": scenario_id,
                "doc_id": doc_id,
                "header_account": extracted.header_account,
                "table_semantics": table_semantics,
                "direction": _direction_for_semantics(table_semantics),
                "table_semantics_source": _serialize_provenance(
                    Provenance(
                        doc_id=doc_id,
                        page=semantics_page,
                        quote=extracted.table_semantics_quote,
                        extractor=EXTRACTOR,
                    ),
                    source_kind=source_kind,
                ),
                "threshold_pct": _decimal_to_str(threshold),
                "threshold_source": _serialize_provenance(
                    Provenance(
                        doc_id=doc_id,
                        page=threshold_page,
                        quote=extracted.threshold_quote,
                        extractor=EXTRACTOR,
                    ),
                    source_kind=source_kind,
                ),
                "ownership": ownership_rows,
                "related_counterparties": related_counterparties,
                "related_normalized": {
                    normalize_counterparty(name): name for name in related_counterparties
                },
                "ledger_map": ledger_map,
                "confidence": _decimal_to_str(confidence),
                "review_fields": review_fields,
                "verification": verification,
            }

            if perimeter is not None and perimeter.threshold_pct is not None:
                peri_threshold = perimeter.threshold_pct
                _, peri_rows = _related_parties_from_extract(
                    perimeter,
                    doc_id=doc_id,
                    pages=pages,
                    threshold=peri_threshold,
                    table_semantics=perimeter.table_semantics,
                    source_kind="ocr",
                )
                scenario_payloads[scenario_id]["perimeter"] = {
                    "table_semantics": perimeter.table_semantics,
                    "direction": _direction_for_semantics(perimeter.table_semantics),
                    "threshold_pct": _decimal_to_str(peri_threshold),
                    "ownership": peri_rows,
                }
                scenario_payloads[scenario_id]["review_fields"] = list(
                    dict.fromkeys(
                        scenario_payloads[scenario_id]["review_fields"]
                        + ["perimeter"],
                    ),
                )

    payload = {
        "scenarios": scenario_payloads,
        "conflicts": conflicts,
        "review": review,
        "summary": {
            "scenario_count": len(scenario_payloads),
            "related_party_rows": sum(len(item["ownership"]) for item in scenario_payloads.values()),
            "review_count": len(review),
        },
    }

    if is_canonical_open_dataset(work_dir):
        _check_payload_structure(payload, conflicts=conflicts)
        payload["summary"]["conflict_count"] = len(conflicts)

    output_path = work_dir / "04b_parties.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"s4b_parties: scenarios={len(scenario_payloads)} "
        f"conflicts={len(conflicts)}",
    )

    return StageResult(item_count=len(scenario_payloads), row_count=len(scenario_payloads))


def run(*, work_dir: Path) -> StageResult:
    return asyncio.run(_run_async(work_dir))
