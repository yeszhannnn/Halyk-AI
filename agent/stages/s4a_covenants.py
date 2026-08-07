from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent.evidence.quotes import verify_extracted_fields, verify_quote
from agent.llm.client import LLMClient
from agent.llm.schemas.covenants import (
    CategorySpecExtract,
    CovenantExtract,
    MetricSpecExtract,
    SpringingConditionExtract,
)
from agent.models import CategorySpec, Covenant, MetricSpec, Provenance, SpringingCondition
from agent.stages import StageResult
from agent.template import load_template, template_cells

ARTICLE_6_HEADING = re.compile(
    r"Статья 6\s*[—–-]\s*Финансовые ковенанты",
    re.IGNORECASE,
)
ARTICLE_7_HEADING = re.compile(r"Статья 7\b", re.IGNORECASE)
PUNKT_MARKER = re.compile(r"Пункт 6\.(\d+)")
REQUIRED_PUNKT_MARKERS = ("Пункт 6.1", "Пункт 6.2", "Пункт 6.3")

EXPECTED_YEAR = 2025
COVENANT_YEAR = (date(2025, 1, 1), date(2025, 12, 31))
SLOTS = ("1", "2", "3")
EXTRACTOR = "s4a_covenants"

SYSTEM_PROMPT = """You extract bank loan financial covenants from Russian/English contract text.

Rules:
- Return only information explicitly stated in the provided clause text.
- Every scalar field must have a matching *_quote field: a verbatim substring copied from the clause.
- direction must be MAX or MIN based on comparison language in the clause (не допускать превышения → MAX;
  не менее / not fall below / обеспечить не ниже → MIN). Never infer direction from the title alone.
- threshold_unit is USD for dollar amounts, RATIO for multipliers like 0.04x or 1.20x.
- metric.scope is BORROWER unless the clause explicitly uses group/consolidated/Группы scope.
- metric.numerator/denominator describe how to compute the figure, including auditor reclass rules.
- springing is non-null only when the test applies conditionally (e.g. "применяется только при условии, что...").
- period_start and period_end must reflect the measurement period stated in the clause.
  Most covenants use 2025-01-01 through 2025-12-31; some specify a sub-period (e.g. fourth quarter).
- Quotes must be exact contiguous substrings from the clause; do not paraphrase or normalize numbers.
"""


def _extract_article_6(pages: list[str]) -> str:
    full_text = "\n".join(pages)
    matches = list(ARTICLE_6_HEADING.finditer(full_text))
    if not matches:
        raise ValueError("Article 6 heading with em dash not found")

    start = matches[-1].start()
    remainder = full_text[start:]
    article_7 = ARTICLE_7_HEADING.search(remainder)
    end = article_7.start() if article_7 else len(remainder)
    section = remainder[:end].strip()

    missing = [marker for marker in REQUIRED_PUNKT_MARKERS if marker not in section]
    if missing:
        raise ValueError(f"Article 6 section missing required markers: {', '.join(missing)}")

    return section


def _split_punkts(section: str) -> dict[str, str]:
    positions: list[tuple[int, str]] = []
    for match in PUNKT_MARKER.finditer(section):
        slot = match.group(1)
        if slot in SLOTS:
            positions.append((match.start(), slot))

    positions.sort(key=lambda item: item[0])
    by_slot: dict[str, str] = {}
    for index, (start, slot) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(section)
        by_slot[slot] = section[start:end].strip()

    missing = [slot for slot in SLOTS if slot not in by_slot]
    if missing:
        raise ValueError(f"Article 6 missing пункты: {', '.join(f'6.{s}' for s in missing)}")

    return by_slot


def _page_span_for_text(pages: list[str], needle: str) -> tuple[int, int, str]:
    normalized_needle = " ".join(needle.split()).casefold()
    if not normalized_needle:
        return 1, len(pages), "\n".join(pages)

    for start_page in range(1, len(pages) + 1):
        for end_page in range(start_page, len(pages) + 1):
            combined = "\n".join(pages[start_page - 1 : end_page])
            if normalized_needle in " ".join(combined.split()).casefold():
                return start_page, end_page, combined

    return 1, len(pages), "\n".join(pages)


def _source_page_for_quote(pages: list[str], quote: str, fallback: int) -> int:
    if not quote:
        return fallback
    for page_number, page_text in enumerate(pages, start=1):
        if verify_quote(quote, page_text):
            return page_number
    return fallback


def _quote_checks(result: CovenantExtract, verification_text: str) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = [
        ("title", result.title_quote, verification_text),
        ("direction", result.direction_quote, verification_text),
        ("threshold", result.threshold_quote, verification_text),
        ("threshold_unit", result.threshold_unit_quote, verification_text),
        ("period", result.period_quote, verification_text),
        ("metric_kind", result.metric.kind_quote, verification_text),
        ("metric_scope", result.metric.scope_quote, verification_text),
        ("metric_notes", result.notes_quote, verification_text),
        (
            "metric_numerator_include",
            result.metric.numerator.include_keywords_quote,
            verification_text,
        ),
        ("metric_numerator_sign", result.metric.numerator.sign_quote, verification_text),
        (
            "metric_numerator_reclass",
            result.metric.numerator.apply_reclass_quote,
            verification_text,
        ),
    ]
    if result.metric.denominator is not None:
        checks.extend(
            [
                (
                    "metric_denominator_include",
                    result.metric.denominator.include_keywords_quote,
                    verification_text,
                ),
                (
                    "metric_denominator_sign",
                    result.metric.denominator.sign_quote,
                    verification_text,
                ),
                (
                    "metric_denominator_reclass",
                    result.metric.denominator.apply_reclass_quote,
                    verification_text,
                ),
            ],
        )
    if result.metric.numerator.exclude_keywords_quote:
        checks.append(
            (
                "metric_numerator_exclude",
                result.metric.numerator.exclude_keywords_quote,
                verification_text,
            ),
        )
    if result.springing is not None:
        checks.extend(_springing_quote_checks(result.springing, verification_text))
    return checks


def _springing_quote_checks(
    springing: SpringingConditionExtract,
    verification_text: str,
) -> list[tuple[str, str, str]]:
    return [
        ("springing_operator", springing.operator_quote, verification_text),
        ("springing_value", springing.value_quote, verification_text),
        ("springing_condition", springing.condition_quote, verification_text),
        ("springing_metric_kind", springing.metric.kind_quote, verification_text),
        ("springing_metric_scope", springing.metric.scope_quote, verification_text),
        ("springing_metric_notes", springing.metric.notes_quote, verification_text),
        (
            "springing_metric_numerator_include",
            springing.metric.numerator.include_keywords_quote,
            verification_text,
        ),
    ]


def _category_from_extract(spec: CategorySpecExtract) -> CategorySpec:
    return CategorySpec(
        include_keywords=spec.include_keywords,
        exclude_keywords=spec.exclude_keywords,
        sign=spec.sign.value,
        apply_reclass=spec.apply_reclass,
    )


def _metric_from_extract(spec: MetricSpecExtract, *, notes: str = "") -> MetricSpec:
    return MetricSpec(
        kind=spec.kind.value,
        numerator=_category_from_extract(spec.numerator),
        denominator=_category_from_extract(spec.denominator) if spec.denominator else None,
        scope=spec.scope.value,
        notes=notes or spec.notes,
    )


def _provenance(
    *,
    doc_id: str,
    pages: list[str],
    quote: str,
    fallback_page: int,
) -> Provenance:
    return Provenance(
        doc_id=doc_id,
        page=_source_page_for_quote(pages, quote, fallback_page),
        quote=quote,
        extractor=EXTRACTOR,
    )


def _springing_from_extract(
    springing: SpringingConditionExtract,
    *,
    doc_id: str,
    pages: list[str],
    fallback_page: int,
) -> SpringingCondition:
    return SpringingCondition(
        metric=_metric_from_extract(springing.metric),
        operator=springing.operator,
        value=springing.value,
        source=_provenance(
            doc_id=doc_id,
            pages=pages,
            quote=springing.condition_quote,
            fallback_page=fallback_page,
        ),
    )


def _covenant_from_extract(
    extracted: CovenantExtract,
    *,
    scenario_id: str,
    slot: str,
    doc_id: str,
    pages: list[str],
    fallback_page: int,
) -> Covenant:
    springing = (
        _springing_from_extract(
            extracted.springing,
            doc_id=doc_id,
            pages=pages,
            fallback_page=fallback_page,
        )
        if extracted.springing is not None
        else None
    )
    return Covenant(
        scenario_id=scenario_id,
        slot=f"6.{slot}",
        title=extracted.title,
        direction=extracted.direction.value,
        threshold=extracted.threshold,
        threshold_unit=extracted.threshold_unit.value,
        metric=_metric_from_extract(extracted.metric, notes=extracted.notes),
        period=(extracted.period_start, extracted.period_end),
        springing=springing,
        source=_provenance(
            doc_id=doc_id,
            pages=pages,
            quote=extracted.threshold_quote,
            fallback_page=fallback_page,
        ),
    )


def _decimal_to_str(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _serialize_provenance(provenance: Provenance) -> dict[str, Any]:
    return {
        "doc_id": provenance.doc_id,
        "page": provenance.page,
        "quote": provenance.quote,
        "extractor": provenance.extractor,
    }


def _serialize_category(category: CategorySpec) -> dict[str, Any]:
    return {
        "include_keywords": category.include_keywords,
        "exclude_keywords": category.exclude_keywords,
        "sign": category.sign,
        "apply_reclass": category.apply_reclass,
    }


def _serialize_metric(metric: MetricSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": metric.kind,
        "numerator": _serialize_category(metric.numerator),
        "scope": metric.scope,
        "notes": metric.notes,
    }
    if metric.denominator is not None:
        payload["denominator"] = _serialize_category(metric.denominator)
    else:
        payload["denominator"] = None
    return payload


def _serialize_springing(springing: SpringingCondition | None) -> dict[str, Any] | None:
    if springing is None:
        return None
    return {
        "metric": _serialize_metric(springing.metric),
        "operator": springing.operator,
        "value": _decimal_to_str(springing.value),
        "source": _serialize_provenance(springing.source),
    }


def _serialize_covenant(covenant: Covenant, verification: dict[str, bool]) -> dict[str, Any]:
    period_start, period_end = covenant.period
    return {
        "scenario_id": covenant.scenario_id,
        "slot": covenant.slot,
        "title": covenant.title,
        "direction": covenant.direction,
        "threshold": _decimal_to_str(covenant.threshold),
        "threshold_unit": covenant.threshold_unit,
        "metric": _serialize_metric(covenant.metric),
        "period": [period_start.isoformat(), period_end.isoformat()],
        "springing": _serialize_springing(covenant.springing),
        "source": _serialize_provenance(covenant.source),
        "verification": verification,
    }


def _collect_verification_flags(payload: dict[str, Any]) -> dict[str, bool]:
    return {
        key.removesuffix("_verified"): bool(value)
        for key, value in payload.items()
        if key.endswith("_verified")
    }


def _placeholder_covenant(scenario_id: str, slot: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "slot": slot,
        "title": "UNAVAILABLE",
        "direction": "MAX",
        "threshold": "0",
        "threshold_unit": "USD",
        "metric": {
            "kind": "SUM",
            "numerator": {
                "include_keywords": [],
                "exclude_keywords": [],
                "sign": "OUTFLOW",
                "apply_reclass": True,
            },
            "denominator": None,
            "scope": "BORROWER",
            "notes": "",
        },
        "period": ["2025-01-01", "2025-12-31"],
        "springing": None,
        "source": {"doc_id": "", "page": 1, "quote": "", "extractor": EXTRACTOR},
        "verification": {},
        "degraded": True,
    }


def _validate_period(
    period: tuple[date, date],
    *,
    scenario_id: str,
    doc_id: str,
    slot: str,
    conflicts: list[dict[str, Any]],
) -> bool:
    start, end = period
    year_start, year_end = COVENANT_YEAR
    if start.year != EXPECTED_YEAR or end.year != EXPECTED_YEAR:
        conflicts.append(
            {
                "kind": "PERIOD_MISMATCH",
                "scenario_id": scenario_id,
                "doc_id": doc_id,
                "slot": f"6.{slot}",
                "expected": str(EXPECTED_YEAR),
                "found": [start.isoformat(), end.isoformat()],
            },
        )
        return False
    if start < year_start or end > year_end:
        conflicts.append(
            {
                "kind": "PERIOD_MISMATCH",
                "scenario_id": scenario_id,
                "doc_id": doc_id,
                "slot": f"6.{slot}",
                "expected": [d.isoformat() for d in COVENANT_YEAR],
                "found": [start.isoformat(), end.isoformat()],
            },
        )
        return False
    return True


async def _extract_covenant_item(
    client: LLMClient,
    *,
    scenario_id: str,
    slot: str,
    item_text: str,
    verification_text: str,
) -> tuple[CovenantExtract, dict[str, bool]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Scenario: {scenario_id}\n"
                f"Slot: 6.{slot}\n\n"
                "Extract the covenant from this clause text:\n\n"
                f"{item_text}"
            ),
        },
    ]
    extracted = await client.complete_verified(
        response_model=CovenantExtract,
        messages=messages,
        quote_checks=lambda result: _quote_checks(result, verification_text),
    )
    payload = extracted.model_dump(mode="python")
    verify_extracted_fields(payload, fields=_quote_checks(extracted, verification_text))
    verification = _collect_verification_flags(payload)
    return extracted, verification


async def _process_scenario(
    client: LLMClient,
    *,
    scenario_id: str,
    doc_id: str,
    pages: list[str],
    conflicts: list[dict[str, Any]],
) -> list[tuple[Covenant, dict[str, bool]]]:
    section = _extract_article_6(pages)
    items = _split_punkts(section)
    results: list[tuple[Covenant, dict[str, bool]]] = []

    for slot in SLOTS:
        item_text = items[slot]
        fallback_page, _, _ = _page_span_for_text(pages, item_text[:120])
        extracted, verification = await _extract_covenant_item(
            client,
            scenario_id=scenario_id,
            slot=slot,
            item_text=item_text,
            verification_text=item_text,
        )

        period = (extracted.period_start, extracted.period_end)
        if not _validate_period(
            period,
            scenario_id=scenario_id,
            doc_id=doc_id,
            slot=slot,
            conflicts=conflicts,
        ):
            continue

        covenant = _covenant_from_extract(
            extracted,
            scenario_id=scenario_id,
            slot=slot,
            doc_id=doc_id,
            pages=pages,
            fallback_page=fallback_page,
        )
        results.append((covenant, verification))

    return results


async def _run_async(work_dir: Path) -> StageResult:
    inventory = json.loads((work_dir / "01_inventory.json").read_text(encoding="utf-8"))
    bound = json.loads((work_dir / "03_bound.json").read_text(encoding="utf-8"))
    template = load_template(work_dir)

    client = LLMClient()
    conflicts: list[dict[str, Any]] = []
    serialized: list[dict[str, Any]] = []

    scenarios = bound["scenarios"]
    extracted_by_scenario: dict[str, dict[str, tuple[Covenant, dict[str, bool]]]] = {}

    for scenario_id in sorted({scenario for scenario, _slot in template_cells(template)}):
        loan_doc_id = scenarios.get(scenario_id, {}).get("loan")
        if not loan_doc_id:
            if scenario_id not in scenarios:
                conflicts.append({"kind": "EXTRA_SCENARIO", "scenario_id": scenario_id})
            continue
        try:
            pages = inventory["documents"][loan_doc_id]["pages"]
            scenario_results = await _process_scenario(
                client,
                scenario_id=scenario_id,
                doc_id=loan_doc_id,
                pages=pages,
                conflicts=conflicts,
            )
            extracted_by_scenario[scenario_id] = {
                covenant.slot: (covenant, verification)
                for covenant, verification in scenario_results
            }
        except Exception as exc:  # noqa: BLE001 — structural gaps must not abort
            conflicts.append(
                {
                    "kind": "COVENANT_EXTRACTION_FAILED",
                    "scenario_id": scenario_id,
                    "error": str(exc),
                },
            )

    for scenario_id, slot in template_cells(template):
        loan_doc_id = scenarios.get(scenario_id, {}).get("loan")
        extracted_by_slot = extracted_by_scenario.get(scenario_id, {})
        if slot in extracted_by_slot:
            covenant, verification = extracted_by_slot[slot]
            serialized.append(_serialize_covenant(covenant, verification))
        else:
            if loan_doc_id:
                conflicts.append(
                    {
                        "kind": "NEW_SLOT",
                        "scenario_id": scenario_id,
                        "slot": slot,
                    },
                )
            serialized.append(_placeholder_covenant(scenario_id, slot))

    springing_count = sum(1 for covenant in serialized if covenant.get("springing") is not None)
    slot_62_directions = {
        covenant["direction"]
        for covenant in serialized
        if covenant["slot"] == "6.2"
    }

    payload = {
        "covenants": serialized,
        "conflicts": conflicts,
        "summary": {
            "count": len(serialized),
            "springing_count": springing_count,
            "slot_6_2_directions": sorted(slot_62_directions),
        },
    }

    output_path = work_dir / "04a_covenants.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"s4a_covenants: extracted={len(serialized)} springing={springing_count} "
        f"6.2_directions={sorted(slot_62_directions)} conflicts={len(conflicts)}",
    )

    return StageResult(item_count=len(serialized), row_count=springing_count)


def run(*, work_dir: Path) -> StageResult:
    return asyncio.run(_run_async(work_dir))
