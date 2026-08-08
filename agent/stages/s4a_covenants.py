from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent.evidence.quotes import verify_extracted_fields, verify_quote
from agent.llm import client as llm_client
from agent.llm.client import LLMClient, LLMTransportExhaustedError, LLMValidationError
from agent.llm.extraction_vote import EXTRACTION_UNSTABLE
from agent.llm.schemas.covenants import (
    CategorySpecExtract,
    CovenantExtract,
    CovenantFlatNoSpringing,
    CovenantFlatWithSpringing,
    CovenantMetricExtract,
    CovenantMetricWithSpringing,
    LEDGER_CATEGORIES_CONTEXT_KEY,
    MetricKind,
    MetricSpecExtract,
    SpringingConditionExtract,
    _text_refers_to_related_parties,
)
from agent.models import CategorySpec, Covenant, MetricSpec, Provenance, SpringingCondition
from agent.parsing.categories import OPEX_SLUGS, derive_leg_sign
from agent.parsing.numbers import (
    ABSENT_SENTINEL_MESSAGE,
    capture_absent_values,
    normalize_decimal,
)
from agent.stages import StageResult
from agent.template import load_template, template_cells

logger = logging.getLogger(__name__)

ARTICLE_6_HEADING = re.compile(
    r"Статья 6\s*[—–-]\s*Финансовые ковенанты",
    re.IGNORECASE,
)
ARTICLE_7_HEADING = re.compile(r"Статья 7\b", re.IGNORECASE)
PUNKT_MARKER = re.compile(r"Пункт 6\.(\d+)")
REQUIRED_PUNKT_MARKERS = ("Пункт 6.1", "Пункт 6.2", "Пункт 6.3")
THRESHOLD_RATIO_RE = re.compile(
    r"(?<![\d,.])(\d+(?:[.,]\d+)?)\s*[xх×X]",
)
THRESHOLD_USD_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:[.,]\d+)?)",
)

EXPECTED_YEAR = 2025
COVENANT_YEAR = (date(2025, 1, 1), date(2025, 12, 31))
SLOTS = ("1", "2", "3")
EXTRACTOR = "s4a_covenants"
THRESHOLD_RATIO_MAX = Decimal("1000")
THRESHOLD_USD_MIN = Decimal("1000")
ZERO_THRESHOLD_MARKERS = ("0", "0.0", "0,0", "ноль", "zero")
MAX_INVALID_COVENANT_FRACTION = Decimal("0.2")
MAX_EXTRACTION_ATTEMPTS = 3

OPEX_KEYWORDS = frozenset({"opex"} | set(OPEX_SLUGS))


class CovenantExtractionError(RuntimeError):
    """Raised when a covenant clause cannot be extracted after all retries."""

    def __init__(
        self,
        *,
        scenario_id: str,
        slot: str,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        self.scenario_id = scenario_id
        self.slot = slot
        super().__init__(f"scenario={scenario_id} slot={slot}: {message}")
        if cause is not None:
            self.__cause__ = cause


SYSTEM_PROMPT = """You extract bank loan financial covenants from Russian/English contract text.

Rules:
- Return only information explicitly stated in the provided clause text.
- Every scalar field must have a matching *_quote field: a verbatim substring copied from the clause.
- Quotes must be exact contiguous substrings from the clause; do not paraphrase or normalize numbers.
"""

FLAT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """
- direction must be MAX or MIN based on comparison language in the clause (не допускать превышения → MAX;
  не менее / not fall below / обеспечить не ниже → MIN). Never infer direction from the title alone.
- threshold_unit is USD for dollar amounts, RATIO for multipliers like 0.04x or 1.20x.
- period_start and period_end must reflect the measurement period stated in the clause.
  Most covenants use 2025-01-01 through 2025-12-31; some specify a sub-period (e.g. fourth quarter).
- When a springing trigger is expected, extract only the trigger operator, value, and condition quote.
  Do not extract the nested springing metric in this step.
"""
)

METRIC_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """
- metric.scope is BORROWER unless the clause explicitly uses group/consolidated/Группы scope.
- For metric.kind RATIO: provide numerator and denominator category legs.
- In a ratio expressing a share, the denominator is the whole and the numerator a
  subset of it: the two legs must never select the same category with the same
  filter (e.g. assets transferred to unrestricted subsidiaries over the borrower's
  total capital assets — the numerator filters the capex subset, the denominator
  selects all capex).
- For metric.kind SUM or COUNT: provide a single category leg (not numerator/denominator).
- Each category leg's include_keywords must contain one or more slugs from the ledger
  category list provided in the prompt. Never invent phrases or translate contract language.
- Related-party payment legs (numerator for payments to affiliates/связанные стороны) do not
  use ledger categories: leave include_keywords empty and quote the related-party clause text
  in include_keywords_quote. Counterparty filtering is resolved from stage 4b.
- Cash-flow sign is derived from category type in code; do not output sign fields.
- notes must capture the full verbatim formulation of how the metric is computed.
- Use only the provided clause text; do not borrow category legs or formulations from
  other covenants in the same borrower agreement (e.g. a revenue-coverage ratio must
  not reuse a related-party payment numerator from a sibling clause).
- When a springing trigger is expected, also extract springing_metric for the nested trigger metric.
"""
)

SPRINGING_TRIGGER_PHRASES: tuple[tuple[str, str], ...] = (
    ("применяется только если", "ru_applies_only_if"),
    ("только при условии, что", "ru_only_if_condition"),
    ("при условии, что", "ru_if_condition"),
    ("в случае если", "ru_in_case_if"),
    ("если совокупные", "ru_if_aggregate"),
    ("only applies if", "en_only_applies_if"),
    ("applies only if", "en_applies_only_if"),
    ("only if", "en_only_if"),
    ("only when", "en_only_when"),
    ("in the event that", "en_in_event"),
    ("if the aggregate", "en_if_aggregate"),
    ("if aggregate", "en_if_aggregate_short"),
)


def _detect_springing_trigger(text: str) -> str | None:
    normalized = " ".join(text.casefold().split())
    for phrase, label in SPRINGING_TRIGGER_PHRASES:
        if phrase.casefold() in normalized:
            return label
    return None


def _log_springing_trigger(scenario_id: str, slot: str, label: str, phrase: str) -> None:
    logger.info(
        "s4a springing trigger %s scenario=%s slot=6.%s phrase=%s",
        label,
        scenario_id,
        slot,
        phrase,
    )


def _springing_trigger_label(text: str) -> tuple[str, str] | None:
    normalized = " ".join(text.casefold().split())
    for phrase, label in SPRINGING_TRIGGER_PHRASES:
        if phrase.casefold() in normalized:
            return label, phrase
    return None


def _load_ledger_categories(work_dir: Path) -> tuple[dict[str, list[str]], list[str]]:
    path = work_dir / "05_ledger.json"
    if not path.exists():
        raise FileNotFoundError(
            f"s4a_covenants requires ledger categories from stage 5: {path} not found",
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_scenario = {
        str(scenario_id): list(categories)
        for scenario_id, categories in (payload.get("categories_by_scenario") or {}).items()
    }
    global_categories = list(payload.get("categories") or [])
    if not global_categories and by_scenario:
        global_categories = sorted({category for cats in by_scenario.values() for category in cats})
    return by_scenario, global_categories


def _metric_primary_category(metric: MetricSpecExtract | Any) -> Any:
    if metric.kind == MetricKind.RATIO:
        assert metric.numerator is not None
        return metric.numerator
    assert metric.category is not None
    return metric.category


def _metric_category_legs(metric: MetricSpecExtract | Any) -> list[tuple[str, Any]]:
    if metric.kind == MetricKind.RATIO:
        assert metric.numerator is not None and metric.denominator is not None
        return [
            ("numerator", metric.numerator),
            ("denominator", metric.denominator),
        ]
    assert metric.category is not None
    return [("category", metric.category)]


def _to_covenant_extract(
    flat: CovenantFlatNoSpringing | CovenantFlatWithSpringing,
    metric: CovenantMetricExtract | CovenantMetricWithSpringing,
    *,
    expect_springing: bool,
) -> CovenantExtract:
    data = flat.model_dump(mode="python")
    data.update(
        {
            "metric": metric.metric,
            "notes": metric.notes,
            "notes_quote": metric.notes_quote,
            "springing": None,
        },
    )
    if expect_springing:
        flat_springing = flat
        assert isinstance(flat_springing, CovenantFlatWithSpringing)
        assert isinstance(metric, CovenantMetricWithSpringing)
        if flat_springing.springing_value is not None:
            data["springing"] = {
                "metric": metric.springing_metric,
                "operator": flat_springing.springing_operator,
                "operator_quote": flat_springing.springing_operator_quote,
                "value": flat_springing.springing_value,
                "value_quote": flat_springing.springing_value_quote,
                "condition_quote": flat_springing.springing_condition_quote,
            }
    return CovenantExtract.model_validate(data)


def _retry_line_from_error(message: str) -> str:
    lowered = message.casefold()
    if "metric" in lowered and "object" in lowered:
        return "metric was missing."
    if "include_keywords" in lowered:
        return "metric category include_keywords must use ledger slugs from the prompt."
    if "numerator" in lowered and "denominator" in lowered:
        return "metric numerator and denominator are required for RATIO."
    if "THRESHOLD_NOT_IN_CLAUSE" in message:
        return message
    first_line = message.split("\n", 1)[0].strip()
    if len(first_line) > 120:
        return first_line[:120]
    return first_line


def _append_retry_line(content: str, retry_line: str | None) -> str:
    if not retry_line:
        return content
    return f"{content}\n\nRetry: {retry_line}"


def _flat_quote_checks(
    result: CovenantFlatNoSpringing | CovenantFlatWithSpringing,
    verification_text: str,
    *,
    expect_springing: bool,
) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = [
        ("title", result.title_quote, verification_text),
        ("direction", result.direction_quote, verification_text),
        ("threshold", result.threshold_quote, verification_text),
        ("threshold_unit", result.threshold_unit_quote, verification_text),
        ("period", result.period_quote, verification_text),
    ]
    if expect_springing and isinstance(result, CovenantFlatWithSpringing):
        checks.extend(
            [
                ("springing_operator", result.springing_operator_quote, verification_text),
                ("springing_value", result.springing_value_quote, verification_text),
                ("springing_condition", result.springing_condition_quote, verification_text),
            ],
        )
    return checks


def _metric_quote_checks(
    result: CovenantMetricExtract | CovenantMetricWithSpringing,
    verification_text: str,
    *,
    expect_springing: bool,
) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = [
        ("metric_kind", result.metric.kind_quote, verification_text),
        ("metric_scope", result.metric.scope_quote, verification_text),
        ("metric_notes", result.notes_quote, verification_text),
    ]
    for leg, category in _metric_category_legs(result.metric):
        checks.extend(
            [
                (
                    f"metric_{leg}_include",
                    category.include_keywords_quote,
                    verification_text,
                ),
                (
                    f"metric_{leg}_reclass",
                    category.apply_reclass_quote,
                    verification_text,
                ),
            ],
        )
        if category.exclude_keywords_quote:
            checks.append(
                (
                    f"metric_{leg}_exclude",
                    category.exclude_keywords_quote,
                    verification_text,
                ),
            )
    if expect_springing and isinstance(result, CovenantMetricWithSpringing):
        checks.extend(
            [
                ("springing_metric_kind", result.springing_metric.kind_quote, verification_text),
                ("springing_metric_scope", result.springing_metric.scope_quote, verification_text),
            ],
        )
        primary = _metric_primary_category(result.springing_metric)
        checks.append(
            (
                "springing_metric_category_include",
                primary.include_keywords_quote,
                verification_text,
            ),
        )
    return checks


def _format_flat_context(
    flat: CovenantFlatNoSpringing | CovenantFlatWithSpringing,
) -> str:
    return (
        f"title={flat.title!r}, direction={flat.direction.value}, "
        f"threshold={_decimal_to_str(flat.threshold)} {flat.threshold_unit.value}, "
        f"period={flat.period_start.isoformat()}..{flat.period_end.isoformat()}"
    )


async def _extract_voted_part(
    client: LLMClient,
    *,
    response_model: type[Any],
    build_user_content: Any,
    quote_checks: Any,
    validation_context: dict[str, Any] | None,
    system_prompt: str,
    context: dict[str, Any],
    use_cache: bool,
    retry_line: str | None = None,
) -> tuple[Any, list[dict[str, Any]], int]:
    unstable_fields: list[dict[str, Any]] = []
    current_retry = retry_line
    for attempt in range(1, MAX_EXTRACTION_ATTEMPTS + 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _append_retry_line(build_user_content(), current_retry),
            },
        ]
        try:
            raw, pass_unstable = await client.complete_verified_voted(
                response_model=response_model,
                messages=messages,
                quote_checks=quote_checks,
                validation_context=validation_context,
                context=context,
                use_cache=use_cache,
            )
            unstable_fields.extend(pass_unstable)
            return raw, unstable_fields, attempt
        except LLMValidationError as exc:
            if ABSENT_SENTINEL_MESSAGE in str(exc):
                raise
            if attempt >= MAX_EXTRACTION_ATTEMPTS:
                raise
            current_retry = _retry_line_from_error(str(exc))
    raise RuntimeError("extraction part returned no result")


def _normalize_threshold_number(raw: str) -> Decimal:
    return normalize_decimal(raw, field_name="threshold")


def _extract_threshold_candidates(text: str) -> list[tuple[Decimal, str]]:
    candidates: list[tuple[Decimal, str]] = []
    seen: set[Decimal] = set()
    for pattern in (THRESHOLD_RATIO_RE, THRESHOLD_USD_RE):
        for match in pattern.finditer(text):
            value = _normalize_threshold_number(match.group(1))
            if value in seen:
                continue
            seen.add(value)
            candidates.append((value, match.group(0)))
    return candidates


def _threshold_matches_candidates(
    threshold: Decimal,
    candidates: list[tuple[Decimal, str]],
) -> bool:
    if not candidates:
        return True
    return any(threshold == candidate for candidate, _token in candidates)


def _filter_threshold_candidates_by_unit(
    candidates: list[tuple[Decimal, str]],
    unit: str,
) -> list[tuple[Decimal, str]]:
    if unit == "RATIO":
        return [
            (value, token)
            for value, token in candidates
            if re.search(r"[xх×X]\s*$", token)
        ]
    if unit == "USD":
        return [(value, token) for value, token in candidates if "$" in token]
    return candidates


def _threshold_anchor_issue(
    extracted: CovenantExtract,
    candidates: list[tuple[Decimal, str]],
) -> str | None:
    unit = extracted.threshold_unit.value
    filtered = _filter_threshold_candidates_by_unit(candidates, unit)
    if not filtered:
        return None
    if _threshold_matches_candidates(extracted.threshold, filtered):
        return None
    return (
        "THRESHOLD_NOT_IN_CLAUSE: threshold must equal one of "
        f"{_format_threshold_candidates(filtered)}; got {_decimal_to_str(extracted.threshold)}"
    )


def _format_threshold_candidates(candidates: list[tuple[Decimal, str]]) -> str:
    return ", ".join(f"{token} ({value})" for value, token in candidates)


def _raise_extraction_failure(
    *,
    scenario_id: str,
    slot: str,
    message: str,
    cause: Exception | None = None,
) -> None:
    raise CovenantExtractionError(
        scenario_id=scenario_id,
        slot=f"6.{slot}",
        message=message,
        cause=cause,
    )


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
    ]
    for leg, category in _metric_category_legs(result.metric):
        checks.extend(
            [
                (
                    f"metric_{leg}_include",
                    category.include_keywords_quote,
                    verification_text,
                ),
                (
                    f"metric_{leg}_reclass",
                    category.apply_reclass_quote,
                    verification_text,
                ),
            ],
        )
        if category.exclude_keywords_quote:
            checks.append(
                (
                    f"metric_{leg}_exclude",
                    category.exclude_keywords_quote,
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
    checks = [
        ("springing_operator", springing.operator_quote, verification_text),
        ("springing_value", springing.value_quote, verification_text),
        ("springing_condition", springing.condition_quote, verification_text),
        ("springing_metric_kind", springing.metric.kind_quote, verification_text),
        ("springing_metric_scope", springing.metric.scope_quote, verification_text),
        ("springing_metric_notes", springing.metric.notes_quote, verification_text),
    ]
    primary = _metric_primary_category(springing.metric)
    checks.append(
        (
            "springing_metric_category_include",
            primary.include_keywords_quote,
            verification_text,
        ),
    )
    return checks


def _prefer_ebitda_denominator_shape(
    metric: MetricSpecExtract,
    notes: str,
) -> str | None:
    """Upgrade a bare opex denominator to derived EBITDA when notes name EBITDA."""
    if "ebitda" not in notes.casefold():
        return None
    if metric.kind != MetricKind.RATIO or metric.denominator is None:
        return None
    keywords = frozenset(str(keyword).casefold() for keyword in metric.denominator.include_keywords)
    if not keywords or keywords == frozenset({"revenue", "opex"}):
        return None
    if keywords <= OPEX_KEYWORDS:
        metric.denominator.include_keywords = ["revenue", "opex"]
        return "derived_ebitda"
    return None


def _category_from_extract(spec: CategorySpecExtract | Any) -> CategorySpec:
    categories = [str(keyword) for keyword in spec.include_keywords]
    return CategorySpec(
        include_keywords=categories,
        exclude_keywords=spec.exclude_keywords,
        sign=derive_leg_sign(categories),
        apply_reclass=spec.apply_reclass,
    )


def _metric_from_extract(spec: MetricSpecExtract, *, notes: str = "") -> MetricSpec:
    if spec.kind == MetricKind.RATIO:
        assert spec.numerator is not None and spec.denominator is not None
        numerator = _category_from_extract(spec.numerator)
        denominator = _category_from_extract(spec.denominator)
    else:
        assert spec.category is not None
        numerator = _category_from_extract(spec.category)
        denominator = None
    return MetricSpec(
        kind=spec.kind.value,
        numerator=numerator,
        denominator=denominator,
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
) -> tuple[Covenant, str | None]:
    denominator_shape = _prefer_ebitda_denominator_shape(extracted.metric, extracted.notes)
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
    covenant = Covenant(
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
    return covenant, denominator_shape


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


def _serialize_metric(
    metric: MetricSpec,
    *,
    denominator_shape: str | None = None,
) -> dict[str, Any]:
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
    if denominator_shape:
        payload["denominator_shape"] = denominator_shape
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


def _serialize_covenant(
    covenant: Covenant,
    verification: dict[str, bool],
    *,
    denominator_shape: str | None = None,
) -> dict[str, Any]:
    period_start, period_end = covenant.period
    return {
        "scenario_id": covenant.scenario_id,
        "slot": covenant.slot,
        "title": covenant.title,
        "direction": covenant.direction,
        "threshold": _decimal_to_str(covenant.threshold),
        "threshold_unit": covenant.threshold_unit,
        "metric": _serialize_metric(
            covenant.metric,
            denominator_shape=denominator_shape,
        ),
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


def _quote_states_zero_threshold(quote: str) -> bool:
    normalized = " ".join(quote.casefold().split())
    return any(marker in normalized for marker in ZERO_THRESHOLD_MARKERS)


def _validate_category_spec(
    spec: CategorySpecExtract | Any,
    *,
    leg: str,
    scenario_id: str,
    slot: str,
    allowed_categories: set[str],
    covenant: CovenantExtract | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    related_party_leg = (
        leg in {"primary", "numerator", "category"}
        and (
            _text_refers_to_related_parties(spec.include_keywords_quote)
            or (
                not spec.include_keywords
                and covenant is not None
                and _text_refers_to_related_parties(covenant.title)
            )
        )
    )
    if related_party_leg:
        return issues

    if not spec.include_keywords:
        issues.append(
            {
                "kind": "EMPTY_CATEGORY_KEYWORDS",
                "scenario_id": scenario_id,
                "slot": f"6.{slot}",
                "leg": leg,
            },
        )
        return issues

    invalid = sorted(
        {
            str(keyword)
            for keyword in spec.include_keywords
            if str(keyword) not in allowed_categories
        },
    )
    if invalid:
        issues.append(
            {
                "kind": "INVALID_LEDGER_CATEGORY",
                "scenario_id": scenario_id,
                "slot": f"6.{slot}",
                "leg": leg,
                "invalid_categories": invalid,
            },
        )
    return issues


def _validate_covenant_extract(
    extracted: CovenantExtract,
    *,
    scenario_id: str,
    slot: str,
    allowed_categories: set[str],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    threshold = extracted.threshold
    unit = extracted.threshold_unit.value
    covenant_key = f"{scenario_id}/6.{slot}"

    if threshold == 0 and not _quote_states_zero_threshold(extracted.threshold_quote):
        issues.append(
            {
                "kind": "ZERO_THRESHOLD",
                "covenant_key": covenant_key,
                "scenario_id": scenario_id,
                "slot": f"6.{slot}",
                "threshold": _decimal_to_str(threshold),
                "threshold_quote": extracted.threshold_quote,
            },
        )

    if unit == "RATIO" and threshold >= THRESHOLD_RATIO_MAX:
        issues.append(
            {
                "kind": "RATIO_THRESHOLD_TOO_LARGE",
                "covenant_key": covenant_key,
                "scenario_id": scenario_id,
                "slot": f"6.{slot}",
                "threshold": _decimal_to_str(threshold),
            },
        )

    if unit == "USD" and threshold < THRESHOLD_USD_MIN:
        issues.append(
            {
                "kind": "USD_THRESHOLD_TOO_SMALL",
                "covenant_key": covenant_key,
                "scenario_id": scenario_id,
                "slot": f"6.{slot}",
                "threshold": _decimal_to_str(threshold),
            },
        )

    metric_kind = extracted.metric.kind.value
    if metric_kind == "RATIO" and unit != "RATIO":
        issues.append(
            {
                "kind": "THRESHOLD_UNIT_KIND_MISMATCH",
                "covenant_key": covenant_key,
                "scenario_id": scenario_id,
                "slot": f"6.{slot}",
                "metric_kind": metric_kind,
                "threshold_unit": unit,
            },
        )
    if metric_kind in {"SUM", "COUNT"} and unit != "USD":
        issues.append(
            {
                "kind": "THRESHOLD_UNIT_KIND_MISMATCH",
                "covenant_key": covenant_key,
                "scenario_id": scenario_id,
                "slot": f"6.{slot}",
                "metric_kind": metric_kind,
                "threshold_unit": unit,
            },
        )

    if extracted.springing is not None and extracted.springing.value == threshold:
        issues.append(
            {
                "kind": "SPRINGING_VALUE_EQUALS_THRESHOLD",
                "covenant_key": covenant_key,
                "scenario_id": scenario_id,
                "slot": f"6.{slot}",
                "threshold": _decimal_to_str(threshold),
                "springing_value": _decimal_to_str(extracted.springing.value),
            },
        )

    issues.extend(
        _validate_category_spec(
            _metric_primary_category(extracted.metric),
            leg="primary",
            scenario_id=scenario_id,
            slot=slot,
            allowed_categories=allowed_categories,
            covenant=extracted,
        ),
    )
    if extracted.metric.kind == MetricKind.RATIO:
        assert extracted.metric.denominator is not None
        issues.extend(
            _validate_category_spec(
                extracted.metric.denominator,
                leg="denominator",
                scenario_id=scenario_id,
                slot=slot,
                allowed_categories=allowed_categories,
                covenant=extracted,
            ),
        )
    if extracted.springing is not None:
        springing_metric = extracted.springing.metric
        issues.extend(
            _validate_category_spec(
                _metric_primary_category(springing_metric),
                leg="springing_primary",
                scenario_id=scenario_id,
                slot=slot,
                allowed_categories=allowed_categories,
                covenant=extracted,
            ),
        )
        if springing_metric.kind == MetricKind.RATIO:
            assert springing_metric.denominator is not None
            issues.extend(
                _validate_category_spec(
                    springing_metric.denominator,
                    leg="springing_denominator",
                    scenario_id=scenario_id,
                    slot=slot,
                    allowed_categories=allowed_categories,
                    covenant=extracted,
                ),
            )

    return issues


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
    ledger_categories: list[str],
    threshold_candidates: list[tuple[Decimal, str]] | None = None,
    expect_springing: bool | None = None,
) -> tuple[CovenantExtract, dict[str, bool], list[dict[str, Any]], bool]:
    unstable_fields: list[dict[str, Any]] = []
    candidates = threshold_candidates or _extract_threshold_candidates(item_text)
    active_candidates = threshold_candidates
    flat_retry: str | None = None
    metric_retry: str | None = None
    flat_attempts = 0
    metric_attempts = 0
    extracted: CovenantExtract | None = None

    if expect_springing is None:
        trigger = _springing_trigger_label(item_text)
        expect_springing = trigger is not None
        if trigger is not None:
            label, phrase = trigger
            _log_springing_trigger(scenario_id, slot, label, phrase)

    flat_model = CovenantFlatWithSpringing if expect_springing else CovenantFlatNoSpringing
    metric_model = CovenantMetricWithSpringing if expect_springing else CovenantMetricExtract
    category_list = ", ".join(sorted(ledger_categories))
    validation_context = {LEDGER_CATEGORIES_CONTEXT_KEY: ledger_categories}
    vote_context = {"scenario_id": scenario_id, "slot": f"6.{slot}"}
    use_cache = active_candidates is None

    flat: CovenantFlatNoSpringing | CovenantFlatWithSpringing | None = None
    metric: CovenantMetricExtract | CovenantMetricWithSpringing | None = None

    def build_flat_content() -> str:
        content = (
            f"Scenario: {scenario_id}\n"
            f"Slot: 6.{slot}\n\n"
            "Extract the covenant header (title, direction, threshold, unit, period"
        )
        if expect_springing:
            content += ", and springing trigger operator/value/condition"
        content += ") from this clause text:\n\n" + item_text
        if active_candidates:
            content += (
                "\n\nThe threshold must be exactly one of these values found in the clause: "
                f"{_format_threshold_candidates(active_candidates)}."
            )
        return content

    def build_metric_content() -> str:
        assert flat is not None
        return (
            f"Scenario: {scenario_id}\n"
            f"Slot: 6.{slot}\n\n"
            f"Already extracted covenant header:\n{_format_flat_context(flat)}\n\n"
            f"Ledger categories (include_keywords must be chosen only from this list): "
            f"{category_list}\n\n"
            "Extract only the metric definition"
            + (" and springing_metric" if expect_springing else "")
            + " from this clause text:\n\n"
            + item_text
        )

    outer_attempt = 0
    for outer_attempt in range(1, MAX_EXTRACTION_ATTEMPTS + 1):
        if flat is None or flat_retry is not None:
            try:
                flat, flat_unstable, flat_attempts = await _extract_voted_part(
                    client,
                    response_model=flat_model,
                    build_user_content=build_flat_content,
                    quote_checks=lambda result: _flat_quote_checks(
                        result,
                        verification_text,
                        expect_springing=expect_springing,
                    ),
                    validation_context=None,
                    system_prompt=FLAT_SYSTEM_PROMPT,
                    context={**vote_context, "part": "flat"},
                    use_cache=use_cache,
                    retry_line=flat_retry,
                )
                unstable_fields.extend(flat_unstable)
                flat_retry = None
            except LLMTransportExhaustedError as exc:
                _raise_extraction_failure(
                    scenario_id=scenario_id,
                    slot=slot,
                    message=str(exc),
                    cause=exc,
                )
            except LLMValidationError as exc:
                if ABSENT_SENTINEL_MESSAGE in str(exc):
                    _raise_extraction_failure(
                        scenario_id=scenario_id,
                        slot=slot,
                        message=str(exc),
                        cause=exc,
                    )
                _raise_extraction_failure(
                    scenario_id=scenario_id,
                    slot=slot,
                    message=str(exc),
                    cause=exc,
                )

        if metric is None or metric_retry is not None:
            try:
                metric, metric_unstable, metric_attempts = await _extract_voted_part(
                    client,
                    response_model=metric_model,
                    build_user_content=build_metric_content,
                    quote_checks=lambda result: _metric_quote_checks(
                        result,
                        verification_text,
                        expect_springing=expect_springing,
                    ),
                    validation_context=validation_context,
                    system_prompt=METRIC_SYSTEM_PROMPT,
                    context={**vote_context, "part": "metric"},
                    use_cache=use_cache,
                    retry_line=metric_retry,
                )
                unstable_fields.extend(metric_unstable)
                metric_retry = None
            except LLMTransportExhaustedError as exc:
                _raise_extraction_failure(
                    scenario_id=scenario_id,
                    slot=slot,
                    message=str(exc),
                    cause=exc,
                )
            except LLMValidationError as exc:
                if ABSENT_SENTINEL_MESSAGE in str(exc):
                    _raise_extraction_failure(
                        scenario_id=scenario_id,
                        slot=slot,
                        message=str(exc),
                        cause=exc,
                    )
                _raise_extraction_failure(
                    scenario_id=scenario_id,
                    slot=slot,
                    message=str(exc),
                    cause=exc,
                )

        extracted = _to_covenant_extract(flat, metric, expect_springing=expect_springing)

        if llm_client.REPLAY_DIR is not None:
            break

        category_issues = _validate_covenant_extract(
            extracted,
            scenario_id=scenario_id,
            slot=slot,
            allowed_categories=set(ledger_categories),
        )
        invalid_categories = [
            issue
            for issue in category_issues
            if issue.get("kind") == "INVALID_LEDGER_CATEGORY"
        ]
        if invalid_categories:
            if outer_attempt >= MAX_EXTRACTION_ATTEMPTS:
                break
            metric_retry = (
                "metric category include_keywords must use ledger category slugs from "
                f"{sorted(ledger_categories)}; invalid: "
                f"{invalid_categories[0].get('invalid_categories')}"
            )
            metric = None
            continue

        anchor_issue = _threshold_anchor_issue(extracted, candidates)
        if anchor_issue is None:
            break
        if outer_attempt >= MAX_EXTRACTION_ATTEMPTS:
            _raise_extraction_failure(
                scenario_id=scenario_id,
                slot=slot,
                message=anchor_issue,
            )
        flat_retry = anchor_issue
        flat = None
        active_candidates = _filter_threshold_candidates_by_unit(
            candidates,
            extracted.threshold_unit.value,
        ) or candidates
        use_cache = False

    if extracted is None:
        _raise_extraction_failure(
            scenario_id=scenario_id,
            slot=slot,
            message="covenant extraction returned no result",
        )

    payload = extracted.model_dump(mode="python")
    verify_extracted_fields(payload, fields=_quote_checks(extracted, verification_text))
    verification = _collect_verification_flags(payload)
    needed_retry = outer_attempt > 1 or flat_attempts > 1 or metric_attempts > 1
    return extracted, verification, unstable_fields, needed_retry


async def _process_scenario(
    client: LLMClient,
    *,
    scenario_id: str,
    doc_id: str,
    pages: list[str],
    ledger_categories: list[str],
    conflicts: list[dict[str, Any]],
) -> tuple[list[tuple[Covenant, dict[str, bool], list[dict[str, Any]], str | None]], int]:
    section = _extract_article_6(pages)
    items = _split_punkts(section)
    results: list[tuple[Covenant, dict[str, bool], list[dict[str, Any]], str | None]] = []
    retry_clause_count = 0

    for slot in SLOTS:
        item_text = items[slot]
        fallback_page, _, _ = _page_span_for_text(pages, item_text[:120])
        with capture_absent_values() as absent_fields:
            extracted, verification, unstable_fields, needed_retry = await _extract_covenant_item(
                client,
                scenario_id=scenario_id,
                slot=slot,
                item_text=item_text,
                verification_text=item_text,
                ledger_categories=ledger_categories,
            )
        if needed_retry:
            retry_clause_count += 1
        conflicts.extend(unstable_fields)
        for field_name in absent_fields:
            conflicts.append(
                {
                    "kind": "ABSENT_VALUE",
                    "field": field_name,
                    "scenario_id": scenario_id,
                    "doc_id": doc_id,
                    "slot": f"6.{slot}",
                },
            )

        validation_issues = _validate_covenant_extract(
            extracted,
            scenario_id=scenario_id,
            slot=slot,
            allowed_categories=set(ledger_categories),
        )
        conflicts.extend(validation_issues)

        period = (extracted.period_start, extracted.period_end)
        if not _validate_period(
            period,
            scenario_id=scenario_id,
            doc_id=doc_id,
            slot=slot,
            conflicts=conflicts,
        ):
            continue

        covenant, denominator_shape = _covenant_from_extract(
            extracted,
            scenario_id=scenario_id,
            slot=slot,
            doc_id=doc_id,
            pages=pages,
            fallback_page=fallback_page,
        )
        results.append((covenant, verification, validation_issues, denominator_shape))

    return results, retry_clause_count


async def _run_async(work_dir: Path) -> StageResult:
    inventory = json.loads((work_dir / "01_inventory.json").read_text(encoding="utf-8"))
    bound = json.loads((work_dir / "03_bound.json").read_text(encoding="utf-8"))
    template = load_template(work_dir)
    categories_by_scenario, global_categories = _load_ledger_categories(work_dir)

    async with LLMClient() as client:
        conflicts: list[dict[str, Any]] = []
        serialized: list[dict[str, Any]] = []

        scenarios = bound["scenarios"]
        extracted_by_scenario: dict[str, dict[str, tuple[Covenant, dict[str, bool], str | None]]] = {}

        retry_clause_count = 0

        for scenario_id in sorted({scenario for scenario, _slot in template_cells(template)}):
            loan_doc_id = scenarios.get(scenario_id, {}).get("loan")
            if not loan_doc_id:
                if scenario_id not in scenarios:
                    conflicts.append({"kind": "EXTRA_SCENARIO", "scenario_id": scenario_id})
                continue
            pages = inventory["documents"][loan_doc_id]["pages"]
            scenario_categories = categories_by_scenario.get(scenario_id) or global_categories
            scenario_results, scenario_retries = await _process_scenario(
                client,
                scenario_id=scenario_id,
                doc_id=loan_doc_id,
                pages=pages,
                ledger_categories=scenario_categories,
                conflicts=conflicts,
            )
            retry_clause_count += scenario_retries
            extracted_by_scenario[scenario_id] = {
                covenant.slot: (covenant, verification, denominator_shape)
                for covenant, verification, _issues, denominator_shape in scenario_results
            }

        for scenario_id, slot in template_cells(template):
            loan_doc_id = scenarios.get(scenario_id, {}).get("loan")
            extracted_by_slot = extracted_by_scenario.get(scenario_id, {})
            if slot in extracted_by_slot:
                covenant, verification, denominator_shape = extracted_by_slot[slot]
                serialized.append(
                    _serialize_covenant(
                        covenant,
                        verification,
                        denominator_shape=denominator_shape,
                    )
                )
            elif loan_doc_id:
                if slot in {f"6.{s}" for s in SLOTS}:
                    _raise_extraction_failure(
                        scenario_id=scenario_id,
                        slot=slot.removeprefix("6."),
                        message="covenant missing after extraction",
                    )
                conflicts.append(
                    {
                        "kind": "NEW_SLOT",
                        "scenario_id": scenario_id,
                        "slot": slot,
                    },
                )
                serialized.append(_placeholder_covenant(scenario_id, slot))
            else:
                serialized.append(_placeholder_covenant(scenario_id, slot))

    extracted_covenants = [covenant for covenant in serialized if not covenant.get("degraded")]
    invalid_covenant_keys = {
        issue["covenant_key"]
        for issue in conflicts
        if issue.get("covenant_key")
        and issue.get("kind")
        in {
            "ZERO_THRESHOLD",
            "RATIO_THRESHOLD_TOO_LARGE",
            "USD_THRESHOLD_TOO_SMALL",
            "SPRINGING_VALUE_EQUALS_THRESHOLD",
            "EMPTY_CATEGORY_KEYWORDS",
            "INVALID_LEDGER_CATEGORY",
            "THRESHOLD_UNIT_KIND_MISMATCH",
        }
    }
    if extracted_covenants:
        invalid_fraction = Decimal(len(invalid_covenant_keys)) / Decimal(len(extracted_covenants))
        if invalid_fraction > MAX_INVALID_COVENANT_FRACTION:
            kinds = sorted({issue["kind"] for issue in conflicts if issue.get("covenant_key")})
            raise AssertionError(
                f"s4a_covenants: {len(invalid_covenant_keys)}/{len(extracted_covenants)} "
                f"covenants failed validation (>{MAX_INVALID_COVENANT_FRACTION:.0%}): {kinds}",
            )

    springing_count = sum(1 for covenant in serialized if covenant.get("springing") is not None)
    slot_62_directions = {
        covenant["direction"]
        for covenant in serialized
        if covenant["slot"] == "6.2"
    }

    stable_conflicts = sorted(
        conflicts,
        key=lambda conflict: (
            conflict.get("kind", ""),
            conflict.get("scenario_id", ""),
            conflict.get("slot", ""),
            conflict.get("leg", ""),
            conflict.get("covenant_key", ""),
        ),
    )

    unstable_field_count = sum(
        1 for conflict in stable_conflicts if conflict.get("kind") == EXTRACTION_UNSTABLE
    )

    payload = {
        "covenants": serialized,
        "conflicts": stable_conflicts,
        "summary": {
            "count": len(serialized),
            "springing_count": springing_count,
            "slot_6_2_directions": sorted(slot_62_directions),
            "unstable_field_count": unstable_field_count,
            "retry_clause_count": retry_clause_count,
        },
    }

    output_path = work_dir / "04a_covenants.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"s4a_covenants: extracted={len(serialized)} springing={springing_count} "
        f"6.2_directions={sorted(slot_62_directions)} conflicts={len(conflicts)} "
        f"unstable_fields={unstable_field_count} retry_clauses={retry_clause_count}",
    )

    return StageResult(
        item_count=len(serialized),
        row_count=springing_count,
        unstable_field_count=unstable_field_count,
        retry_clause_count=retry_clause_count,
    )


def run(*, work_dir: Path) -> StageResult:
    return asyncio.run(_run_async(work_dir))
