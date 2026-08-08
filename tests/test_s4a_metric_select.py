from __future__ import annotations

from datetime import date
from decimal import Decimal

from agent.llm.schemas.covenants import (
    CategorySpecExtract,
    CovenantMetricExtract,
    DERIVED_LEG_SHAPES,
    MetricKind,
    MetricScope,
    MetricSpecExtract,
    LEDGER_CATEGORIES_CONTEXT_KEY,
)

LEDGER_CATEGORIES = [
    "consulting",
    "expense",
    "insurance",
    "interest",
    "marketing",
    "opex",
    "personnel",
    "rent",
    "revenue",
    "tax",
    "utilities",
]


def _category(keywords: list[str]) -> CategorySpecExtract:
    return CategorySpecExtract(
        include_keywords=keywords,
        include_keywords_quote="quote",
        apply_reclass=True,
        apply_reclass_quote="quote",
    )


def _ratio_metric(
    numerator_keywords: list[str],
    denominator_keywords: list[str],
    *,
    notes: str = "",
) -> MetricSpecExtract:
    return MetricSpecExtract(
        kind=MetricKind.RATIO,
        kind_quote="отношение",
        numerator=_category(numerator_keywords),
        denominator=_category(denominator_keywords),
        scope=MetricScope.BORROWER,
        scope_quote="заемщика",
        notes=notes,
    )


def test_schema_allows_ebitda_derived_leg() -> None:
    metric = _ratio_metric(["revenue"], ["EBITDA"], notes="отношение долга к EBITDA")
    context = {LEDGER_CATEGORIES_CONTEXT_KEY: LEDGER_CATEGORIES}
    validated = MetricSpecExtract.model_validate(metric.model_dump(), context=context)
    assert validated.denominator is not None
    assert validated.denominator.include_keywords == ["EBITDA"]


def test_schema_allows_empty_notes_quote() -> None:
    metric = _ratio_metric(["revenue"], ["opex"])
    payload = CovenantMetricExtract(metric=metric).model_dump()
    assert payload["notes_quote"] == ""
    restored = CovenantMetricExtract.model_validate(payload)
    assert restored.notes_quote == ""


def test_schema_remaps_max_kind_to_ratio() -> None:
    metric = MetricSpecExtract.model_validate(
        {
            "kind": "MAX",
            "kind_quote": "не более",
            "numerator": {
                "include_keywords": ["revenue"],
                "include_keywords_quote": "выручка",
                "apply_reclass": True,
                "apply_reclass_quote": "quote",
            },
            "denominator": {
                "include_keywords": ["opex"],
                "include_keywords_quote": "opex",
                "apply_reclass": True,
                "apply_reclass_quote": "quote",
            },
            "scope": "BORROWER",
            "scope_quote": "заемщика",
        },
        context={LEDGER_CATEGORIES_CONTEXT_KEY: LEDGER_CATEGORIES},
    )
    assert metric.kind == MetricKind.RATIO


def test_derived_shapes_constant() -> None:
    assert DERIVED_LEG_SHAPES == frozenset({"EBITDA", "ADJUSTED_EBITDA"})
