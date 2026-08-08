from __future__ import annotations

from datetime import date
from decimal import Decimal

from agent.llm.schemas.covenants import (
    CategorySpecExtract,
    CovenantFlatNoSpringing,
    CovenantMetricExtract,
    Direction,
    MetricKind,
    MetricScope,
    MetricSpecExtract,
    ThresholdUnit,
)
from agent.stages.s4a_covenants import (
    _append_retry_line,
    _retry_line_from_error,
    _to_covenant_extract,
)


def _sample_metric() -> MetricSpecExtract:
    category = CategorySpecExtract(
        include_keywords=["revenue"],
        include_keywords_quote="выручка",
        apply_reclass=True,
        apply_reclass_quote="с учетом реклассификаций",
    )
    return MetricSpecExtract(
        kind=MetricKind.RATIO,
        kind_quote="отношение",
        numerator=category,
        denominator=category,
        scope=MetricScope.BORROWER,
        scope_quote="заемщика",
    )


def test_to_covenant_extract_merges_flat_and_metric() -> None:
    flat = CovenantFlatNoSpringing(
        title="Leverage",
        title_quote="Leverage",
        direction=Direction.MAX,
        direction_quote="не более",
        threshold=Decimal("1.2"),
        threshold_quote="1.20x",
        threshold_unit=ThresholdUnit.RATIO,
        threshold_unit_quote="1.20x",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        period_quote="2025",
    )
    metric = CovenantMetricExtract(
        metric=_sample_metric(),
        notes="ratio of debt to EBITDA",
        notes_quote="отношение долга к EBITDA",
    )

    merged = _to_covenant_extract(flat, metric, expect_springing=False)

    assert merged.threshold == Decimal("1.2")
    assert merged.metric.kind == MetricKind.RATIO
    assert merged.notes == "ratio of debt to EBITDA"


def test_retry_line_from_metric_null_error() -> None:
    message = "1 validation error for CovenantExtract\nmetric\n  Input should be an object"
    assert _retry_line_from_error(message) == "metric was missing."


def test_append_retry_line_keeps_original_prompt() -> None:
    base = "Extract the covenant header from this clause."
    retried = _append_retry_line(base, "metric was missing.")
    assert retried.startswith(base)
    assert retried.endswith("Retry: metric was missing.")
    assert retried.count("Extract the covenant header") == 1
