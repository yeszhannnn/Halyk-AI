from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Direction(str, Enum):
    MAX = "MAX"
    MIN = "MIN"


class ThresholdUnit(str, Enum):
    USD = "USD"
    RATIO = "RATIO"


class MetricKind(str, Enum):
    RATIO = "RATIO"
    SUM = "SUM"
    COUNT = "COUNT"


class MetricScope(str, Enum):
    BORROWER = "BORROWER"
    GROUP = "GROUP"


class CategorySign(str, Enum):
    OUTFLOW = "OUTFLOW"
    INFLOW = "INFLOW"
    BOTH = "BOTH"


def _uppercase_str(value: Any) -> Any:
    if isinstance(value, str):
        upper = value.upper()
        if upper in {"OUTFLOWS", "NEGATIVE", "OUTFLOW"}:
            return "OUTFLOW"
        if upper in {"INFLOWS", "POSITIVE", "INFLOW"}:
            return "INFLOW"
        if upper in {"VALUE", "BOTH", "ABSOLUTE"}:
            return "BOTH"
        return upper
    return value


def _parse_decimal_field(value: Any, *, field_name: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace(" ", "")
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty")
        return Decimal(cleaned)
    raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}")


def _parse_date_field(value: Any, *, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty")
        return date.fromisoformat(cleaned)
    raise ValueError(f"{field_name} must be an ISO date string, got {type(value).__name__}")


class CategorySpecExtract(BaseModel):
    include_keywords: list[str] = Field(
        description="Keywords or category labels included in the numerator or sum.",
    )
    include_keywords_quote: str = Field(
        description="Verbatim quote from the clause naming included categories or keywords.",
    )
    exclude_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords or categories explicitly excluded.",
    )
    exclude_keywords_quote: str = Field(
        default="",
        description="Verbatim quote for exclusions, or empty string if none stated.",
    )
    sign: CategorySign = Field(description="Cash-flow sign of included amounts.")
    sign_quote: str = Field(
        description="Verbatim quote indicating inflows, outflows, or both.",
    )
    apply_reclass: bool = Field(
        description="Whether auditor reclassifications apply to this category leg.",
    )
    apply_reclass_quote: str = Field(
        description="Verbatim quote about auditor reclassifications for this leg.",
    )

    @field_validator("sign", mode="before")
    @classmethod
    def _normalize_sign(cls, value: Any) -> Any:
        return _uppercase_str(value)


class MetricSpecExtract(BaseModel):
    kind: MetricKind
    kind_quote: str = Field(
        description="Verbatim quote showing ratio, sum, or count computation.",
    )
    numerator: CategorySpecExtract
    denominator: CategorySpecExtract | None = None
    scope: MetricScope = Field(
        description="BORROWER if only borrower financials; GROUP if consolidated group scope.",
    )
    scope_quote: str = Field(
        description="Verbatim quote stating borrower-only or group/consolidated scope.",
    )
    notes: str = Field(
        default="",
        description="Optional notes for nested trigger metrics (springing conditions).",
    )
    notes_quote: str = Field(
        default="",
        description="Optional verbatim quote for nested trigger metric notes.",
    )

    @field_validator("kind", "scope", mode="before")
    @classmethod
    def _normalize_enums(cls, value: Any) -> Any:
        return _uppercase_str(value)


class SpringingConditionExtract(BaseModel):
    metric: MetricSpecExtract
    operator: Literal[">", "<"]
    operator_quote: str = Field(
        description="Verbatim quote containing the comparison operator for the trigger.",
    )
    value: Decimal
    value_quote: str = Field(
        description="Verbatim quote containing the trigger threshold value.",
    )
    condition_quote: str = Field(
        description="Verbatim quote for the full springing applicability condition.",
    )

    @field_validator("value", mode="before")
    @classmethod
    def _parse_value(cls, value: Any) -> Any:
        if value is None:
            return value
        return _parse_decimal_field(value, field_name="springing.value")


class CovenantExtract(BaseModel):
    title: str
    title_quote: str = Field(
        description="Verbatim quote containing the covenant title or heading.",
    )
    direction: Direction = Field(
        description=(
            "MAX if compliance requires staying at or below the threshold; "
            "MIN if compliance requires meeting or exceeding the threshold. "
            "Determine from comparison language (не более/не менее/exceed/not fall below), "
            "never from the title alone."
        ),
    )
    direction_quote: str = Field(
        description="Verbatim quote containing the comparison direction language.",
    )
    threshold: Decimal
    threshold_quote: str = Field(
        description="Verbatim quote containing the numeric threshold.",
    )
    threshold_unit: ThresholdUnit
    threshold_unit_quote: str = Field(
        description="Verbatim quote showing USD amount or ratio/multiplier unit.",
    )
    metric: MetricSpecExtract
    notes: str = Field(
        description="Full verbatim formulation of how the metric is computed.",
    )
    notes_quote: str = Field(
        description="Verbatim quote backing the metric definition (may match notes).",
    )
    period_start: date
    period_end: date
    period_quote: str = Field(
        description="Verbatim quote containing the covenant period dates.",
    )
    springing: SpringingConditionExtract | None = Field(
        default=None,
        description=(
            "Present only when the covenant test applies conditionally "
            "(e.g. only if some quantity exceeds a value). Otherwise null."
        ),
    )

    @field_validator("direction", "threshold_unit", mode="before")
    @classmethod
    def _normalize_enums(cls, value: Any) -> Any:
        return _uppercase_str(value)

    @field_validator("threshold", mode="before")
    @classmethod
    def _parse_threshold(cls, value: Any) -> Any:
        return _parse_decimal_field(value, field_name="threshold")

    @field_validator("period_start", "period_end", mode="before")
    @classmethod
    def _parse_period_dates(cls, value: Any) -> Any:
        return _parse_date_field(value, field_name="period")

    @model_validator(mode="before")
    @classmethod
    def _promote_metric_notes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        metric = data.get("metric")
        if isinstance(metric, dict):
            if data.get("notes") is None and metric.get("notes"):
                data["notes"] = metric["notes"]
            if data.get("notes_quote") is None and metric.get("notes_quote"):
                data["notes_quote"] = metric["notes_quote"]
        return data
