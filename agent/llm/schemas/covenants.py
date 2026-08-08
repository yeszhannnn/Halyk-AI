from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from agent.parsing.categories import derive_leg_sign
from agent.parsing.numbers import normalize_decimal, normalize_optional_decimal

LEDGER_CATEGORIES_CONTEXT_KEY = "ledger_categories"

RELATED_PARTY_MARKERS = (
    "связан",
    "аффилир",
    "related party",
    "related-party",
    "affiliate",
    "ограниченные платежи",
)


def _text_refers_to_related_parties(text: str) -> bool:
    normalized = str(text or "").casefold()
    return any(marker in normalized for marker in RELATED_PARTY_MARKERS)


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
        if upper in {"VALUE", "BOTH", "ABSOLUTE", "NONE", "N/A", "NA", "ВСЕ", "ALL"}:
            return "BOTH"
        return upper
    return value


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
    """Category leg using closed ledger vocabulary (include_keywords = category slugs)."""

    include_keywords: list[str] = Field(
        description=(
            "One or more ledger category slugs included in this leg. "
            "For related-party payment legs, this may be empty; counterparty "
            "resolution from stage 4b applies instead of category filtering."
        ),
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
    apply_reclass: bool = Field(
        description="Whether auditor reclassifications apply to this category leg.",
    )
    apply_reclass_quote: str = Field(
        description="Verbatim quote about auditor reclassifications for this leg.",
    )

    @model_validator(mode="after")
    def _validate_category_vocabulary(self) -> Self:
        if _text_refers_to_related_parties(self.include_keywords_quote):
            return self
        if not self.include_keywords:
            raise ValueError("include_keywords must contain at least one ledger category")
        return self

    @field_validator("include_keywords")
    @classmethod
    def _validate_include_keywords(cls, value: list[str], info: ValidationInfo) -> list[str]:
        quote = ""
        if isinstance(info.data, dict):
            quote = str(info.data.get("include_keywords_quote") or "")
        if _text_refers_to_related_parties(quote):
            return value
        context = info.context or {}
        allowed = context.get(LEDGER_CATEGORIES_CONTEXT_KEY)
        if allowed is None:
            return value
        if not value:
            return value
        allowed_set = {str(category) for category in allowed}
        invalid = sorted({str(keyword) for keyword in value if str(keyword) not in allowed_set})
        if invalid:
            raise ValueError(
                "include_keywords must be chosen from the ledger category list "
                f"{sorted(allowed_set)}; invalid: {invalid}",
            )
        return value


class MetricSpecExtract(BaseModel):
    kind: MetricKind = Field(
        description="RATIO for quotient metrics; SUM for summed USD caps; COUNT for counted metrics.",
    )
    kind_quote: str = Field(
        description="Verbatim quote showing ratio, sum, or count computation.",
    )
    numerator: CategorySpecExtract | None = Field(
        default=None,
        description="Numerator category when kind is RATIO.",
    )
    denominator: CategorySpecExtract | None = Field(
        default=None,
        description="Denominator category when kind is RATIO.",
    )
    category: CategorySpecExtract | None = Field(
        default=None,
        description="Aggregate category when kind is SUM or COUNT.",
    )
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
        if value is None or (isinstance(value, str) and not value.strip()):
            return value
        return _uppercase_str(value)

    @model_validator(mode="before")
    @classmethod
    def _infer_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        kind = data.get("kind")
        if kind is None or (isinstance(kind, str) and not kind.strip()):
            if data.get("denominator") is not None:
                data["kind"] = "RATIO"
            else:
                data["kind"] = "SUM"
        else:
            data["kind"] = _uppercase_str(kind)
        kind = data["kind"]
        if kind in {"SUM", "COUNT"} and data.get("category") is None and data.get("numerator") is not None:
            data["category"] = data["numerator"]
        if not data.get("scope"):
            data["scope"] = "BORROWER"
        if not str(data.get("scope_quote", "")).strip():
            data["scope_quote"] = str(data.get("kind_quote", ""))
        return data

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        if self.kind == MetricKind.RATIO:
            if self.numerator is None or self.denominator is None:
                raise ValueError("RATIO metric requires numerator and denominator")
        elif self.category is None:
            raise ValueError(f"{self.kind.value} metric requires category")
        return self


class SpringingConditionExtract(BaseModel):
    metric: MetricSpecExtract
    operator: Literal[">", "<"]
    operator_quote: str = Field(
        description="Verbatim quote containing the comparison operator for the trigger.",
    )
    value: Decimal | None = Field(
        default=None,
        description="Trigger threshold value; null when the clause states no value.",
    )
    value_quote: str = Field(
        description="Verbatim quote containing the trigger threshold value.",
    )
    condition_quote: str = Field(
        description="Verbatim quote for the full springing applicability condition.",
    )

    @field_validator("value", mode="before")
    @classmethod
    def _parse_value(cls, value: Any) -> Any:
        return normalize_optional_decimal(value, field_name="springing.value")


def _infer_metric_kind(metric: dict[str, Any], covenant_data: dict[str, Any]) -> str:
    kind = metric.get("kind")
    if kind is not None and str(kind).strip():
        return str(_uppercase_str(kind))
    unit = covenant_data.get("threshold_unit")
    if unit is not None and str(unit).strip():
        normalized_unit = str(_uppercase_str(unit))
        if normalized_unit == "RATIO":
            return "RATIO"
        if normalized_unit == "USD":
            return "SUM"
    if metric.get("denominator") is not None:
        return "RATIO"
    return "SUM"


def _normalize_metric_dict(metric: dict[str, Any], covenant_data: dict[str, Any]) -> dict[str, Any]:
    metric = dict(metric)
    metric["kind"] = _infer_metric_kind(metric, covenant_data)
    kind = metric["kind"]
    if kind in {"SUM", "COUNT"} and metric.get("category") is None and metric.get("numerator") is not None:
        metric["category"] = metric["numerator"]
    if not metric.get("scope"):
        metric["scope"] = "BORROWER"
    if not str(metric.get("scope_quote", "")).strip():
        metric["scope_quote"] = str(metric.get("kind_quote", ""))
    return metric


class CovenantExtractBase(BaseModel):
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

    @field_validator("direction", "threshold_unit", mode="before")
    @classmethod
    def _normalize_enums(cls, value: Any) -> Any:
        return _uppercase_str(value)

    @field_validator("threshold", mode="before")
    @classmethod
    def _parse_threshold(cls, value: Any) -> Any:
        return normalize_decimal(value, field_name="threshold")

    @field_validator("period_start", "period_end", mode="before")
    @classmethod
    def _parse_period_dates(cls, value: Any) -> Any:
        return _parse_date_field(value, field_name="period")

    @model_validator(mode="after")
    def _drop_valueless_springing(self) -> Self:
        # A trigger without a value is untestable; treat the condition as absent.
        springing = getattr(self, "springing", None)
        if springing is not None and springing.value is None:
            self.springing = None
        return self

    @model_validator(mode="before")
    @classmethod
    def _promote_metric_notes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        metric = data.get("metric")
        if isinstance(metric, dict):
            metric = _normalize_metric_dict(metric, data)
            data["metric"] = metric
            if data.get("notes") is None and metric.get("notes"):
                data["notes"] = metric["notes"]
            if data.get("notes_quote") is None and metric.get("notes_quote"):
                data["notes_quote"] = metric["notes_quote"]
        springing = data.get("springing")
        if isinstance(springing, dict):
            springing_metric = springing.get("metric")
            if isinstance(springing_metric, dict):
                springing = dict(springing)
                springing["metric"] = _normalize_metric_dict(springing_metric, data)
                data["springing"] = springing
        return data


class CovenantExtractNoSpringing(CovenantExtractBase):
    """Covenant extraction when the clause has no springing trigger phrase."""


class CovenantExtractWithSpringing(CovenantExtractBase):
    springing: SpringingConditionExtract


class CovenantFlatExtractBase(BaseModel):
    """Header fields for a covenant clause (no metric shape)."""

    title: str
    title_quote: str = Field(
        description="Verbatim quote containing the covenant title or heading.",
    )
    direction: Direction = Field(
        description=(
            "MAX if compliance requires staying at or below the threshold; "
            "MIN if compliance requires meeting or exceeding the threshold."
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
    period_start: date
    period_end: date
    period_quote: str = Field(
        description="Verbatim quote containing the covenant period dates.",
    )

    @field_validator("direction", "threshold_unit", mode="before")
    @classmethod
    def _normalize_enums(cls, value: Any) -> Any:
        return _uppercase_str(value)

    @field_validator("threshold", mode="before")
    @classmethod
    def _parse_threshold(cls, value: Any) -> Any:
        return normalize_decimal(value, field_name="threshold")

    @field_validator("period_start", "period_end", mode="before")
    @classmethod
    def _parse_period_dates(cls, value: Any) -> Any:
        return _parse_date_field(value, field_name="period")


class CovenantFlatNoSpringing(CovenantFlatExtractBase):
    """Flat covenant header when the clause has no springing trigger phrase."""


class CovenantFlatWithSpringing(CovenantFlatExtractBase):
    springing_operator: Literal[">", "<"]
    springing_operator_quote: str = Field(
        description="Verbatim quote containing the springing comparison operator.",
    )
    springing_value: Decimal | None = Field(
        default=None,
        description="Trigger threshold value; null when the clause states no value.",
    )
    springing_value_quote: str = Field(
        default="",
        description="Verbatim quote containing the springing trigger threshold value.",
    )
    springing_condition_quote: str = Field(
        description="Verbatim quote for the full springing applicability condition.",
    )

    @field_validator("springing_value", mode="before")
    @classmethod
    def _parse_springing_value(cls, value: Any) -> Any:
        return normalize_optional_decimal(value, field_name="springing_value")


class CovenantMetricExtract(BaseModel):
    """Metric definition for a covenant clause."""

    metric: MetricSpecExtract
    notes: str = Field(
        description="Full verbatim formulation of how the metric is computed.",
    )
    notes_quote: str = Field(
        description="Verbatim quote backing the metric definition (may match notes).",
    )


class CovenantMetricWithSpringing(CovenantMetricExtract):
    springing_metric: MetricSpecExtract = Field(
        description="Nested metric for the springing trigger condition.",
    )


class CovenantExtract(CovenantExtractBase):
    springing: SpringingConditionExtract | None = Field(
        default=None,
        description=(
            "Present only when the covenant test applies conditionally "
            "(e.g. only if some quantity exceeds a value). Otherwise null."
        ),
    )
