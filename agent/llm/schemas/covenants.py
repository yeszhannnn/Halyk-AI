from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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
        description="Full verbatim formulation of how the metric is computed.",
    )
    notes_quote: str = Field(
        description="Verbatim quote backing the metric definition (may match notes).",
    )


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
