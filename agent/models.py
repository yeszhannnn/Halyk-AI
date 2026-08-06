from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class Provenance:
    doc_id: str
    page: int
    quote: str  # дословно, верифицируется как подстрока
    extractor: str


@dataclass(frozen=True)
class CategorySpec:
    """Что входит в статью. Определяется ковенантом, не глобальным словарём."""

    include_keywords: list[str]
    exclude_keywords: list[str]
    sign: str  # OUTFLOW | INFLOW | BOTH
    apply_reclass: bool = True  # учитывать переклассификации аудитора


@dataclass(frozen=True)
class MetricSpec:
    """Как считать показатель. Определение живёт в тексте ковенанта."""

    kind: str  # RATIO | SUM | COUNT
    numerator: CategorySpec
    denominator: CategorySpec | None
    scope: str  # BORROWER | GROUP
    notes: str  # дословная формулировка из договора


@dataclass(frozen=True)
class SpringingCondition:
    """Ковенант применяется только при выполнении условия."""

    metric: MetricSpec
    operator: str  # ">" | "<"
    value: Decimal
    source: Provenance


@dataclass(frozen=True)
class Covenant:
    scenario_id: str  # P1 … P10, B1, B4
    slot: str  # "6.1" | "6.2" | "6.3"
    title: str
    direction: str  # MAX (≤ порога) | MIN (≥ порога)
    threshold: Decimal
    threshold_unit: str  # USD | RATIO
    metric: MetricSpec
    period: tuple[date, date]  # ковенантный период из договора
    springing: SpringingCondition | None
    source: Provenance


@dataclass(frozen=True)
class RelatedParty:
    counterparty: str
    ownership_pct: Decimal
    is_related: bool  # ownership_pct >= порога из досье
    source: Provenance


@dataclass(frozen=True)
class Adjustment:
    """Корректировка из примечаний аудитора или источника корректировок."""

    kind: str  # RECLASS | CUTOFF | EXCLUDE | OFF_LEDGER | AMOUNT_FILL | FX | EBITDA_ADDBACK | NONE
    scenario_id: str
    txn_id: str | None  # если привязана к операции
    amount: Decimal | None  # если задана суммой + контрагентом
    counterparty: str | None
    from_category: str | None
    to_category: str | None
    source: Provenance


@dataclass(frozen=True)
class Finding:
    scenario_id: str
    slot: str
    status: str
    evaluated: Decimal
    rounded: Decimal
    evidence_txn_id: str | None
    strategy: str
    confidence: Decimal
    flags: list[str]
