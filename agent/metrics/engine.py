"""Covenant metric computation against an adjusted ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent.metrics.group_figures import resolve_group_figure
from agent.parsing.categories import (
    INFLOW_CATEGORIES,
    OPEX_SLUGS,
    category_sign,
    derive_leg_sign,
    infer_category,
)
from agent.stages.s4b_parties import normalize_counterparty

ZERO = Decimal("0")
EMPTY_CATEGORY_SPEC = "EMPTY_CATEGORY_SPEC"
EMPTY_LEG = "EMPTY_LEG"
ZERO_DENOMINATOR = "ZERO_DENOMINATOR"
FLOOR_MISSING_REVIEW = "FLOOR_MISSING_REVIEW"
GROUP_FIGURE_NOT_FOUND = "GROUP_FIGURE_NOT_FOUND"
WIDE_LEG_REVIEW = "WIDE_LEG_REVIEW"
SCENARIO_SCOPE_VIOLATION = "SCENARIO_SCOPE_VIOLATION"
LEG_SUBTOTAL_MISMATCH = "LEG_SUBTOTAL_MISMATCH"
ADJUSTMENT_APPLIED_TWICE = "ADJUSTMENT_APPLIED_TWICE"
IDENTICAL_LEGS = "IDENTICAL_LEGS"
EBITDA_CONSTRUCTION_FAILED = "EBITDA_CONSTRUCTION_FAILED"

FUNDING_EXCLUSION_MARKERS = (
    "refund",
    "credit received",
    "rebate",
    "reversal",
    "recovery",
    "reimbursement",
    "deposit returned",
    "deposit refunded",
    "overbilling refund",
    "overpayment refunded",
    "tax assessment reversal",
    "tax overpayment refunded",
    "tax credit received",
    "experience refund",
    "broker rebate",
    "free period credit",
    "utility rebate",
    "utility deposit returned",
    "insurance claim reimbursement",
    "insurance deductible recovery",
    "payroll advance recovered",
    "interest on escrow",
    "incentive",
)


def _d(value: Any) -> Decimal:
    if value is None:
        return ZERO
    if isinstance(value, Decimal):
        return value
    text = str(value).strip()
    if not text:
        return ZERO
    return Decimal(text)


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def compare_values(actual: Decimal, operator: str, threshold: Decimal) -> bool:
    if operator == ">":
        return actual > threshold
    if operator == ">=":
        return actual >= threshold
    if operator == "<":
        return actual < threshold
    if operator == "<=":
        return actual <= threshold
    raise ValueError(f"unsupported operator: {operator}")


def breaches(actual: Decimal, direction: str, threshold: Decimal) -> bool:
    if direction == "MAX":
        return actual > threshold
    if direction == "MIN":
        return actual < threshold
    raise ValueError(f"unsupported direction: {direction}")


def _ledger_counterparties(parties: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for mapping in parties.get("ledger_map", {}).values():
        names.update(mapping)
    return names


def _related_counterparties(parties: dict[str, Any]) -> set[str]:
    return _ledger_counterparties(parties)


def _unrestricted_counterparties(parties: dict[str, Any]) -> set[str]:
    perimeter = parties.get("perimeter")
    if not perimeter:
        return set()
    names: set[str] = set()
    for row in perimeter.get("ownership", []):
        if row.get("is_related"):
            continue
        for ledger_names in parties.get("ledger_map", {}).values():
            if row["counterparty"] in ledger_names or row["counterparty"] in parties.get(
                "ledger_map", {}
            ):
                names.update(ledger_names)
        # fallback: direct counterparty string
        names.add(row["counterparty"])
    # P9 stores perimeter without ledger_map entries — match by normalized name later.
    for row in perimeter.get("ownership", []):
        if not row.get("is_related"):
            names.add(row["counterparty"])
    return names


def _counterparty_matches(row: dict[str, Any], names: set[str]) -> bool:
    cp = str(row.get("counterparty") or "")
    if cp in names:
        return True
    key = normalize_counterparty(cp)
    for name in names:
        if normalize_counterparty(name) == key:
            return True
    return False


def _is_capex_row(row: dict[str, Any], *, group_scope: bool = False) -> bool:
    category = _effective_category(row, True)
    desc = str(row.get("description") or "").casefold()
    if category == "capex":
        return True
    if group_scope and "capitalised interest" in desc:
        return True
    return False


def _rows_for_scope(
    ledger: list[dict[str, Any]],
    *,
    scenario_id: str,
    scope: str,
) -> list[dict[str, Any]]:
    if scope == "GROUP":
        return [row for row in ledger if row.get("scenario_id") == scenario_id]
    return [row for row in ledger if row.get("scenario_id") == scenario_id]


def _is_excluded_inflow(description: str) -> bool:
    text = str(description or "").casefold()
    return any(marker in text for marker in FUNDING_EXCLUSION_MARKERS)


def _assert_leg_scenario(
    rows: list[dict[str, Any]],
    *,
    scenario_id: str,
    leg: str,
) -> None:
    for row in rows:
        row_scenario = row.get("scenario_id")
        if row_scenario is None:
            continue
        if row_scenario != scenario_id:
            txn_id = row.get("txn_id") or row.get("adjustment_ref") or "?"
            raise ValueError(
                f"{SCENARIO_SCOPE_VIOLATION}: {leg} leg for {scenario_id} "
                f"includes row {txn_id} from {row_scenario}"
            )


@dataclass
class LegBreakdown:
    kind: str
    value: Decimal
    rows: list[dict[str, Any]] = field(default_factory=list)
    expression: str | None = None
    terms: list[tuple[str, Decimal]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def category_count(self) -> int:
        return len(self.categories)


def _ledger_category(row: dict[str, Any]) -> str:
    """Post-adjustment category from s5_ledger (destination leg sees RECLASS)."""
    return str(row.get("category") or "other")


def _effective_category(row: dict[str, Any], apply_reclass: bool) -> str:
    # apply_reclass only gates exclusion of rows reclassified out of this leg;
    # inclusion always uses the adjusted ledger category.
    return _ledger_category(row)


def _in_period(row: dict[str, Any], period: tuple[date, date]) -> bool:
    row_date = _parse_date(row["date"])
    return period[0] <= row_date <= period[1]


def _sign_ok(amount: Decimal, sign: str) -> bool:
    if sign == "INFLOW":
        return amount > ZERO
    if sign == "OUTFLOW":
        return amount < ZERO
    return True


def _amount_for_aggregation(amount: Decimal) -> Decimal:
    # Legs aggregate signed amounts as posted: inflows positive, outflows
    # negative. Only the final reported actual is made positive, never the
    # intermediate terms — abs() here would turn every difference into a total.
    return amount


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


def _requires_related_party_filter(covenant_context: dict[str, Any] | None) -> bool:
    if not covenant_context:
        return False
    text = " ".join(
        [
            str(covenant_context.get("title") or ""),
            str(covenant_context.get("notes") or ""),
            str((covenant_context.get("metric") or {}).get("notes") or ""),
        ],
    )
    return _text_refers_to_related_parties(text)


def _is_related_party_leg(
    covenant_context: dict[str, Any] | None,
    *,
    leg: str | None,
) -> bool:
    """Payment legs for related-party covenants filter by counterparty, not category."""
    if leg != "numerator" or not _requires_related_party_filter(covenant_context):
        return False
    return True


def _requires_unrestricted_subsidiary_capex(covenant_context: dict[str, Any] | None) -> bool:
    if not covenant_context:
        return False
    text = " ".join(
        [
            str(covenant_context.get("title") or ""),
            str(covenant_context.get("notes") or ""),
            str((covenant_context.get("metric") or {}).get("notes") or ""),
        ],
    ).casefold()
    return "неограниченн" in text and "дочерн" in text and "капитал" in text


def _category_matches(category: str, include_keywords: list[str]) -> bool:
    return category in {str(keyword) for keyword in include_keywords}


def _exclude_matches(
    keyword: str,
    *,
    row: dict[str, Any],
    category: str,
    apply_reclass: bool,
    spec: dict[str, Any],
) -> bool:
    key = keyword.casefold()
    if "финансовых или иных неоперационных статей" in key:
        return category == "financing"
    if "переклассифицированные аудиторами" in key:
        if not apply_reclass:
            return False
        if not row.get("adjustment_ref"):
            return False
        original = str(row.get("original_category") or row.get("category") or "other")
        current = _ledger_category(row)
        if original == current:
            return False
        for include_keyword in spec.get("include_keywords") or []:
            if _category_matches(original, [str(include_keyword)]):
                return True
        return False
    if "аффилированных и связанных сторон" in key:
        return False
    if "ограниченной" in key:
        return "restricted" in str(row.get("description") or "").casefold()
    if "наибольшей из величин расходов на оплату tруда и налогов" in key.replace("т", "t"):
        return category in {"personnel", "tax"}
    if "прямо согласованных кредитором" in key:
        return False
    return key in str(row.get("description") or "").casefold()


def _row_matches_spec(
    row: dict[str, Any],
    spec: dict[str, Any],
    *,
    period: tuple[date, date],
    parties: dict[str, Any] | None,
    group_scope: bool = False,
    covenant_context: dict[str, Any] | None = None,
    leg: str | None = None,
) -> bool:
    if row.get("excluded"):
        return False
    if not _in_period(row, period):
        return False
    amount = _d(row.get("amount_usd"))
    if amount == ZERO:
        return False

    apply_reclass = bool(spec.get("apply_reclass", True))
    category = _effective_category(row, apply_reclass)
    for keyword in spec.get("exclude_keywords") or []:
        if _exclude_matches(
            keyword,
            row=row,
            category=category,
            apply_reclass=apply_reclass,
            spec=spec,
        ):
            return False

    related_party_leg = _is_related_party_leg(covenant_context, leg=leg)
    if related_party_leg:
        if amount >= ZERO:
            return False
        if parties is None or not _counterparty_matches(row, _related_counterparties(parties)):
            return False
        return True

    include_keywords = [str(keyword) for keyword in spec.get("include_keywords") or []]
    if not include_keywords:
        return False

    if group_scope and "capex" in include_keywords and _is_capex_row(row, group_scope=True):
        matched = True
    elif _category_matches(category, include_keywords):
        matched = True
    else:
        matched = False

    if not matched:
        return False

    if category in INFLOW_CATEGORIES and _is_excluded_inflow(str(row.get("description") or "")):
        return False

    if not _sign_ok(amount, category_sign(category)):
        return False

    if (
        leg == "numerator"
        and _requires_unrestricted_subsidiary_capex(covenant_context)
        and category == "capex"
    ):
        if parties is None or not _counterparty_matches(
            row,
            _unrestricted_counterparties(parties),
        ):
            return False

    return True


def _filter_rows(
    ledger: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    period: tuple[date, date],
    parties: dict[str, Any] | None,
    group_scope: bool = False,
    covenant_context: dict[str, Any] | None = None,
    leg: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in ledger
        if _row_matches_spec(
            row,
            spec,
            period=period,
            parties=parties,
            group_scope=group_scope,
            covenant_context=covenant_context,
            leg=leg,
        )
    ]


def _sum_rows(rows: list[dict[str, Any]]) -> Decimal:
    total = ZERO
    for row in rows:
        total += _amount_for_aggregation(_d(row.get("amount_usd")))
    return total


def _ebitda_addback_adjustments(
    adjustments: dict[str, Any],
    scenario_id: str,
) -> list[tuple[str, Decimal, list[dict[str, Any]], bool]]:
    """EBITDA add-back tables as (adjustment id, above-floor total, all rows, floor_missing).

    Every row is a one-off expense: all of them are subtracted inside opex
    first, and only rows at or above the materiality floor are added back.
    When the materiality floor sentence was not extracted (materiality_floor
    is None) every listed row is added back and the cell is flagged for
    review — losing one field must never discard the whole record.
    """
    items: list[tuple[str, Decimal, list[dict[str, Any]], bool]] = []
    for adj_id, adj in adjustments.items():
        if adj.get("scenario_id") != scenario_id or adj.get("kind") != "EBITDA_ADDBACK":
            continue
        rows = [row for row in (adj.get("rows") or []) if _d(row.get("amount")) != ZERO]
        if not rows:
            continue
        floor_missing = adj.get("materiality_floor") is None
        if floor_missing:
            above_floor = sum((_d(row.get("amount")) for row in rows), ZERO)
        else:
            above_floor = sum(
                (_d(row.get("amount")) for row in rows if row.get("above_floor")),
                ZERO,
            )
        items.append((adj_id, above_floor, rows, floor_missing))
    return items


def _addback_total(adjustments: dict[str, Any], scenario_id: str) -> Decimal:
    return sum(
        (
            above_floor
            for _, above_floor, _rows, _floor_missing
            in _ebitda_addback_adjustments(adjustments, scenario_id)
        ),
        ZERO,
    )


def _record_adjustment_application(
    metadata: dict[str, Any] | None,
    *,
    adjustment_id: str,
    leg: str,
    scenario_id: str,
    slot: str,
) -> None:
    if metadata is None:
        return
    applied = metadata.setdefault("adjustments_applied", {})
    prior_leg = applied.get(adjustment_id)
    if prior_leg is not None and prior_leg != leg:
        raise ValueError(
            f"{ADJUSTMENT_APPLIED_TWICE}: {adjustment_id} applied to "
            f"{prior_leg} and {leg} in {scenario_id}/{slot}"
        )
    applied[adjustment_id] = leg


def _assert_leg_subtotal(
    breakdown: LegBreakdown,
    *,
    scenario_id: str,
    slot: str,
    leg: str,
) -> None:
    if breakdown.kind == "empty":
        if breakdown.value != ZERO:
            raise ValueError(
                f"{LEG_SUBTOTAL_MISMATCH}: {scenario_id}/{slot} {leg} "
                f"empty leg subtotal {breakdown.value} != 0"
            )
        return

    expected = _sum_rows(breakdown.rows)
    for _, amount in breakdown.terms:
        expected += amount

    if breakdown.value != expected:
        raise ValueError(
            f"{LEG_SUBTOTAL_MISMATCH}: {scenario_id}/{slot} {leg} "
            f"subtotal {breakdown.value} != sum of terms {expected} "
            f"(rows={_sum_rows(breakdown.rows)}, "
            f"adjustment_terms={sum((a for _, a in breakdown.terms), ZERO)})"
        )


def _ebitda_revenue_rows(
    ledger: list[dict[str, Any]],
    *,
    period: tuple[date, date],
    apply_reclass: bool,
) -> list[dict[str, Any]]:
    revenue_spec = {
        "include_keywords": ["revenue"],
        "exclude_keywords": [],
        "apply_reclass": apply_reclass,
    }
    return _filter_rows(ledger, revenue_spec, period=period, parties=None)


def _ebitda_opex_rows(
    ledger: list[dict[str, Any]],
    *,
    period: tuple[date, date],
    apply_reclass: bool,
) -> list[dict[str, Any]]:
    """Operating expenses for EBITDA: opex-category rows plus misclassified outflows.

    FX-normalised foreign-currency rows can still carry category ``other`` when
    the description maps to an operating-expense slug; include them here so the
    converted amount reaches the EBITDA denominator.
    """
    opex_spec = {
        "include_keywords": ["opex"],
        "exclude_keywords": [],
        "apply_reclass": apply_reclass,
    }
    matched = _filter_rows(ledger, opex_spec, period=period, parties=None)
    seen = {row.get("txn_id") for row in matched}
    for row in ledger:
        txn_id = row.get("txn_id")
        if txn_id in seen:
            continue
        if row.get("excluded"):
            continue
        if not _in_period(row, period):
            continue
        amount = _d(row.get("amount_usd"))
        if amount == ZERO:
            continue
        category = _effective_category(row, apply_reclass)
        if category != "other":
            continue
        inferred = infer_category(str(row.get("description") or ""))
        if inferred not in OPEX_SLUGS:
            continue
        if amount > ZERO and category_sign(inferred) == "OUTFLOW":
            amount = -abs(amount)
        if not _sign_ok(amount, category_sign(inferred)):
            continue
        matched.append({**row, "amount_usd": str(amount)})
        seen.add(txn_id)
    return matched


def _compute_ebitda(
    ledger: list[dict[str, Any]],
    *,
    period: tuple[date, date],
    apply_reclass: bool,
    addbacks: Decimal = ZERO,
) -> Decimal:
    revenue = _sum_rows(_ebitda_revenue_rows(ledger, period=period, apply_reclass=apply_reclass))
    opex = _sum_rows(_ebitda_opex_rows(ledger, period=period, apply_reclass=apply_reclass))
    return revenue + opex + addbacks


def _metric_notes(notes: str) -> str:
    return " ".join(str(notes).split()).casefold()


EBITDA_COMPONENT_SLUGS = frozenset({"revenue", "opex"} | set(OPEX_SLUGS))


def _is_adjusted_ebitda_notes(notes: str) -> bool:
    return "скорректированная" in notes or "adjusted" in notes


def _is_ebitda_leg(spec: dict[str, Any], notes: str, *, leg: str) -> bool:
    """True when the leg must be built as derived EBITDA, never category rows.

    Plain EBITDA is revenue minus opex. Adjusted EBITDA also applies add-backs.
    The covenant notes and leg role decide which applies; mis-tagged revenue
    slugs on an EBITDA numerator still take the derived path so the leg cannot
    collapse onto a plain revenue denominator.
    """
    if "ebitda" not in notes:
        return False
    include_keywords = {str(keyword).casefold() for keyword in spec.get("include_keywords") or []}
    keyword_text = " ".join(sorted(include_keywords))
    if "capex" in keyword_text or "капитал" in keyword_text:
        return False
    if "financing" in include_keywords:
        return False
    if "ebitda" in keyword_text:
        return True
    # Remapped EBITDA vocabulary is exactly revenue plus the opex slug.
    if include_keywords == frozenset({"revenue", "opex"}):
        return True
    # Revenue-only denominator is выручка, not EBITDA (e.g. P4 margin).
    if leg == "denominator" and include_keywords <= frozenset({"revenue"}):
        return False
    # EBITDA margin numerators can be mis-tagged with revenue slugs only.
    if leg == "numerator" and include_keywords <= frozenset({"revenue"}):
        return True
    return False


def _special_metric(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    period: tuple[date, date],
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
) -> Decimal | None:
    scenario_id = covenant["scenario_id"]
    slot = covenant["slot"]
    metric = covenant["metric"]
    notes = _metric_notes(metric.get("notes", ""))
    title = covenant.get("title", "")

    if slot == "6.2" and "individual overhead line ceiling" in title.casefold():
        payroll_spec = {
            "include_keywords": ["personnel"],
            "exclude_keywords": [],
            "apply_reclass": True,
        }
        util_spec = {
            "include_keywords": ["utilities"],
            "exclude_keywords": [],
            "apply_reclass": True,
        }
        payroll_total = _sum_rows(
            _filter_rows(ledger, payroll_spec, period=period, parties=parties)
        )
        util_total = _sum_rows(
            _filter_rows(ledger, util_spec, period=period, parties=parties)
        )
        # Overhead lines are outflows (negative as posted); the ceiling is the
        # largest line by magnitude.
        return max(abs(payroll_total), abs(util_total))

    if slot == "6.2" and "за вычетом наибольшей" in notes:
        revenue = _sum_rows(
            _filter_rows(
                ledger,
                {
                    "include_keywords": ["revenue"],
                    "exclude_keywords": [],
                    "apply_reclass": False,
                },
                period=period,
                parties=parties,
            )
        )
        payroll = _sum_rows(
            _filter_rows(
                ledger,
                {
                    "include_keywords": ["personnel"],
                    "exclude_keywords": [],
                    "apply_reclass": False,
                },
                period=period,
                parties=parties,
            )
        )
        tax = _sum_rows(
            _filter_rows(
                ledger,
                {
                    "include_keywords": ["tax"],
                    "exclude_keywords": [],
                    "apply_reclass": False,
                },
                period=period,
                parties=parties,
            )
        )
        # payroll/tax are negative as posted; min() picks the larger expense,
        # so revenue + min(...) subtracts the largest of the two lines.
        return revenue + min(payroll, tax)

    if metric["kind"] == "RATIO":
        pass  # ratio legs handle EBITDA below

    return None


def _leg_categories(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[str]:
    return sorted(
        {
            _effective_category(row, bool(spec.get("apply_reclass", True)))
            for row in rows
        }
    )


def _record_leg_metadata(
    metadata: dict[str, Any] | None,
    *,
    leg: str,
    breakdown: LegBreakdown,
) -> None:
    if metadata is None:
        return
    legs = metadata.setdefault("legs", {})
    legs[leg] = {
        "kind": breakdown.kind,
        "value": str(breakdown.value),
        "row_count": breakdown.row_count,
        "category_count": breakdown.category_count,
        "categories": breakdown.categories,
        "flags": breakdown.flags,
        "expression": breakdown.expression,
        "terms": [(label, str(amount)) for label, amount in breakdown.terms],
    }
    flags = metadata.setdefault("flags", [])
    for flag in breakdown.flags:
        if flag not in flags:
            flags.append(flag)
    if breakdown.category_count > 3:
        if WIDE_LEG_REVIEW not in flags:
            flags.append(WIDE_LEG_REVIEW)


def _ebitda_breakdown(
    ledger: list[dict[str, Any]],
    *,
    period: tuple[date, date],
    apply_reclass: bool,
    addback_items: list[tuple[str, Decimal, list[dict[str, Any]], bool]],
    label: str,
) -> LegBreakdown:
    revenue_rows = _ebitda_revenue_rows(ledger, period=period, apply_reclass=apply_reclass)
    opex_rows = _ebitda_opex_rows(ledger, period=period, apply_reclass=apply_reclass)
    revenue = _sum_rows(revenue_rows)
    opex = _sum_rows(opex_rows)
    value = revenue + opex
    terms: list[tuple[str, Decimal]] = [
        ("revenue", revenue),
        ("opex", opex),
    ]
    flags: list[str] = []
    opex_txn_ids = {str(row.get("txn_id")) for row in opex_rows if row.get("txn_id")}
    for adj_id, above_floor, addback_rows, floor_missing in addback_items:
        # One-off rows are expenses: subtract every row that the opex leg has
        # not already captured, then add back only the above-floor ones. When
        # the floor sentence was not extracted, every listed row is treated as
        # above-floor and the cell is flagged for review instead of dropping
        # the adjustment.
        if floor_missing and FLOOR_MISSING_REVIEW not in flags:
            flags.append(FLOOR_MISSING_REVIEW)
        unsubtracted = sum(
            (
                _d(row.get("amount"))
                for row in addback_rows
                if str(row.get("matched_txn") or "") not in opex_txn_ids
            ),
            ZERO,
        )
        if unsubtracted != ZERO:
            value -= unsubtracted
            terms.append((f"one_off_expense:{adj_id}", -unsubtracted))
        if above_floor != ZERO:
            value += above_floor
            terms.append((f"addback:{adj_id}", above_floor))
    if addback_items:
        expression = f"{label} = revenue - (opex + one-offs) + addbacks"
    else:
        expression = f"{label} = revenue - opex"
    return LegBreakdown(
        kind="derived",
        value=value,
        expression=expression,
        terms=terms,
        flags=flags,
        categories=sorted({row.get("category", "other") for row in revenue_rows + opex_rows}),
    )


def _leg_breakdown(
    ledger: list[dict[str, Any]],
    spec: dict[str, Any] | None,
    *,
    period: tuple[date, date],
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
    scenario_id: str,
    slot: str,
    leg: str,
    metric: dict[str, Any],
    full_ledger: list[dict[str, Any]],
    work_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
    covenant_context: dict[str, Any] | None = None,
) -> LegBreakdown:
    if spec is None:
        return LegBreakdown(kind="empty", value=ZERO)

    include_keywords = spec.get("include_keywords") or []
    flags: list[str] = []
    notes = _metric_notes(metric.get("notes", ""))

    if _is_ebitda_leg(spec, notes, leg=leg):
        addback_items: list[tuple[str, Decimal, list[dict[str, Any]], bool]] = []
        if _is_adjusted_ebitda_notes(notes):
            addback_items = _ebitda_addback_adjustments(adjustments, scenario_id)
            if not addback_items:
                flags.append(EBITDA_CONSTRUCTION_FAILED)
                breakdown = LegBreakdown(kind="empty", value=ZERO, flags=flags)
                _record_leg_metadata(metadata, leg=leg, breakdown=breakdown)
                _assert_leg_subtotal(breakdown, scenario_id=scenario_id, slot=slot, leg=leg)
                return breakdown
            for adj_id, _, _rows, _floor_missing in addback_items:
                _record_adjustment_application(
                    metadata,
                    adjustment_id=adj_id,
                    leg=leg,
                    scenario_id=scenario_id,
                    slot=slot,
                )
        borrower_ledger = _rows_for_scope(full_ledger, scenario_id=scenario_id, scope="BORROWER")
        label = "adjusted EBITDA" if addback_items else "EBITDA"
        breakdown = _ebitda_breakdown(
            borrower_ledger,
            period=period,
            apply_reclass=bool(spec.get("apply_reclass", True)),
            addback_items=addback_items,
            label=label,
        )
        breakdown.flags.extend(flags)
        _record_leg_metadata(metadata, leg=leg, breakdown=breakdown)
        _assert_leg_subtotal(breakdown, scenario_id=scenario_id, slot=slot, leg=leg)
        return breakdown

    related_party_leg = _is_related_party_leg(covenant_context, leg=leg)
    if not include_keywords and not related_party_leg:
        flags.append(EMPTY_CATEGORY_SPEC)
        breakdown = LegBreakdown(kind="empty", value=ZERO, flags=flags)
        _record_leg_metadata(metadata, leg=leg, breakdown=breakdown)
        _assert_leg_subtotal(breakdown, scenario_id=scenario_id, slot=slot, leg=leg)
        return breakdown

    scope = metric.get("scope", "BORROWER")
    effective_scope = scope if leg == "numerator" else "BORROWER"
    leg_ledger = _rows_for_scope(full_ledger, scenario_id=scenario_id, scope=effective_scope)
    group_scope = scope == "GROUP" and leg == "numerator"

    if leg == "numerator" and scope == "GROUP" and group_scope:
        group_value, source = resolve_group_figure(
            scenario_id=scenario_id,
            include_keywords=include_keywords,
            work_dir=work_dir,
        )
        if group_value is not None:
            breakdown = LegBreakdown(
                kind="document",
                value=group_value,
                expression=f"group figure from {source}",
                terms=[(f"group:{source}", group_value)],
                flags=flags,
            )
            _record_leg_metadata(metadata, leg=leg, breakdown=breakdown)
            _assert_leg_subtotal(breakdown, scenario_id=scenario_id, slot=slot, leg=leg)
            return breakdown
        flags.append(GROUP_FIGURE_NOT_FOUND)
        if metadata is not None:
            metadata["strategy"] = "group_scope_borrower_fallback"

    rows = _filter_rows(
        leg_ledger,
        spec,
        period=period,
        parties=parties,
        group_scope=group_scope,
        covenant_context=covenant_context,
        leg=leg,
    )
    _assert_leg_scenario(rows, scenario_id=scenario_id, leg=leg)
    categories = _leg_categories(rows, spec)
    value = _sum_rows(rows)

    kind = "rows" if rows else "empty"
    if kind == "empty" and (include_keywords or related_party_leg) and EMPTY_LEG not in flags:
        flags.append(EMPTY_LEG)
        if metadata is not None:
            metadata.setdefault("empty_legs", []).append(
                {
                    "scenario_id": scenario_id,
                    "slot": slot,
                    "leg": leg,
                    "keywords": list(include_keywords),
                    "row_count": 0,
                    "category_count": 0,
                },
            )
    breakdown = LegBreakdown(
        kind=kind,
        value=value,
        rows=rows,
        categories=categories,
        flags=flags,
    )
    _record_leg_metadata(metadata, leg=leg, breakdown=breakdown)
    _assert_leg_subtotal(breakdown, scenario_id=scenario_id, slot=slot, leg=leg)
    return breakdown


def _leg_value(
    ledger: list[dict[str, Any]],
    spec: dict[str, Any] | None,
    *,
    period: tuple[date, date],
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
    scenario_id: str,
    slot: str,
    leg: str,
    metric: dict[str, Any],
    full_ledger: list[dict[str, Any]],
    work_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
    covenant_context: dict[str, Any] | None = None,
) -> Decimal:
    return _leg_breakdown(
        ledger,
        spec,
        period=period,
        parties=parties,
        adjustments=adjustments,
        scenario_id=scenario_id,
        slot=slot,
        leg=leg,
        metric=metric,
        full_ledger=full_ledger,
        work_dir=work_dir,
        metadata=metadata,
        covenant_context=covenant_context,
    ).value


def describe_leg_breakdown(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    leg: str,
    parties: dict[str, Any] | None = None,
    adjustments: dict[str, Any] | None = None,
    work_dir: Path | None = None,
) -> LegBreakdown:
    """Return a structured breakdown for diagnose --cell."""
    adjustments = adjustments or {}
    scenario_id = covenant["scenario_id"]
    period = (_parse_date(covenant["period"][0]), _parse_date(covenant["period"][1]))
    metric = covenant["metric"]
    spec = metric["numerator"] if leg == "numerator" else metric.get("denominator")
    if leg == "denominator" and metric.get("kind") == "RATIO":
        metric = {**metric, "scope": "BORROWER"}
    scenario_ledger = [row for row in ledger if row.get("scenario_id") == scenario_id]
    covenant_context = {
        "title": covenant.get("title"),
        "notes": covenant.get("metric", {}).get("notes"),
        "metric": covenant.get("metric"),
    }
    return _leg_breakdown(
        scenario_ledger,
        spec,
        period=period,
        parties=parties,
        adjustments=adjustments,
        scenario_id=scenario_id,
        slot=covenant["slot"],
        leg=leg,
        metric=metric,
        full_ledger=ledger,
        work_dir=work_dir,
        covenant_context=covenant_context,
    )


def _leg_row_ids(breakdown: LegBreakdown) -> set[str]:
    ids: set[str] = set()
    for row in breakdown.rows:
        row_id = row.get("txn_id") or row.get("adjustment_ref")
        if row_id is not None:
            ids.add(str(row_id))
    return ids


def _identical_leg_rows(
    numerator: LegBreakdown,
    denominator: LegBreakdown,
) -> set[str]:
    """Row ids when a ratio's two legs resolve to an identical row set.

    In a ratio expressing a share, the denominator is the whole and the
    numerator a subset of it; identical selections are an extraction failure.
    Returns an empty set when the legs are distinct.
    """
    numerator_ids = _leg_row_ids(numerator)
    if not numerator_ids:
        return set()
    if numerator_ids == _leg_row_ids(denominator):
        return numerator_ids
    return set()


def compute_covenant_metric(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    parties: dict[str, Any] | None = None,
    adjustments: dict[str, Any] | None = None,
    work_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Decimal:
    """Compute the covenant metric at full Decimal precision.

    Legs aggregate signed amounts as posted; only the final reported actual
    returned here is made positive.
    """
    adjustments = adjustments or {}
    scenario_id = covenant["scenario_id"]
    slot = covenant["slot"]
    period = (_parse_date(covenant["period"][0]), _parse_date(covenant["period"][1]))
    scenario_ledger = [row for row in ledger if row.get("scenario_id") == scenario_id]
    covenant_context = {
        "title": covenant.get("title"),
        "notes": covenant.get("metric", {}).get("notes"),
        "metric": covenant.get("metric"),
    }

    special = _special_metric(
        covenant,
        scenario_ledger,
        period=period,
        parties=parties,
        adjustments=adjustments,
    )
    if special is not None:
        return special

    metric = covenant["metric"]
    if metric["kind"] == "SUM":
        return abs(
            _leg_value(
                scenario_ledger,
                metric["numerator"],
                period=period,
                parties=parties,
                adjustments=adjustments,
                scenario_id=scenario_id,
                slot=slot,
                leg="numerator",
                metric=metric,
                full_ledger=ledger,
                work_dir=work_dir,
                metadata=metadata,
                covenant_context=covenant_context,
            )
        )

    numerator_breakdown = _leg_breakdown(
        scenario_ledger,
        metric["numerator"],
        period=period,
        parties=parties,
        adjustments=adjustments,
        scenario_id=scenario_id,
        slot=slot,
        leg="numerator",
        metric=metric,
        full_ledger=ledger,
        work_dir=work_dir,
        metadata=metadata,
        covenant_context=covenant_context,
    )
    denominator_breakdown = _leg_breakdown(
        scenario_ledger,
        metric.get("denominator"),
        period=period,
        parties=parties,
        adjustments=adjustments,
        scenario_id=scenario_id,
        slot=slot,
        leg="denominator",
        metric={**metric, "scope": "BORROWER"},
        full_ledger=ledger,
        work_dir=work_dir,
        metadata=metadata,
        covenant_context=covenant_context,
    )
    identical_rows = _identical_leg_rows(numerator_breakdown, denominator_breakdown)
    if identical_rows:
        # Extraction failure: flag the cell for degradation and move on —
        # one bad covenant must never cost the whole run.
        if metadata is not None:
            flags = metadata.setdefault("flags", [])
            if IDENTICAL_LEGS not in flags:
                flags.append(IDENTICAL_LEGS)
            metadata["identical_leg_rows"] = sorted(identical_rows)
        return ZERO

    leg_flags = list(numerator_breakdown.flags) + list(denominator_breakdown.flags)
    if EMPTY_LEG in leg_flags or EBITDA_CONSTRUCTION_FAILED in leg_flags:
        if metadata is not None:
            flags = metadata.setdefault("flags", [])
            for flag in (EMPTY_LEG, EBITDA_CONSTRUCTION_FAILED):
                if flag in leg_flags and flag not in flags:
                    flags.append(flag)
        return ZERO
    if denominator_breakdown.value == ZERO:
        if metadata is not None:
            flags = metadata.setdefault("flags", [])
            if ZERO_DENOMINATOR not in flags:
                flags.append(ZERO_DENOMINATOR)
        return ZERO
    return abs(numerator_breakdown.value / denominator_breakdown.value)


def _explain_row_match(
    row: dict[str, Any],
    spec: dict[str, Any],
    *,
    parties: dict[str, Any] | None,
    group_scope: bool = False,
    covenant_context: dict[str, Any] | None = None,
    leg: str | None = None,
) -> str:
    category = _effective_category(row, bool(spec.get("apply_reclass", True)))
    if row.get("synthetic"):
        ref = row.get("adjustment_ref") or "off_ledger"
        return f"off_ledger~{ref}"
    if _is_related_party_leg(covenant_context, leg=leg):
        return f"related_party_outflow counterparty={row.get('counterparty')}"
    for keyword in spec.get("include_keywords") or []:
        if _category_matches(category, [str(keyword)]):
            return f"category={category}"
    return f"category={category}"


def collect_covenant_inputs(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    parties: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Rows contributing to the covenant metric with categorisation rationale."""
    period = (_parse_date(covenant["period"][0]), _parse_date(covenant["period"][1]))
    scenario_id = covenant["scenario_id"]
    metric = covenant["metric"]
    scope = metric.get("scope", "BORROWER")
    covenant_context = {
        "title": covenant.get("title"),
        "notes": metric.get("notes"),
        "metric": metric,
    }
    inputs: list[dict[str, str]] = []
    seen: set[str | None] = set()

    for row in ledger:
        if row.get("scenario_id") != scenario_id:
            continue
        if row.get("excluded"):
            continue
        if not _in_period(row, period):
            continue
        amount = _d(row.get("amount_usd"))
        if amount == ZERO:
            continue

        matched_spec: dict[str, Any] | None = None
        matched_leg: str | None = None
        group_scope = scope == "GROUP"
        for leg_name, spec in (
            ("numerator", metric["numerator"]),
            ("denominator", metric.get("denominator")),
        ):
            if spec is None:
                continue
            if _row_matches_spec(
                row,
                spec,
                period=period,
                parties=parties,
                group_scope=group_scope,
                covenant_context=covenant_context,
                leg=leg_name,
            ):
                matched_spec = spec
                matched_leg = leg_name
                break
        if matched_spec is None:
            continue

        key = row.get("txn_id") or row.get("adjustment_ref")
        if key in seen:
            continue
        seen.add(key)
        inputs.append(
            {
                "txn_id": str(row.get("txn_id") or f"SYN-{row.get('adjustment_ref')}"),
                "amount_usd": str(amount),
                "category": _effective_category(row, bool(matched_spec.get("apply_reclass", True))),
                "why": _explain_row_match(
                    row,
                    matched_spec,
                    parties=parties,
                    group_scope=group_scope,
                    covenant_context=covenant_context,
                    leg=matched_leg,
                ),
            }
        )
    return inputs


def relevant_row_indices(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    parties: dict[str, Any] | None = None,
) -> list[int]:
    """Indices of rows that can affect the covenant metric (for evidence search)."""
    period = (_parse_date(covenant["period"][0]), _parse_date(covenant["period"][1]))
    scenario_id = covenant["scenario_id"]
    metric = covenant["metric"]
    scope = metric.get("scope", "BORROWER")
    group_scope = scope == "GROUP"
    covenant_context = {
        "title": covenant.get("title"),
        "notes": metric.get("notes"),
        "metric": metric,
    }
    specs: list[tuple[str, dict[str, Any] | None]]
    if metric.get("kind") == "RATIO":
        specs = [
            ("numerator", metric["numerator"]),
            ("denominator", metric.get("denominator")),
        ]
    else:
        specs = [
            ("numerator", metric["numerator"]),
            ("denominator", metric.get("denominator")),
        ]

    indices: list[int] = []
    for index, row in enumerate(ledger):
        if row.get("scenario_id") != scenario_id:
            continue
        if row.get("excluded"):
            continue
        if not _in_period(row, period):
            continue
        matched = False
        for leg_name, spec in specs:
            if spec is None:
                continue
            if _row_matches_spec(
                row,
                spec,
                period=period,
                parties=parties,
                group_scope=group_scope,
                covenant_context=covenant_context,
                leg=leg_name,
            ):
                matched = True
                break
        if matched:
            indices.append(index)
    return indices
