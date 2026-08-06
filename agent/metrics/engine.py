"""Covenant metric computation against an adjusted ledger."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from agent.parsing.categories import OPEX_SLUGS
from agent.stages.s4b_parties import normalize_counterparty

ZERO = Decimal("0")


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
        return ledger
    return [row for row in ledger if row.get("scenario_id") == scenario_id]


def _effective_category(row: dict[str, Any], apply_reclass: bool) -> str:
    if apply_reclass:
        return str(row.get("category") or "other")
    return str(row.get("original_category") or row.get("category") or "other")


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
    return abs(amount)


def _keyword_matches(
    keyword: str,
    *,
    row: dict[str, Any],
    category: str,
    parties: dict[str, Any] | None,
    group_scope: bool = False,
) -> bool:
    key = keyword.casefold()
    desc = str(row.get("description") or "").casefold()

    if key in {"ebitda", "выручка за вычетом операционных расходов"}:
        return False
    if key in {"скорректированная ebitda"}:
        return False

    if key in {"выручка", "выручки"}:
        return category == "revenue"

    if key in {"процентные расходы"}:
        return category == "interest"

    if key in {
        "капитальные затраты",
        "совокупных капитальных затрат",
        "совокупные капитальные затраты заёмщика",
    }:
        return _is_capex_row(row, group_scope=group_scope)

    if key in {
        "расходы на оплату труда",
        "расходов на оплату труда",
        "всех расходов на оплату труда",
    }:
        return category == "personnel" or str(row.get("category") or "") == "severance"

    if key in {
        "расходы на коммунальные услуги",
        "коммунальные расходы",
        "коммунальных расходов",
    }:
        return category == "utilities"

    if key in {"налоги"}:
        return category == "tax"

    if key in {"страховые премии"}:
        return category == "insurance"

    if key in {"арендные и коммунальные расходы"}:
        return category in {"rent", "utilities"}

    if key in {"арендных платежей"}:
        return category == "rent"

    if key in {"операционных расходов"}:
        return category == "opex"

    if key in {"операционных и капитальных затрат"}:
        return category in OPEX_SLUGS or category == "capex"

    if key in {"поступления по финансированию", "поступлений по финансированию"}:
        return category == "financing"

    if key in {
        "платежи",
        "платежей в пользу связанных сторон",
        "связанные стороны",
        "ограниченные платежи",
        "ограниченные платежи в пользу аффилированных лиц",
        "аффилированные лица",
    }:
        if parties is None:
            return False
        return _counterparty_matches(row, _related_counterparties(parties))

    if "капитальные активы, переданные неограниченным дочерним организациям" in key:
        if category != "capex" or parties is None:
            return False
        return _counterparty_matches(row, _unrestricted_counterparties(parties))

    if "совокупного обязательства" in key and "выходных пособий" in key:
        return str(row.get("category") or "") == "severance"

    if key in desc or key in category:
        return True
    return False


def _exclude_matches(keyword: str, *, row: dict[str, Any], category: str) -> bool:
    key = keyword.casefold()
    if "финансовых или иных неоперационных статей" in key:
        return category == "financing"
    if "переклассифицированные аудиторами" in key:
        return bool(row.get("adjustment_ref")) and str(row.get("adjustment_ref", "")).startswith(
            "adj_"
        ) and row.get("original_category") != row.get("category")
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
) -> bool:
    if row.get("excluded"):
        return False
    if not _in_period(row, period):
        return False
    amount = _d(row.get("amount_usd"))
    if amount == ZERO:
        return False
    if not _sign_ok(amount, spec["sign"]):
        return False

    category = _effective_category(row, bool(spec.get("apply_reclass", True)))
    for keyword in spec.get("exclude_keywords") or []:
        if _exclude_matches(keyword, row=row, category=category):
            return False

    include_keywords = spec.get("include_keywords") or []
    if not include_keywords:
        return False
    return any(
        _keyword_matches(
            keyword,
            row=row,
            category=category,
            parties=parties,
            group_scope=group_scope,
        )
        for keyword in include_keywords
    )


def _filter_rows(
    ledger: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    period: tuple[date, date],
    parties: dict[str, Any] | None,
    group_scope: bool = False,
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
        )
    ]


def _sum_rows(rows: list[dict[str, Any]]) -> Decimal:
    total = ZERO
    for row in rows:
        total += _amount_for_aggregation(_d(row.get("amount_usd")))
    return total


def _addback_total(adjustments: dict[str, Any], scenario_id: str) -> Decimal:
    total = ZERO
    for adj in adjustments.values():
        if adj.get("scenario_id") != scenario_id or adj.get("kind") != "EBITDA_ADDBACK":
            continue
        for row in adj.get("rows") or []:
            if row.get("above_floor"):
                total += _d(row.get("amount"))
    return total


def _compute_ebitda(
    ledger: list[dict[str, Any]],
    *,
    period: tuple[date, date],
    apply_reclass: bool,
    addbacks: Decimal = ZERO,
) -> Decimal:
    revenue_spec = {
        "include_keywords": ["Выручка"],
        "exclude_keywords": [],
        "sign": "INFLOW",
        "apply_reclass": apply_reclass,
    }
    opex_spec = {
        "include_keywords": ["Операционных расходов"],
        "exclude_keywords": [],
        "sign": "OUTFLOW",
        "apply_reclass": apply_reclass,
    }
    revenue = _sum_rows(_filter_rows(ledger, revenue_spec, period=period, parties=None))
    opex = _sum_rows(_filter_rows(ledger, opex_spec, period=period, parties=None))
    return revenue - opex + addbacks


def _metric_notes(notes: str) -> str:
    return " ".join(str(notes).split()).casefold()


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
            "include_keywords": ["расходы на оплату труда"],
            "exclude_keywords": [],
            "sign": "OUTFLOW",
            "apply_reclass": True,
        }
        util_spec = {
            "include_keywords": ["расходы на коммунальные услуги"],
            "exclude_keywords": [],
            "sign": "OUTFLOW",
            "apply_reclass": True,
        }
        return max(
            _sum_rows(_filter_rows(ledger, payroll_spec, period=period, parties=parties)),
            _sum_rows(_filter_rows(ledger, util_spec, period=period, parties=parties)),
        )

    if slot == "6.2" and "за вычетом наибольшей" in notes:
        revenue = _sum_rows(
            _filter_rows(
                ledger,
                {
                    "include_keywords": ["Выручка"],
                    "exclude_keywords": [],
                    "sign": "INFLOW",
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
                    "include_keywords": ["Расходов на оплату труда"],
                    "exclude_keywords": [],
                    "sign": "OUTFLOW",
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
                    "include_keywords": ["Налоги"],
                    "exclude_keywords": [],
                    "sign": "OUTFLOW",
                    "apply_reclass": False,
                },
                period=period,
                parties=parties,
            )
        )
        return revenue - max(payroll, tax)

    if metric["kind"] == "RATIO":
        pass  # ratio legs handle EBITDA below

    return None


def _leg_value(
    ledger: list[dict[str, Any]],
    spec: dict[str, Any] | None,
    *,
    period: tuple[date, date],
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
    scenario_id: str,
    leg: str,
    metric: dict[str, Any],
    full_ledger: list[dict[str, Any]],
) -> Decimal:
    if spec is None:
        return ZERO
    scope = metric.get("scope", "BORROWER")
    leg_ledger = _rows_for_scope(full_ledger, scenario_id=scenario_id, scope=scope if leg == "numerator" else "BORROWER")
    group_scope = scope == "GROUP" and leg == "numerator"
    keywords = [k.casefold() for k in spec.get("include_keywords") or []]
    if leg == "numerator" and any("ebitda" in k for k in keywords):
        addbacks = ZERO
        if "скорректированная" in _metric_notes(metric.get("notes", "")):
            addbacks = _addback_total(adjustments, scenario_id)
        borrower_ledger = _rows_for_scope(full_ledger, scenario_id=scenario_id, scope="BORROWER")
        return _compute_ebitda(
            borrower_ledger,
            period=period,
            apply_reclass=bool(spec.get("apply_reclass", True)),
            addbacks=addbacks,
        )
    if leg == "denominator" and any("ebitda" in k for k in keywords):
        borrower_ledger = _rows_for_scope(full_ledger, scenario_id=scenario_id, scope="BORROWER")
        return _compute_ebitda(
            borrower_ledger,
            period=period,
            apply_reclass=bool(spec.get("apply_reclass", True)),
            addbacks=ZERO,
        )
    total = _sum_rows(
        _filter_rows(
            leg_ledger,
            spec,
            period=period,
            parties=parties,
            group_scope=group_scope,
        )
    )
    if leg == "denominator" and any("выручка" in k for k in keywords):
        total += _addback_total(adjustments, scenario_id)
    return total


def compute_covenant_metric(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    parties: dict[str, Any] | None = None,
    adjustments: dict[str, Any] | None = None,
) -> Decimal:
    """Compute the covenant metric at full Decimal precision."""
    adjustments = adjustments or {}
    scenario_id = covenant["scenario_id"]
    period = (_parse_date(covenant["period"][0]), _parse_date(covenant["period"][1]))
    scenario_ledger = [row for row in ledger if row.get("scenario_id") == scenario_id]

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
        return _leg_value(
            scenario_ledger,
            metric["numerator"],
            period=period,
            parties=parties,
            adjustments=adjustments,
            scenario_id=scenario_id,
            leg="numerator",
            metric=metric,
            full_ledger=ledger,
        )

    numerator = _leg_value(
        scenario_ledger,
        metric["numerator"],
        period=period,
        parties=parties,
        adjustments=adjustments,
        scenario_id=scenario_id,
        leg="numerator",
        metric=metric,
        full_ledger=ledger,
    )
    denominator = _leg_value(
        scenario_ledger,
        metric.get("denominator"),
        period=period,
        parties=parties,
        adjustments=adjustments,
        scenario_id=scenario_id,
        leg="denominator",
        metric={**metric, "scope": "BORROWER"},
        full_ledger=ledger,
    )
    if denominator == ZERO:
        return ZERO
    return numerator / denominator


def _explain_row_match(
    row: dict[str, Any],
    spec: dict[str, Any],
    *,
    parties: dict[str, Any] | None,
    group_scope: bool = False,
) -> str:
    category = _effective_category(row, bool(spec.get("apply_reclass", True)))
    desc = str(row.get("description") or "")
    if row.get("synthetic"):
        ref = row.get("adjustment_ref") or "off_ledger"
        return f"off_ledger~{ref}"
    for keyword in spec.get("include_keywords") or []:
        if _keyword_matches(
            keyword,
            row=row,
            category=category,
            parties=parties,
            group_scope=group_scope,
        ):
            key = keyword.casefold()
            if key in desc.casefold():
                token = desc.strip().split()[0][:24] if desc.strip() else keyword[:24]
                return f"description~{token.casefold()}"
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
    inputs: list[dict[str, str]] = []
    seen: set[str | None] = set()

    for row in ledger:
        if scope == "BORROWER" and row.get("scenario_id") != scenario_id:
            continue
        if row.get("excluded"):
            continue
        if not _in_period(row, period):
            continue
        amount = _d(row.get("amount_usd"))
        if amount == ZERO:
            continue

        matched_spec: dict[str, Any] | None = None
        group_scope = scope == "GROUP"
        for spec in (metric["numerator"], metric.get("denominator")):
            if spec is None:
                continue
            if _row_matches_spec(
                row,
                spec,
                period=period,
                parties=parties,
                group_scope=group_scope,
            ):
                matched_spec = spec
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
    indices: list[int] = []
    for index, row in enumerate(ledger):
        if row.get("scenario_id") != scenario_id or row.get("excluded"):
            continue
        if not _in_period(row, period):
            continue
        matched = False
        for spec in (metric["numerator"], metric.get("denominator")):
            if spec is None:
                continue
            if _row_matches_spec(row, spec, period=period, parties=parties):
                matched = True
                break
        if matched:
            indices.append(index)
    return indices
