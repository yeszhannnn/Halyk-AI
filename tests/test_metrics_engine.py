from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from agent.metrics.engine import (
    EMPTY_CATEGORY_SPEC,
    FLOOR_MISSING_REVIEW,
    GROUP_FIGURE_NOT_FOUND,
    IDENTICAL_LEGS,
    LEG_SIGN_CONTRADICTION,
    LEG_SUBTOTAL_MISMATCH,
    SCENARIO_SCOPE_VIOLATION,
    ZERO_DENOMINATOR,
    _assert_leg_scenario,
    _category_matches,
    _ebitda_addback_adjustments,
    _is_ebitda_leg,
    _is_excluded_inflow,
    _metric_notes,
    compute_covenant_metric,
    describe_leg_breakdown,
)
from agent.metrics.group_figures import resolve_group_figure
from agent.parsing.numbers import round_half_up
from scripts.remap_covenant_categories import remap_covenant

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT / "data" / "open"


def _ledger() -> list[dict]:
    con = duckdb.connect()
    try:
        return con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(OPEN / "05_ledger.parquet")],
        ).df().to_dict("records")
    finally:
        con.close()


@pytest.mark.parametrize(
    "description",
    [
        "Interest rebate on early repayment",
        "Excise tax credit received",
        "Telecom service credit received",
        "Insurance broker rebate",
    ],
)
def test_excluded_inflow_markers(description: str) -> None:
    assert _is_excluded_inflow(description)


def test_financing_category_rejects_rebate_row() -> None:
    row = {
        "description": "Interest rebate on early repayment",
        "category": "financing",
        "amount_usd": "100.00",
        "date": "2025-06-01",
        "excluded": False,
    }
    from agent.metrics.engine import _row_matches_spec

    assert not _row_matches_spec(
        row,
        {"include_keywords": ["financing"], "exclude_keywords": [], "apply_reclass": True},
        period=(__import__("datetime").date(2025, 1, 1), __import__("datetime").date(2025, 12, 31)),
        parties=None,
    )


def test_operating_and_capex_categories_match_slugs() -> None:
    assert not _category_matches("insurance", ["opex", "capex"])
    assert _category_matches("opex", ["opex", "capex"])


def test_describe_leg_breakdown_shows_derived_terms() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = next(c for c in covenants if c["scenario_id"] == "P4" and c["slot"] == "6.1")
    breakdown = describe_leg_breakdown(covenant, _ledger(), leg="numerator", work_dir=OPEN)
    assert breakdown.kind == "derived"
    assert breakdown.terms
    assert breakdown.expression is not None


def test_empty_category_spec_flag() -> None:
    covenant = {
        "scenario_id": "P1",
        "slot": "6.1",
        "period": ["2025-01-01", "2025-12-31"],
        "metric": {
            "kind": "SUM",
            "scope": "BORROWER",
            "notes": "",
            "numerator": {
                "include_keywords": [],
                "exclude_keywords": [],
                "sign": "OUTFLOW",
                "apply_reclass": True,
            },
            "denominator": None,
        },
    }
    metadata: dict = {}
    value = compute_covenant_metric(covenant, _ledger(), metadata=metadata)
    assert value == Decimal("0")
    assert EMPTY_CATEGORY_SPEC in metadata.get("flags", [])


def test_group_figure_resolves_from_consolidated_disclosure() -> None:
    figure, source = resolve_group_figure(
        scenario_id="P5",
        include_keywords=["capex"],
        work_dir=OPEN,
    )
    assert figure is not None
    assert source is not None
    assert figure > Decimal("20000000")


def test_group_scope_falls_back_to_borrower_rows() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = next(c for c in covenants if c["scenario_id"] == "P5" and c["slot"] == "6.1")
    covenant = {
        **covenant,
        "metric": {
            **covenant["metric"],
            "scope": "GROUP",
        },
    }
    breakdown = describe_leg_breakdown(
        covenant,
        _ledger(),
        leg="numerator",
        work_dir=Path("/nonexistent"),
    )
    assert GROUP_FIGURE_NOT_FOUND in breakdown.flags
    assert breakdown.row_count >= 1
    assert all(row.get("scenario_id") == "P5" for row in breakdown.rows)


def test_scenario_scope_invariant_raises() -> None:
    with pytest.raises(ValueError, match=SCENARIO_SCOPE_VIOLATION):
        _assert_leg_scenario(
            [{"scenario_id": "P1", "txn_id": "TXN-P1-0001"}],
            scenario_id="P2",
            leg="numerator",
        )


def _adjustments() -> dict:
    return json.loads((OPEN / "04c_adjustments.json").read_text(encoding="utf-8"))["adjustments"]


def test_reclass_row_reaches_destination_leg() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = next(c for c in covenants if c["scenario_id"] == "P2" and c["slot"] == "6.1")
    breakdown = describe_leg_breakdown(
        covenant,
        _ledger(),
        leg="denominator",
        adjustments=_adjustments(),
        work_dir=OPEN,
    )
    txn_ids = {row["txn_id"] for row in breakdown.rows}
    assert "TXN-P2-0040" in txn_ids
    # Legs aggregate signed amounts as posted; opex/capex rows are outflows.
    assert breakdown.value == Decimal("-10227549.20")


def test_ebitda_addback_stays_on_numerator_leg() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = copy.deepcopy(
        next(c for c in covenants if c["scenario_id"] == "P4" and c["slot"] == "6.1")
    )
    remap_covenant(covenant)
    ledger = _ledger()
    adjustments = _adjustments()
    numerator = describe_leg_breakdown(
        covenant,
        ledger,
        leg="numerator",
        adjustments=adjustments,
        work_dir=OPEN,
    )
    denominator = describe_leg_breakdown(
        covenant,
        ledger,
        leg="denominator",
        adjustments=adjustments,
        work_dir=OPEN,
    )
    row_sum = sum(Decimal(str(row["amount_usd"])) for row in denominator.rows)
    assert denominator.value == row_sum
    assert denominator.value == Decimal("7004318.47")
    assert any(label.startswith("addback:") for label, _ in numerator.terms)
    assert any(label.startswith("one_off_expense:") for label, _ in numerator.terms)
    # One-off rows are subtracted inside opex first; only above-floor rows
    # are added back: 7004318.47 - 4431662.19 - 1075491.85 + 824152.91.
    assert numerator.value == Decimal("2321317.34")
    actual = compute_covenant_metric(
        covenant,
        ledger,
        adjustments=adjustments,
        work_dir=OPEN,
    )
    assert actual == Decimal("2321317.34") / Decimal("7004318.47")


def test_p2_6_1_breach_with_reclass_evidence() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = next(c for c in covenants if c["scenario_id"] == "P2" and c["slot"] == "6.1")
    actual = compute_covenant_metric(
        covenant,
        _ledger(),
        adjustments=_adjustments(),
        work_dir=OPEN,
    )
    assert actual > Decimal("1.18") - Decimal("0.01")
    assert actual < Decimal("1.20")


@pytest.mark.parametrize(
    ("scenario_id", "expected"),
    [
        ("P5", Decimal("273418.66")),
        ("P7", Decimal("291663.82")),
        ("P9", Decimal("268447.19")),
    ],
)
def test_related_party_6_3_uses_counterparty_filter(scenario_id: str, expected: Decimal) -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    parties = json.loads((OPEN / "04b_parties.json").read_text(encoding="utf-8"))
    covenant = next(c for c in covenants if c["scenario_id"] == scenario_id and c["slot"] == "6.3")
    actual = compute_covenant_metric(
        covenant,
        _ledger(),
        parties=parties["scenarios"][scenario_id],
    )
    assert actual == expected


def test_leg_subtotal_mismatch_raises() -> None:
    from agent.metrics.engine import LegBreakdown, _assert_leg_subtotal

    breakdown = LegBreakdown(
        kind="rows",
        value=Decimal("100"),
        rows=[{"amount_usd": "50"}],
    )
    with pytest.raises(ValueError, match=LEG_SUBTOTAL_MISMATCH):
        _assert_leg_subtotal(breakdown, scenario_id="P4", slot="6.1", leg="denominator")


def _synthetic_ledger() -> list[dict]:
    return [
        {
            "txn_id": "TXN-T1-0001",
            "scenario_id": "T1",
            "date": "2025-06-01",
            "amount_usd": "1000.00",
            "category": "revenue",
            "excluded": False,
        },
        {
            "txn_id": "TXN-T1-0002",
            "scenario_id": "T1",
            "date": "2025-06-02",
            "amount_usd": "-400.00",
            "category": "opex",
            "excluded": False,
        },
        {
            "txn_id": "TXN-T1-0003",
            "scenario_id": "T1",
            "date": "2025-06-03",
            "amount_usd": "-250.00",
            "category": "capex",
            "excluded": False,
        },
    ]


def _synthetic_covenant(
    numerator_keywords: list[str],
    denominator_keywords: list[str] | None,
) -> dict:
    return {
        "scenario_id": "T1",
        "slot": "6.1",
        "title": "synthetic",
        "period": ["2025-01-01", "2025-12-31"],
        "metric": {
            "kind": "RATIO" if denominator_keywords is not None else "SUM",
            "scope": "BORROWER",
            "notes": "",
            "numerator": {
                "include_keywords": numerator_keywords,
                "exclude_keywords": [],
                "apply_reclass": True,
            },
            "denominator": (
                {
                    "include_keywords": denominator_keywords,
                    "exclude_keywords": [],
                    "apply_reclass": True,
                }
                if denominator_keywords is not None
                else None
            ),
        },
    }


def test_mixed_sign_leg_nets_instead_of_totalling() -> None:
    covenant = _synthetic_covenant(["revenue", "opex"], None)
    # 1000.00 - 400.00 as posted; abs-per-row would wrongly give 1400.00.
    assert compute_covenant_metric(covenant, _synthetic_ledger()) == Decimal("600.00")


def test_identical_legs_recorded_and_degraded() -> None:
    covenant = _synthetic_covenant(["capex"], ["capex"])
    metadata: dict = {}
    actual = compute_covenant_metric(covenant, _synthetic_ledger(), metadata=metadata)
    assert actual == Decimal("0")
    assert IDENTICAL_LEGS in metadata.get("flags", [])
    assert metadata.get("identical_leg_rows") == ["TXN-T1-0003"]


def test_subset_ratio_legs_pass_invariant() -> None:
    covenant = _synthetic_covenant(["capex"], ["capex", "opex"])
    actual = compute_covenant_metric(covenant, _synthetic_ledger())
    assert actual == Decimal("250.00") / Decimal("650.00")


def _ebitda_covenant(numerator_keywords: list[str] | None = None) -> dict:
    covenant = _synthetic_covenant(numerator_keywords or ["revenue", "opex"], None)
    covenant["metric"]["notes"] = "скорректированная EBITDA"
    return covenant


def _floor_missing_adjustments() -> dict:
    return {
        "adj_t1_1": {
            "id": "adj_t1_1",
            "kind": "EBITDA_ADDBACK",
            "scenario_id": "T1",
            "materiality_floor": None,
            "rows": [
                {
                    "item": "Очистка прибрежного дна",
                    "counterparty": "Zhailyk Dredging LLP",
                    "amount": "251338.94",
                    "above_floor": None,
                    "matched_txn": None,
                },
                {
                    "item": "Урегулирование спора",
                    "counterparty": "Aral Freight",
                    "amount": "342905.28",
                    "above_floor": None,
                    "matched_txn": "TXN-T1-0023",
                },
                {
                    "item": "Устранение последствий паводка",
                    "counterparty": "Ilek Restoration",
                    "amount": "481247.63",
                    "above_floor": None,
                    "matched_txn": "TXN-T1-0025",
                },
            ],
        },
    }


def test_ebitda_addback_floor_missing_adds_back_every_listed_row() -> None:
    items = _ebitda_addback_adjustments(_floor_missing_adjustments(), "T1")
    assert len(items) == 1
    adj_id, above_floor, rows, floor_missing = items[0]
    assert adj_id == "adj_t1_1"
    assert floor_missing is True
    assert len(rows) == 3
    # With no floor, every listed row is added back rather than dropped.
    assert above_floor == Decimal("251338.94") + Decimal("342905.28") + Decimal("481247.63")


def test_ebitda_addback_floor_present_keeps_above_floor_filter() -> None:
    adjustments = _floor_missing_adjustments()
    adjustments["adj_t1_1"]["materiality_floor"] = "300000"
    adjustments["adj_t1_1"]["rows"] = [
        {**adjustments["adj_t1_1"]["rows"][0], "above_floor": False},
        {**adjustments["adj_t1_1"]["rows"][1], "above_floor": True},
        {**adjustments["adj_t1_1"]["rows"][2], "above_floor": True},
    ]
    items = _ebitda_addback_adjustments(adjustments, "T1")
    assert len(items) == 1
    _adj_id, above_floor, _rows, floor_missing = items[0]
    assert floor_missing is False
    assert above_floor == Decimal("342905.28") + Decimal("481247.63")


def test_ebitda_addback_floor_missing_flags_cell_for_review() -> None:
    covenant = _ebitda_covenant()
    metadata: dict = {}
    compute_covenant_metric(
        covenant,
        _synthetic_ledger(),
        adjustments=_floor_missing_adjustments(),
        metadata=metadata,
    )
    assert FLOOR_MISSING_REVIEW in metadata.get("flags", [])


def test_ebitda_addback_floor_present_does_not_flag_cell() -> None:
    adjustments = _floor_missing_adjustments()
    adjustments["adj_t1_1"]["materiality_floor"] = "300000"
    adjustments["adj_t1_1"]["rows"] = [
        {**adjustments["adj_t1_1"]["rows"][0], "above_floor": False},
        {**adjustments["adj_t1_1"]["rows"][1], "above_floor": True},
        {**adjustments["adj_t1_1"]["rows"][2], "above_floor": True},
    ]
    covenant = _ebitda_covenant()
    metadata: dict = {}
    compute_covenant_metric(
        covenant,
        _synthetic_ledger(),
        adjustments=adjustments,
        metadata=metadata,
    )
    assert FLOOR_MISSING_REVIEW not in metadata.get("flags", [])


def test_zero_denominator_flags_and_skips_ratio() -> None:
    ledger = [
        {
            "txn_id": "TXN-T1-0001",
            "scenario_id": "T1",
            "date": "2025-06-01",
            "amount_usd": "1000.00",
            "category": "revenue",
            "excluded": False,
        },
        {
            "txn_id": "TXN-T1-0002",
            "scenario_id": "T1",
            "date": "2025-06-02",
            "amount_usd": "-1000.00",
            "category": "opex",
            "excluded": False,
        },
    ]
    covenant = {
        "scenario_id": "T1",
        "slot": "6.1",
        "title": "synthetic",
        "period": ["2025-01-01", "2025-12-31"],
        "metric": {
            "kind": "RATIO",
            "scope": "BORROWER",
            "notes": "",
            "numerator": {
                "include_keywords": ["revenue"],
                "exclude_keywords": [],
                "apply_reclass": True,
            },
            "denominator": {
                "include_keywords": ["revenue", "opex"],
                "exclude_keywords": [],
                "apply_reclass": True,
            },
        },
    }
    metadata: dict = {}
    actual = compute_covenant_metric(covenant, ledger, metadata=metadata)
    assert actual == Decimal("0")
    assert ZERO_DENOMINATOR in metadata.get("flags", [])


def _p3_ledger_with_fx_opex() -> list[dict]:
    return _ledger()


def test_remapped_ebitda_denominator_uses_derived_path() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = next(c for c in covenants if c["scenario_id"] == "P3" and c["slot"] == "6.1")
    remap_covenant(covenant)
    spec = covenant["metric"]["denominator"]
    notes = _metric_notes(covenant["metric"]["notes"])
    assert _is_ebitda_leg(spec, notes, leg="denominator")
    breakdown = describe_leg_breakdown(
        covenant,
        _p3_ledger_with_fx_opex(),
        leg="denominator",
        adjustments=_adjustments(),
        work_dir=OPEN,
    )
    assert breakdown.kind == "derived"
    assert breakdown.value == Decimal("3175820.12")


def test_p3_6_1_financing_to_ebitda_ratio() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = next(c for c in covenants if c["scenario_id"] == "P3" and c["slot"] == "6.1")
    remap_covenant(covenant)
    ledger = _p3_ledger_with_fx_opex()
    adjustments = _adjustments()
    denominator = describe_leg_breakdown(
        covenant,
        ledger,
        leg="denominator",
        adjustments=adjustments,
        work_dir=OPEN,
    )
    assert denominator.value == Decimal("3175820.12")
    txn = next(row for row in ledger if row.get("txn_id") == "TXN-P3-0024")
    assert Decimal(str(txn["amount_usd"])) == Decimal("710945.73")
    metadata: dict = {}
    actual = compute_covenant_metric(
        covenant,
        ledger,
        adjustments=adjustments,
        work_dir=OPEN,
        metadata=metadata,
    )
    assert round_half_up(abs(actual), 6) == Decimal("1.713611")
    assert round_half_up(abs(actual), 2) == Decimal("1.71")


def test_p6_6_2_revenue_numerator_ignores_related_party_notes_bleed() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    parties = json.loads((OPEN / "04b_parties.json").read_text(encoding="utf-8"))["scenarios"]["P6"]
    covenant = copy.deepcopy(
        next(c for c in covenants if c["scenario_id"] == "P6" and c["slot"] == "6.2")
    )
    remap_covenant(covenant)
    covenant["metric"]["notes"] += " Платёж в пользу связанной стороны."
    covenant["metric"]["denominator"]["include_keywords"] = ["personnel", "utilities"]
    ledger = _ledger()
    numerator = describe_leg_breakdown(
        covenant,
        ledger,
        leg="numerator",
        parties=parties,
    )
    denominator = describe_leg_breakdown(
        covenant,
        ledger,
        leg="denominator",
        parties=parties,
    )
    assert numerator.value == Decimal("6918204.37")
    assert denominator.value == Decimal("-1900867.56")
    metadata: dict = {}
    actual = compute_covenant_metric(
        covenant,
        ledger,
        parties=parties,
        metadata=metadata,
    )
    assert LEG_SIGN_CONTRADICTION not in metadata.get("flags", [])
    assert actual == Decimal("6918204.37") / Decimal("1900867.56")


def test_leg_sign_contradiction_flags_revenue_numerator_resolving_to_outflows() -> None:
    covenant = {
        "scenario_id": "P6",
        "slot": "6.2",
        "title": "Revenue coverage",
        "period": ["2025-01-01", "2025-12-31"],
        "metric": {
            "kind": "RATIO",
            "scope": "BORROWER",
            "notes": "revenue must cover payroll and utilities",
            "numerator": {
                "include_keywords": ["consulting"],
                "exclude_keywords": [],
                "sign": "INFLOW",
                "apply_reclass": True,
            },
            "denominator": {
                "include_keywords": ["personnel", "utilities"],
                "exclude_keywords": [],
                "sign": "OUTFLOW",
                "apply_reclass": True,
            },
        },
    }
    ledger = [
        {
            "txn_id": "TXN-P6-0040",
            "scenario_id": "P6",
            "date": "2025-06-01",
            "amount_usd": "-418662.44",
            "category": "consulting",
            "excluded": False,
            "counterparty": "Taraz Holding Group LLP",
        },
        {
            "txn_id": "TXN-P6-0031",
            "scenario_id": "P6",
            "date": "2025-06-02",
            "amount_usd": "6918204.37",
            "category": "revenue",
            "excluded": False,
        },
    ]
    metadata: dict = {}
    actual = compute_covenant_metric(covenant, ledger, metadata=metadata)
    assert actual == Decimal("0")
    assert LEG_SIGN_CONTRADICTION in metadata.get("flags", [])


def test_p4_6_1_mistagged_revenue_numerator_uses_adjusted_ebitda() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = copy.deepcopy(
        next(c for c in covenants if c["scenario_id"] == "P4" and c["slot"] == "6.1")
    )
    covenant["metric"]["numerator"]["include_keywords"] = ["revenue"]
    covenant["metric"]["denominator"]["include_keywords"] = ["revenue"]
    ledger = _ledger()
    adjustments = _adjustments()
    numerator = describe_leg_breakdown(
        covenant,
        ledger,
        leg="numerator",
        adjustments=adjustments,
        work_dir=OPEN,
    )
    denominator = describe_leg_breakdown(
        covenant,
        ledger,
        leg="denominator",
        adjustments=adjustments,
        work_dir=OPEN,
    )
    assert numerator.kind == "derived"
    assert numerator.value == Decimal("2321317.34")
    assert denominator.value == Decimal("7004318.47")
    metadata: dict = {}
    actual = compute_covenant_metric(
        covenant,
        ledger,
        adjustments=adjustments,
        work_dir=OPEN,
        metadata=metadata,
    )
    assert IDENTICAL_LEGS not in metadata.get("flags", [])
    assert actual == Decimal("2321317.34") / Decimal("7004318.47")
    assert actual > Decimal("0.33") - Decimal("0.01")
    assert actual < Decimal("0.33") + Decimal("0.01")


def test_p7_6_1_opex_denominator_uses_derived_ebitda() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = copy.deepcopy(
        next(c for c in covenants if c["scenario_id"] == "P7" and c["slot"] == "6.1")
    )
    covenant["metric"]["denominator"]["include_keywords"] = ["opex"]
    covenant["metric"]["numerator"]["include_keywords"] = ["tax", "utilities"]
    notes = _metric_notes(covenant["metric"]["notes"])
    assert _is_ebitda_leg(covenant["metric"]["denominator"], notes, leg="denominator")
    denominator = describe_leg_breakdown(
        covenant,
        _ledger(),
        leg="denominator",
        adjustments=_adjustments(),
        work_dir=OPEN,
    )
    assert denominator.kind == "derived"
    assert denominator.shape == "derived_ebitda"
    assert denominator.value == Decimal("2728878.76")


def test_fx_rates_prefers_settlement_over_stored_rate() -> None:
    from agent.stages.s5_ledger import _fx_rates

    adjustments = {
        "adj_p3_fx": {
            "kind": "FX",
            "scenario_id": "P3",
            "fx_source_amount": "72146.75",
            "fx_settlement_usd": "83690.23",
            "rate": "1.16",
        },
    }
    rate, _ = _fx_rates(adjustments)["P3"]
    expected = Decimal("83690.23") / Decimal("72146.75")
    assert rate == expected


def test_fx_normalisation_preserves_sign_and_covers_all_rows() -> None:
    from agent.stages.s5_ledger import _fx_rates, _normalize_fx

    adjustments = {
        "adj_t1_fx": {"kind": "FX", "scenario_id": "T1", "rate": "1.16"},
    }
    rows = [
        {
            "txn_id": "TXN-T1-0001",
            "scenario_id": "T1",
            "amount_usd": "-612884.25",
            "currency": "EUR",
            "adjustment_ref": None,
        },
        {
            "txn_id": "TXN-T1-0002",
            "scenario_id": "T1",
            "amount_usd": "100",
            "currency": "EUR",
            "adjustment_ref": None,
        },
        {
            "txn_id": "TXN-T2-0001",
            "scenario_id": "T2",
            "amount_usd": "-50",
            "currency": "EUR",
            "adjustment_ref": None,
        },
    ]
    conflicts: list[dict] = []
    _normalize_fx(rows, _fx_rates(adjustments), conflicts)

    assert rows[0]["amount_usd"] == str(Decimal("-612884.25") * Decimal("1.16"))
    assert rows[0]["currency"] == "USD"
    assert rows[1]["amount_usd"] == str(Decimal("100") * Decimal("1.16"))
    # No disclosed rate for T2: row left untouched and flagged.
    assert rows[2]["currency"] == "EUR"
    assert conflicts == [
        {
            "kind": "FX_RATE_MISSING",
            "scenario_id": "T2",
            "txn_id": "TXN-T2-0001",
            "currency": "EUR",
        }
    ]
