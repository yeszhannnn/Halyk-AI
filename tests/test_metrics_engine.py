from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from agent.metrics.engine import (
    EMPTY_CATEGORY_SPEC,
    GROUP_FIGURE_NOT_FOUND,
    LEG_SUBTOTAL_MISMATCH,
    SCENARIO_SCOPE_VIOLATION,
    _assert_leg_scenario,
    _category_matches,
    _is_excluded_inflow,
    compute_covenant_metric,
    describe_leg_breakdown,
)
from agent.metrics.group_figures import resolve_group_figure

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
    assert breakdown.value == Decimal("10227549.20")


def test_ebitda_addback_stays_on_numerator_leg() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = next(c for c in covenants if c["scenario_id"] == "P4" and c["slot"] == "6.1")
    adjustments = _adjustments()
    numerator = describe_leg_breakdown(
        covenant,
        _ledger(),
        leg="numerator",
        adjustments=adjustments,
        work_dir=OPEN,
    )
    denominator = describe_leg_breakdown(
        covenant,
        _ledger(),
        leg="denominator",
        adjustments=adjustments,
        work_dir=OPEN,
    )
    row_sum = sum(abs(Decimal(str(row["amount_usd"]))) for row in denominator.rows)
    assert denominator.value == row_sum
    assert denominator.value > Decimal("0")
    assert any(label.startswith("addback:") for label, _ in numerator.terms)
    assert numerator.value == Decimal("3396809.19")


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
