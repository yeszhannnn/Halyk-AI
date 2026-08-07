from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from agent.metrics.engine import (
    EMPTY_CATEGORY_SPEC,
    GROUP_FIGURE_NOT_FOUND,
    SCENARIO_SCOPE_VIOLATION,
    _assert_leg_scenario,
    _is_excluded_inflow,
    _keyword_matches,
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


def test_financing_keyword_rejects_rebate_row() -> None:
    row = {
        "description": "Interest rebate on early repayment",
        "category": "financing",
        "amount_usd": "100.00",
    }
    assert not _keyword_matches(
        "поступления по финансированию",
        row=row,
        category="financing",
        parties=None,
    )


def test_operating_and_capex_keywords_narrow_to_slug_categories() -> None:
    row = {"description": "Property insurance premium", "category": "insurance"}
    assert not _keyword_matches(
        "операционных и капитальных затрат",
        row=row,
        category="insurance",
        parties=None,
    )
    row = {"description": "Cold store servicing and operating costs", "category": "opex"}
    assert _keyword_matches(
        "операционных и капитальных затрат",
        row=row,
        category="opex",
        parties=None,
    )


def test_describe_leg_breakdown_shows_derived_terms() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = next(c for c in covenants if c["scenario_id"] == "P4" and c["slot"] == "6.1")
    breakdown = describe_leg_breakdown(covenant, _ledger(), leg="numerator", work_dir=OPEN)
    assert breakdown.kind == "derived"
    assert breakdown.terms
    assert breakdown.value > Decimal("0")


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
        include_keywords=["капитальные затраты"],
        work_dir=OPEN,
    )
    assert figure is not None
    assert source is not None
    assert figure > Decimal("20000000")


def test_group_scope_falls_back_to_borrower_rows() -> None:
    covenants = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    covenant = next(c for c in covenants if c["scenario_id"] == "P5" and c["slot"] == "6.1")
    metadata: dict = {}
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
