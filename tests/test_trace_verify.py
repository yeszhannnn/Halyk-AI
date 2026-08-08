from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from agent.trace import _computation_block, _format_comparison, verify


def _cell(
    *,
    scenario_id: str = "P3",
    slot: str = "6.1",
    status: str = "COMPLIANT",
    strategy: str = "computed",
    evaluated: str = "5",
    rounded: str = "5",
    threshold: str = "1",
    direction: str = "MAX",
    quote: str = "threshold quote",
) -> dict:
    comparison = _format_comparison(
        Decimal(evaluated),
        Decimal(threshold),
        direction,
        status,
    )
    status_from_evaluated = (
        "BREACH" if Decimal(evaluated) > Decimal(threshold) else "COMPLIANT"
    )
    return {
        "scenario_id": scenario_id,
        "slot": slot,
        "status": status,
        "actual": float(rounded),
        "strategy": strategy,
        "covenant": {
            "threshold": threshold,
            "direction": direction,
            "source": {"doc": "loan.pdf", "page": 1, "quote": quote},
        },
        "computation": {
            "evaluated": evaluated,
            "rounded": rounded,
            "comparison": comparison,
            "status_from_evaluated": status_from_evaluated,
            "comparison_from_evaluated": _format_comparison(
                Decimal(evaluated),
                Decimal(threshold),
                direction,
                status_from_evaluated,
            ),
            "adjustments_applied": [],
        },
    }


def _template() -> dict:
    return {"answers": {"P3": {"6.1": {}}}}


def _inventory_with_quote(quote: str) -> dict:
    return {"documents": {"loan": {"pages": [quote]}}}


def test_verify_records_status_mismatch_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent.trace._ledger_rows", lambda _work_dir: [])
    inventory = _inventory_with_quote("threshold quote")
    (tmp_path / "01_inventory.json").write_text(
        __import__("json").dumps(inventory),
        encoding="utf-8",
    )

    trace = {
        "cells": [
            _cell(status="COMPLIANT", evaluated="5", threshold="1"),
        ],
        "adjustments": {},
    }

    conflicts = verify(trace, work_dir=tmp_path, template=_template())

    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["check"] == "status_matches_evaluated"
    assert conflict["scenario_id"] == "P3"
    assert conflict["slot"] == "6.1"
    assert conflict["status"] == "COMPLIANT"
    assert conflict["status_from_evaluated"] == "BREACH"
    assert conflict["comparison"].endswith("COMPLIANT")
    assert conflict["comparison_from_evaluated"].endswith("BREACH")


def test_verify_continues_after_cell_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent.trace._ledger_rows", lambda _work_dir: [])
    inventory = _inventory_with_quote("good quote")
    (tmp_path / "01_inventory.json").write_text(
        __import__("json").dumps(inventory),
        encoding="utf-8",
    )

    trace = {
        "cells": [
            _cell(status="COMPLIANT", evaluated="5", threshold="1", quote="missing"),
            _cell(
                scenario_id="P4",
                slot="6.2",
                status="COMPLIANT",
                evaluated="0.5",
                threshold="1",
                quote="good quote",
            ),
        ],
        "adjustments": {},
    }
    template = {"answers": {"P3": {"6.1": {}}, "P4": {"6.2": {}}}}

    conflicts = verify(trace, work_dir=tmp_path, template=template)

    checks = {conflict["check"] for conflict in conflicts}
    assert "quote_on_page" in checks
    assert "status_matches_evaluated" in checks
    assert len(conflicts) >= 2


def test_computation_block_records_both_comparisons() -> None:
    covenant = {
        "scenario_id": "P3",
        "slot": "6.1",
        "threshold": "1",
        "direction": "MAX",
        "period": ["2025-01-01", "2025-12-31"],
        "metric": {"kind": "SUM", "numerator": {"include_keywords": ["revenue"]}},
    }
    block = _computation_block(
        covenant,
        [],
        parties=None,
        adjustments={},
        evaluated=Decimal("5"),
        rounded=Decimal("5"),
        status="COMPLIANT",
    )
    assert block["comparison"].endswith("COMPLIANT")
    assert block["status_from_evaluated"] == "BREACH"
    assert block["comparison_from_evaluated"].endswith("BREACH")
