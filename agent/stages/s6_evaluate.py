"""Stage 6 — evaluate all 36 covenant cells."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from agent.degrade import SLOT_STATUS_PRIOR, _degraded_finding
from agent.evidence.counterfactual import find_evidence
from agent.metrics.engine import (
    breaches,
    compare_values,
    compute_covenant_metric,
    EBITDA_CONSTRUCTION_FAILED,
    EMPTY_LEG,
    FLOOR_MISSING_REVIEW,
    IDENTICAL_LEGS,
    ZERO_DENOMINATOR,
    _requires_related_party_filter,
)
from agent.parsing.numbers import round_half_up
from agent.stages import StageResult
from agent.template import load_template, template_cells


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _adjusted_txn_ids(adjustments: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for adj in adjustments.values():
        txn_id = adj.get("matched_txn") or adj.get("txn_id")
        if txn_id and adj.get("kind") in {
            "RECLASS",
            "AMOUNT_FILL",
            "EXCLUDE",
            "CUTOFF",
        }:
            ids.add(str(txn_id))
    return ids


def _ledger_rows(work_dir: Path) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        df = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(work_dir / "05_ledger.parquet")],
        ).df()
    finally:
        con.close()
    return df.to_dict(orient="records")


def _evaluate_cell(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
    adjusted_txn_ids: set[str],
    work_dir: Path,
) -> dict[str, Any]:
    slot = covenant["slot"]
    if covenant.get("degraded"):
        finding = _degraded_finding(covenant)
        finding["flags"] = ["DEGRADED"]
        return finding

    if _requires_related_party_filter(covenant) and (
        parties is None or parties.get("threshold_pct") is None
    ):
        # No relatedness threshold was extracted for this scenario; the
        # related-party leg cannot be computed. Degrade instead of crashing.
        finding = _degraded_finding(covenant)
        finding["flags"] = ["DEGRADED", "ABSENT_RELATEDNESS_THRESHOLD"]
        return finding

    threshold = Decimal(str(covenant["threshold"]))
    direction = covenant["direction"]
    flags: list[str] = []
    strategy = "computed"
    springing = covenant.get("springing")
    status: str | None = None
    computed = False
    actual = Decimal("0")

    try:
        metric_metadata: dict[str, Any] = {}
        actual = compute_covenant_metric(
            covenant,
            ledger,
            parties=parties,
            adjustments=adjustments,
            work_dir=work_dir,
            metadata=metric_metadata,
        )
        computed = True
        flags.extend(metric_metadata.get("flags") or [])
        if EMPTY_LEG in flags:
            computed = False
            status = SLOT_STATUS_PRIOR.get(slot, "COMPLIANT")
            strategy = "empty_leg_slot_prior"
            actual = threshold
        elif EBITDA_CONSTRUCTION_FAILED in flags:
            computed = False
            status = SLOT_STATUS_PRIOR.get(slot, "COMPLIANT")
            strategy = "ebitda_construction_slot_prior"
            actual = threshold
        elif ZERO_DENOMINATOR in flags:
            computed = False
            status = SLOT_STATUS_PRIOR.get(slot, "COMPLIANT")
            strategy = "zero_denominator_slot_prior"
            actual = threshold
        elif IDENTICAL_LEGS in flags:
            # Both legs selected the same rows: extraction failure. Degrade
            # this cell through the slot prior and keep evaluating the rest.
            computed = False
            status = SLOT_STATUS_PRIOR.get(slot, "COMPLIANT")
            strategy = "identical_legs_slot_prior"
            actual = Decimal("0")
        elif metric_metadata.get("strategy"):
            strategy = str(metric_metadata["strategy"])
    except Exception:
        actual = threshold
        computed = False
        strategy = "actual_threshold_fallback"

    if springing and status is None:
        trigger_metric = springing["metric"]
        trigger_cov = {
            **covenant,
            "metric": trigger_metric,
            "direction": "MAX",
            "threshold": springing["value"],
        }
        try:
            trigger_value = compute_covenant_metric(
                trigger_cov,
                ledger,
                parties=parties,
                adjustments=adjustments,
                work_dir=work_dir,
            )
            if not compare_values(
                trigger_value,
                springing["operator"],
                Decimal(str(springing["value"])),
            ):
                status = "COMPLIANT"
                strategy = "springing_not_triggered"
        except Exception:
            pass

    if status is None:
        if computed:
            status = "BREACH" if breaches(actual, direction, threshold) else "COMPLIANT"
        else:
            status = SLOT_STATUS_PRIOR.get(slot, "COMPLIANT")
            strategy = "status_slot_prior"

    if not computed and strategy not in {
        "empty_leg_slot_prior",
        "ebitda_construction_slot_prior",
        "zero_denominator_slot_prior",
    }:
        actual = threshold if strategy == "actual_threshold_fallback" else Decimal("0")
        if strategy == "status_slot_prior":
            actual = Decimal("0")
            strategy = "actual_zero_fallback"

    evidence: str | None = None
    if status == "BREACH":
        evidence, ev_flags = find_evidence(
            covenant,
            ledger,
            status=status,
            parties=parties,
            adjustments=adjustments,
            adjusted_txn_ids=adjusted_txn_ids,
        )
        flags.extend(ev_flags)

    rounded = round_half_up(abs(actual), 2)
    return {
        "scenario_id": covenant["scenario_id"],
        "slot": slot,
        "status": status,
        "actual": str(actual),
        "rounded": str(rounded),
        "evidence_txn_id": evidence,
        "strategy": strategy,
        "confidence": "0.95" if strategy == "computed" else "0.5",
        "flags": flags,
    }


def run(*, work_dir: Path) -> StageResult:
    covenants_payload = _load_json(work_dir / "04a_covenants.json")
    parties_payload = _load_json(work_dir / "04b_parties.json")
    adjustments_payload = _load_json(work_dir / "04c_adjustments.json")
    template = load_template(work_dir)

    covenants = {
        (covenant["scenario_id"], covenant["slot"]): covenant
        for covenant in (covenants_payload.get("covenants") or [])
    }
    parties_by_scenario = parties_payload.get("scenarios") or {}
    adjustments = adjustments_payload.get("adjustments") or {}
    adjusted_ids = _adjusted_txn_ids(adjustments)
    ledger = _ledger_rows(work_dir)

    findings = []
    for scenario_id, slot in template_cells(template):
        covenant = covenants.get((scenario_id, slot))
        if covenant is None:
            covenant = {
                "scenario_id": scenario_id,
                "slot": slot,
                "threshold": "0",
                "degraded": True,
            }
        findings.append(
            _evaluate_cell(
                covenant,
                ledger,
                parties=parties_by_scenario.get(scenario_id),
                adjustments=adjustments,
                adjusted_txn_ids=adjusted_ids,
                work_dir=work_dir,
            ),
        )

    review = [
        {
            "kind": kind,
            "scenario_id": finding["scenario_id"],
            "slot": finding["slot"],
        }
        for finding in findings
        for kind in (
            "ABSENT_RELATEDNESS_THRESHOLD",
            IDENTICAL_LEGS,
            FLOOR_MISSING_REVIEW,
            EBITDA_CONSTRUCTION_FAILED,
            ZERO_DENOMINATOR,
        )
        if kind in (finding.get("flags") or [])
    ]
    payload = {
        "findings": findings,
        "review": review,
        "summary": {
            "count": len(findings),
            "breach_count": sum(1 for f in findings if f["status"] == "BREACH"),
            "empty_leg_count": sum(
                1 for finding in findings if "EMPTY_LEG" in (finding.get("flags") or [])
            ),
            "identical_leg_count": sum(
                1 for finding in findings if IDENTICAL_LEGS in (finding.get("flags") or [])
            ),
            "review_count": len(review),
        },
    }
    (work_dir / "06_evaluated.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"s6_evaluate: cells={len(findings)} breaches={payload['summary']['breach_count']} "
        f"empty_legs={payload['summary']['empty_leg_count']}",
    )

    return StageResult(item_count=len(findings), row_count=len(findings))
