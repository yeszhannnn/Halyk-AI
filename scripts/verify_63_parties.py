"""Verify prompt-6 check: related-party outflows == 6.3 actual (related-party covenants only)."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from agent.stages.s4b_parties import _sum_related_outflows

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "open"

RELATED_MARKERS = ("связан", "related")


def is_related_party_63(cov: dict) -> bool:
    blob = " ".join(
        [
            cov.get("title", ""),
            cov["metric"].get("notes", ""),
            " ".join(cov["metric"]["numerator"].get("include_keywords", [])),
        ],
    ).casefold()
    return cov["metric"]["kind"] == "SUM" and any(marker in blob for marker in RELATED_MARKERS)


def main() -> None:
    parties = json.loads((WORK / "04b_parties.json").read_text(encoding="utf-8"))
    gt = json.loads((WORK / "ground_truth.json").read_text(encoding="utf-8"))
    covenants = json.loads((WORK / "04a_covenants.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(WORK / "master_ledger_2025.csv")

    cov_63 = {
        c["scenario_id"]: c
        for c in covenants["covenants"]
        if c["slot"] == "6.3"
    }

    lines: list[str] = []
    failures: list[str] = []

    for scenario_id in sorted(parties["scenarios"].keys()):
        cov = cov_63[scenario_id]
        expected = Decimal(str(gt["scenarios"][scenario_id]["covenants"]["6.3"]["actual"]))
        rec = parties["scenarios"][scenario_id]
        related = [r for r in rec["ownership"] if r["is_related"]]
        outflows = _sum_related_outflows(
            ledger,
            scenario_id=scenario_id,
            ledger_map=rec["ledger_map"],
        )

        if is_related_party_63(cov):
            ok = outflows == expected
            mark = "OK" if ok else "FAIL"
            lines.append(
                f"[{mark}] {scenario_id}: outflows={outflows} gt_6.3={expected} "
                f"related={[r['counterparty'] for r in related]}",
            )
            if not ok:
                failures.append(scenario_id)
        else:
            lines.append(
                f"[--] {scenario_id}: 6.3 is not related-party metric "
                f"(gt={expected}, related_outflows={outflows}) — skip",
            )

    print("\n".join(lines))

    if failures:
        raise SystemExit(f"FAIL: {', '.join(failures)}")

    print("\nPASS: every related-party 6.3 actual equals sum of related-party outflows.")


if __name__ == "__main__":
    main()
