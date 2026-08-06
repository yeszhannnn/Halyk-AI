"""Hard-check P5: one related party, outflows match 6.3 actual."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from agent.stages.s4b_parties import _build_ledger_map, _sum_related_outflows, normalize_counterparty

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "open"


def main() -> None:
    parties = json.loads((WORK / "04b_parties.json").read_text(encoding="utf-8"))
    gt = json.loads((WORK / "ground_truth.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(WORK / "master_ledger_2025.csv")

    scenario_id: str | None = None
    for sid, rec in parties["scenarios"].items():
        pcts = sorted(Decimal(str(r["ownership_pct"])) for r in rec["ownership"])
        if pcts == [Decimal("9.4"), Decimal("33.8"), Decimal("41.2")]:
            scenario_id = sid
            break

    if scenario_id is None:
        raise SystemExit("scenario with 41.2/33.8/9.4 not found")

    rec = parties["scenarios"][scenario_id]
    threshold = Decimal(str(rec["threshold_pct"]))
    related = [r for r in rec["ownership"] if r["is_related"]]

    print(f"scenario: {scenario_id}")
    print(f"threshold_pct: {threshold}")
    print(f"related_count: {len(related)}")
    print()

    for row in rec["ownership"]:
        pct = Decimal(str(row["ownership_pct"]))
        flag = pct >= threshold
        ok = flag == row["is_related"]
        mark = "OK" if ok else "FAIL"
        print(
            f"  [{mark}] {row['counterparty']}: {pct}% >= {threshold}% "
            f"=> {flag} (is_related={row['is_related']})",
        )

    print()
    print(f"related_counterparties: {rec['related_counterparties']}")
    print(f"ledger_map: {rec['ledger_map']}")

    outflows = _sum_related_outflows(
        ledger,
        scenario_id=scenario_id,
        ledger_map=rec["ledger_map"],
    )
    expected = Decimal(str(gt["scenarios"][scenario_id]["covenants"]["6.3"]["actual"]))

    print()
    print(f"yearly outflows (related only): {outflows}")
    print(f"ground truth 6.3 actual:       {expected}")
    print(f"match: {outflows == expected}")

    matched_names = {name for names in rec["ledger_map"].values() for name in names}
    rows = ledger[
        (ledger["txn_id"].str.startswith(f"TXN-{scenario_id}-"))
        & (ledger["counterparty"].isin(matched_names))
        & (ledger["amount"] < 0)
    ]
    print(f"\nmatching outflow transactions: {len(rows)}")
    for _, row in rows.iterrows():
        print(f"  {row['txn_id']}  {row['counterparty']}  {row['amount']}")

    pav = next(r for r in rec["ownership"] if "Pavlodar" in r["counterparty"])
    pav_key = normalize_counterparty(pav["counterparty"])
    decoy = ledger[
        (ledger["txn_id"].str.startswith(f"TXN-{scenario_id}-"))
        & (ledger["counterparty"].map(normalize_counterparty) == pav_key)
        & (ledger["amount"] < 0)
    ]
    decoy_sum = abs(Decimal(str(decoy["amount"].sum()))) if len(decoy) else Decimal("0")
    print(f"\ndecoy Pavlodar (33.8%, NOT related) outflows: {decoy_sum} across {len(decoy)} txns")
    if decoy_sum > 0 and decoy_sum != outflows:
        print("  (correctly excluded from related set)")

    assert len(related) == 1, f"expected 1 related party, got {len(related)}"
    assert outflows == expected, f"outflows {outflows} != expected {expected}"
    print("\nPASS: threshold and name matching both work.")


if __name__ == "__main__":
    main()
