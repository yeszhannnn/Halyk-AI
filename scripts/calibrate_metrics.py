"""Calibrate covenant metrics against ground truth."""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"c:\Start Up\Halyk AI")
sys.path.insert(0, str(ROOT))

from agent.metrics.engine import breaches, compute_covenant_metric
from agent.stages import s5_ledger, s6_evaluate

GT = json.loads((ROOT / "eval" / "ground_truth.json").read_text(encoding="utf-8"))
WD = ROOT / "data" / "open"


def main() -> None:
    s5_ledger.run(work_dir=WD)
    s6_evaluate.run(work_dir=WD)

    covs = json.loads((WD / "04a_covenants.json").read_text(encoding="utf-8"))["covenants"]
    parties = json.loads((WD / "04b_parties.json").read_text(encoding="utf-8"))["scenarios"]
    adjs = json.loads((WD / "04c_adjustments.json").read_text(encoding="utf-8"))["adjustments"]
    import duckdb

    con = duckdb.connect()
    ledger = con.execute("SELECT * FROM read_parquet(?)", [str(WD / "05_ledger.parquet")]).df().to_dict(
        orient="records"
    )
    con.close()

    wrong_status = 0
    for cov in covs:
        sid, slot = cov["scenario_id"], cov["slot"]
        gt = GT["scenarios"][sid]["covenants"][slot]
        val = compute_covenant_metric(
            cov,
            ledger,
            parties=parties.get(sid),
            adjustments=adjs,
            work_dir=WD,
        )
        st = "BREACH" if breaches(val, cov["direction"], Decimal(str(cov["threshold"]))) else "COMPLIANT"
        ok = st == gt["status"]
        if not ok:
            wrong_status += 1
            print(
                f"STATUS {sid} {slot}: got {st} ({float(val):.4f}) "
                f"key {gt['status']} ({gt['actual']})"
            )
    print(f"\nWrong statuses: {wrong_status}/36")


if __name__ == "__main__":
    main()
