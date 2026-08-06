"""Generate all-COMPLIANT / null-actual submission and verify scorer baseline."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from eval.score import score_submission
GT_PATH = ROOT / "ground_truth.json"
SUB_PATH = ROOT / "all_compliant_null_actual.json"


def main() -> None:
    with GT_PATH.open(encoding="utf-8") as handle:
        ground_truth = json.load(handle)

    answers = {
        scenario_id: {
            slot: {"status": "COMPLIANT", "actual": None, "evidence_txn_id": None}
            for slot in scenario["covenants"]
        }
        for scenario_id, scenario in ground_truth["scenarios"].items()
    }

    submission = {
        "team": "baseline",
        "contact_email": "test@example.com",
        "model": "all-compliant-null-actual",
        "answers": answers,
    }
    SUB_PATH.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    print(f"Written: {SUB_PATH}")
    print()

    report = score_submission(SUB_PATH, GT_PATH)
    print()
    print("--- ASSERT ---")

    if report["status_correct"] != 19:
        raise SystemExit(
            f"FAIL: status_correct={report['status_correct']}, expected 19"
        )
    if report["total_cells"] != 36:
        raise SystemExit(f"FAIL: total_cells={report['total_cells']}, expected 36")
    if abs(report["total_score"] - 9.5) >= 0.01:
        raise SystemExit(
            f"FAIL: total_score={report['total_score']}, expected 9.5"
        )

    print("OK: 19/36 statuses, total score 9.50")


if __name__ == "__main__":
    main()
