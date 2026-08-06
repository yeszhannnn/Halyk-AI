"""Local scorer against ground truth."""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Any


STATUS_WEIGHT = 0.50
ACTUAL_WEIGHT = 0.30
EVIDENCE_WEIGHT = 0.20
TOLERANCE = 0.05
SLOTS = ("6.1", "6.2", "6.3")


def _is_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, Decimal):
        return True
    return isinstance(value, Real) and not isinstance(value, complex)


def _to_float(value: Any) -> float:
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def relative_error(answer_actual: Any, key_actual: Any) -> float:
    if not _is_number(answer_actual):
        return 1.0
    key_value = abs(_to_float(key_actual))
    if key_value == 0.0:
        return 0.0 if _to_float(answer_actual) == 0.0 else 1.0
    return abs(_to_float(answer_actual) - _to_float(key_actual)) / key_value


def _actual_accuracy_factor(error: float) -> float:
    return max(0.0, 1.0 - error / TOLERANCE)


def _cell_components(answer: dict[str, Any], key: dict[str, Any]) -> tuple[float, float, float]:
    if answer.get("status") != key["status"]:
        return 0.0, 0.0, 0.0

    status_points = STATUS_WEIGHT
    error = relative_error(answer.get("actual"), key["actual"])
    accuracy = _actual_accuracy_factor(error)
    actual_points = ACTUAL_WEIGHT * accuracy

    if key["evidence_txn_id"] is None:
        evidence_points = EVIDENCE_WEIGHT * accuracy
    elif answer.get("evidence_txn_id") == key["evidence_txn_id"]:
        evidence_points = EVIDENCE_WEIGHT
    else:
        evidence_points = 0.0

    return status_points, actual_points, evidence_points


def score_cell(answer: dict[str, Any], key: dict[str, Any]) -> float:
    status_points, actual_points, evidence_points = _cell_components(answer, key)
    return status_points + actual_points + evidence_points


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _empty_answer() -> dict[str, Any]:
    return {"status": None, "actual": None, "evidence_txn_id": None}


def _get_answer(submission: dict[str, Any], scenario_id: str, slot: str) -> dict[str, Any]:
    return submission.get("answers", {}).get(scenario_id, {}).get(slot, _empty_answer())


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    return str(value)


def score_submission(submission_path: str | Path, ground_truth_path: str | Path) -> dict[str, Any]:
    submission = _load_json(submission_path)
    ground_truth = _load_json(ground_truth_path)

    total_score = 0.0
    status_points = 0.0
    actual_points = 0.0
    evidence_points = 0.0
    status_correct = 0
    slot_scores = {slot: 0.0 for slot in SLOTS}
    slot_counts = {slot: 0 for slot in SLOTS}
    imperfect_cells: list[dict[str, Any]] = []

    for scenario_id, scenario in ground_truth["scenarios"].items():
        for slot, key in scenario["covenants"].items():
            answer = _get_answer(submission, scenario_id, slot)
            cell_score = score_cell(answer, key)
            status_component, actual_component, evidence_component = _cell_components(answer, key)

            total_score += cell_score
            status_points += status_component
            actual_points += actual_component
            evidence_points += evidence_component
            slot_scores[slot] += cell_score
            slot_counts[slot] += 1

            if answer.get("status") == key["status"]:
                status_correct += 1

            if cell_score < 1.0:
                imperfect_cells.append(
                    {
                        "scenario": scenario_id,
                        "slot": slot,
                        "score": cell_score,
                        "expected": key,
                        "received": answer,
                    }
                )

    total_cells = sum(slot_counts.values())
    max_total = float(total_cells)
    max_status = STATUS_WEIGHT * total_cells
    max_actual = ACTUAL_WEIGHT * total_cells
    max_evidence = EVIDENCE_WEIGHT * total_cells

    print("=== Scoring Report ===")
    print(f"Total score: {total_score:.2f} / {max_total:.2f}")
    print()
    print("Component accuracy:")
    print(
        f"  status:   {status_points:.2f} / {max_status:.2f} "
        f"({status_correct}/{total_cells} correct)"
    )
    print(f"  actual:   {actual_points:.2f} / {max_actual:.2f}")
    print(f"  evidence: {evidence_points:.2f} / {max_evidence:.2f}")
    print()
    print("Per-slot accuracy:")
    for slot in SLOTS:
        count = slot_counts[slot]
        print(f"  {slot}: {slot_scores[slot]:.2f} / {count:.2f}")
    print()

    if imperfect_cells:
        print("Cells scoring below 1.0:")
        print(f"{'scenario':<8} {'slot':<5} {'score':>6}  expected / received")
        for cell in sorted(imperfect_cells, key=lambda item: (item["scenario"], item["slot"])):
            expected = cell["expected"]
            received = cell["received"]
            print(
                f"{cell['scenario']:<8} {cell['slot']:<5} {cell['score']:6.2f}  "
                f"status={_format_value(expected['status'])} actual={_format_value(expected['actual'])} "
                f"evidence={_format_value(expected['evidence_txn_id'])} / "
                f"status={_format_value(received.get('status'))} "
                f"actual={_format_value(received.get('actual'))} "
                f"evidence={_format_value(received.get('evidence_txn_id'))}"
            )
    else:
        print("All cells scored 1.0.")

    return {
        "total_score": total_score,
        "total_cells": total_cells,
        "status_correct": status_correct,
        "status_points": status_points,
        "actual_points": actual_points,
        "evidence_points": evidence_points,
        "slot_scores": slot_scores,
        "imperfect_cells": imperfect_cells,
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("Usage: python -m eval.score <submission.json> <ground_truth.json>", file=sys.stderr)
        return 2

    score_submission(args[0], args[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
