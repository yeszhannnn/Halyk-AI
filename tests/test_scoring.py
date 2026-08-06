from decimal import Decimal

import pytest

from eval.score import score_cell, score_submission


def test_status_mismatch_returns_zero():
    answer = {"status": "COMPLIANT", "actual": 1.0, "evidence_txn_id": None}
    key = {"status": "BREACH", "actual": 1.0, "evidence_txn_id": None}
    assert score_cell(answer, key) == 0.0


def test_exact_match_returns_one():
    answer = {
        "status": "COMPLIANT",
        "actual": 283664.18,
        "evidence_txn_id": None,
    }
    key = {
        "status": "COMPLIANT",
        "actual": 283664.18,
        "evidence_txn_id": None,
    }
    assert score_cell(answer, key) == pytest.approx(1.0)


def test_two_and_half_percent_error_halves_actual_points():
    key_actual = 100.0
    answer_actual = 102.5  # 2.5% high
    answer = {
        "status": "COMPLIANT",
        "actual": answer_actual,
        "evidence_txn_id": None,
    }
    key = {
        "status": "COMPLIANT",
        "actual": key_actual,
        "evidence_txn_id": None,
    }
    # 0.50 + 0.30*0.5 + 0.20*0.5 = 0.75
    assert score_cell(answer, key) == pytest.approx(0.75)
    actual_component = 0.30 * 0.5
    assert actual_component == pytest.approx(0.15)


def test_missing_actual_zeros_actual_and_evidence_when_key_evidence_none():
    answer = {"status": "COMPLIANT", "actual": None, "evidence_txn_id": None}
    key = {
        "status": "COMPLIANT",
        "actual": 283664.18,
        "evidence_txn_id": None,
    }
    assert score_cell(answer, key) == pytest.approx(0.50)


def test_evidence_exact_match_adds_twenty_points():
    answer = {
        "status": "BREACH",
        "actual": 1.68,
        "evidence_txn_id": "TXN-B1-0020",
    }
    key = {
        "status": "BREACH",
        "actual": 1.68,
        "evidence_txn_id": "TXN-B1-0020",
    }
    assert score_cell(answer, key) == pytest.approx(1.0)


def test_wrong_evidence_gets_no_evidence_points():
    answer = {
        "status": "BREACH",
        "actual": 1.68,
        "evidence_txn_id": "TXN-B1-0001",
    }
    key = {
        "status": "BREACH",
        "actual": 1.68,
        "evidence_txn_id": "TXN-B1-0020",
    }
    assert score_cell(answer, key) == pytest.approx(0.80)


def test_non_numeric_actual_treated_as_missing():
    answer = {
        "status": "COMPLIANT",
        "actual": "not-a-number",
        "evidence_txn_id": None,
    }
    key = {
        "status": "COMPLIANT",
        "actual": 100.0,
        "evidence_txn_id": None,
    }
    assert score_cell(answer, key) == pytest.approx(0.50)


def test_score_submission_all_compliant_null_actual(tmp_path):
    import json
    from pathlib import Path

    ground_truth_path = Path(__file__).resolve().parents[1] / "eval" / "ground_truth.json"
    with ground_truth_path.open(encoding="utf-8") as f:
        ground_truth = json.load(f)

    answers = {}
    for scenario_id, scenario in ground_truth["scenarios"].items():
        answers[scenario_id] = {}
        for slot in scenario["covenants"]:
            answers[scenario_id][slot] = {
                "status": "COMPLIANT",
                "actual": None,
                "evidence_txn_id": None,
            }

    submission = {
        "team": "test",
        "contact_email": "test@example.com",
        "model": "test",
        "answers": answers,
    }
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")

    report = score_submission(str(submission_path), str(ground_truth_path))
    assert report["status_correct"] == 19
    assert report["total_cells"] == 36
    assert report["total_score"] == pytest.approx(9.5, abs=0.01)
