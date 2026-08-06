from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from agent.evidence.quotes import locate_bbox, normalize_quote_text, verify_quote
from agent.stages import s1_ingest, s2_classify, s3_bind

OPEN_DATA = Path(__file__).resolve().parents[1] / "data" / "open"


@pytest.fixture(scope="module")
def bound_work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("pipeline")
    s1_ingest.run(input_dir=OPEN_DATA, work_dir=work)
    s2_classify.run(work_dir=work)
    shutil.copy(OPEN_DATA / "master_ledger_2025.csv", work / "master_ledger_2025.csv")
    s3_bind.run(work_dir=work)
    return work


def test_bind_twelve_scenarios_one_loan_each(bound_work_dir: Path) -> None:
    bound = json.loads((bound_work_dir / "03_bound.json").read_text(encoding="utf-8"))
    scenarios = bound["scenarios"]
    account_to_scenario = bound["account_to_scenario"]

    assert len(scenarios) == 12
    assert len(account_to_scenario) == 12
    assert all(record["loan"] for record in scenarios.values())
    assert bound["conflicts"] == []

    loan_doc_ids = {record["loan"] for record in scenarios.values()}
    assert len(loan_doc_ids) == 12

    classified = json.loads((bound_work_dir / "02_classified.json").read_text(encoding="utf-8"))
    loan_types = Counter(
        classified["documents"][doc_id]["doc_type"]
        for doc_id in loan_doc_ids
    )
    assert loan_types == {"LOAN": 12}


def test_account_to_scenario_derived_from_ledger(bound_work_dir: Path) -> None:
    bound = json.loads((bound_work_dir / "03_bound.json").read_text(encoding="utf-8"))
    mapping = bound["account_to_scenario"]
    assert set(mapping.values()) == {
        "B1",
        "B4",
        "P1",
        "P10",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
    }
    assert all(account_id.startswith("ACC-") for account_id in mapping)


def test_verify_quote_normalizes_whitespace_and_case() -> None:
    page = "Threshold must not exceed  $4,000,000.00  in 2025"
    assert verify_quote("threshold must not exceed $4,000,000.00", page)
    assert not verify_quote("threshold must not exceed $5,000,000.00", page)
    assert normalize_quote_text("  Foo\n Bar ") == "foo bar"


def test_locate_bbox_finds_verified_quote(bound_work_dir: Path) -> None:
    inventory = json.loads((bound_work_dir / "01_inventory.json").read_text(encoding="utf-8"))
    bound = json.loads((bound_work_dir / "03_bound.json").read_text(encoding="utf-8"))
    loan_doc_id = bound["scenarios"]["P1"]["loan"]
    assert loan_doc_id is not None
    pages = inventory["documents"][loan_doc_id]["pages"]
    quote = pages[0][:40].strip()
    pdf_path = OPEN_DATA / "documents" / f"{loan_doc_id}.pdf"
    assert verify_quote(quote, pages[0])
    bbox = locate_bbox(pdf_path, page=1, quote=quote)
    assert bbox is not None
    assert len(bbox) == 4
