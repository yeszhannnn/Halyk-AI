"""Fast offline robustness tests using the mini fixture and LLM replay."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agent.llm.client import LLMClient, ReplayMissError, set_record_dir, set_replay_dir
from agent.stages import s1_ingest, s2_classify, s3_bind, s4_extract, s5_ledger, s6_evaluate, s7_emit
from agent.template import load_template, template_cells

ROOT = Path(__file__).resolve().parents[1]
MINI_FIXTURE = ROOT / "tests" / "fixtures" / "mini"
REPLAY_FIXTURE = ROOT / "tests" / "fixtures" / "llm"
META = json.loads((MINI_FIXTURE / "fixture_meta.json").read_text(encoding="utf-8"))
P1 = META["documents"]["P1"]
P2 = META["documents"]["P2"]

STAGE4_PLUS = (
    "04a_covenants.json",
    "04b_parties.json",
    "04c_adjustments.json",
    "05_ledger.parquet",
    "05_ledger.json",
    "06_evaluated.json",
    "trace.json",
    "submission.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_conflict(conflicts: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    if entry not in conflicts:
        conflicts.append(entry)


def _mutate_no_kyc(work_dir: Path) -> None:
    bound = _load_json(work_dir / "03_bound.json")
    bound["scenarios"]["P1"]["kyc"] = None
    _append_conflict(bound["conflicts"], {"kind": "MISSING_KYC", "scenario_id": "P1"})
    _write_json(work_dir / "03_bound.json", bound)


def _mutate_no_audit(work_dir: Path) -> None:
    bound = _load_json(work_dir / "03_bound.json")
    bound["scenarios"]["P1"]["audit_notes"] = None
    _append_conflict(bound["conflicts"], {"kind": "MISSING_AUDIT", "scenario_id": "P1"})
    _write_json(work_dir / "03_bound.json", bound)


def _mutate_extra_scenario(work_dir: Path) -> None:
    template = _load_json(work_dir / "submission_template.json")
    template["answers"]["B99"] = {
        "6.1": {"status": None, "actual": None, "evidence_txn_id": None},
        "6.2": {"status": None, "actual": None, "evidence_txn_id": None},
        "6.3": {"status": None, "actual": None, "evidence_txn_id": None},
    }
    _write_json(work_dir / "submission_template.json", template)


def _mutate_new_slot(work_dir: Path) -> None:
    template = _load_json(work_dir / "submission_template.json")
    for scenario in template["answers"].values():
        if "6.3" in scenario:
            scenario["6.4"] = scenario.pop("6.3")
    _write_json(work_dir / "submission_template.json", template)


def _mutate_unmarked_pdf(work_dir: Path) -> None:
    classified = _load_json(work_dir / "02_classified.json")
    inventory = _load_json(work_dir / "01_inventory.json")
    donor_id = P1["loan"]
    classified["documents"]["deadbeef0001"] = {
        "doc_type": "NOISE",
        "acc_ids": [P1["account"]],
        "unbound": False,
        "file_type": "pdf",
    }
    inventory["documents"]["deadbeef0001"] = copy.deepcopy(inventory["documents"][donor_id])
    inventory["documents"]["deadbeef0001"]["source_path"] = "documents/deadbeef0001.pdf"
    _append_conflict(
        classified.setdefault("conflicts", []),
        {"kind": "UNMARKED_PDF", "doc_id": "deadbeef0001"},
    )
    _write_json(work_dir / "02_classified.json", classified)
    _write_json(work_dir / "01_inventory.json", inventory)


def _mutate_empty_ledger(work_dir: Path) -> None:
    ledger_path = work_dir / "master_ledger_2025.csv"
    ledger = pd.read_csv(ledger_path)
    ledger = ledger[~ledger["txn_id"].astype(str).str.contains("-P1-", regex=False)]
    ledger.to_csv(ledger_path, index=False)
    bound = _load_json(work_dir / "03_bound.json")
    _append_conflict(bound["conflicts"], {"kind": "EMPTY_LEDGER", "scenario_id": "P1"})
    _write_json(work_dir / "03_bound.json", bound)


def _mutate_no_ocr(work_dir: Path) -> None:
    doc_id = P2["kyc"]
    inventory = _load_json(work_dir / "01_inventory.json")
    doc = inventory["documents"][doc_id]
    doc["ocr_pages"] = []
    classified = _load_json(work_dir / "02_classified.json")
    conflicts = classified.setdefault("conflicts", [])
    for page_number, page_text in enumerate(doc["pages"], start=1):
        if len(page_text.strip()) < 100:
            _append_conflict(
                conflicts,
                {"kind": "NO_OCR", "doc_id": doc_id, "page": page_number},
            )
    _write_json(work_dir / "01_inventory.json", inventory)
    _write_json(work_dir / "02_classified.json", classified)


def _mutate_bad_amount(work_dir: Path) -> None:
    ledger_path = work_dir / "master_ledger_2025.csv"
    ledger = pd.read_csv(ledger_path)
    candidates = ledger[ledger["amount"].notna()].head(5)
    to_blank = candidates["txn_id"].astype(str).tolist()[:3]
    ledger.loc[ledger["txn_id"].isin(to_blank), "amount"] = pd.NA
    ledger.to_csv(ledger_path, index=False)


def _mutate_duplicate_loan(work_dir: Path) -> None:
    loan_id = P1["loan"]
    classified = _load_json(work_dir / "02_classified.json")
    inventory = _load_json(work_dir / "01_inventory.json")
    classified["documents"]["cafebabe0001"] = copy.deepcopy(classified["documents"][loan_id])
    inventory["documents"]["cafebabe0001"] = copy.deepcopy(inventory["documents"][loan_id])
    inventory["documents"]["cafebabe0001"]["source_path"] = "documents/cafebabe0001.pdf"
    bound = _load_json(work_dir / "03_bound.json")
    _append_conflict(
        bound["conflicts"],
        {
            "kind": "MULTIPLE_ACTIVE_LOANS",
            "scenario_id": "P1",
            "slot": "loan",
            "doc_ids": [loan_id, "cafebabe0001"],
        },
    )
    _write_json(work_dir / "02_classified.json", classified)
    _write_json(work_dir / "01_inventory.json", inventory)
    _write_json(work_dir / "03_bound.json", bound)


MUTATIONS: dict[str, tuple[str, Callable[[Path], None]]] = {
    "no_kyc": ("MISSING_KYC", _mutate_no_kyc),
    "no_audit": ("MISSING_AUDIT", _mutate_no_audit),
    "extra_scenario": ("EXTRA_SCENARIO", _mutate_extra_scenario),
    "new_slot": ("NEW_SLOT", _mutate_new_slot),
    "unmarked_pdf": ("UNMARKED_PDF", _mutate_unmarked_pdf),
    "empty_ledger": ("EMPTY_LEDGER", _mutate_empty_ledger),
    "no_ocr": ("NO_OCR", _mutate_no_ocr),
    "bad_amount": ("BAD_AMOUNT", _mutate_bad_amount),
    "duplicate_loan": ("MULTIPLE_ACTIVE_LOANS", _mutate_duplicate_loan),
}


def _clear_tail_artifacts(work_dir: Path) -> None:
    for name in STAGE4_PLUS:
        path = work_dir / name
        if path.is_file():
            path.unlink()


def _run_stages_4_to_7(work_dir: Path) -> None:
    _clear_tail_artifacts(work_dir)
    s4_extract.run(work_dir=work_dir)
    s5_ledger.run(work_dir=work_dir)
    from agent.stages import s4a_covenants

    s4a_covenants.run(work_dir=work_dir)
    s6_evaluate.run(work_dir=work_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    s7_emit.run(
        work_dir=work_dir,
        output_path=work_dir / "submission.json",
        started_at=started_at,
    )


def _conflict_kinds(work_dir: Path) -> set[str]:
    trace = _load_json(work_dir / "trace.json")
    return {conflict["kind"] for conflict in trace.get("conflicts", [])}


def _assert_submission_filled(work_dir: Path) -> None:
    template = load_template(work_dir)
    submission = _load_json(work_dir / "submission.json")
    for scenario_id, slot in template_cells(template):
        cell = submission["answers"][scenario_id][slot]
        assert cell["status"] in {"COMPLIANT", "BREACH"}
        assert cell["actual"] is not None


def _assert_strategies_recorded(work_dir: Path) -> None:
    evaluated = _load_json(work_dir / "06_evaluated.json")
    for finding in evaluated["findings"]:
        assert finding.get("strategy")


@pytest.fixture(scope="module")
def base_work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("mini_base")
    shutil.copytree(MINI_FIXTURE, work, dirs_exist_ok=True)
    s1_ingest.run(input_dir=work, work_dir=work)
    s2_classify.run(work_dir=work)
    s3_bind.run(work_dir=work)
    return work


@pytest.fixture(autouse=True)
def llm_replay_mode() -> None:
    set_record_dir(None)
    set_replay_dir(REPLAY_FIXTURE)
    yield
    set_replay_dir(None)


@pytest.mark.parametrize("case_name", list(MUTATIONS))
def test_robustness_mutation(case_name: str, base_work_dir: Path, tmp_path: Path) -> None:
    expected_conflict, mutate = MUTATIONS[case_name]
    work_dir = tmp_path / case_name
    shutil.copytree(base_work_dir, work_dir)
    mutate(work_dir)

    _run_stages_4_to_7(work_dir)

    _assert_submission_filled(work_dir)
    _assert_strategies_recorded(work_dir)
    assert expected_conflict in _conflict_kinds(work_dir)


def test_replay_raises_on_missing_fixture() -> None:
    from pydantic import BaseModel

    class ProbeModel(BaseModel):
        answer: str

    set_replay_dir(REPLAY_FIXTURE)

    async def _probe() -> None:
        client = LLMClient()
        await client.complete(
            response_model=ProbeModel,
            messages=[
                {
                    "role": "user",
                    "content": "this prompt hash must not exist in replay fixtures",
                },
            ],
            use_cache=False,
        )

    import asyncio

    with pytest.raises(ReplayMissError):
        asyncio.run(_probe())
