from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from agent.stages import StageResult

SCENARIO_ID_PATTERN = re.compile(r"^(P\d+|B\d+)$")

DOC_TYPE_SLOTS: dict[str, str] = {
    "LOAN": "loan",
    "AUDIT_NOTES": "audit_notes",
    "KYC": "kyc",
}


def _derive_account_to_scenario(ledger: pd.DataFrame) -> dict[str, str]:
    scenario_ids = ledger["txn_id"].str.split("-").str[1]
    grouped = (
        pd.DataFrame({"account_id": ledger["account_id"], "scenario_id": scenario_ids})
        .groupby("account_id")["scenario_id"]
        .agg(lambda values: values.mode().iloc[0])
    )
    return {
        account_id: scenario_id
        for account_id, scenario_id in grouped.items()
        if SCENARIO_ID_PATTERN.match(scenario_id)
    }


def _empty_scenario_record() -> dict[str, str | None]:
    return {"loan": None, "audit_notes": None, "kyc": None}


def run(*, work_dir: Path) -> StageResult:
    ledger_path = work_dir / "master_ledger_2025.csv"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"ledger not found: {ledger_path}")

    classified_path = work_dir / "02_classified.json"
    classified = json.loads(classified_path.read_text(encoding="utf-8"))

    ledger = pd.read_csv(ledger_path)
    account_to_scenario = _derive_account_to_scenario(ledger)
    scenario_ids = sorted(set(account_to_scenario.values()))

    scenarios: dict[str, dict[str, str | None]] = {
        scenario_id: _empty_scenario_record() for scenario_id in scenario_ids
    }
    scenario_accounts: dict[str, set[str]] = defaultdict(set)
    conflicts: list[dict] = []

    for doc_id, record in classified["documents"].items():
        doc_type = record["doc_type"]
        slot = DOC_TYPE_SLOTS.get(doc_type)
        if slot is None:
            continue

        acc_ids = record.get("acc_ids") or []
        if not acc_ids:
            conflicts.append(
                {
                    "kind": "UNBOUND_DOCUMENT",
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                },
            )
            continue

        mapped_scenarios = {
            acc_id: account_to_scenario.get(acc_id) for acc_id in acc_ids
        }
        unknown = [acc_id for acc_id, scenario_id in mapped_scenarios.items() if scenario_id is None]
        if unknown:
            conflicts.append(
                {
                    "kind": "UNBOUND_DOCUMENT",
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "acc_ids": unknown,
                },
            )
            continue

        unique_scenarios = {scenario_id for scenario_id in mapped_scenarios.values() if scenario_id}
        if len(unique_scenarios) != 1:
            conflicts.append(
                {
                    "kind": "AMBIGUOUS_DOCUMENT_BINDING",
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "acc_ids": acc_ids,
                    "scenarios": sorted(unique_scenarios),
                },
            )
            continue

        scenario_id = next(iter(unique_scenarios))
        for acc_id in acc_ids:
            scenario_accounts[scenario_id].add(acc_id)

        current = scenarios[scenario_id][slot]
        if current is not None and current != doc_id:
            kind = "MULTIPLE_ACTIVE_LOANS" if slot == "loan" else "DUPLICATE_DOCUMENT"
            conflicts.append(
                {
                    "kind": kind,
                    "scenario_id": scenario_id,
                    "slot": slot,
                    "doc_ids": [current, doc_id],
                },
            )
            continue

        scenarios[scenario_id][slot] = doc_id

    for scenario_id in scenario_ids:
        if scenarios[scenario_id]["loan"] is None:
            conflicts.append(
                {
                    "kind": "MISSING_LOAN",
                    "scenario_id": scenario_id,
                },
            )

    bound = {
        "account_to_scenario": account_to_scenario,
        "scenarios": scenarios,
        "scenario_accounts": {
            scenario_id: sorted(accounts)
            for scenario_id, accounts in sorted(scenario_accounts.items())
        },
        "conflicts": conflicts,
    }

    output_path = work_dir / "03_bound.json"
    output_path.write_text(
        json.dumps(bound, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    loans_bound = sum(1 for record in scenarios.values() if record["loan"] is not None)
    print(
        f"bind: scenarios={len(scenario_ids)} loans_bound={loans_bound} "
        f"conflicts={len(conflicts)}",
    )

    return StageResult(item_count=len(scenario_ids), row_count=loans_bound)
