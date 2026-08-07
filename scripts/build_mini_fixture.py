"""Build tests/fixtures/mini from the open dataset (P1 + P2)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT / "data" / "open"
OUT = ROOT / "tests" / "fixtures" / "mini"

SCENARIOS = ("P1", "P2")
ROWS_PER_SCENARIO = 15
REQUIRED_TXNS = {
    "P1": ["TXN-P1-0045"],
    "P2": ["TXN-P2-0040"],
}

DOCS = {
    "P1": {
        "loan": "8d878af064f2",
        "kyc": "2dd8671ac405",
        "audit_notes": "896b7933db48",
        "account": "ACC-7801",
    },
    "P2": {
        "loan": "b2519c8e3ea4",
        "kyc": "63e162bd710b",
        "audit_notes": "3fd0d34546b5",
        "account": "ACC-7802",
    },
}


def _cell() -> dict:
    return {"status": None, "actual": None, "evidence_txn_id": None}


def main() -> None:
    bound = json.loads((OPEN / "03_bound.json").read_text(encoding="utf-8"))
    template = json.loads((OPEN / "submission_template.json").read_text(encoding="utf-8"))

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "documents").mkdir(parents=True)

    for scenario_id, docs in DOCS.items():
        for key in ("loan", "kyc", "audit_notes"):
            doc_id = bound["scenarios"][scenario_id][key]
            if doc_id is None:
                raise RuntimeError(f"{scenario_id} missing {key}")
            src = OPEN / "documents" / f"{doc_id}.pdf"
            shutil.copy2(src, OUT / "documents" / f"{doc_id}.pdf")

    ledger = pd.read_csv(OPEN / "master_ledger_2025.csv")
    rows: list[pd.DataFrame] = []
    for scenario_id in SCENARIOS:
        mask = ledger["txn_id"].astype(str).str.contains(f"-{scenario_id}-", regex=False)
        part = ledger[mask]
        required = part[part["txn_id"].isin(REQUIRED_TXNS.get(scenario_id, []))]
        remainder = part[~part["txn_id"].isin(REQUIRED_TXNS.get(scenario_id, []))].head(
            max(0, ROWS_PER_SCENARIO - len(required)),
        )
        rows.append(pd.concat([required, remainder], ignore_index=True))
    mini_ledger = pd.concat(rows, ignore_index=True)
    mini_ledger.to_csv(OUT / "master_ledger_2025.csv", index=False)

    mini_template = {
        "team": template.get("team", ""),
        "contact_email": template.get("contact_email", ""),
        "model": template.get("model", ""),
        "answers": {
            scenario_id: {
                "6.1": _cell(),
                "6.2": _cell(),
                "6.3": _cell(),
            }
            for scenario_id in SCENARIOS
        },
    }
    (OUT / "submission_template.json").write_text(
        json.dumps(mini_template, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    meta = {
        "scenarios": list(SCENARIOS),
        "documents": DOCS,
        "ledger_rows": len(mini_ledger),
        "ocr_doc": DOCS["P2"]["kyc"],
    }
    (OUT / "fixture_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"mini fixture: {len(mini_ledger)} ledger rows, 6 PDFs -> {OUT}")


if __name__ == "__main__":
    main()
