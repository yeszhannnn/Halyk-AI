"""Stage 5 — filter, adjust, and persist the covenant ledger."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from agent.parsing.categories import infer_category
from agent.stages import StageResult

ACTIONABLE_KINDS = (
    "AMOUNT_FILL",
    "EXCLUDE",
    "CUTOFF",
    "RECLASS",
    "FX",
    "OFF_LEDGER",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Decimal(text)


def _scenario_from_txn(txn_id: str) -> str:
    parts = str(txn_id).split("-")
    if len(parts) >= 2:
        return parts[1]
    raise ValueError(f"cannot derive scenario_id from txn_id: {txn_id}")


def _build_scenario_accounts(bind: dict[str, Any]) -> dict[str, list[str]]:
    return bind.get("scenario_accounts") or {}


def _account_to_scenario(bind: dict[str, Any]) -> dict[str, str]:
    mapping = bind.get("account_to_scenario")
    if mapping:
        return mapping
    reverse: dict[str, str] = {}
    for scenario_id, accounts in _build_scenario_accounts(bind).items():
        for account_id in accounts:
            reverse[account_id] = scenario_id
    return reverse


def _apply_adjustments(
    rows: list[dict[str, Any]],
    adjustments: dict[str, Any],
    *,
    scenario_accounts: dict[str, list[str]],
    conflicts: list[dict[str, Any]],
) -> None:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    by_txn: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario_id"], []).append(row)
        if row.get("txn_id"):
            by_txn[str(row["txn_id"])] = row

    fx_rates: dict[str, Decimal] = {}
    for adj_id, adj in adjustments.items():
        if adj.get("kind") != "FX":
            continue
        rate = _decimal_or_none(adj.get("rate"))
        if rate is None:
            source_amt = _decimal_or_none(adj.get("fx_source_amount"))
            settlement = _decimal_or_none(adj.get("fx_settlement_usd"))
            if source_amt and settlement and source_amt != 0:
                rate = settlement / source_amt
        if rate is not None:
            fx_rates[adj["scenario_id"]] = rate

    ordered = sorted(
        adjustments.items(),
        key=lambda item: (
            0
            if item[1].get("kind") == "AMOUNT_FILL"
            else 1
            if item[1].get("kind") in {"EXCLUDE", "CUTOFF"}
            else 2
            if item[1].get("kind") == "RECLASS"
            else 3
            if item[1].get("kind") == "FX"
            else 4
            if item[1].get("kind") == "OFF_LEDGER"
            else 9
        ),
    )

    synthetic_counter = 0
    for adj_id, adj in ordered:
        kind = adj.get("kind")
        if kind not in ACTIONABLE_KINDS:
            continue
        scenario_id = adj["scenario_id"]
        matched_txn = adj.get("matched_txn") or adj.get("txn_id")

        if kind == "AMOUNT_FILL":
            if not matched_txn or matched_txn not in by_txn:
                conflicts.append(
                    {
                        "kind": "ADJUSTMENT_TXN_NOT_FOUND",
                        "adj_id": adj_id,
                        "txn_id": matched_txn,
                    },
                )
                continue
            row = by_txn[matched_txn]
            amount = _decimal_or_none(adj.get("amount"))
            if amount is None:
                raise ValueError(f"AMOUNT_FILL {adj_id}: missing amount")
            signed = -abs(amount)
            row["amount_usd"] = str(signed)
            if adj.get("category"):
                row["category"] = adj["category"]
            row["adjustment_ref"] = adj_id
            continue

        if kind in {"EXCLUDE", "CUTOFF"}:
            if not matched_txn or matched_txn not in by_txn:
                conflicts.append(
                    {
                        "kind": "ADJUSTMENT_TXN_NOT_FOUND",
                        "adj_id": adj_id,
                        "txn_id": matched_txn,
                    },
                )
                continue
            row = by_txn[matched_txn]
            row["excluded"] = True
            row["adjustment_ref"] = adj_id
            continue

        if kind == "RECLASS":
            if not matched_txn or matched_txn not in by_txn:
                conflicts.append(
                    {
                        "kind": "ADJUSTMENT_TXN_NOT_FOUND",
                        "adj_id": adj_id,
                        "txn_id": matched_txn,
                    },
                )
                continue
            row = by_txn[matched_txn]
            if adj.get("to_category"):
                row["category"] = adj["to_category"]
            row["adjustment_ref"] = adj_id
            continue

        if kind == "FX":
            rate = fx_rates.get(scenario_id)
            if rate is None:
                continue
            for row in by_scenario.get(scenario_id, []):
                if row.get("currency") == "EUR" and not row.get("excluded"):
                    amount = _decimal_or_none(row.get("amount_usd"))
                    if amount is not None:
                        row["amount_usd"] = str(abs(amount) * rate if amount < 0 else amount * rate)
                        row["currency"] = "USD"
                        row["adjustment_ref"] = adj_id
            continue

        if kind == "OFF_LEDGER":
            amount = _decimal_or_none(adj.get("amount"))
            if amount is None:
                raise ValueError(f"OFF_LEDGER {adj_id}: missing amount")
            synthetic_counter += 1
            rows.append(
                {
                    "txn_id": None,
                    "date": "2025-12-31",
                    "counterparty": adj.get("counterparty") or "",
                    "description": f"Synthetic off-ledger ({adj_id})",
                    "amount_usd": str(-abs(amount)),
                    "currency": "USD",
                    "category": adj.get("category") or "other",
                    "original_category": adj.get("category") or "other",
                    "scenario_id": scenario_id,
                    "account_id": scenario_accounts.get(scenario_id, [None])[0],
                    "excluded": False,
                    "synthetic": True,
                    "adjustment_ref": adj_id,
                }
            )


def _resolve_null_amounts(
    rows: list[dict[str, Any]],
    adjustments: dict[str, Any],
    conflicts: list[dict[str, Any]],
) -> None:
    for row in rows:
        value = row.get("amount_usd")
        if value is not None and str(value).strip():
            continue
        txn_id = row.get("txn_id")
        conflicts.append(
            {
                "kind": "BAD_AMOUNT",
                "txn_id": txn_id,
                "scenario_id": row.get("scenario_id"),
            },
        )
        filled = False
        for adj in adjustments.values():
            if adj.get("kind") != "AMOUNT_FILL":
                continue
            matched_txn = adj.get("matched_txn") or adj.get("txn_id")
            if matched_txn and str(matched_txn) == str(txn_id):
                amount = _decimal_or_none(adj.get("amount"))
                if amount is not None:
                    row["amount_usd"] = str(-abs(amount))
                    filled = True
                    break
        if not filled:
            row["excluded"] = True
            row["amount_usd"] = "0"


def _assert_no_null_amounts(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        value = row.get("amount_usd")
        if value is None or str(value).strip() == "":
            raise AssertionError(
                f"null amount after adjustments: txn_id={row.get('txn_id')!r} "
                f"scenario={row.get('scenario_id')!r}"
            )
        try:
            Decimal(str(value))
        except InvalidOperation as exc:
            raise AssertionError(
                f"invalid amount after adjustments: txn_id={row.get('txn_id')!r} value={value!r}"
            ) from exc


def run(*, work_dir: Path) -> StageResult:
    bind = _load_json(work_dir / "03_bound.json")
    adjustments_payload = _load_json(work_dir / "04c_adjustments.json")
    adjustments = adjustments_payload.get("adjustments") or {}
    conflicts: list[dict[str, Any]] = []

    account_to_scenario = _account_to_scenario(bind)
    scenario_accounts = _build_scenario_accounts(bind)
    account_ids = list(account_to_scenario.keys())

    ledger_csv = work_dir / "master_ledger_2025.csv"
    if not ledger_csv.exists():
        raise FileNotFoundError(f"ledger not found: {ledger_csv}")

    con = duckdb.connect()
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE raw_ledger AS
            SELECT * FROM read_csv_auto(?, header=true)
            """,
            [str(ledger_csv)],
        )
        placeholders = ", ".join("?" for _ in account_ids)
        filtered = con.execute(
            f"""
            SELECT *
            FROM raw_ledger
            WHERE account_id IN ({placeholders})
            ORDER BY txn_id
            """,
            account_ids,
        ).df()
    finally:
        con.close()

    rows: list[dict[str, Any]] = []
    for record in filtered.to_dict(orient="records"):
        txn_id = str(record["txn_id"])
        scenario_id = account_to_scenario.get(str(record["account_id"])) or _scenario_from_txn(
            txn_id
        )
        category = infer_category(str(record["description"]))
        amount_raw = record.get("amount")
        amount_usd: str | None
        if pd.isna(amount_raw):
            amount_usd = None
        else:
            amount_usd = str(Decimal(str(amount_raw)))

        rows.append(
            {
                "txn_id": txn_id,
                "date": str(record["date"])[:10],
                "counterparty": str(record.get("counterparty") or ""),
                "description": str(record.get("description") or ""),
                "amount_usd": amount_usd,
                "currency": str(record.get("currency") or "USD"),
                "category": category,
                "original_category": category,
                "scenario_id": scenario_id,
                "account_id": str(record["account_id"]),
                "excluded": False,
                "synthetic": False,
                "adjustment_ref": None,
            }
        )

    _apply_adjustments(rows, adjustments, scenario_accounts=scenario_accounts, conflicts=conflicts)
    _resolve_null_amounts(rows, adjustments, conflicts)
    _assert_no_null_amounts(rows)

    output = pd.DataFrame(rows)[
        [
            "txn_id",
            "date",
            "counterparty",
            "description",
            "amount_usd",
            "category",
            "excluded",
            "synthetic",
            "adjustment_ref",
            "scenario_id",
            "currency",
            "original_category",
        ]
    ]
    output["adjustment_ref"] = output["adjustment_ref"].where(
        output["adjustment_ref"].notna(),
        None,
    )
    parquet_path = work_dir / "05_ledger.parquet"
    con = duckdb.connect()
    try:
        con.register("ledger_df", output)
        con.execute(
            "COPY ledger_df TO ? (FORMAT PARQUET)",
            [str(parquet_path)],
        )
    finally:
        con.close()

    meta_path = work_dir / "05_ledger.json"
    meta_path.write_text(
        json.dumps({"conflicts": conflicts}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return StageResult(item_count=len(rows), row_count=len(rows))
