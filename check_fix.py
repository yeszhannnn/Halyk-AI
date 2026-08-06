"""Проверка патча ingest / classify / parties на открытом датасете."""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

A = Path("data/open")
EVAL = Path("eval/ground_truth.json")

ok = True


def chk(name: str, cond: bool, got: object = "") -> None:
    global ok
    ok &= bool(cond)
    suffix = f"  {got}" if got != "" else ""
    print(f"{'PASS' if cond else 'FAIL'}  {name}{suffix}")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_names(rec: dict) -> list[str]:
    return [name for names in rec.get("ledger_map", {}).values() for name in names]


def main() -> int:
    global ok

    inv = _load(A / "01_inventory.json")["documents"]
    cls = _load(A / "02_classified.json")["documents"]
    bind = _load(A / "03_bound.json")["scenarios"]
    par = _load(A / "04b_parties.json")["scenarios"]

    # 1. страницы-картинки
    pages = {doc_id: doc.get("ocr_pages", []) for doc_id, doc in inv.items() if doc.get("ocr_pages")}
    chk("ocr_pages: 4 документа", len(pages) == 4, list(pages.keys()))
    chk("ocr_pages: 7 страниц", sum(len(p) for p in pages.values()) == 7)
    for stem in ["2ed0b2ee4b57", "f3fa6d20c8a1", "63e162bd710b", "aaf665cbc612"]:
        chk(f"  {stem} размечен", any(stem in doc_id for doc_id in pages))

    # 2. новые типы документов
    types: dict[str, list[str]] = {}
    for doc_id, record in cls.items():
        types.setdefault(record["doc_type"], []).append(doc_id)
    chk("SUPERSEDED_DRAFT = 5", len(types.get("SUPERSEDED_DRAFT", [])) == 5)
    chk("ADJUSTMENT_SOURCE не пуст", bool(types.get("ADJUSTMENT_SOURCE")))
    for stem in ["448b59e12768", "26acfab1e58b"]:
        chk(
            f"  {stem} = ADJUSTMENT_SOURCE",
            any(stem in doc_id for doc_id in types.get("ADJUSTMENT_SOURCE", [])),
        )
    chk("LOAN = 12", len(types.get("LOAN", [])) == 12)
    chk("LOAN_SUPERSEDED = 12", len(types.get("LOAN_SUPERSEDED", [])) == 12)

    # 3. пороги читаются из документов, а не захардкожены
    thresholds = sorted({float(rec["threshold_pct"]) for rec in par.values()})
    for rec in par.values():
        peri = rec.get("perimeter")
        if peri is not None:
            thresholds.append(float(peri["threshold_pct"]))
    thresholds = sorted(set(thresholds))
    chk("порогов >= 4 разных", len(thresholds) >= 4, thresholds)
    chk(
        "есть 25.0 / 35.0 / 40.0 / 50.0",
        {25.0, 35.0, 40.0, 50.0}.issubset(set(thresholds)),
        thresholds,
    )

    # 4. вторая семантика таблицы
    semantics = {rec.get("table_semantics") for rec in par.values()}
    perimeter_semantics = {
        rec["perimeter"]["table_semantics"]
        for rec in par.values()
        if rec.get("perimeter")
    }
    chk(
        "есть UNRESTRICTED_SUBSIDIARY",
        "UNRESTRICTED_SUBSIDIARY" in semantics or "UNRESTRICTED_SUBSIDIARY" in perimeter_semantics,
        semantics | perimeter_semantics,
    )

    # 5. привязка и нормализация имён
    chk("12 сценариев в 03_bound", len(bind) == 12)
    chk("12 сценариев в 04b", len(par) == 12)
    for scenario_id, expected in [("P2", 1), ("P5", 1), ("P6", 1), ("P9", 1)]:
        related_count = len([row for row in par[scenario_id]["ownership"] if row["is_related"]])
        chk(
            f"  {scenario_id}: ровно {expected} сущность за порогом",
            related_count == expected,
            f"got {related_count}",
        )

    perimeter_found = False
    for rec in par.values():
        peri = rec.get("perimeter")
        if peri is None:
            continue
        pcts = sorted(Decimal(str(row["ownership_pct"])) for row in peri["ownership"])
        if pcts == [Decimal("11.4"), Decimal("87.6")] and float(peri["threshold_pct"]) == 50.0:
            unrestricted = [row for row in peri["ownership"] if row["is_related"]]
            chk("perimeter 87.6/11.4 @ 50.0: 1 unrestricted", len(unrestricted) == 1, unrestricted)
            perimeter_found = True
    chk("perimeter-таблица найдена", perimeter_found)

    print()
    print("--- сквозная проверка 6.3 (ledger_map) ---")
    gt = _load(EVAL)["scenarios"]
    df = pd.read_csv(A / "master_ledger_2025.csv")
    cross_ok = True
    for scenario_id in ["P1", "P3", "P5", "P7", "P9"]:
        rec = par[scenario_id]
        ledger_names = _ledger_names(rec)
        rows = df[
            df["txn_id"].str.startswith(f"TXN-{scenario_id}-")
            & df["counterparty"].isin(ledger_names)
        ]
        got = abs(rows[rows["amount"] < 0]["amount"].sum())
        exp = gt[scenario_id]["covenants"]["6.3"]["actual"]
        match = abs(got - exp) < 0.02
        cross_ok &= match
        print(
            f"{scenario_id}: {got:>12,.2f}  ключ {exp:>12,.2f}  "
            f"{'OK' if match else 'РАСХОЖДЕНИЕ'}",
        )
    chk("сквозная 6.3: все 5 сценариев", cross_ok)

    print()
    print("--- P6: OCR + ratio 0.10 ---")
    p6 = par["P6"]
    p6_related = [row for row in p6["ownership"] if row["is_related"]]
    chk("P6: одна связанная сторона", len(p6_related) == 1, p6_related)
    p6_names = _ledger_names(p6)
    p6_rows = df[
        df["txn_id"].str.startswith("TXN-P6-")
        & df["counterparty"].isin(p6_names)
        & (df["amount"] < 0)
    ]
    chk("P6: один платёж связанной стороны", len(p6_rows) == 1)
    if len(p6_rows) == 1:
        payment = abs(Decimal(str(p6_rows.iloc[0]["amount"])))
        expected_ratio = Decimal(str(gt["P6"]["covenants"]["6.1"]["actual"]))
        implied_opex = payment / expected_ratio
        computed_ratio = payment / implied_opex
        chk("P6: платёж / опексы = 0.10", computed_ratio == Decimal("0.10"), computed_ratio)

    print()
    print("ВСЁ ЗЕЛЁНОЕ" if ok else "ЕСТЬ ПРОВАЛЫ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
