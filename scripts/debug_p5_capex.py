import duckdb
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "open"
con = duckdb.connect()
all_rows = con.execute(
    "SELECT amount, description FROM read_csv_auto(?) WHERE amount < 0",
    [str(ROOT / "master_ledger_2025.csv")],
).df()
scenario = con.execute(
    "SELECT scenario_id, category, amount_usd FROM read_parquet(?)",
    [str(ROOT / "05_ledger.parquet")],
).df()
con.close()

def is_capex(desc: str) -> bool:
    d = desc.casefold()
    return "purchase of" in d or ("transfer" in d and "equipment" in d)

all_capex = sum(abs(Decimal(str(a))) for a, d in zip(all_rows["amount"], all_rows["description"]) if is_capex(d))
sc_capex = sum(
    abs(Decimal(str(r.amount_usd)))
    for r in scenario.itertuples()
    if r.category == "capex" and Decimal(str(r.amount_usd)) < 0
)
transfers = sum(
    abs(Decimal(str(r.amount_usd)))
    for r in scenario.itertuples()
    if "transfer" in str(r.category) or (
        "transfer" in str(getattr(r, "description", "")).casefold()
        and Decimal(str(r.amount_usd)) < 0
    )
)

p5 = scenario[scenario["scenario_id"] == "P5"]
rev = sum(abs(Decimal(str(r.amount_usd))) for r in p5.itertuples() if r.category == "revenue" and Decimal(str(r.amount_usd)) > 0)
opex = sum(abs(Decimal(str(r.amount_usd))) for r in p5.itertuples() if r.category == "opex" and Decimal(str(r.amount_usd)) < 0)
ebitda = rev - opex

print("all ledger capex+transfer pattern", float(all_capex))
print("scenario capex slug", float(sc_capex))
print("p5 ebitda narrow", float(ebitda))
print("ratio all/scenario", float(all_capex / ebitda), float(sc_capex / ebitda))
