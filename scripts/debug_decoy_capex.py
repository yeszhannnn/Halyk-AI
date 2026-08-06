import duckdb
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "open"
bind = __import__("json").loads((ROOT / "03_bound.json").read_text(encoding="utf-8"))
scenario_accounts = set(bind["account_to_scenario"])

con = duckdb.connect()
rows = con.execute(
    "SELECT account_id, amount, description FROM read_csv_auto(?)",
    [str(ROOT / "master_ledger_2025.csv")],
).df()
con.close()

decoy = Decimal(0)
scenario = Decimal(0)
for account_id, amount, desc in zip(rows["account_id"], rows["amount"], rows["description"]):
    d = str(desc).casefold()
    if "purchase of" not in d and not ("transfer" in d and "equipment" in d):
        continue
    val = abs(Decimal(str(amount)))
    if str(account_id) in scenario_accounts:
        scenario += val
    else:
        decoy += val

print("scenario capex pattern", float(scenario))
print("decoy capex pattern", float(decoy))
print("total", float(scenario + decoy))
print("ratio/2312216", float((scenario + decoy) / Decimal("2312216.15")))
