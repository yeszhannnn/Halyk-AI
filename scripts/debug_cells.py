import json
import sys
from decimal import Decimal
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1] / "data" / "open"

con = duckdb.connect()
rows = con.execute("SELECT * FROM read_parquet(?)", [str(ROOT / "05_ledger.parquet")]).df()
con.close()

p9 = rows[rows["scenario_id"] == "P9"]
print("P9 capex:")
print(p9[p9["category"] == "capex"][["txn_id", "counterparty", "description", "amount_usd"]].to_string())
print("\nP9 Zhezkazgan counterparties:")
print(p9[p9["counterparty"].str.contains("Zhezkazgan", case=False, na=False)][["txn_id", "counterparty", "description", "amount_usd", "category"]].to_string())

p5 = rows[rows["scenario_id"] == "P5"]
print("\nP5 group/capex:")
print(p5[p5["description"].str.contains("group", case=False, na=False)][["txn_id", "description", "amount_usd", "category"]].to_string())
print(p5[p5["category"] == "capex"][["txn_id", "description", "amount_usd"]].to_string())

p8 = rows[rows["scenario_id"] == "P8"]
print("\nP8 personnel/severance:")
print(p8[p8["category"].isin(["personnel", "severance"]) | p8["synthetic"]][["txn_id", "description", "amount_usd", "category", "synthetic"]].to_string())

p3 = rows[rows["scenario_id"] == "P3"]
print("\nP3 financing inflows:")
print(p3[(p3["category"] == "financing") & (p3["amount_usd"].astype(float) > 0)][["txn_id", "description", "amount_usd"]].to_string())
