import duckdb
from pathlib import Path
ROOT = Path(r"c:\Start Up\Halyk AI/data/open")
con = duckdb.connect()
p5 = con.execute(
    "SELECT txn_id, counterparty, description, amount_usd, category FROM read_parquet(?) WHERE scenario_id='P5' ORDER BY txn_id",
    [str(ROOT / "05_ledger.parquet")],
).df()
con.close()
print(p5.to_string())
print("\nTransfers:")
print(p5[p5["description"].str.contains("transfer", case=False, na=False)].to_string())
