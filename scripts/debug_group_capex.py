import duckdb
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "open"
con = duckdb.connect()
all_rows = con.execute(
    """
    SELECT amount, description
    FROM read_csv_auto(?)
    WHERE CAST(amount AS DOUBLE) < 0
    """,
    [str(ROOT / "master_ledger_2025.csv")],
).df()
con.close()

purchase = Decimal(0)
transfer = Decimal(0)
for amount, desc in zip(all_rows["amount"], all_rows["description"]):
    d = str(desc).casefold()
    val = abs(Decimal(str(amount)))
    if "purchase of" in d or "acquisition of" in d:
        purchase += val
    elif "transfer" in d and "equipment" in d:
        transfer += val

print("purchase", float(purchase))
print("transfer", float(transfer))
print("sum", float(purchase + transfer))
ebitda = Decimal("2312216.15")
print("ratio scenario capex", float((purchase + transfer) / ebitda))

# target numerator for 9.45
target = ebitda * Decimal("9.45")
print("target num", float(target), "gap", float(target - purchase - transfer))
