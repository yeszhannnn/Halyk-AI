import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

ASTANA_TZ = ZoneInfo("Asia/Almaty")

DEFAULT_INPUT = Path("data/open")
DEFAULT_OUTPUT = Path("submission.json")

DEADLINE = datetime(2026, 8, 9, 23, 59, 59, tzinfo=ASTANA_TZ)
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "eszhan.e4051@gmail.com")

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "3"))
BUDGET_USD = Decimal(os.getenv("BUDGET_USD", "50.00"))

MODEL_ID = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 0
OPENAI_SEED = int(os.getenv("OPENAI_SEED", "42"))

ARTIFACTS = (
    "00_manifest.json",
    "01_inventory.json",
    "02_classified.json",
    "03_bound.json",
    "04a_covenants.json",
    "04b_parties.json",
    "04c_adjustments.json",
    "05_ledger.parquet",
    "06_evaluated.json",
    "trace.json",
)
