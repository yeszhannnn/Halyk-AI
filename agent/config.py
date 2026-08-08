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
TPM_LIMIT = int(os.getenv("TPM_LIMIT", "150000"))
BUDGET_USD = Decimal(os.getenv("BUDGET_USD", "50.00"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
_PROVIDER_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}
if LLM_PROVIDER not in _PROVIDER_MODELS:
    supported = ", ".join(sorted(_PROVIDER_MODELS))
    raise ValueError(f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; expected one of: {supported}")

MODEL_ID = _PROVIDER_MODELS[LLM_PROVIDER]
TEMPERATURE = 0
OPENAI_SEED = int(os.getenv("OPENAI_SEED", "42"))
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))

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
