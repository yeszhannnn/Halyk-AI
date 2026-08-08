"""Infer ledger row categories from English transaction descriptions."""

from __future__ import annotations

import re

# Operating expense slugs used when computing EBITDA-style metrics.
OPEX_SLUGS = frozenset(
    {
        "consulting",
        "insurance",
        "marketing",
        "opex",
        "personnel",
        "rent",
        "utilities",
    }
)

INFLOW_CATEGORIES = frozenset(
    {
        "revenue",
        "financing",
        "interest_income",
    }
)


def category_sign(category: str) -> str:
    """Derive cash-flow sign from a ledger category slug."""
    return "INFLOW" if category in INFLOW_CATEGORIES else "OUTFLOW"


def derive_leg_sign(categories: list[str]) -> str:
    """Aggregate sign for a leg that may include multiple ledger categories."""
    if not categories:
        return "OUTFLOW"
    signs = {category_sign(category) for category in categories}
    if len(signs) == 1:
        return signs.pop()
    return "BOTH"

INTEREST_INCOME_MARKERS = (
    "interest income",
    "interest credited",
    "interest recovery",
)

REVENUE_MARKERS = (
    "sales settlement",
    "distribution sales",
    "handling and stevedoring sales",
    "refrigerated distribution sales",
)

# Financing means funding drawdowns only. Incentives, rebates, credits,
# refunds and recoveries are never financing, regardless of counterparty.
FINANCING_MARKERS = (
    "drawdown",
    "facility drawdown",
)


def infer_category(description: str) -> str:
    """Assign a short English slug from the transaction description."""
    text = " ".join(str(description).split()).casefold()

    if re.search(r"\b(purchase of|acquisition of)\b", text):
        return "capex"
    if "transfer" in text and "equipment" in text:
        return "capex"
    if "management advisory" in text or "advisory engagement" in text:
        return "consulting"
    if "payroll" in text or "personnel" in text:
        return "personnel"
    if "insurance" in text:
        return "insurance"
    if re.search(
        r"\b(tax|levy|excise|withholding tax|franchise tax|vehicle tax|vat instalment)\b",
        text,
    ):
        return "tax"
    if any(marker in text for marker in FINANCING_MARKERS):
        return "financing"
    if "interest" in text or "coupon" in text:
        if any(marker in text for marker in INTEREST_INCOME_MARKERS):
            return "interest_income"
        return "interest"
    if any(marker in text for marker in REVENUE_MARKERS):
        return "revenue"
    if any(
        token in text
        for token in (
            "marketing",
            "media buy",
            "ad campaign",
            "sponsorship",
            "exhibition stand marketing",
            "newsletter marketing",
        )
    ):
        return "marketing"
    if any(
        token in text
        for token in (
            "electricity",
            "water charge",
            "water supply",
            "heating",
            "utility",
            "telecom",
            "compressed air",
            "district heating",
            "waste water",
            "metering charge",
        )
    ):
        return "utilities"
    if " rent" in text or text.startswith("rent ") or " lease" in text or "lease " in text:
        return "rent"
    if (
        "servicing and operating costs" in text
        or "operating and maintenance" in text
        or "servicing contract" in text
        or "servicing agreement" in text
    ):
        return "opex"
    return "other"
