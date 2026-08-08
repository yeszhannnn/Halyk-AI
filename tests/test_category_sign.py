from __future__ import annotations

from agent.parsing.categories import category_sign, derive_leg_sign


def test_expense_categories_are_outflow() -> None:
    for category in ("capex", "opex", "personnel", "interest", "tax", "utilities"):
        assert category_sign(category) == "OUTFLOW"


def test_revenue_and_financing_are_inflow() -> None:
    assert category_sign("revenue") == "INFLOW"
    assert category_sign("financing") == "INFLOW"
    assert category_sign("interest_income") == "INFLOW"


def test_mixed_leg_sign_is_both() -> None:
    assert derive_leg_sign(["revenue", "opex"]) == "BOTH"
