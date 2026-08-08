from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.llm.schemas.covenants import (
    CategorySpecExtract,
    LEDGER_CATEGORIES_CONTEXT_KEY,
)


def test_include_keywords_rejects_unknown_categories() -> None:
    context = {LEDGER_CATEGORIES_CONTEXT_KEY: ["revenue", "capex", "opex"]}
    with pytest.raises(ValidationError, match="invalid: \\['совокупные капитальные затраты'\\]"):
        CategorySpecExtract.model_validate(
            {
                "include_keywords": ["совокупные капитальные затраты"],
                "include_keywords_quote": "quote",
                "apply_reclass": True,
                "apply_reclass_quote": "quote",
            },
            context=context,
        )


def test_include_keywords_accepts_ledger_categories() -> None:
    context = {LEDGER_CATEGORIES_CONTEXT_KEY: ["revenue", "capex", "opex"]}
    spec = CategorySpecExtract.model_validate(
        {
            "include_keywords": ["revenue", "capex"],
            "include_keywords_quote": "quote",
            "apply_reclass": True,
            "apply_reclass_quote": "quote",
        },
        context=context,
    )
    assert spec.include_keywords == ["revenue", "capex"]


def test_include_keywords_exempts_related_party_quote() -> None:
    context = {LEDGER_CATEGORIES_CONTEXT_KEY: ["revenue", "capex", "opex"]}
    spec = CategorySpecExtract.model_validate(
        {
            "include_keywords": [],
            "include_keywords_quote": "платежей в пользу связанных сторон",
            "apply_reclass": False,
            "apply_reclass_quote": "quote",
        },
        context=context,
    )
    assert spec.include_keywords == []
