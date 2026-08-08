from decimal import Decimal

import pytest

from agent.parsing.numbers import (
    AbsentValueError,
    capture_absent_values,
    is_absent_sentinel,
    normalize_optional_decimal,
    parse_money,
    round_half_up,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1 104 663.28", Decimal("1104663.28")),
        ("1,104,663.28", Decimal("1104663.28")),
        ("(918 447.52)", Decimal("-918447.52")),
        ("$83 690.23", Decimal("83690.23")),
        ("273 418,66", Decimal("273418.66")),
        ("1.104.663,28", Decimal("1104663.28")),
        ("0.04", Decimal("0.04")),
        ("-1 500.00", Decimal("-1500.00")),
        ("($142 118.64)", Decimal("-142118.64")),
        ("72\u00a0146.75", Decimal("72146.75")),
        ("$300,000.00", Decimal("300000.00")),
        ("0.42x", Decimal("0.42")),
        ("1.70x", Decimal("1.70")),
        ("41.2%", Decimal("41.2")),
    ],
)
def test_parse_money(text, expected):
    from agent.parsing.numbers import normalize_decimal

    assert normalize_decimal(text) == expected


def test_normalize_decimal_rejects_unparseable() -> None:
    from agent.parsing.numbers import normalize_decimal

    with pytest.raises(ValueError):
        normalize_decimal("not-a-number")


SENTINEL_STRINGS = [
    "<UNKNOWN>",
    "UNKNOWN",
    "unknown",
    "Unknown",
    "N/A",
    "n/a",
    "NULL",
    "null",
    "NONE",
    "None",
    "NOT FOUND",
    "not found",
    "НЕ НАЙДЕНО",
    "не найдено",
    "",
    "   ",
]


@pytest.mark.parametrize("sentinel", SENTINEL_STRINGS)
def test_sentinel_is_detected_case_insensitively(sentinel: str) -> None:
    assert is_absent_sentinel(sentinel)


@pytest.mark.parametrize("sentinel", SENTINEL_STRINGS)
def test_optional_decimal_treats_sentinel_as_absence(sentinel: str) -> None:
    with capture_absent_values() as absent:
        assert normalize_optional_decimal(sentinel, field_name="threshold_pct") is None
    assert absent == ["threshold_pct"]


@pytest.mark.parametrize("sentinel", SENTINEL_STRINGS)
def test_required_decimal_raises_absent_value_error_on_sentinel(sentinel: str) -> None:
    from agent.parsing.numbers import normalize_decimal

    with pytest.raises(AbsentValueError, match="absent sentinel value"):
        normalize_decimal(sentinel, field_name="threshold")


def test_optional_decimal_passes_none_through() -> None:
    assert normalize_optional_decimal(None, field_name="amount") is None


def test_optional_decimal_still_parses_numbers() -> None:
    assert normalize_optional_decimal("35.0", field_name="threshold_pct") == Decimal("35.0")
    assert normalize_optional_decimal(41.2, field_name="threshold_pct") == Decimal("41.2")


def test_optional_decimal_still_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        normalize_optional_decimal("approximately fifty", field_name="amount")


def test_kyc_schema_accepts_sentinel_threshold() -> None:
    from agent.llm.schemas.parties import KycPartiesExtract

    extracted = KycPartiesExtract.model_validate(
        {
            "header_account": "ACC-7805",
            "header_account_quote": "Счёт ACC-7805",
            "ownership_rows": [],
            "table_semantics": "RELATED_PARTY",
            "table_semantics_quote": "rule",
            "threshold_pct": "<UNKNOWN>",
            "threshold_quote": "rule",
        }
    )
    assert extracted.threshold_pct is None


def test_round_half_up_two_places():
    assert round_half_up(Decimal("0.045"), 2) == Decimal("0.05")
    assert round_half_up(Decimal("0.044"), 2) == Decimal("0.04")
    assert round_half_up(Decimal("283664.185"), 2) == Decimal("283664.19")


def test_round_half_up_from_float():
    assert round_half_up(1.005, 2) == Decimal("1.01")
