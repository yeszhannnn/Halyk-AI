from decimal import Decimal

import pytest

from agent.parsing.numbers import parse_money, round_half_up


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


def test_round_half_up_two_places():
    assert round_half_up(Decimal("0.045"), 2) == Decimal("0.05")
    assert round_half_up(Decimal("0.044"), 2) == Decimal("0.04")
    assert round_half_up(Decimal("283664.185"), 2) == Decimal("283664.19")


def test_round_half_up_from_float():
    assert round_half_up(1.005, 2) == Decimal("1.01")
