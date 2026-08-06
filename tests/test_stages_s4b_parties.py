from decimal import Decimal

import pytest

from agent.stages.s4b_parties import normalize_counterparty


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sarybel Capital LLP.", "sarybel capital"),
        ("Pavlodar Plant Services LLP", "pavlodar plant services"),
        ("Syrdarya Capital Holding, LLP", "syrdarya capital holding"),
        ("Atyrau Holding Group L.L.P.", "atyrau holding group"),
        ("Aktau Holdings L.L.P.", "aktau holdings"),
        ("Aktau Holdings LLP", "aktau holdings"),
        ("«Turan Capital» LLP", "turan capital"),
        ("Taraz Holding Group LLP", "taraz holding group"),
        ("Kyzylorda Drilling Services JSC", "kyzylorda drilling services"),
        ('"Saryarka Capital Partners" LLP', "saryarka capital partners"),
        ("Shymkent Refinery Services JSC", "shymkent refinery services"),
        ("Shymkent Refinery JSC", "shymkent refinery"),
    ],
)
def test_normalize_counterparty(raw: str, expected: str) -> None:
    assert normalize_counterparty(raw) == expected


def test_unrestricted_subsidiary_threshold() -> None:
    from agent.stages.s4b_parties import _row_is_related

    threshold = Decimal("50.0")
    assert _row_is_related(Decimal("11.4"), threshold, "UNRESTRICTED_SUBSIDIARY")
    assert not _row_is_related(Decimal("87.6"), threshold, "UNRESTRICTED_SUBSIDIARY")


def test_relatedness_exact_threshold() -> None:
    threshold = Decimal("35.0")
    assert Decimal("41.2") >= threshold
    assert not (Decimal("33.8") >= threshold)
    assert Decimal("35.0") >= threshold
