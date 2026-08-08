"""Регрессионные тесты парсинга и детекции springing.

Строки взяты дословно из логов провалившихся прогонов 7 августа.
Ключевое требование: непарсящееся падает или возвращает None, но НИКОГДА не 0.
"""

import importlib
from decimal import Decimal as D

import pytest

from agent.parsing.numbers import normalize_decimal, parse_money
from agent.stages.s4a_covenants import _detect_springing_trigger


# --------------------------------------------------------------------------
# 1. Денежные строки — так они напечатаны в документах
# --------------------------------------------------------------------------

MONEY_CASES = [
    ("$300,000.00",            D("300000.00")),
    ("$251,338.94",            D("251338.94")),
    ("$342,905.28",            D("342905.28")),
    ("$481,247.63",            D("481247.63")),
    ("$4,000,000.00",          D("4000000.00")),
    ("$260,000.00",            D("260000.00")),
    ("72,146.75",              D("72146.75")),
    ("83,690.23",              D("83690.23")),
    ("918,447.52",             D("918447.52")),
    ("3 204 881.55",           D("3204881.55")),      # обычный пробел
    ("3\u00a0204\u00a0881.55", D("3204881.55")),      # неразрывный пробел
    ("0.04",                   D("0.04")),
    ("  $1,234.56  ",          D("1234.56")),         # обрамляющие пробелы
]


@pytest.mark.parametrize("raw,expected", MONEY_CASES)
def test_normalize_decimal_parses_money(raw, expected):
    assert normalize_decimal(raw) == expected


@pytest.mark.parametrize("raw,expected", MONEY_CASES)
def test_parse_money_parses_money(raw, expected):
    assert parse_money(raw) == expected


# --------------------------------------------------------------------------
# 2. Коэффициенты с суффиксом x — зона ответственности normalize_decimal.
#    parse_money их отвергает намеренно: это не деньги.
# --------------------------------------------------------------------------

RATIO_CASES = [
    ("0.42x", D("0.42")),
    ("1.70x", D("1.70")),
    ("9.00x", D("9.00")),
    ("3.50x", D("3.50")),
]


@pytest.mark.parametrize("raw,expected", RATIO_CASES)
def test_normalize_decimal_strips_ratio_suffix(raw, expected):
    assert normalize_decimal(raw) == expected


@pytest.mark.parametrize("raw,_expected", RATIO_CASES)
def test_parse_money_rejects_ratio(raw, _expected):
    """Денежный парсер не должен молча съедать коэффициент."""
    with pytest.raises(Exception):
        parse_money(raw)


# --------------------------------------------------------------------------
# 3. Точность: никаких float по дороге
# --------------------------------------------------------------------------

def test_decimal_precision_preserved():
    assert normalize_decimal("0.041177") == D("0.041177")
    assert str(normalize_decimal("$7,004,318.47")) == "7004318.47"
    assert normalize_decimal("$2,321,317.34") == D("2321317.34")


def test_negative_forms():
    """Скобки и минус — отрицательные суммы."""
    assert parse_money("(1 234.56)") == D("-1234.56")
    assert parse_money("-$1,234.56") == D("-1234.56")


# --------------------------------------------------------------------------
# 4. Непарсящееся. Главный блок — вчерашняя регрессия была здесь.
# --------------------------------------------------------------------------

UNPARSEABLE = ["", "   ", "н/д", "не применимо", "N/A", "abc", "—", None]


@pytest.mark.parametrize("raw", UNPARSEABLE)
def test_normalize_decimal_never_invents_zero(raw):
    try:
        result = normalize_decimal(raw)
    except Exception:
        return  # корректно: отвергло
    assert result is None, f"вернуло {result!r} вместо отказа для {raw!r}"


@pytest.mark.parametrize("raw", UNPARSEABLE)
def test_parse_money_never_invents_zero(raw):
    try:
        result = parse_money(raw)
    except Exception:
        return
    assert result is None, f"вернуло {result!r} вместо отказа для {raw!r}"


# --------------------------------------------------------------------------
# 5. Детекция springing до обращения к модели
# --------------------------------------------------------------------------

SPRINGING_PRESENT = [
    "Ограничение применяется только если совокупные поступления по "
    "финансированию превышают $4,000,000.00",
    "Настоящий ковенант применяется при условии, что объём привлечённого "
    "финансирования за период превышает $4,000,000.00",
    "В случае если совокупные поступления превышают установленный порог, "
    "отношение не должно превышать 1.70x",
]

SPRINGING_ABSENT = [
    "Заёмщик обязуется не допускать превышения отношением Долг/EBITDA "
    "значения 3.50x",
    "Выручка Заёмщика за Ковенантный период должна составлять не менее "
    "$7,500,000.00",
    "Совокупный объём платежей в пользу связанных сторон не должен превышать "
    "$260,000.00",
]


@pytest.mark.parametrize("txt", SPRINGING_PRESENT)
def test_springing_detected(txt):
    assert _detect_springing_trigger(txt) is not None


@pytest.mark.parametrize("txt", SPRINGING_ABSENT)
def test_springing_absent(txt):
    assert _detect_springing_trigger(txt) is None


# --------------------------------------------------------------------------
# 6. Согласованность парсеров. В проекте их шесть в разных модулях —
#    расхождение даёт разные результаты между прогонами.
# --------------------------------------------------------------------------

PARSER_PATHS = [
    "agent.parsing.numbers.parse_money",
    "agent.parsing.numbers.normalize_decimal",
    "agent.stages.s4c_adjustments._parse_amount",
]


@pytest.mark.parametrize("path", PARSER_PATHS)
def test_all_parsers_agree_on_money(path):
    module_name, fn_name = path.rsplit(".", 1)
    try:
        fn = getattr(importlib.import_module(module_name), fn_name)
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"{path} недоступен: {exc}")

    assert fn("$300,000.00") == D("300000.00")
    assert fn("72,146.75") == D("72146.75")
    assert fn("$4,000,000.00") == D("4000000.00")