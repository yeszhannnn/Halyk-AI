import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

_TRAILING_MULTIPLIER_RE = re.compile(r"[xх×X]\s*$")
_PERCENT_SUFFIX_RE = re.compile(r"%\s*$")


def parse_money(text: str) -> Decimal:
    """Parse a monetary string into a Decimal."""
    raw = str(text).strip()
    if not raw:
        raise ValueError("empty monetary string")

    s = raw.replace("\u00a0", " ").replace("\u202f", " ")

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        negative = True
        s = s[1:].strip()

    if s.startswith("$"):
        s = s[1:].strip()

    s = s.replace(" ", "")

    last_dot = s.rfind(".")
    last_comma = s.rfind(",")

    if last_dot >= 0 and last_comma >= 0:
        if last_dot > last_comma:
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif last_comma >= 0:
        integer_part, fractional_part = s.rsplit(",", 1)
        if len(fractional_part) <= 2 and fractional_part.isdigit():
            s = f"{integer_part}.{fractional_part}"
        else:
            s = s.replace(",", "")
    elif last_dot >= 0:
        parts = s.split(".")
        if len(parts) > 2:
            if len(parts[-1]) <= 2 and parts[-1].isdigit():
                s = "".join(parts[:-1]) + "." + parts[-1]
            else:
                s = s.replace(".", "")

    try:
        value = Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"cannot parse monetary string: {text!r}") from exc

    return -value if negative else value


def normalize_decimal(value: Any, *, field_name: str = "value") -> Decimal:
    """Parse model output into Decimal without inventing defaults for bad input."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty")
        cleaned = _TRAILING_MULTIPLIER_RE.sub("", cleaned).strip()
        cleaned = _PERCENT_SUFFIX_RE.sub("", cleaned).strip()
        return parse_money(cleaned)
    raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}")


def round_half_up(value, places: int) -> Decimal:
    """Round to a fixed number of decimal places using ROUND_HALF_UP."""
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    quantizer = Decimal("1").scaleb(-places)
    return number.quantize(quantizer, rounding=ROUND_HALF_UP)
