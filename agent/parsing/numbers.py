import contextlib
import contextvars
import logging
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

logger = logging.getLogger(__name__)

_TRAILING_MULTIPLIER_RE = re.compile(r"[xх×X]\s*$")
_PERCENT_SUFFIX_RE = re.compile(r"%\s*$")

# Strings the model uses to signal "value not present in the document".
# Matched case-insensitively after whitespace normalisation.
ABSENT_SENTINELS: frozenset[str] = frozenset(
    {
        "",
        "<unknown>",
        "unknown",
        "n/a",
        "null",
        "none",
        "not found",
        "не найдено",
    }
)

ABSENT_SENTINEL_MESSAGE = "absent sentinel value"

# Stage code installs a sink here to learn which fields were coerced to None.
_absent_sink: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "absent_value_sink",
    default=None,
)


class AbsentValueError(ValueError):
    """Raised when a required numeric field comes back as an absence sentinel."""

    def __init__(self, field_name: str, raw: Any) -> None:
        super().__init__(f"{ABSENT_SENTINEL_MESSAGE} for {field_name}: {raw!r}")
        self.field_name = field_name
        self.raw = raw


def is_absent_sentinel(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return " ".join(value.split()).casefold() in ABSENT_SENTINELS


def _record_absent(field_name: str) -> None:
    sink = _absent_sink.get()
    if sink is not None:
        sink.append(field_name)
    logger.warning("absent sentinel coerced to None for field %s", field_name)


@contextlib.contextmanager
def capture_absent_values():
    """Collect field names coerced to None by sentinel values within the block."""
    sink: list[str] = []
    token = _absent_sink.set(sink)
    try:
        yield sink
    finally:
        _absent_sink.reset(token)


def exception_mentions_absent_sentinel(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if ABSENT_SENTINEL_MESSAGE in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


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
        if is_absent_sentinel(cleaned):
            raise AbsentValueError(field_name, value)
        cleaned = _TRAILING_MULTIPLIER_RE.sub("", cleaned).strip()
        cleaned = _PERCENT_SUFFIX_RE.sub("", cleaned).strip()
        return parse_money(cleaned)
    raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}")


def normalize_optional_decimal(value: Any, *, field_name: str = "value") -> Decimal | None:
    """Parse model output into Decimal, treating absence sentinels as None."""
    if value is None:
        return None
    if isinstance(value, str) and is_absent_sentinel(value):
        _record_absent(field_name)
        return None
    return normalize_decimal(value, field_name=field_name)


def round_half_up(value, places: int) -> Decimal:
    """Round to a fixed number of decimal places using ROUND_HALF_UP."""
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    quantizer = Decimal("1").scaleb(-places)
    return number.quantize(quantizer, rounding=ROUND_HALF_UP)
