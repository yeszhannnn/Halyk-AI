"""Per-field majority voting across independent extraction passes."""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

EXTRACTION_UNSTABLE = "EXTRACTION_UNSTABLE"
VOTE_PASS_COUNT = 3
VOTE_PASS_DELAY_SECONDS = 1.0


class _Missing:
    __slots__ = ()


MISSING = _Missing()


def _json_normalize(value: Any) -> Any:
    if value is MISSING:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_normalize(item) for item in value]
    return value


def _vote_key(value: Any) -> str:
    return json.dumps(_json_normalize(value), sort_keys=True, ensure_ascii=False, default=str)


def _serialize_variant(value: Any) -> Any:
    if value is MISSING:
        return None
    return _json_normalize(value)


def collect_leaf_paths(obj: Any, *, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(obj, dict):
        if not obj:
            if prefix:
                paths.add(prefix)
            return paths
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.update(collect_leaf_paths(value, prefix=child))
        return paths
    if isinstance(obj, list):
        if not obj:
            if prefix:
                paths.add(prefix)
            return paths
        for index, value in enumerate(obj):
            child = f"{prefix}[{index}]"
            paths.update(collect_leaf_paths(value, prefix=child))
        return paths
    if prefix:
        paths.add(prefix)
    return paths


def _tokenize_path(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    current = ""
    index = 0
    while index < len(path):
        char = path[index]
        if char == ".":
            if current:
                tokens.append(current)
                current = ""
            index += 1
            continue
        if char == "[":
            if current:
                tokens.append(current)
                current = ""
            close = path.index("]", index)
            tokens.append(int(path[index + 1 : close]))
            index = close + 1
            continue
        current += char
        index += 1
    if current:
        tokens.append(current)
    return tokens


def get_at_path(obj: Any, path: str) -> Any:
    current = obj
    for token in _tokenize_path(path):
        if current is MISSING:
            return MISSING
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return MISSING
            current = current[token]
            continue
        if not isinstance(current, dict) or token not in current:
            return MISSING
        current = current[token]
    return current


def _ensure_container(parent: Any, token: str | int) -> Any:
    if isinstance(token, int):
        if not isinstance(parent, list):
            raise TypeError("expected list parent for index token")
        while len(parent) <= token:
            parent.append({})
        return parent[token]
    if token not in parent:
        parent[token] = {}
    return parent[token]


def set_at_path(obj: dict[str, Any], path: str, value: Any) -> None:
    tokens = _tokenize_path(path)
    if not tokens:
        return
    current: Any = obj
    for token in tokens[:-1]:
        current = _ensure_container(current, token)
    last = tokens[-1]
    if isinstance(last, int):
        if not isinstance(current, list):
            raise TypeError("expected list parent for index token")
        while len(current) <= last:
            current.append(None)
        current[last] = copy.deepcopy(value)
        return
    current[last] = copy.deepcopy(value)


def vote_fields(
    passes: list[BaseModel],
    response_model: type[T],
    *,
    context: dict[str, Any] | None = None,
    validation_context: dict[str, Any] | None = None,
) -> tuple[T, list[dict[str, Any]]]:
    """Vote leaf fields independently across extraction passes."""
    if not passes:
        raise ValueError("vote_fields requires at least one pass")
    if len(passes) == 1:
        return passes[0], []

    dumps = [item.model_dump(mode="python") for item in passes]
    result = copy.deepcopy(dumps[0])
    unstable: list[dict[str, Any]] = []

    all_paths: set[str] = set()
    for payload in dumps:
        all_paths.update(collect_leaf_paths(payload))

    for path in sorted(all_paths):
        values = [get_at_path(payload, path) for payload in dumps]
        vote_counts = Counter(_vote_key(value) for value in values)
        winning_key, winning_count = vote_counts.most_common(1)[0]

        if winning_count >= 2:
            chosen_index = next(
                index for index, value in enumerate(values) if _vote_key(value) == winning_key
            )
            chosen = values[chosen_index]
            if chosen is MISSING:
                continue
        else:
            chosen = values[0]
            if chosen is MISSING:
                continue
            entry: dict[str, Any] = {
                "kind": EXTRACTION_UNSTABLE,
                "field": path,
                "pass_0": _serialize_variant(values[0]),
                "pass_1": _serialize_variant(values[1]),
                "pass_2": _serialize_variant(values[2]),
            }
            if context:
                entry.update(context)
            unstable.append(entry)

        set_at_path(result, path, chosen)

    validated = response_model.model_validate(result, context=validation_context)
    return validated, unstable
