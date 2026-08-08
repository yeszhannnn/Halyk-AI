"""Dual-pass vision extraction and numeric structure comparison."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, TypeVar

from pydantic import BaseModel

from agent.llm.client import LLMClient, REPLAY_DIR

T = TypeVar("T", bound=BaseModel)

DUAL_PASS_DELAY_SECONDS = 1.0
VOTE_PASS_DELAY_SECONDS = DUAL_PASS_DELAY_SECONDS


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, Decimal)):
        return True
    if isinstance(value, str):
        try:
            Decimal(value)
        except Exception:
            return False
        return True
    return False


def _normalize_numeric(value: Any) -> Decimal:
    return Decimal(str(value))


def collect_numeric_paths(obj: Any, *, prefix: str = "") -> dict[str, Decimal]:
    paths: dict[str, Decimal] = {}
    if isinstance(obj, BaseModel):
        return collect_numeric_paths(obj.model_dump(mode="python"), prefix=prefix)
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.update(collect_numeric_paths(value, prefix=child))
        return paths
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            child = f"{prefix}[{index}]"
            paths.update(collect_numeric_paths(value, prefix=child))
        return paths
    if _is_numeric(obj):
        paths[prefix or "$"] = _normalize_numeric(obj)
    return paths


def digit_mismatches_between(
    left: BaseModel,
    right: BaseModel,
    *,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    left_paths = collect_numeric_paths(left)
    right_paths = collect_numeric_paths(right)
    mismatches: list[dict[str, Any]] = []
    for path in sorted(set(left_paths) | set(right_paths)):
        left_value = left_paths.get(path)
        right_value = right_paths.get(path)
        if left_value == right_value:
            continue
        entry: dict[str, Any] = {
            "kind": "DIGIT_MISMATCH",
            "field": path,
            "pass_a": str(left_value) if left_value is not None else None,
            "pass_b": str(right_value) if right_value is not None else None,
        }
        if context:
            entry.update(context)
        mismatches.append(entry)
    return mismatches


async def complete_vision_dual(
    client: LLMClient,
    *,
    response_model: type[T],
    prompt: str,
    image_paths: list[Any],
    system_prompt: str | None = None,
    context: dict[str, Any] | None = None,
    **params: Any,
) -> tuple[T, list[dict[str, Any]]]:
    """Run identical vision extraction twice and flag differing numeric fields."""
    from pathlib import Path

    paths = [Path(path) for path in image_paths]

    if REPLAY_DIR is not None:
        result = await client.complete_vision(
            response_model=response_model,
            prompt=prompt,
            image_paths=paths,
            system_prompt=system_prompt,
            **params,
        )
        return result, []

    pass_a = await client.complete_vision(
        response_model=response_model,
        prompt=prompt,
        image_paths=paths,
        system_prompt=system_prompt,
        use_cache=False,
        **params,
    )
    await asyncio.sleep(DUAL_PASS_DELAY_SECONDS)
    pass_b = await client.complete_vision(
        response_model=response_model,
        prompt=prompt,
        image_paths=paths,
        system_prompt=system_prompt,
        use_cache=False,
        **params,
    )
    return pass_a, digit_mismatches_between(pass_a, pass_b, context=context)
