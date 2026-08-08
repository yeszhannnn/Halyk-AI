from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field

from agent.llm.client import _cache_key
from agent.llm.extraction_vote import (
    EXTRACTION_UNSTABLE,
    collect_leaf_paths,
    get_at_path,
    set_at_path,
    vote_fields,
)


class _Metric(BaseModel):
    category: str
    threshold: Decimal


class _Covenant(BaseModel):
    direction: str
    threshold: Decimal
    metric: _Metric


def test_cache_key_differs_by_pass_index() -> None:
    base_kwargs = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "params": {"temperature": 0},
    }
    voted_key = _cache_key(**base_kwargs)
    pass_keys = [_cache_key(**base_kwargs, pass_index=index) for index in range(3)]
    assert len(set(pass_keys)) == 3
    assert voted_key not in pass_keys


def test_vote_fields_majority_per_field_not_object() -> None:
    pass_a = _Covenant(
        direction="MAX",
        threshold=Decimal("0.42"),
        metric=_Metric(category="ebitda", threshold=Decimal("0.42")),
    )
    pass_b = _Covenant(
        direction="MAX",
        threshold=Decimal("0.42"),
        metric=_Metric(category="interest", threshold=Decimal("0.42")),
    )
    pass_c = _Covenant(
        direction="MIN",
        threshold=Decimal("0.42"),
        metric=_Metric(category="opex", threshold=Decimal("0.42")),
    )

    voted, unstable = vote_fields([pass_a, pass_b, pass_c], _Covenant)

    assert voted.threshold == Decimal("0.42")
    assert voted.direction == "MAX"
    assert voted.metric.category == "ebitda"
    assert {entry["field"] for entry in unstable} == {"metric.category"}


def test_vote_fields_all_three_differ_keeps_first_and_flags() -> None:
    pass_a = _Covenant(
        direction="MAX",
        threshold=Decimal("0.40"),
        metric=_Metric(category="a", threshold=Decimal("0.40")),
    )
    pass_b = _Covenant(
        direction="MIN",
        threshold=Decimal("0.41"),
        metric=_Metric(category="b", threshold=Decimal("0.41")),
    )
    pass_c = _Covenant(
        direction="MAX",
        threshold=Decimal("0.42"),
        metric=_Metric(category="c", threshold=Decimal("0.42")),
    )

    voted, unstable = vote_fields([pass_a, pass_b, pass_c], _Covenant)

    assert voted.threshold == Decimal("0.40")
    assert voted.metric.category == "a"
    assert {entry["field"] for entry in unstable} == {
        "threshold",
        "metric.category",
        "metric.threshold",
    }
    assert all(entry["kind"] == EXTRACTION_UNSTABLE for entry in unstable)
    threshold_entry = next(entry for entry in unstable if entry["field"] == "threshold")
    assert threshold_entry["pass_0"] == "0.40"


class _Row(BaseModel):
    amount: Decimal


class _VisionItem(BaseModel):
    kind: str
    materiality_floor: Decimal | None = None
    ebitda_rows: list[_Row] = Field(default_factory=list)


class _VisionExtract(BaseModel):
    items: list[_VisionItem]


def test_vote_fields_handles_nested_list_paths() -> None:
    passes = [
        _VisionExtract(
            items=[
                _VisionItem(kind="EBITDA_ADDBACK", materiality_floor=Decimal("1000"), ebitda_rows=[]),
            ],
        ),
        _VisionExtract(
            items=[
                _VisionItem(kind="EBITDA_ADDBACK", materiality_floor=Decimal("1000"), ebitda_rows=[]),
                _VisionItem(kind="NONE", materiality_floor=None, ebitda_rows=[]),
            ],
        ),
        _VisionExtract(
            items=[
                _VisionItem(kind="EBITDA_ADDBACK", materiality_floor=Decimal("2000"), ebitda_rows=[]),
            ],
        ),
    ]

    voted, unstable = vote_fields(passes, _VisionExtract)

    assert voted.items[0].kind == "EBITDA_ADDBACK"
    assert voted.items[0].materiality_floor == Decimal("1000")
    assert len(voted.items) == 1
    assert not any(entry["field"] == "items[0].materiality_floor" for entry in unstable)


def test_path_helpers_round_trip() -> None:
    payload = {"items": [{"kind": "FX", "amount": "10"}]}
    paths = collect_leaf_paths(payload)
    assert "items[0].kind" in paths
    assert get_at_path(payload, "items[0].kind") == "FX"
    set_at_path(payload, "items[0].kind", "EBITDA_ADDBACK")
    assert get_at_path(payload, "items[0].kind") == "EBITDA_ADDBACK"
