from __future__ import annotations

from decimal import Decimal

from agent.stages.s4a_covenants import (
    _extract_threshold_candidates,
    _filter_threshold_candidates_by_unit,
    _format_threshold_candidates,
    _threshold_matches_candidates,
)


def test_format_threshold_candidates() -> None:
    formatted = _format_threshold_candidates(
        [(Decimal("0.42"), "0.42x"), (Decimal("4000000"), "$4,000,000.00")],
    )
    assert "0.42x" in formatted
    assert "$4,000,000.00" in formatted


def test_threshold_unit_filter_prefers_ratio_tokens() -> None:
    text = "threshold 1.70x applies only if drawdowns exceed $4,000,000.00"
    candidates = _extract_threshold_candidates(text)
    ratio_only = _filter_threshold_candidates_by_unit(candidates, "RATIO")
    usd_only = _filter_threshold_candidates_by_unit(candidates, "USD")
    assert len(ratio_only) == 1
    assert ratio_only[0][1] == "1.70x"
    assert len(usd_only) == 1
    assert "$" in usd_only[0][1]


def test_threshold_anchor_rejects_hallucinated_value() -> None:
    text = "Maximum ratio shall not exceed 0.42x during 2025."
    candidates = _extract_threshold_candidates(text)
    assert _threshold_matches_candidates(Decimal("0.42"), candidates)
    assert not _threshold_matches_candidates(Decimal("0.43"), candidates)
