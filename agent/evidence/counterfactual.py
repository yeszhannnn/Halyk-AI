"""Leave-one-out counterfactual evidence search."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agent.metrics.engine import breaches, compute_covenant_metric, relevant_row_indices


def find_evidence(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    status: str,
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
    adjusted_txn_ids: set[str],
) -> tuple[str | None, list[str]]:
    """Return (evidence_txn_id, flags)."""
    if status != "BREACH":
        return None, []

    threshold = Decimal(str(covenant["threshold"]))
    direction = covenant["direction"]
    indices = relevant_row_indices(covenant, ledger, parties=parties)
    flipping: list[str] = []

    for index in indices:
        row = ledger[index]
        txn_id = row.get("txn_id")
        if not txn_id:
            continue
        reduced = [r for i, r in enumerate(ledger) if i != index]
        reduced_value = compute_covenant_metric(
            covenant,
            reduced,
            parties=parties,
            adjustments=adjustments,
        )
        reduced_status = "BREACH" if breaches(reduced_value, direction, threshold) else "COMPLIANT"
        if reduced_status != status:
            flipping.append(str(txn_id))

    if len(flipping) == 1:
        return flipping[0], []
    if len(flipping) == 0:
        return None, []
    adjusted = [txn for txn in flipping if txn in adjusted_txn_ids]
    if len(adjusted) == 1:
        return adjusted[0], []
    return None, ["MULTIPLE_FLIPPING_TXNS"]
