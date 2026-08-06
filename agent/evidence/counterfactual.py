"""Leave-one-out counterfactual evidence search."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agent.metrics.engine import breaches, compute_covenant_metric, relevant_row_indices


def _flipping_txns(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    status: str,
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
) -> tuple[list[str], int]:
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

    return flipping, len(indices)


def _pick_evidence(
    flipping: list[str],
    adjusted_txn_ids: set[str],
) -> tuple[str | None, list[str]]:
    if len(flipping) == 1:
        return flipping[0], []
    if len(flipping) == 0:
        return None, []
    adjusted = [txn for txn in flipping if txn in adjusted_txn_ids]
    if len(adjusted) == 1:
        return adjusted[0], []
    return None, ["MULTIPLE_FLIPPING_TXNS"]


def _search_reason(
    *,
    status: str,
    flipping: list[str],
    result: str | None,
    flags: list[str],
) -> str | None:
    if status != "BREACH":
        return "статус COMPLIANT, улика не требуется"
    if result is not None:
        return None
    if not flipping:
        return "агрегатный лимит, одна операция вердикт не переворачивает"
    if "MULTIPLE_FLIPPING_TXNS" in flags:
        return "несколько операций переворачивают вердикт"
    return "контрфактуал не выделил единственную улику"


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

    flipping, _ = _flipping_txns(
        covenant,
        ledger,
        status=status,
        parties=parties,
        adjustments=adjustments,
    )
    return _pick_evidence(flipping, adjusted_txn_ids)


def evidence_search(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    status: str,
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
    adjusted_txn_ids: set[str],
    evidence_txn_id: str | None = None,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    """Return the evidence_search block for trace.json."""
    if status != "BREACH":
        return {
            "method": "counterfactual",
            "candidates_tested": 0,
            "flipping": [],
            "result": None,
            "reason": _search_reason(status=status, flipping=[], result=None, flags=[]),
        }

    flipping, candidates = _flipping_txns(
        covenant,
        ledger,
        status=status,
        parties=parties,
        adjustments=adjustments,
    )
    result = evidence_txn_id
    search_flags = list(flags or [])
    if result is None:
        result, extra_flags = _pick_evidence(flipping, adjusted_txn_ids)
        search_flags.extend(extra_flags)

    return {
        "method": "counterfactual",
        "candidates_tested": candidates,
        "flipping": flipping,
        "result": result,
        "reason": _search_reason(
            status=status,
            flipping=flipping,
            result=result,
            flags=search_flags,
        ),
    }
