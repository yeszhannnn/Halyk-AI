"""Leave-one-out counterfactual evidence search."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agent.metrics.engine import breaches, compute_covenant_metric, relevant_row_indices, _d


def _scenario_ledger(ledger: list[dict[str, Any]], scenario_id: str) -> list[dict[str, Any]]:
    return [row for row in ledger if row.get("scenario_id") == scenario_id]


def _evidence_prefix(scenario_id: str) -> str:
    return f"TXN-{scenario_id}-"


def _assert_evidence_prefix(evidence: str | None, scenario_id: str) -> str | None:
    if evidence is None:
        return None
    prefix = _evidence_prefix(scenario_id)
    if not str(evidence).startswith(prefix):
        raise ValueError(
            f"evidence_txn_id {evidence!r} does not start with scenario prefix {prefix!r}",
        )
    return evidence


def _flipping_txns(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    status: str,
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
) -> tuple[list[str], int, dict[str, Decimal]]:
    scenario_id = covenant["scenario_id"]
    scenario_ledger = _scenario_ledger(ledger, scenario_id)
    threshold = Decimal(str(covenant["threshold"]))
    direction = covenant["direction"]
    indices = relevant_row_indices(covenant, scenario_ledger, parties=parties)
    flipping: list[str] = []
    amounts: dict[str, Decimal] = {}

    for index in indices:
        row = scenario_ledger[index]
        txn_id = row.get("txn_id")
        if not txn_id:
            continue
        txn_key = str(txn_id)
        amounts[txn_key] = abs(_d(row.get("amount_usd")))
        reduced = [r for r in ledger if r.get("txn_id") != txn_id]
        reduced_value = compute_covenant_metric(
            covenant,
            reduced,
            parties=parties,
            adjustments=adjustments,
        )
        reduced_status = "BREACH" if breaches(reduced_value, direction, threshold) else "COMPLIANT"
        if reduced_status != status:
            flipping.append(txn_key)

    return flipping, len(indices), amounts


def _pick_evidence(
    flipping: list[str],
    adjusted_txn_ids: set[str],
    amounts: dict[str, Decimal],
) -> tuple[str | None, list[str]]:
    if len(flipping) == 0:
        return None, []
    if len(flipping) == 1:
        return flipping[0], []
    adjusted = [txn for txn in flipping if txn in adjusted_txn_ids]
    if len(adjusted) == 1:
        return adjusted[0], []
    smallest = min(flipping, key=lambda txn: amounts.get(txn, Decimal("Infinity")))
    return smallest, []


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

    flipping, _, amounts = _flipping_txns(
        covenant,
        ledger,
        status=status,
        parties=parties,
        adjustments=adjustments,
    )
    evidence, flags = _pick_evidence(flipping, adjusted_txn_ids, amounts)
    return _assert_evidence_prefix(evidence, covenant["scenario_id"]), flags


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

    flipping, candidates, amounts = _flipping_txns(
        covenant,
        ledger,
        status=status,
        parties=parties,
        adjustments=adjustments,
    )
    result = evidence_txn_id
    search_flags = list(flags or [])
    if result is None:
        result, extra_flags = _pick_evidence(flipping, adjusted_txn_ids, amounts)
        search_flags.extend(extra_flags)

    result = _assert_evidence_prefix(result, covenant["scenario_id"])

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
