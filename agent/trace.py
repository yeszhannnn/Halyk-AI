"""Build, verify, and project trace.json into submission.json."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from agent.config import CONTACT_EMAIL, LLM_PROVIDER, MODEL_ID, TEMPERATURE
from agent.template import load_template, template_cells
from agent.evidence.counterfactual import evidence_search
from agent.evidence.quotes import verify_quote
from agent.metrics.engine import (
    breaches,
    collect_covenant_inputs,
    compute_covenant_metric,
)
from agent.parsing.numbers import round_half_up

ZERO = Decimal("0")
OFF_LEDGER_KIND = "OFF_LEDGER"
EBITDA_KIND = "EBITDA_ADDBACK"


def _is_adjustment_ref(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.casefold() != "nan"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _d(value: Any) -> Decimal:
    if value is None:
        return ZERO
    return Decimal(str(value))


def _ledger_rows(work_dir: Path) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        df = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(work_dir / "05_ledger.parquet")],
        ).df()
    finally:
        con.close()
    return df.to_dict(orient="records")


def _adjusted_txn_ids(adjustments: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for adj in adjustments.values():
        txn_id = adj.get("matched_txn") or adj.get("txn_id")
        if txn_id and adj.get("kind") in {
            "RECLASS",
            "AMOUNT_FILL",
            "EXCLUDE",
            "CUTOFF",
        }:
            ids.add(str(txn_id))
    return ids


def _adjustments_without_kinds(
    adjustments: dict[str, Any],
    *kinds: str,
) -> dict[str, Any]:
    excluded = set(kinds)
    return {
        adj_id: adj
        for adj_id, adj in adjustments.items()
        if adj.get("kind") not in excluded
    }


def _off_ledger_amount(
    covenant: dict[str, Any],
    adjustments: dict[str, Any],
) -> Decimal:
    scenario_id = covenant["scenario_id"]
    total = ZERO
    for adj in adjustments.values():
        if adj.get("kind") != OFF_LEDGER_KIND:
            continue
        if adj.get("scenario_id") != scenario_id:
            continue
        amount = adj.get("amount")
        if amount is not None:
            total += abs(_d(amount))
    return total


def _adjustments_applied(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    adjustments: dict[str, Any],
) -> list[str]:
    scenario_id = covenant["scenario_id"]
    applied: list[str] = []
    seen: set[str] = set()
    for row in ledger:
        if row.get("scenario_id") != scenario_id:
            continue
        ref = row.get("adjustment_ref")
        if _is_adjustment_ref(ref) and ref not in seen:
            seen.add(str(ref))
            applied.append(str(ref))
    for adj_id, adj in adjustments.items():
        if adj.get("scenario_id") != scenario_id:
            continue
        if adj.get("kind") in {OFF_LEDGER_KIND, EBITDA_KIND} and adj_id not in seen:
            seen.add(adj_id)
            applied.append(adj_id)
    return sorted(applied)


def _metric_expression(covenant: dict[str, Any]) -> str:
    metric = covenant["metric"]
    kind = metric.get("kind", "SUM")
    if kind == "RATIO":
        return "RATIO(numerator/denominator)"
    keywords = metric.get("numerator", {}).get("include_keywords") or []
    if keywords:
        slug = keywords[0].casefold().replace(" ", "_")[:32]
        return f"SUM({slug})"
    return f"SUM({kind.lower()})"


def _format_comparison(
    evaluated: Decimal,
    threshold: Decimal,
    direction: str,
    status: str,
) -> str:
    if direction == "MAX":
        breach_op = ">"
        compliant_op = "<="
    else:
        breach_op = "<"
        compliant_op = ">="
    op = breach_op if status == "BREACH" else compliant_op
    return f"{evaluated} {op} {threshold} → {status}"


def _computation_block(
    covenant: dict[str, Any],
    ledger: list[dict[str, Any]],
    *,
    parties: dict[str, Any] | None,
    adjustments: dict[str, Any],
    evaluated: Decimal,
    rounded: Decimal,
    status: str,
) -> dict[str, str | list[str]]:
    non_synthetic = [row for row in ledger if not row.get("synthetic")]
    reduced_adjustments = _adjustments_without_kinds(
        adjustments,
        OFF_LEDGER_KIND,
        EBITDA_KIND,
    )
    ledger_component = compute_covenant_metric(
        covenant,
        non_synthetic,
        parties=parties,
        adjustments=reduced_adjustments,
    )
    off_ledger = _off_ledger_amount(covenant, adjustments)
    threshold = _d(covenant["threshold"])
    direction = covenant["direction"]
    status_from_evaluated = (
        "BREACH" if breaches(evaluated, direction, threshold) else "COMPLIANT"
    )
    return {
        "expression": _metric_expression(covenant),
        "ledger_component": str(ledger_component),
        "adjustments_applied": _adjustments_applied(covenant, ledger, adjustments),
        "off_ledger_added": str(off_ledger),
        "evaluated": str(evaluated),
        "rounded": str(rounded),
        "comparison": _format_comparison(evaluated, threshold, direction, status),
        "status_from_evaluated": status_from_evaluated,
        "comparison_from_evaluated": _format_comparison(
            evaluated,
            threshold,
            direction,
            status_from_evaluated,
        ),
    }


def _doc_ref(doc_id: str, *, file_type: str = "pdf") -> str:
    if doc_id.endswith(f".{file_type}"):
        return doc_id
    return f"{doc_id}.{file_type}"


def _trace_covenant(covenant: dict[str, Any]) -> dict[str, Any]:
    source = covenant["source"]
    springing = covenant.get("springing")
    springing_payload = None
    if springing:
        springing_payload = {
            "metric": springing.get("metric"),
            "operator": springing.get("operator"),
            "value": str(springing.get("value")),
            "source": {
                "doc": _doc_ref(springing["source"]["doc_id"]),
                "page": springing["source"]["page"],
                "quote": springing["source"]["quote"],
            },
        }
    return {
        "title": covenant["title"],
        "direction": covenant["direction"],
        "threshold": str(covenant["threshold"]),
        "period": list(covenant["period"]),
        "springing": springing_payload,
        "source": {
            "doc": _doc_ref(source["doc_id"]),
            "page": source["page"],
            "quote": source["quote"],
        },
    }


def _trace_adjustments(adjustments: dict[str, Any]) -> dict[str, Any]:
    trace_adjustments: dict[str, Any] = {}
    for adj_id, adj in adjustments.items():
        source = adj.get("source") or {}
        entry: dict[str, Any] = {
            "kind": adj["kind"],
            "source": {
                "doc": _doc_ref(str(source.get("doc_id", ""))),
                "page": source.get("page", 1),
                "quote": source.get("quote", ""),
            },
        }
        if adj.get("amount") is not None:
            entry["amount"] = str(adj["amount"])
        if adj.get("category") is not None:
            entry["category"] = adj["category"]
        if adj.get("counterparty") is not None:
            entry["counterparty"] = adj["counterparty"]
        if adj.get("from_category") is not None:
            entry["from_category"] = adj["from_category"]
        if adj.get("to_category") is not None:
            entry["to_category"] = adj["to_category"]
        if adj.get("matched_txn") is not None:
            entry["matched_txn"] = adj["matched_txn"]
        if adj.get("match_method") is not None:
            entry["match_method"] = adj["match_method"]
        trace_adjustments[adj_id] = entry
    return trace_adjustments


def _aggregate_conflicts(work_dir: Path) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for name in (
        "02_classified.json",
        "03_bound.json",
        "04a_covenants.json",
        "04b_parties.json",
        "04c_adjustments.json",
        "05_ledger.json",
    ):
        path = work_dir / name
        if not path.exists():
            continue
        payload = _load_json(path)
        conflicts.extend(payload.get("conflicts") or [])
    return conflicts


def _aggregate_review(parties_payload: dict[str, Any]) -> list[dict[str, Any]]:
    review: list[dict[str, Any]] = []
    for scenario_id, scenario in (parties_payload.get("scenarios") or {}).items():
        fields = scenario.get("review_fields") or []
        if fields:
            review.append({"scenario_id": scenario_id, "fields": fields})
    return review


def _quotes_rejected(work_dir: Path) -> int:
    rejected = 0
    covenants_path = work_dir / "04a_covenants.json"
    if covenants_path.exists():
        for covenant in _load_json(covenants_path).get("covenants") or []:
            verification = covenant.get("verification") or {}
            rejected += sum(1 for value in verification.values() if value is False)
    adjustments_path = work_dir / "04c_adjustments.json"
    if adjustments_path.exists():
        for adj in (_load_json(adjustments_path).get("adjustments") or {}).values():
            verification = adj.get("verification") or {}
            rejected += sum(1 for value in verification.values() if value is False)
    return rejected


def _dataset_sha256(work_dir: Path) -> str:
    digest = hashlib.sha256()
    ledger = work_dir / "master_ledger_2025.csv"
    if ledger.exists():
        digest.update(ledger.read_bytes())
    documents = work_dir / "documents"
    if documents.is_dir():
        for path in sorted(documents.rglob("*")):
            if path.is_file():
                digest.update(path.name.encode("utf-8"))
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_block(
    work_dir: Path,
    *,
    mode: str,
    started_at: str | None,
    llm_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    classified = _load_json(work_dir / "02_classified.json") if (work_dir / "02_classified.json").exists() else {}
    adjustments_payload = (
        _load_json(work_dir / "04c_adjustments.json") if (work_dir / "04c_adjustments.json").exists() else {}
    )
    bound = _load_json(work_dir / "03_bound.json") if (work_dir / "03_bound.json").exists() else {}
    inventory = _load_json(work_dir / "01_inventory.json") if (work_dir / "01_inventory.json").exists() else {}

    pdf_counts = (classified.get("summary") or {}).get("pdf_counts") or {}
    classified_noise = int(pdf_counts.get("NOISE", 0))
    loans_active = len((bound.get("scenarios") or {}))
    adjustments = adjustments_payload.get("adjustments") or {}
    actionable = [
        adj for adj in adjustments.values() if adj.get("kind") not in {"NONE"}
    ]

    ledger_rows = 0
    ledger_path = work_dir / "05_ledger.parquet"
    if ledger_path.exists():
        ledger_rows = len(_ledger_rows(work_dir))

    stats = llm_stats or {}
    manifest_path = work_dir / "00_manifest.json"
    git_sha = None
    if manifest_path.exists():
        git_sha = _load_json(manifest_path).get("git_sha")

    return {
        "run_id": str(uuid.uuid4()),
        "started_at": started_at or datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "code": {"git_sha": git_sha},
        "inputs": {
            "dataset_sha256": _dataset_sha256(work_dir),
            "pdf_count": int((classified.get("summary") or {}).get("pdf_total", 0)),
            "ledger_rows": ledger_rows,
        },
        "models": [{"provider": LLM_PROVIDER, "id": MODEL_ID, "temperature": TEMPERATURE}],
        "counters": {
            "classified_noise": classified_noise,
            "loans_active": loans_active,
            "adjustments_found": len(actionable),
            "quotes_rejected": _quotes_rejected(work_dir),
        },
        "cost": {"usd": float(stats.get("cost_usd", "0") or 0)},
        "cache": {"hits": int(stats.get("cache_hits", 0))},
    }


def _finding_map(evaluated: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (finding["scenario_id"], finding["slot"]): finding
        for finding in evaluated.get("findings") or []
    }


def build_trace(
    work_dir: Path,
    *,
    mode: str = "full",
    started_at: str | None = None,
    llm_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble trace.json from pipeline artifacts."""
    covenants_payload = _load_json(work_dir / "04a_covenants.json")
    parties_payload = _load_json(work_dir / "04b_parties.json")
    adjustments_payload = _load_json(work_dir / "04c_adjustments.json")
    evaluated_payload = _load_json(work_dir / "06_evaluated.json")

    covenants = {
        (covenant["scenario_id"], covenant["slot"]): covenant
        for covenant in (covenants_payload.get("covenants") or [])
    }
    parties_by_scenario = parties_payload.get("scenarios") or {}
    adjustments = adjustments_payload.get("adjustments") or {}
    adjusted_ids = _adjusted_txn_ids(adjustments)
    ledger = _ledger_rows(work_dir)
    findings = _finding_map(evaluated_payload)
    template = load_template(work_dir)

    cells: list[dict[str, Any]] = []
    for scenario_id, slot in template_cells(template):
        covenant = covenants.get((scenario_id, slot))
        if covenant is None:
            continue
        key = (scenario_id, slot)
        finding = findings[key]
        evaluated = _d(finding["actual"])
        rounded = _d(finding["rounded"])
        status = finding["status"]
        parties = parties_by_scenario.get(covenant["scenario_id"])

        cells.append(
            {
                "scenario_id": covenant["scenario_id"],
                "slot": covenant["slot"],
                "status": status,
                "actual": float(rounded),
                "evidence_txn_id": finding.get("evidence_txn_id"),
                "covenant": _trace_covenant(covenant),
                "computation": _computation_block(
                    covenant,
                    ledger,
                    parties=parties,
                    adjustments=adjustments,
                    evaluated=evaluated,
                    rounded=rounded,
                    status=status,
                ),
                "inputs": collect_covenant_inputs(
                    covenant,
                    ledger,
                    parties=parties,
                ),
                "evidence_search": evidence_search(
                    covenant,
                    ledger,
                    status=status,
                    parties=parties,
                    adjustments=adjustments,
                    adjusted_txn_ids=adjusted_ids,
                    evidence_txn_id=finding.get("evidence_txn_id"),
                    flags=finding.get("flags") or [],
                ),
                "confidence": float(finding.get("confidence", "0.5")),
                "strategy": finding.get("strategy", "computed"),
                "flags": finding.get("flags") or [],
            }
        )

    trace = {
        "run": _run_block(work_dir, mode=mode, started_at=started_at, llm_stats=llm_stats),
        "cells": cells,
        "adjustments": _trace_adjustments(adjustments),
        "conflicts": _aggregate_conflicts(work_dir),
        "review": _aggregate_review(parties_payload),
    }
    return trace


def _template_cell_count(template: dict[str, Any]) -> int:
    count = 0
    for scenario in template.get("answers", {}).values():
        count += len(scenario)
    return count


def _page_text(inventory: dict[str, Any], doc_ref: str, page: int) -> str:
    doc_id = doc_ref.removesuffix(".pdf").removesuffix(".txt").removesuffix(".csv")
    document = (inventory.get("documents") or {}).get(doc_id)
    if not document:
        return ""
    pages = document.get("pages") or []
    if page < 1 or page > len(pages):
        return ""
    return pages[page - 1]


def _verification_conflict(
    *,
    check: str,
    message: str,
    scenario_id: str | None = None,
    slot: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    conflict: dict[str, Any] = {
        "kind": "TRACE_VERIFY_FAILED",
        "check": check,
        "message": message,
    }
    if scenario_id is not None:
        conflict["scenario_id"] = scenario_id
    if slot is not None:
        conflict["slot"] = slot
    conflict.update(extra)
    return conflict


def verify(trace: dict[str, Any], *, work_dir: Path, template: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the seven TRACE_SPEC section-4 checks; record conflicts and continue."""
    conflicts: list[dict[str, Any]] = []
    inventory = _load_json(work_dir / "01_inventory.json")
    adjustments = trace.get("adjustments") or {}
    ledger = _ledger_rows(work_dir)
    ledger_txn_ids = {str(row["txn_id"]) for row in ledger if row.get("txn_id")}

    expected_cells = _template_cell_count(template)
    actual_cells = len(trace.get("cells") or [])
    if actual_cells != expected_cells:
        conflicts.append(
            _verification_conflict(
                check="cell_count",
                message=(
                    f"trace cell count {actual_cells} != template cell count {expected_cells}"
                ),
                expected_cells=expected_cells,
                actual_cells=actual_cells,
            ),
        )

    for cell in trace.get("cells") or []:
        scenario_id = str(cell["scenario_id"])
        slot = str(cell["slot"])
        covenant = cell.get("covenant") or {}
        for quote_holder in (covenant.get("source"), (covenant.get("springing") or {}).get("source")):
            if not quote_holder:
                continue
            page_text = _page_text(
                inventory,
                str(quote_holder.get("doc", "")),
                int(quote_holder.get("page", 1)),
            )
            quote = str(quote_holder.get("quote", ""))
            if quote and not verify_quote(quote, page_text):
                conflicts.append(
                    _verification_conflict(
                        check="quote_on_page",
                        scenario_id=scenario_id,
                        slot=slot,
                        message=f"quote not found on page: {quote!r}",
                        quote=quote,
                    ),
                )

        computation = cell.get("computation") or {}
        evaluated = _d(computation.get("evaluated"))
        rounded = _d(computation.get("rounded"))
        expected_rounded = round_half_up(abs(evaluated), 2)
        if rounded != expected_rounded:
            conflicts.append(
                _verification_conflict(
                    check="rounded_matches_evaluated",
                    scenario_id=scenario_id,
                    slot=slot,
                    message=f"rounded mismatch: {rounded} != {expected_rounded}",
                    evaluated=str(evaluated),
                    rounded=str(rounded),
                    expected_rounded=str(expected_rounded),
                ),
            )

        threshold = _d(covenant.get("threshold"))
        direction = covenant.get("direction", "MAX")
        status = cell.get("status")
        strategy = cell.get("strategy", "computed")
        status_from_evaluated = computation.get("status_from_evaluated")
        if status_from_evaluated is None:
            status_from_evaluated = (
                "BREACH" if breaches(evaluated, direction, threshold) else "COMPLIANT"
            )
        comparison = str(computation.get("comparison", ""))
        comparison_from_evaluated = str(
            computation.get("comparison_from_evaluated", ""),
        )
        if strategy in {"computed", "springing_not_triggered"}:
            if status != status_from_evaluated:
                conflicts.append(
                    _verification_conflict(
                        check="status_matches_evaluated",
                        scenario_id=scenario_id,
                        slot=slot,
                        message=(
                            f"status {status} disagrees with evaluated value "
                            f"(expected {status_from_evaluated})"
                        ),
                        status=status,
                        status_from_evaluated=status_from_evaluated,
                        evaluated=str(evaluated),
                        rounded=str(rounded),
                        threshold=str(threshold),
                        direction=direction,
                        strategy=strategy,
                        comparison=comparison,
                        comparison_from_evaluated=comparison_from_evaluated,
                        flags=cell.get("flags") or [],
                    ),
                )

            if status not in comparison:
                conflicts.append(
                    _verification_conflict(
                        check="comparison_contains_status",
                        scenario_id=scenario_id,
                        slot=slot,
                        message=f"comparison string missing recorded status {status!r}",
                        status=status,
                        comparison=comparison,
                    ),
                )
            if status_from_evaluated not in comparison_from_evaluated:
                conflicts.append(
                    _verification_conflict(
                        check="comparison_from_evaluated_contains_status",
                        scenario_id=scenario_id,
                        slot=slot,
                        message=(
                            "comparison_from_evaluated missing evaluated status "
                            f"{status_from_evaluated!r}"
                        ),
                        status_from_evaluated=status_from_evaluated,
                        comparison_from_evaluated=comparison_from_evaluated,
                    ),
                )

        evidence = cell.get("evidence_txn_id")
        if evidence is not None and str(evidence) not in ledger_txn_ids:
            conflicts.append(
                _verification_conflict(
                    check="evidence_in_ledger",
                    scenario_id=scenario_id,
                    slot=slot,
                    message=f"evidence_txn_id not in ledger: {evidence}",
                    evidence_txn_id=str(evidence),
                ),
            )
        if evidence and status != "BREACH":
            conflicts.append(
                _verification_conflict(
                    check="evidence_requires_breach",
                    scenario_id=scenario_id,
                    slot=slot,
                    message="non-null evidence on COMPLIANT cell",
                    status=status,
                    evidence_txn_id=str(evidence),
                ),
            )

        for adj_id in computation.get("adjustments_applied") or []:
            if not _is_adjustment_ref(adj_id):
                continue
            if adj_id not in adjustments and adj_id not in (
                trace.get("adjustments") or {}
            ):
                conflicts.append(
                    _verification_conflict(
                        check="adjustment_resolves",
                        scenario_id=scenario_id,
                        slot=slot,
                        message=f"unknown adjustment in trace: {adj_id}",
                        adjustment_id=str(adj_id),
                    ),
                )

    return conflicts


def project(trace: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    """Pure projection of trace cells into the submission template."""
    submission = copy.deepcopy(template)
    submission["team"] = "macintosh"
    submission["contact_email"] = CONTACT_EMAIL
    submission["model"] = MODEL_ID

    for cell in trace.get("cells") or []:
        scenario_id = cell["scenario_id"]
        slot = cell["slot"]
        rounded = round_half_up(abs(_d((cell.get("computation") or {}).get("rounded", cell["actual"]))), 2)
        submission["answers"][scenario_id][slot] = {
            "status": cell["status"],
            "actual": float(rounded),
            "evidence_txn_id": cell.get("evidence_txn_id"),
        }
    return submission


def load_template(work_dir: Path) -> dict[str, Any]:
    from agent.template import load_template as _load_template

    return _load_template(work_dir)
