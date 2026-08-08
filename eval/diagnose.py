"""Failure attribution diagnostic for covenant submissions."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.evidence.quotes import verify_quote
from agent.metrics.engine import (
    IDENTICAL_LEGS,
    breaches,
    collect_covenant_inputs,
    compute_covenant_metric,
    describe_leg_breakdown,
)
from agent.parsing.numbers import round_half_up
from eval.score import score_cell

SLOTS = ("6.1", "6.2", "6.3")
FALLBACK_STRATEGIES = {
    "actual_threshold_fallback",
    "actual_zero_fallback",
    "status_slot_prior",
    "deadline_threshold_fallback",
    "deadline_zero_fallback",
}
RATIO_MARKERS: list[tuple[float, str]] = [
    (1000.0, "scale error (x1000)"),
    (0.001, "scale error (/1000)"),
    (1.16, "unconverted EUR row"),
    (-1.0, "missing abs()"),
]
RELATED_MARKERS = ("связанн", "аффилир", "ограниченн")
EXPECTED_KYC_THRESHOLDS: dict[str, tuple[str, float]] = {
    "P2": ("relatedness", 25.0),
    "P5": ("relatedness", 35.0),
    "P6": ("relatedness", 40.0),
    "P9": ("perimeter", 50.0),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _is_two_decimal_places(value: Any) -> bool:
    try:
        quantized = Decimal(str(value)).quantize(Decimal("0.01"))
        return quantized == Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False


def _covenant_index(covenants: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(item["scenario_id"], item["slot"]): item for item in covenants}


def _scenario_adjustments(
    adjustments: dict[str, Any],
    scenario_id: str,
) -> list[dict[str, Any]]:
    return [
        adj
        for adj in adjustments.values()
        if adj.get("scenario_id") == scenario_id and adj.get("kind") != "NONE"
    ]


def _adjustment_amounts(adj: dict[str, Any]) -> list[tuple[str, Decimal]]:
    amounts: list[tuple[str, Decimal]] = []
    amount = _to_decimal(adj.get("amount"))
    if amount is not None:
        amounts.append((adj.get("kind", "?"), abs(amount)))
    for row in adj.get("rows") or []:
        row_amount = _to_decimal(row.get("amount"))
        if row_amount is not None:
            label = adj.get("kind", "?")
            if row.get("above_floor") is False:
                label = f"{label} below-floor"
            amounts.append((label, abs(row_amount)))
    return amounts


def _metric_involves_related_parties(covenant: dict[str, Any]) -> bool:
    metric = covenant.get("metric") or {}
    chunks: list[str] = [str(metric.get("notes", ""))]
    for spec in (metric.get("numerator"), metric.get("denominator")):
        if not spec:
            continue
        chunks.extend(spec.get("include_keywords") or [])
    text = " ".join(chunks).casefold()
    return any(marker in text for marker in RELATED_MARKERS)


def _related_party_lines(parties: dict[str, Any] | None) -> list[str]:
    if not parties:
        return ["  (no parties payload for scenario)"]

    lines: list[str] = []
    threshold = parties.get("threshold_pct")
    if threshold is not None:
        lines.append(f"  threshold: {threshold}%")

    for row in parties.get("ownership") or []:
        qualifier = "qualifying" if row.get("is_related") else "not related"
        lines.append(
            f"  {row.get('counterparty')}: {row.get('ownership_pct')}% ({qualifier})",
        )

    perimeter = parties.get("perimeter")
    if perimeter:
        lines.append(f"  perimeter threshold: {perimeter.get('threshold_pct')}%")
        for row in perimeter.get("ownership") or []:
            qualifier = "unrestricted" if not row.get("is_related") else "restricted"
            lines.append(
                f"  {row.get('counterparty')}: {row.get('ownership_pct')}% ({qualifier})",
            )
    return lines


def _delta_match(got: Decimal, key: Decimal, adjustments: list[dict[str, Any]]) -> str | None:
    delta = abs(got - key)
    rounded_delta = round(delta, 2)
    for adj in adjustments:
        for kind, amount in _adjustment_amounts(adj):
            if round(float(amount), 2) == float(rounded_delta):
                return f"delta equals {kind} {amount} - adjustment not applied"
    return None


def _ratio_matches(got: Decimal, key: Decimal) -> list[str]:
    if key == 0:
        return []
    ratio = float(got / key)
    labels: list[str] = []
    for target, label in RATIO_MARKERS:
        if target == 0:
            continue
        if abs(ratio - target) / abs(target) <= 0.02:
            labels.append(label)
    return labels


def _springing_line(covenant: dict[str, Any]) -> str | None:
    springing = covenant.get("springing")
    if not springing:
        return None
    return (
        f"springing: {springing.get('metric', {}).get('kind', 'metric')} "
        f"{springing.get('operator')} {springing.get('value')}"
    )


def _adjustment_lines(adjustments: list[dict[str, Any]]) -> list[str]:
    if not adjustments:
        return ["  (none bound to scenario)"]
    lines: list[str] = []
    for adj in adjustments:
        amount = adj.get("amount")
        suffix = f" amount={amount}" if amount is not None else ""
        lines.append(f"  {adj.get('id', '?')}: {adj.get('kind')}{suffix}")
        if adj.get("kind") == "EBITDA_ADDBACK":
            above = sum(
                1 for row in adj.get("rows") or [] if row.get("above_floor")
            )
            lines.append(f"    above-floor rows: {above}")
    return lines


def _points_lost(answer: dict[str, Any], key: dict[str, Any]) -> float:
    return 1.0 - score_cell(answer, key)


def _status_rounding_note(
    answer: dict[str, Any],
    key: dict[str, Any],
    covenant: dict[str, Any],
) -> str | None:
    if answer.get("status") == key.get("status"):
        return None
    got = _to_decimal(answer.get("actual"))
    if got is None:
        return None
    threshold = Decimal(str(covenant["threshold"]))
    direction = covenant["direction"]
    rounded = round_half_up(abs(got), 2)
    if breaches(rounded, direction, threshold) != breaches(got, direction, threshold):
        rounded_status = "BREACH" if breaches(rounded, direction, threshold) else "COMPLIANT"
        if rounded_status == answer.get("status"):
            return (
                "status matches rounded comparison but not full-precision "
                "(rounding-before-compare defect)"
            )
    return None


def _page_text(inventory: dict[str, Any], doc_id: str, page: int) -> str:
    document = (inventory.get("documents") or {}).get(doc_id)
    if not document:
        return ""
    pages = document.get("pages") or []
    if page < 1 or page > len(pages):
        return ""
    return pages[page - 1]


def _ledger_null_amounts(work_dir: Path) -> list[str]:
    parquet = work_dir / "05_ledger.parquet"
    if not parquet.exists():
        return []
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT txn_id, amount_usd FROM read_parquet(?)",
            [str(parquet)],
        ).fetchall()
    finally:
        con.close()
    nulls: list[str] = []
    for txn_id, amount in rows:
        if amount is None or str(amount).strip() == "":
            nulls.append(str(txn_id))
    return nulls


def _finding_map(work_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = work_dir / "06_evaluated.json"
    if not path.exists():
        return {}
    findings = _load_json(path).get("findings") or []
    return {(item["scenario_id"], item["slot"]): item for item in findings}


def _diagnose_cell(
    scenario_id: str,
    slot: str,
    answer: dict[str, Any],
    key: dict[str, Any],
    *,
    covenant: dict[str, Any] | None,
    parties: dict[str, Any] | None,
    adjustments: list[dict[str, Any]],
    pipeline_finding: dict[str, Any] | None = None,
) -> list[str]:
    lines: list[str] = []
    got = _to_decimal(answer.get("actual"))
    key_actual = _to_decimal(key.get("actual"))
    pipeline_actual = None
    if pipeline_finding:
        pipeline_actual = _to_decimal(pipeline_finding.get("actual"))

    lines.append(
        f"  got: status={answer.get('status')} actual={answer.get('actual')} "
        f"evidence={answer.get('evidence_txn_id')}",
    )
    lines.append(
        f"  key: status={key.get('status')} actual={key.get('actual')} "
        f"evidence={key.get('evidence_txn_id')}",
    )
    if pipeline_actual is not None and pipeline_actual != got:
        lines.append(
            f"  pipeline evaluated (full precision): {pipeline_actual} "
            f"-> rounded {pipeline_finding.get('rounded')}",
        )

    if covenant:
        lines.append(
            f"  covenant: {covenant.get('direction')} {covenant.get('threshold')} "
            f"{covenant.get('threshold_unit', '')}".rstrip(),
        )
        springing = _springing_line(covenant)
        if springing:
            lines.append(f"  {springing}")
        lines.append("  adjustments bound to scenario:")
        lines.extend(_adjustment_lines(adjustments))

    if got is not None and key_actual is not None and key_actual != 0:
        error = float(abs(got - key_actual) / abs(key_actual))
        lines.append(f"  relative error e = {error:.4f}")

        delta_line = _delta_match(got, key_actual, adjustments)
        if delta_line:
            lines.append(f"  {delta_line}")

        for label in _ratio_matches(got, key_actual):
            lines.append(f"  ratio match: {label}")

    if covenant and _metric_involves_related_parties(covenant):
        lines.append("  related-party resolution:")
        lines.extend(_related_party_lines(parties))

    if answer.get("status") != key.get("status"):
        lines.append(
            f"  status mismatch: costs full cell ({_points_lost(answer, key):.2f} pt lost)",
        )
        if covenant and pipeline_actual is not None:
            threshold = Decimal(str(covenant["threshold"]))
            direction = covenant["direction"]
            pipeline_status = (
                "BREACH" if breaches(pipeline_actual, direction, threshold) else "COMPLIANT"
            )
            key_status = key.get("status")
            if pipeline_status == key_status and pipeline_status != answer.get("status"):
                lines.append(
                    "  pipeline full-precision status matches key; submission status differs",
                )
            note = _status_rounding_note(answer, key, covenant)
            if note:
                lines.append(f"  {note}")
            elif (
                pipeline_status != key_status
                and got is not None
                and key_actual is not None
                and key_actual != 0
            ):
                epsilon = float(abs(pipeline_actual - key_actual) / abs(key_actual))
                lines.append(
                    f"  computed value differs from key by ε (relative error {epsilon:.4f})",
                )

    if answer.get("evidence_txn_id") != key.get("evidence_txn_id"):
        lines.append("  evidence_txn_id mismatch")

    if (
        key.get("evidence_txn_id") is None
        and answer.get("status") == key.get("status")
        and got is not None
        and key_actual is not None
        and got != key_actual
    ):
        lines.append("  actual error also drains evidence component (key evidence is null)")

    return lines


def mode_a(
    submission_path: Path,
    ground_truth_path: Path,
    work_dir: Path,
) -> float:
    submission = _load_json(submission_path)
    ground_truth = _load_json(ground_truth_path)

    covenants_payload = _load_json(work_dir / "04a_covenants.json")
    parties_payload = _load_json(work_dir / "04b_parties.json")
    adjustments_payload = _load_json(work_dir / "04c_adjustments.json")

    covenants = _covenant_index(covenants_payload.get("covenants") or [])
    parties_by_scenario = parties_payload.get("scenarios") or {}
    adjustments = adjustments_payload.get("adjustments") or {}
    pipeline_findings = _finding_map(work_dir)

    imperfect: list[tuple[float, str, str, list[str]]] = []
    total_score = 0.0

    for scenario_id, scenario in ground_truth.get("scenarios", {}).items():
        for slot in SLOTS:
            key = scenario["covenants"][slot]
            answer = (
                submission.get("answers", {})
                .get(scenario_id, {})
                .get(slot, {"status": None, "actual": None, "evidence_txn_id": None})
            )
            cell_score = score_cell(answer, key)
            total_score += cell_score
            lost = 1.0 - cell_score
            if lost <= 1e-9:
                continue
            covenant = covenants.get((scenario_id, slot))
            scenario_adjustments = _scenario_adjustments(adjustments, scenario_id)
            detail = _diagnose_cell(
                scenario_id,
                slot,
                answer,
                key,
                covenant=covenant,
                parties=parties_by_scenario.get(scenario_id),
                adjustments=scenario_adjustments,
                pipeline_finding=pipeline_findings.get((scenario_id, slot)),
            )
            imperfect.append((lost, scenario_id, slot, detail))

    imperfect.sort(key=lambda item: (-item[0], item[1], item[2]))

    print("=== Failure Attribution (Mode A) ===")
    print(f"Total score: {total_score:.2f} / 36.00")
    print()
    if not imperfect:
        print("All 36 cells match the key.")
        return total_score

    print("Cells ranked by points lost (descending):")
    for lost, scenario_id, slot, detail in imperfect:
        answer = submission["answers"][scenario_id][slot]
        key = ground_truth["scenarios"][scenario_id]["covenants"][slot]
        score = score_cell(answer, key)
        print(f"\n{scenario_id}/{slot}  lost {lost:.2f} pt  (score {score:.2f})")
        print("\n".join(detail))

    return total_score


def _print_check(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{suffix}")


def mode_b(submission_path: Path, work_dir: Path) -> None:
    submission = _load_json(submission_path)
    template_path = work_dir / "submission_template.json"
    template = _load_json(template_path) if template_path.exists() else None

    print("=== Invariant Checks (Mode B) ===")
    violations: list[str] = []

    if template is None:
        violations.append("submission_template.json not found")
    else:
        template_answers = template.get("answers") or {}
        submission_answers = submission.get("answers") or {}
        for scenario_id, slots in template_answers.items():
            if scenario_id not in submission_answers:
                violations.append(f"missing scenario {scenario_id}")
                continue
            for slot in slots:
                if slot not in submission_answers[scenario_id]:
                    violations.append(f"missing cell {scenario_id}/{slot}")
                    continue
                cell = submission_answers[scenario_id][slot]
                for field in ("status", "actual", "evidence_txn_id"):
                    if field not in cell or cell[field] is None:
                        if field == "evidence_txn_id":
                            continue
                        violations.append(f"{scenario_id}/{slot}: {field} not filled")

    total_cells = 0
    breach_count = 0
    for scenario_id, slots in submission.get("answers", {}).items():
        for slot, cell in slots.items():
            total_cells += 1
            status = cell.get("status")
            if status == "BREACH":
                breach_count += 1

            evidence = cell.get("evidence_txn_id")
            if evidence is not None and status != "BREACH":
                violations.append(f"{scenario_id}/{slot}: evidence on COMPLIANT cell")

            actual = cell.get("actual")
            if actual is None:
                violations.append(f"{scenario_id}/{slot}: actual is null")
            elif isinstance(actual, bool) or not isinstance(actual, (int, float)):
                violations.append(f"{scenario_id}/{slot}: actual not numeric")
            else:
                if actual < 0:
                    violations.append(f"{scenario_id}/{slot}: actual negative")
                if not _is_two_decimal_places(actual):
                    violations.append(f"{scenario_id}/{slot}: actual not two decimals")

    findings = _finding_map(work_dir)
    fallback_cells: list[str] = []
    for key, finding in sorted(findings.items()):
        strategy = finding.get("strategy")
        if strategy in FALLBACK_STRATEGIES:
            fallback_cells.append(f"{key[0]}/{key[1]} ({strategy})")
    if fallback_cells:
        violations.append("fallback strategy cells: " + ", ".join(fallback_cells))

    null_amounts = _ledger_null_amounts(work_dir)
    if null_amounts:
        violations.append(
            "ledger rows with null amount after adjustments: "
            + ", ".join(null_amounts[:10]),
        )

    adjustments_path = work_dir / "04c_adjustments.json"
    if adjustments_path.exists():
        adjustments_payload = _load_json(adjustments_path)
        if "unrecognised" not in adjustments_payload:
            violations.append("04c_adjustments.json missing unrecognised list")
        else:
            unrecognised = adjustments_payload.get("unrecognised") or []
            print(f"unrecognised adjustments ({len(unrecognised)}):")
            if unrecognised:
                for item in unrecognised:
                    print(f"  - {item}")
            else:
                print("  (empty)")
    else:
        violations.append("04c_adjustments.json not found")

    inventory_path = work_dir / "01_inventory.json"
    ocr_page_count = 0
    ocr_doc_count = 0
    if inventory_path.exists():
        inventory = _load_json(inventory_path)
        for document in (inventory.get("documents") or {}).values():
            pages = document.get("ocr_pages") or []
            if pages:
                ocr_doc_count += 1
                ocr_page_count += len(pages)
        if ocr_page_count == 0:
            violations.append("ocr_pages empty across inventory")
    else:
        violations.append("01_inventory.json not found")

    covenants_path = work_dir / "04a_covenants.json"
    if covenants_path.exists() and inventory_path.exists():
        inventory = _load_json(inventory_path)
        for covenant in _load_json(covenants_path).get("covenants") or []:
            source = covenant.get("source") or {}
            quote = str(source.get("quote", ""))
            page_text = _page_text(
                inventory,
                str(source.get("doc_id", "")),
                int(source.get("page", 1)),
            )
            if quote and not verify_quote(quote, page_text):
                violations.append(
                    f"{covenant['scenario_id']}/{covenant['slot']}: threshold quote unverified",
                )

    breach_share = breach_count / total_cells if total_cells else 0.0
    if breach_share > 0.80:
        violations.append(
            f"breach share {breach_share:.1%} > 80% (possible inverted operator)",
        )

    print()
    if violations:
        print("Violations:")
        for item in violations:
            print(f"  - {item}")
    else:
        print("All invariants satisfied.")

    print()
    print(
        f"Summary: {total_cells} cells, {breach_count} BREACH "
        f"({breach_share:.1%}), ocr_pages={ocr_page_count} in {ocr_doc_count} documents",
    )


def _ledger_rows(work_dir: Path) -> list[dict[str, Any]]:
    parquet = work_dir / "05_ledger.parquet"
    if not parquet.exists():
        return []
    con = duckdb.connect()
    try:
        df = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(parquet)],
        ).df()
    finally:
        con.close()
    return df.to_dict(orient="records")



def _print_leg_breakdown(
    breakdown,
    *,
    covenant: dict[str, Any],
    parties: dict[str, Any] | None,
) -> None:
    print(
        f"  {breakdown.kind} ({breakdown.row_count} rows, "
        f"categories={breakdown.category_count}, subtotal={breakdown.value}):"
    )
    if breakdown.shape:
        print(f"    shape: {breakdown.shape}")
    if breakdown.flags:
        print(f"    flags: {', '.join(breakdown.flags)}")
    if breakdown.category_count > 3:
        print(
            f"    review: leg spans {breakdown.category_count} categories "
            f"({', '.join(breakdown.categories)})"
        )
    if breakdown.kind == "derived":
        print(f"    expression: {breakdown.expression}")
        for label, amount in breakdown.terms:
            print(f"      {label}: {amount}")
        return
    if breakdown.kind == "document":
        print(f"    expression: {breakdown.expression}")
        for label, amount in breakdown.terms:
            print(f"      {label}: {amount}")
        return
    if breakdown.kind == "empty":
        if "EMPTY_CATEGORY_SPEC" in breakdown.flags:
            print("    (empty CategorySpec — extraction failure)")
        else:
            print("    (no matching rows)")
        return
    for row in breakdown.rows:
        amount = row.get("amount_usd")
        why_inputs = collect_covenant_inputs(covenant, [row], parties=parties)
        why = why_inputs[0].get("why", "matched") if why_inputs else "matched"
        print(
            f"    {row.get('txn_id')}: amount={amount} "
            f"counterparty={row.get('counterparty')!r} reason={why}",
        )


def print_cell_breakdown(work_dir: Path, scenario_id: str, slot: str) -> None:
    covenants_payload = _load_json(work_dir / "04a_covenants.json")
    parties_payload = _load_json(work_dir / "04b_parties.json")
    adjustments_payload = _load_json(work_dir / "04c_adjustments.json")
    covenants = _covenant_index(covenants_payload.get("covenants") or [])
    covenant = covenants.get((scenario_id, slot))
    if covenant is None:
        print(f"No covenant found for {scenario_id}/{slot}")
        return

    ledger = _ledger_rows(work_dir)
    parties = (parties_payload.get("scenarios") or {}).get(scenario_id)
    adjustments = adjustments_payload.get("adjustments") or {}
    metric = covenant["metric"]

    print(f"=== Cell breakdown {scenario_id}/{slot} ===")
    print(
        f"  {covenant.get('direction')} {covenant.get('threshold')} "
        f"{covenant.get('threshold_unit', '')}".rstrip(),
    )

    leg_totals: dict[str, Decimal] = {}
    for leg in ("numerator", "denominator"):
        spec = metric["numerator"] if leg == "numerator" else metric.get("denominator")
        if spec is None:
            continue
        breakdown = describe_leg_breakdown(
            covenant,
            ledger,
            leg=leg,
            parties=parties,
            adjustments=adjustments,
            work_dir=work_dir,
        )
        leg_totals[leg] = breakdown.value
        print(f"  {leg}:")
        _print_leg_breakdown(breakdown, covenant=covenant, parties=parties)

    metric_metadata: dict[str, Any] = {}
    actual = compute_covenant_metric(
        covenant,
        ledger,
        parties=parties,
        adjustments=adjustments,
        work_dir=work_dir,
        metadata=metric_metadata,
    )
    kind = metric.get("kind", "SUM")
    if kind == "RATIO" and "denominator" in leg_totals:
        numerator = leg_totals.get("numerator", Decimal("0"))
        denominator = leg_totals.get("denominator", Decimal("0"))
        metric_flags = metric_metadata.get("flags") or []
        if IDENTICAL_LEGS in metric_flags:
            print(
                f"  expression: {numerator} / {denominator} "
                f"→ not evaluated ({IDENTICAL_LEGS}); scored actual = {actual} "
                f"(metric kind={kind})",
            )
        else:
            print(
                f"  expression: {numerator} / {denominator} = {actual} "
                f"(metric kind={kind})",
            )
    elif kind == "SUM":
        numerator = leg_totals.get("numerator", actual)
        print(f"  expression: sum(numerator) = {numerator} = {actual} (metric kind={kind})")
    else:
        print(f"  computed actual: {actual} (metric kind={kind})")

    threshold = Decimal(str(covenant["threshold"]))
    direction = covenant["direction"]
    status = "BREACH" if breaches(actual, direction, threshold) else "COMPLIANT"
    print(f"  verdict: {status} ({direction} {threshold})")
    if metric_metadata.get("flags"):
        print(f"  metric flags: {', '.join(metric_metadata['flags'])}")
    if metric_metadata.get("strategy"):
        print(f"  strategy: {metric_metadata['strategy']}")
    legs_meta = metric_metadata.get("legs") or {}
    for leg_name, leg_info in legs_meta.items():
        shape = leg_info.get("shape")
        if shape:
            print(f"  {leg_name} shape: {shape}")


def run_reference_checks(work_dir: Path) -> None:
    print()
    print("=== Reference Checks ===")

    adjustments_path = work_dir / "04c_adjustments.json"
    parties_path = work_dir / "04b_parties.json"
    inventory_path = work_dir / "01_inventory.json"

    if not adjustments_path.exists():
        _print_check("04c_adjustments.json present", False)
        return

    adjustments = _load_json(adjustments_path).get("adjustments") or {}

    off_ledger = [adj for adj in adjustments.values() if adj.get("kind") == "OFF_LEDGER"]
    if off_ledger:
        amount = _to_decimal(off_ledger[0].get("amount"))
        _print_check(
            "OFF_LEDGER 918447.52",
            amount is not None and abs(amount - Decimal("918447.52")) <= Decimal("0.01"),
            str(amount),
        )
    else:
        _print_check("OFF_LEDGER 918447.52", False, "missing")

    fills = {
        adj.get("matched_txn") or adj.get("txn_id"): adj
        for adj in adjustments.values()
        if adj.get("kind") == "AMOUNT_FILL"
    }
    for txn_id, expected in [
        ("TXN-P7-0033", Decimal("486204.19")),
        ("TXN-P8-0031", Decimal("884204.16")),
    ]:
        adj = fills.get(txn_id)
        amount = _to_decimal(adj.get("amount")) if adj else None
        _print_check(
            f"AMOUNT_FILL {txn_id} {expected}",
            amount is not None and abs(amount - expected) <= Decimal("0.01"),
            str(amount) if amount is not None else "missing",
        )

    ebitda = [adj for adj in adjustments.values() if adj.get("kind") == "EBITDA_ADDBACK"]
    if ebitda:
        above_floor_total = sum(
            _to_decimal(row.get("amount")) or Decimal("0")
            for row in ebitda[0].get("rows") or []
            if row.get("above_floor")
        )
        _print_check(
            "EBITDA_ADDBACK above-floor sum 824152.91",
            above_floor_total == Decimal("824152.91"),
            str(above_floor_total),
        )
    else:
        _print_check("EBITDA_ADDBACK above-floor sum 824152.91", False, "missing")

    fx = [adj for adj in adjustments.values() if adj.get("kind") == "FX"]
    if fx:
        adj = fx[0]
        source = _to_decimal(adj.get("fx_source_amount"))
        settlement = _to_decimal(adj.get("fx_settlement_usd"))
        rate = _to_decimal(adj.get("rate"))
        if source and settlement and source != 0:
            expected = settlement / source
            _print_check(
                "FX rate from settlement/source",
                rate == expected,
                f"stored={rate} expected={expected}",
            )
        else:
            _print_check("FX rate from settlement/source", False, "missing amounts")
    else:
        _print_check("FX rate from settlement/source", False, "missing")

    for expected in [Decimal("1104663.28"), Decimal("592296.10")]:
        matches = [
            adj
            for adj in adjustments.values()
            if adj.get("kind") == "RECLASS"
            and _to_decimal(adj.get("amount")) == expected
        ]
        _print_check(
            f"RECLASS {expected}",
            bool(matches),
            "found" if matches else "missing",
        )

    if parties_path.exists():
        parties = _load_json(parties_path).get("scenarios") or {}
        for scenario_id, (kind, expected) in EXPECTED_KYC_THRESHOLDS.items():
            record = parties.get(scenario_id)
            if record is None:
                _print_check(
                    f"KYC {kind} threshold {scenario_id} -> {expected}",
                    False,
                    "scenario missing",
                )
                continue
            if kind == "perimeter":
                perimeter = record.get("perimeter") or {}
                actual = perimeter.get("threshold_pct")
            else:
                actual = record.get("threshold_pct")
            passed = actual is not None and float(actual) == expected
            _print_check(
                f"KYC {kind} threshold {scenario_id} -> {expected}",
                passed,
                str(actual),
            )
    else:
        for scenario_id, (kind, expected) in EXPECTED_KYC_THRESHOLDS.items():
            _print_check(
                f"KYC {kind} threshold {scenario_id} -> {expected}",
                False,
                "04b_parties.json missing",
            )

    if inventory_path.exists():
        inventory = _load_json(inventory_path)
        ocr_docs = 0
        ocr_pages = 0
        for document in (inventory.get("documents") or {}).values():
            pages = document.get("ocr_pages") or []
            if pages:
                ocr_docs += 1
                ocr_pages += len(pages)
        _print_check(
            "seven ocr pages across four documents",
            ocr_pages == 7 and ocr_docs == 4,
            f"{ocr_pages} pages in {ocr_docs} documents",
        )
    else:
        _print_check("seven ocr pages across four documents", False, "inventory missing")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Covenant submission failure attribution")
    parser.add_argument("submission", type=Path, help="Path to submission.json")
    parser.add_argument("--key", type=Path, default=None, help="Ground truth JSON (Mode A)")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Pipeline artifact directory (default: parent of submission)",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Assert open-dataset reference measurements",
    )
    parser.add_argument(
        "--cell",
        metavar="SCENARIO/SLOT",
        help="Print numerator/denominator rows for one covenant cell",
    )
    args = parser.parse_args(argv)

    work_dir = args.work_dir or args.submission.parent

    if args.cell:
        if "/" not in args.cell:
            parser.error("--cell must be SCENARIO/SLOT, e.g. P6/6.1")
        scenario_id, slot = args.cell.split("/", 1)
        print_cell_breakdown(work_dir, scenario_id, slot)
        return 0

    if args.key:
        mode_a(args.submission, args.key, work_dir)
    else:
        mode_b(args.submission, work_dir)

    if args.reference:
        run_reference_checks(work_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
