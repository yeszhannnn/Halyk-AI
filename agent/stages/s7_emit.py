"""Stage 7 — emit trace.json and project submission.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agent.llm.client import RUN_COUNTER
from agent.stages import StageResult
from agent.trace import build_trace, load_template, project, verify
from agent.validate import validate

logger = logging.getLogger(__name__)


def run(
    *,
    work_dir: Path,
    output_path: Path,
    mode: str = "full",
    started_at: str | None = None,
) -> StageResult:
    template = load_template(work_dir)
    trace = build_trace(
        work_dir,
        mode=mode,
        started_at=started_at,
        llm_stats=RUN_COUNTER.to_dict(),
    )
    verification_conflicts = verify(trace, work_dir=work_dir, template=template)
    trace["verification"] = {
        "failed_count": len(verification_conflicts),
        "conflicts": verification_conflicts,
    }
    counters = trace.setdefault("run", {}).setdefault("counters", {})
    counters["verification_failed"] = len(verification_conflicts)

    trace_path = work_dir / "trace.json"
    trace_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    submission = project(trace, template)
    submission_json = json.dumps(submission, indent=2, ensure_ascii=False) + "\n"
    (work_dir / "submission.json").write_text(submission_json, encoding="utf-8")
    output_path.write_text(submission_json, encoding="utf-8")

    try:
        validate(work_dir, submission, template=template)
    except ValueError as exc:
        logger.warning("submission validation reported issues: %s", exc)

    print(
        f"s7_emit: cells={len(trace['cells'])} "
        f"verification_failed={len(verification_conflicts)}",
    )

    return StageResult(
        item_count=len(trace["cells"]),
        row_count=len(trace["cells"]),
        verification_failed_count=len(verification_conflicts),
    )
