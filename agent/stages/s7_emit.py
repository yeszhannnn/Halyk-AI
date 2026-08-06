"""Stage 7 — emit trace.json and project submission.json."""

from __future__ import annotations

import json
from pathlib import Path

from agent.llm.client import RUN_COUNTER
from agent.stages import StageResult
from agent.trace import build_trace, load_template, project, verify
from agent.validate import validate


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
    verify(trace, work_dir=work_dir, template=template)

    trace_path = work_dir / "trace.json"
    trace_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    submission = project(trace, template)
    submission_json = json.dumps(submission, indent=2, ensure_ascii=False) + "\n"
    (work_dir / "submission.json").write_text(submission_json, encoding="utf-8")
    output_path.write_text(submission_json, encoding="utf-8")

    validate(work_dir, submission, template=template)

    return StageResult(item_count=len(trace["cells"]), row_count=len(trace["cells"]))
