from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.config import DEADLINE, MODEL_ID, TEMPERATURE
from agent.stages import StageResult
from agent.stages import s1_ingest, s2_classify, s3_bind, s4_extract, s5_ledger, s6_evaluate, s7_emit


@dataclass(frozen=True)
class StageSpec:
    name: str
    outputs: tuple[str, ...]
    run: Callable[..., StageResult]


STAGES: tuple[StageSpec, ...] = (
    StageSpec("s1_ingest", ("01_inventory.json",), s1_ingest.run),
    StageSpec("s2_classify", ("02_classified.json",), s2_classify.run),
    StageSpec("s3_bind", ("03_bound.json",), s3_bind.run),
    StageSpec(
        "s4_extract",
        ("04a_covenants.json", "04b_parties.json", "04c_adjustments.json"),
        s4_extract.run,
    ),
    StageSpec("s5_ledger", ("05_ledger.parquet",), s5_ledger.run),
    StageSpec("s6_evaluate", ("06_evaluated.json",), s6_evaluate.run),
    StageSpec("s7_emit", ("trace.json",), s7_emit.run),
)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_manifest(work_dir: Path, *, input_dir: Path, output_path: Path) -> None:
    manifest = {
        "git_sha": _git_sha(),
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "deadline": DEADLINE.isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    (work_dir / "00_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _outputs_exist(work_dir: Path, outputs: Sequence[str]) -> bool:
    return all((work_dir / name).exists() for name in outputs)


def _log_stage(stage_name: str, elapsed_ms: int, result: StageResult) -> None:
    row_part = f" rows={result.row_count}" if result.row_count is not None else ""
    print(f"{stage_name}: {elapsed_ms}ms items={result.item_count}{row_part}")


def _past_deadline() -> bool:
    now = datetime.now(DEADLINE.tzinfo)
    return now >= DEADLINE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Covenant checking agent pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input data directory (e.g. data/open)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Final submission JSON path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even when output artifacts already exist",
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input
    output_path: Path = args.out
    work_dir = input_dir

    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    _write_manifest(work_dir, input_dir=input_dir, output_path=output_path)

    for spec in STAGES:
        if _past_deadline():
            print(
                f"deadline reached ({DEADLINE.isoformat()}); stopping before {spec.name}",
                file=sys.stderr,
            )
            break

        if not args.force and _outputs_exist(work_dir, spec.outputs):
            print(f"{spec.name}: skipped (outputs exist)")
            continue

        started = time.perf_counter()
        try:
            if spec.name == "s1_ingest":
                result = spec.run(input_dir=input_dir, work_dir=work_dir)
            elif spec.name == "s7_emit":
                result = spec.run(work_dir=work_dir, output_path=output_path)
            else:
                result = spec.run(work_dir=work_dir)
        except NotImplementedError as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            print(f"{spec.name}: failed after {elapsed_ms}ms", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 1

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _log_stage(spec.name, elapsed_ms, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
