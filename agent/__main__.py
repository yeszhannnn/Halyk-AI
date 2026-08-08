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

from agent.config import DEADLINE, LLM_PROVIDER, MODEL_ID, TEMPERATURE
from agent.degrade import apply_degradation_ladder
from agent.preflight import print_preflight_report, run_preflight
from agent.stages import StageResult
from agent.stages import s1_ingest, s2_classify, s3_bind, s4_extract, s4a_covenants, s5_ledger, s6_evaluate, s7_emit


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
    StageSpec("s5_ledger", ("05_ledger.parquet", "05_ledger.json"), s5_ledger.run),
    StageSpec("s4a_covenants", ("04a_covenants.json",), s4a_covenants.run),
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


def _write_manifest(
    work_dir: Path,
    *,
    input_dir: Path,
    output_path: Path,
    mode: str,
    started_at: str,
    stages: dict[str, dict[str, int | str | None]],
) -> None:
    manifest = {
        "git_sha": _git_sha(),
        "llm_provider": LLM_PROVIDER,
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "deadline": DEADLINE.isoformat(),
        "started_at": started_at,
        "mode": mode,
        "stages": stages,
    }
    (work_dir / "00_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _outputs_exist(work_dir: Path, outputs: Sequence[str]) -> bool:
    missing = [name for name in outputs if not (work_dir / name).exists()]
    return not missing


def _missing_outputs(work_dir: Path, outputs: Sequence[str]) -> list[str]:
    return [name for name in outputs if not (work_dir / name).exists()]


def _log_stage(stage_name: str, elapsed_ms: int, result: StageResult) -> None:
    row_part = f" rows={result.row_count}" if result.row_count is not None else ""
    print(f"{stage_name}: {elapsed_ms}ms items={result.item_count}{row_part}")


def _past_deadline() -> bool:
    now = datetime.now(DEADLINE.tzinfo)
    return now >= DEADLINE


def _run_stage(spec: StageSpec, *, input_dir: Path, work_dir: Path, output_path: Path, mode: str, started_at: str) -> StageResult:
    if spec.name == "s1_ingest":
        return spec.run(input_dir=input_dir, work_dir=work_dir)
    if spec.name == "s7_emit":
        return spec.run(
            work_dir=work_dir,
            output_path=output_path,
            mode=mode,
            started_at=started_at,
        )
    return spec.run(work_dir=work_dir)


def _build_pipeline_parser() -> argparse.ArgumentParser:
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
    return parser


def _run_pipeline(argv: Sequence[str]) -> int:
    args = _build_pipeline_parser().parse_args(argv)

    input_dir: Path = args.input
    output_path: Path = args.out
    work_dir = input_dir

    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    started_at = datetime.now(timezone.utc).isoformat()
    mode = "full"
    stage_timings: dict[str, dict[str, int | str | None]] = {}

    for spec in STAGES:
        if _past_deadline():
            mode = "degraded"
            print(
                f"deadline reached ({DEADLINE.isoformat()}); stopping before {spec.name}",
                file=sys.stderr,
            )
            break

        if not args.force and _outputs_exist(work_dir, spec.outputs):
            print(f"{spec.name}: skipped (outputs exist)")
            continue

        missing = _missing_outputs(work_dir, spec.outputs)
        if missing and not args.force:
            print(f"{spec.name}: running (missing {', '.join(missing)})")

        started = time.perf_counter()
        try:
            result = _run_stage(
                spec,
                input_dir=input_dir,
                work_dir=work_dir,
                output_path=output_path,
                mode=mode,
                started_at=started_at,
            )
        except Exception as exc:  # noqa: BLE001 — structural failures must not abort the run
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            print(f"{spec.name}: failed after {elapsed_ms}ms", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 1

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        stage_timings[spec.name] = {
            "elapsed_ms": elapsed_ms,
            "items": result.item_count,
            "rows": result.row_count,
        }
        _log_stage(spec.name, elapsed_ms, result)

    if mode == "degraded":
        apply_degradation_ladder(work_dir)
        if not args.force and (work_dir / "trace.json").exists():
            print("s7_emit: skipped (outputs exist)")
        else:
            started = time.perf_counter()
            try:
                result = s7_emit.run(
                    work_dir=work_dir,
                    output_path=output_path,
                    mode=mode,
                    started_at=started_at,
                )
            except Exception as exc:  # noqa: BLE001 — surface emit failures clearly
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                print(f"s7_emit: failed after {elapsed_ms}ms", file=sys.stderr)
                print(str(exc), file=sys.stderr)
                return 1
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            stage_timings["s7_emit"] = {
                "elapsed_ms": elapsed_ms,
                "items": result.item_count,
                "rows": result.row_count,
            }
            _log_stage("s7_emit", elapsed_ms, result)

    _write_manifest(
        work_dir,
        input_dir=input_dir,
        output_path=output_path,
        mode=mode,
        started_at=started_at,
        stages=stage_timings,
    )

    return 0


def _run_preflight(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run stages 1–3 and print a shape report before LLM spend",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input data directory (e.g. data/open)",
    )
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        print(f"error: input directory not found: {args.input}", file=sys.stderr)
        return 1

    report = run_preflight(args.input)
    print_preflight_report(report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["preflight"]:
        return _run_preflight(argv[1:])
    return _run_pipeline(argv)


if __name__ == "__main__":
    raise SystemExit(main())
