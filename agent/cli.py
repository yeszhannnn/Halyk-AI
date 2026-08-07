"""CLI entry point for covenant agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import __main__ as pipeline
from agent.preflight import print_preflight_report, run_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="covenant", description="Covenant checking agent")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("--input", type=Path, required=True, help="Input data directory")
    run_parser.add_argument("--out", type=Path, required=True, help="Submission JSON path")
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even when output artifacts already exist",
    )

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Run stages 1–3 and print a shape report before LLM spend",
    )
    preflight_parser.add_argument("--input", type=Path, required=True, help="Input data directory")

    args = parser.parse_args(argv)
    if args.command == "preflight":
        if not args.input.is_dir():
            print(f"error: input directory not found: {args.input}", file=sys.stderr)
            return 1
        report = run_preflight(args.input)
        print_preflight_report(report)
        return 0

    if args.command == "run":
        return pipeline.main(
            ["--input", str(args.input), "--out", str(args.out)]
            + (["--force"] if args.force else []),
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
