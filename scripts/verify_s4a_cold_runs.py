"""Run three cold s4a_covenants extractions and verify byte-identical output."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from agent.stages.s4a_covenants import run

WORK_DIR = Path("data/open")
RUNS = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "springing": payload["summary"]["springing_count"],
        "conflicts": len(payload["conflicts"]),
    }


def main() -> int:
    cache_dir = Path(".cache/llm")
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)

    digests: list[str] = []
    for index in range(1, RUNS + 1):
        output = WORK_DIR / "04a_covenants.json"
        if output.is_file():
            output.unlink()
        print(f"=== cold run {index}/{RUNS} ===", flush=True)
        run(work_dir=WORK_DIR)
        digest = _sha256(output)
        summary = _summary(output)
        print(
            f"run {index}: sha256={digest[:16]}... "
            f"springing={summary['springing']} conflicts={summary['conflicts']}",
            flush=True,
        )
        digests.append(digest)
        shutil.copy2(output, WORK_DIR / f"04a_covenants_run{index}.json")

    if len(set(digests)) != 1:
        print("FAIL: outputs differ across runs", file=sys.stderr)
        for index, digest in enumerate(digests, start=1):
            print(f"  run {index}: {digest}", file=sys.stderr)
        return 1

    print(
        f"PASS: {RUNS} runs produced identical 04a_covenants.json "
        f"({_summary(WORK_DIR / '04a_covenants.json')})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
