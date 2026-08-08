"""Record LLM replay fixtures for the mini dataset (run once when API key is set)."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from agent.llm.client import set_record_dir
from agent.stages import s1_ingest, s2_classify, s3_bind, s4_extract, s4a_covenants, s5_ledger, s6_evaluate, s7_emit

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "fixtures" / "mini"
REPLAY = ROOT / "tests" / "fixtures" / "llm"


def main() -> None:
    work = ROOT / "data" / "mini_record_work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(MINI, work)

    REPLAY.mkdir(parents=True, exist_ok=True)
    set_record_dir(REPLAY)

    s1_ingest.run(input_dir=work, work_dir=work)
    s2_classify.run(work_dir=work)
    s3_bind.run(work_dir=work)
    s4_extract.run(work_dir=work)
    s5_ledger.run(work_dir=work)
    s4a_covenants.run(work_dir=work)
    s6_evaluate.run(work_dir=work)
    started_at = datetime.now(timezone.utc).isoformat()
    s7_emit.run(
        work_dir=work,
        output_path=work / "submission.json",
        started_at=started_at,
    )

    count = len(list(REPLAY.glob("*.json")))
    print(f"recorded {count} LLM fixtures under {REPLAY}")


if __name__ == "__main__":
    main()
