from pathlib import Path

from agent.stages import StageResult
from agent.stages import s4a_covenants


def run(*, work_dir: Path) -> StageResult:
    return s4a_covenants.run(work_dir=work_dir)
