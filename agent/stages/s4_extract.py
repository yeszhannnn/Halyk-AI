from pathlib import Path

from agent.stages import StageResult
from agent.stages import s4a_covenants, s4b_parties, s4c_adjustments


def run(*, work_dir: Path) -> StageResult:
    covenants = s4a_covenants.run(work_dir=work_dir)
    parties = s4b_parties.run(work_dir=work_dir)
    adjustments = s4c_adjustments.run(work_dir=work_dir)
    return StageResult(
        item_count=covenants.item_count + parties.item_count + adjustments.item_count,
        row_count=parties.row_count,
    )
