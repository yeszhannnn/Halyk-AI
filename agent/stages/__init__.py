from dataclasses import dataclass


@dataclass(frozen=True)
class StageResult:
    item_count: int
    row_count: int | None = None
