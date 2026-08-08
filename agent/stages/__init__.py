from dataclasses import dataclass


@dataclass(frozen=True)
class StageResult:
    item_count: int
    row_count: int | None = None
    unstable_field_count: int | None = None
    retry_clause_count: int | None = None
    verification_failed_count: int | None = None
