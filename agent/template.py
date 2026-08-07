"""Helpers for reading submission_template.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_template(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "submission_template.json"
    if not path.is_file():
        raise FileNotFoundError(f"submission template not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def template_scenarios(template: dict[str, Any]) -> list[str]:
    return sorted(template.get("answers", {}))


def template_slots(template: dict[str, Any]) -> list[str]:
    slots: set[str] = set()
    for scenario in template.get("answers", {}).values():
        slots.update(scenario)
    return sorted(slots, key=lambda slot: (float(slot), slot))


def template_cells(template: dict[str, Any]) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for scenario_id in template_scenarios(template):
        for slot in sorted(
            template["answers"][scenario_id],
            key=lambda value: (float(value), value),
        ):
            cells.append((scenario_id, slot))
    return cells


def template_cell_count(template: dict[str, Any]) -> int:
    return len(template_cells(template))
