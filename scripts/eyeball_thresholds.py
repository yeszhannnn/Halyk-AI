"""Eyeball-check five 6.x thresholds against source PDF text."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OPEN = Path("data/open")
payload = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))
inventory = json.loads((OPEN / "01_inventory.json").read_text(encoding="utf-8"))

ART6 = re.compile(r"Статья 6\s*[—–-]\s*Финансовые ковенанты", re.I)
ART7 = re.compile(r"Статья 7\b", re.I)
PUNKT = re.compile(r"Пункт 6\.(\d+)")


def get_punkt(doc_id: str, slot: str) -> str:
    pages = inventory["documents"][doc_id]["pages"]
    full = "\n".join(pages)
    matches = list(ART6.finditer(full))
    rest = full[matches[-1].start() :]
    article_7 = ART7.search(rest)
    section = rest[: article_7.start()] if article_7 else rest
    positions = sorted(
        (match.start(), match.group(1))
        for match in PUNKT.finditer(section)
        if match.group(1) in "123"
    )
    for index, (pos, found_slot) in enumerate(positions):
        if found_slot == slot:
            end = positions[index + 1][0] if index + 1 < len(positions) else len(section)
            return section[pos:end].strip()
    return ""


def body_text(clause: str) -> str:
    lines = clause.splitlines()
    return "\n".join(lines[1:]) if len(lines) > 1 else clause


def direction_from_body(clause: str) -> str:
    body = body_text(clause)
    if re.search(
        r"не менее|not fall below|на уровне не менее|составляло не менее|обеспечивает, чтобы.*не менее",
        body,
        re.I,
    ):
        return "MIN"
    if re.search(
        r"не допускать|не вправе|превысил|превышал|превышали|превышала|ceiling",
        body,
        re.I,
    ):
        return "MAX"
    return "?"


CHECKS = [
    ("P1", "6.2"),
    ("B1", "6.2"),
    ("P2", "6.2"),
    ("P3", "6.2"),
    ("P6", "6.2"),
]

print("=== Slot 6.2: all directions ===")
for covenant in payload["covenants"]:
    if covenant["slot"] != "6.2":
        continue
    print(
        f"{covenant['scenario_id']:4} {covenant['direction']:3} "
        f"{covenant['threshold']:>10} {covenant['threshold_unit']:5} | {covenant['title'][:60]}"
    )

mins = sum(1 for c in payload["covenants"] if c["slot"] == "6.2" and c["direction"] == "MIN")
maxs = sum(1 for c in payload["covenants"] if c["slot"] == "6.2" and c["direction"] == "MAX")
print(f"\nTotals: MIN={mins} MAX={maxs}\n")

print("=== Five PDF spot-checks ===")
for scenario_id, slot in CHECKS:
    covenant = next(
        item
        for item in payload["covenants"]
        if item["scenario_id"] == scenario_id and item["slot"] == slot
    )
    doc_id = covenant["source"]["doc_id"]
    slot_num = slot.split(".")[1]
    clause = get_punkt(doc_id, slot_num)
    inferred = direction_from_body(clause)
    quote = covenant["source"]["quote"]
    quote_ok = quote in clause or quote.replace(",", "") in clause.replace(",", "")

    print(f"\n--- {scenario_id} {slot} | {doc_id}.pdf ---")
    print(f"Extracted: direction={covenant['direction']} threshold={covenant['threshold']} {covenant['threshold_unit']}")
    print(f"Title:     {covenant['title']}")
    print(f"From body: direction={inferred}  quote_ok={quote_ok}  quote={quote!r}")
    if covenant["direction"] != inferred:
        print("WARNING: extracted direction differs from body-language heuristic")
    print("Clause:")
    print(clause)
