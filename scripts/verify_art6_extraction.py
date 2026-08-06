"""Verify Article 6 extraction: em-dash heading, last occurrence, required punkts."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OPEN = Path("data/open")
inventory = json.loads((OPEN / "01_inventory.json").read_text(encoding="utf-8"))
bound = json.loads((OPEN / "03_bound.json").read_text(encoding="utf-8"))

# Patterns from s4a_covenants.py
ARTICLE_6_HEADING = re.compile(
    r"Статья 6\s*[—–-]\s*Финансовые ковенанты",
    re.IGNORECASE,
)
ARTICLE_7_HEADING = re.compile(r"Статья 7\b", re.IGNORECASE)
TOC_HEADING = re.compile(r"Статья 6\s+Финансовые ковенанты", re.IGNORECASE)
PUNKT_MARKER = re.compile(r"Пункт 6\.(\d+)")
REQUIRED_PUNKT_MARKERS = ("Пункт 6.1", "Пункт 6.2", "Пункт 6.3")


def extract_article_6(pages: list[str]) -> str:
    full_text = "\n".join(pages)
    matches = list(ARTICLE_6_HEADING.finditer(full_text))
    if not matches:
        raise ValueError("Article 6 heading with em dash not found")

    start = matches[-1].start()
    remainder = full_text[start:]
    article_7 = ARTICLE_7_HEADING.search(remainder)
    end = article_7.start() if article_7 else len(remainder)
    section = remainder[:end].strip()

    missing = [marker for marker in REQUIRED_PUNKT_MARKERS if marker not in section]
    if missing:
        raise ValueError(f"Article 6 section missing required markers: {', '.join(missing)}")

    return section


def split_punkts(section: str) -> dict[str, str]:
    positions: list[tuple[int, str]] = []
    for match in PUNKT_MARKER.finditer(section):
        slot = match.group(1)
        if slot in ("1", "2", "3"):
            positions.append((match.start(), slot))
    positions.sort(key=lambda item: item[0])
    by_slot: dict[str, str] = {}
    for index, (start, slot) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(section)
        by_slot[slot] = section[start:end].strip()
    return by_slot


print("=== Article 6 extraction audit (12 active loans) ===\n")

all_ok = True
for scenario_id in sorted(bound["scenarios"].keys()):
    doc_id = bound["scenarios"][scenario_id]["loan"]
    pages = inventory["documents"][doc_id]["pages"]
    full = "\n".join(pages)

    em_dash_matches = list(ARTICLE_6_HEADING.finditer(full))
    toc_matches = list(TOC_HEADING.finditer(full))
    # TOC entries should NOT match em-dash pattern
    toc_only = [m for m in toc_matches if not ARTICLE_6_HEADING.match(full[m.start() : m.end() + 5])]

    try:
        section = extract_article_6(pages)
        items = split_punkts(section)
        ok = True
        err = ""
    except ValueError as exc:
        section = ""
        items = {}
        ok = False
        err = str(exc)
        all_ok = False

    # Verify we took LAST em-dash match, not first
    used_last = ok and em_dash_matches and em_dash_matches[-1].start() == full.find(section[:40])
    has_toc = len(toc_matches) > 0
    has_punkts = all(m in section for m in REQUIRED_PUNKT_MARKERS) if ok else False

    status = "OK" if ok else "FAIL"
    print(f"{scenario_id} ({doc_id}) [{status}]")
    print(f"  TOC hits (no em-dash): {len(toc_matches)}")
    print(f"  Em-dash hits: {len(em_dash_matches)}")
    if em_dash_matches:
        print(f"  Using match #{len(em_dash_matches)} at offset {em_dash_matches[-1].start()}")
    if ok:
        print(f"  Section length: {len(section)} chars")
        print(f"  Punkts found: {sorted(items.keys())}")
        print(f"  Starts with: {section[:70].replace(chr(10), ' ')}...")
        # Sanity: section must NOT be just the TOC line
        if "Пункт 6.1" not in section:
            print("  WARNING: no Пункт 6.1 — might have grabbed TOC")
            all_ok = False
    else:
        print(f"  Error: {err}")
    print()

# Negative test: superseded 2024 contract should have Article 6 but we check em-dash logic on a loan doc
print("=== Negative check: TOC line alone must NOT match em-dash regex ===")
toc_sample = "Содержание\nСтатья 6 Финансовые ковенанты\nСтатья 7"
real_sample = "Статья 6 — Финансовые ковенанты\nПункт 6.1 ..."
print(f"  TOC only matches em-dash pattern: {bool(ARTICLE_6_HEADING.search(toc_sample))}")
print(f"  Real heading matches em-dash pattern: {bool(ARTICLE_6_HEADING.search(real_sample))}")

print("\n=== Naive 'Статья 6' (first hit) would grab TOC ===")
NAIVE = re.compile(r"Статья 6", re.IGNORECASE)
for scenario_id in ("P1", "B1", "P3"):
    doc_id = bound["scenarios"][scenario_id]["loan"]
    full = "\n".join(inventory["documents"][doc_id]["pages"])
    first = NAIVE.search(full)
    assert first is not None
    snippet = full[first.start() : first.start() + 80].replace("\n", " ")
    has_punkt = "Пункт 6.1" in full[first.start() : first.start() + 500]
    print(f"  {scenario_id}: {snippet!r}")
    print(f"    Punkt 6.1 within 500 chars: {has_punkt}")

