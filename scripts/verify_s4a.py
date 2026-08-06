"""Verify 04a_covenants.json extraction results."""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from agent.evidence.quotes import verify_quote

OPEN = Path("data/open")
ART6 = re.compile(r"Статья 6\s*[—–-]\s*Финансовые ковенанты", re.I)
ART7 = re.compile(r"Статья 7\b", re.I)
PUNKT = re.compile(r"Пункт 6\.(\d+)")


def extract_threshold_from_text(text: str) -> list[str]:
    return re.findall(r"\$[\d,]+\.\d{2}|\d+\.\d{2}x|\d+\.\d+x", text)


def main() -> None:
    payload = json.loads((OPEN / "04a_covenants.json").read_text(encoding="utf-8"))
    inventory = json.loads((OPEN / "01_inventory.json").read_text(encoding="utf-8"))
    covenants = payload["covenants"]

    assert len(covenants) == 36, len(covenants)
    assert payload["summary"]["springing_count"] == 1
    assert payload["summary"]["count"] == 36

    slot_62 = [c for c in covenants if c["slot"] == "6.2"]
    directions = Counter(c["direction"] for c in slot_62)
    assert "MIN" in directions and "MAX" in directions, directions
    print(f"6.2 directions: {dict(directions)}")

    springing = [c for c in covenants if c["springing"] is not None]
    print(f"Springing covenant: {springing[0]['scenario_id']} {springing[0]['slot']}")
    assert len(springing) == 1

    sample = random.Random(42).sample(covenants, 5)
    print("\nSpot-check thresholds against source PDF text:")
    for cov in sample:
        doc_id = cov["source"]["doc_id"]
        pages = inventory["documents"][doc_id]["pages"]
        quote = cov["source"]["quote"]
        threshold = cov["threshold"]
        verified = any(verify_quote(quote, page) for page in pages)
        pdf_path = OPEN / "documents" / f"{doc_id}.pdf"
        print(
            f"  {cov['scenario_id']} {cov['slot']}: threshold={threshold} "
            f"direction={cov['direction']} quote_ok={verified} pdf={pdf_path.name}",
        )
        assert verified, f"quote failed for {cov['scenario_id']} {cov['slot']}"

    unverified = [
        f"{c['scenario_id']} {c['slot']}.{field}"
        for c in covenants
        for field, ok in c.get("verification", {}).items()
        if not ok
    ]
    if unverified:
        print(f"\nUnverified fields ({len(unverified)}): {unverified[:10]}")
    else:
        print("\nAll quoted fields verified.")

    print("\nOK: 36 covenants, springing=1, 6.2 has MIN and MAX")


if __name__ == "__main__":
    main()
