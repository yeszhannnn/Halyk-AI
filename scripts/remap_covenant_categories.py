"""Remap legacy Russian include_keywords in 04a_covenants.json to ledger category slugs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.parsing.categories import OPEX_SLUGS, derive_leg_sign

OUTFLOW_CATEGORIES = sorted(
    {
        "capex",
        "consulting",
        "expense",
        "insurance",
        "interest",
        "marketing",
        "opex",
        "other",
        "personnel",
        "rent",
        "tax",
        "utilities",
    },
)

KEYWORD_TO_CATEGORIES: dict[str, list[str]] = {
    "выручка": ["revenue"],
    "выручки": ["revenue"],
    "поступления": ["financing", "revenue"],
    "поступления по финансированию": ["financing"],
    "поступлений по финансированию": ["financing"],
    "операционных расходов": sorted(OPEX_SLUGS),
    "операционные расходы": sorted(OPEX_SLUGS),
    "операционных и капитальных затрат": ["opex", "capex"],
    "операционных капитальных затрат": ["opex", "capex"],
    "сумме операционных и капитальных затрат": ["opex", "capex"],
    "ebitda": ["revenue", "opex"],
    "скорректированная ebitda": ["revenue", "opex"],
    "капитальные затраты": ["capex"],
    "капитальных затрат": ["capex"],
    "совокупных капитальных затрат": ["capex"],
    "совокупные капитальные затраты": ["capex"],
    "процентные расходы": ["interest"],
    "расходы на оплату труда": ["personnel"],
    "расходов на оплату труда": ["personnel"],
    "расходы на коммунальные услуги": ["utilities"],
    "коммуналь": ["utilities"],
    "налоги": ["tax"],
    "платежи": OUTFLOW_CATEGORIES,
    "связанным сторонам": OUTFLOW_CATEGORIES,
    "ограниченные платежи": OUTFLOW_CATEGORIES,
    "аффилированные лица": OUTFLOW_CATEGORIES,
    "суммы": OUTFLOW_CATEGORIES,
}


def _map_keywords(keywords: list[str]) -> list[str]:
    mapped: set[str] = set()
    patterns = sorted(KEYWORD_TO_CATEGORIES.items(), key=lambda item: len(item[0]), reverse=True)
    for keyword in keywords:
        key = str(keyword).casefold().strip()
        if key in {category.casefold() for category in OUTFLOW_CATEGORIES} or key in {
            "revenue",
            "financing",
            "interest_income",
        }:
            mapped.add(key)
            continue
        hit = KEYWORD_TO_CATEGORIES.get(key)
        if hit:
            mapped.update(hit)
            continue
        for pattern, categories in patterns:
            if pattern in key or key in pattern:
                mapped.update(categories)
                break
    return sorted(mapped)


def _remap_category(spec: dict) -> None:
    if not spec:
        return
    spec["include_keywords"] = _map_keywords(spec.get("include_keywords") or [])
    spec["sign"] = derive_leg_sign(spec["include_keywords"])


def remap_covenant(covenant: dict) -> None:
    metric = covenant.get("metric") or {}
    notes = str(metric.get("notes") or "")
    for leg in ("numerator", "denominator", "category"):
        _remap_category(metric.get(leg))
    if "ebitda" in notes.casefold() and metric.get("numerator") is not None:
        if not metric["numerator"].get("include_keywords"):
            metric["numerator"]["include_keywords"] = ["revenue", "opex"]
    springing = covenant.get("springing")
    if isinstance(springing, dict):
        spring_metric = springing.get("metric") or {}
        for leg in ("numerator", "denominator", "category"):
            _remap_category(spring_metric.get(leg))


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/open/04a_covenants.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for covenant in payload.get("covenants") or []:
        remap_covenant(covenant)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"remapped categories in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
