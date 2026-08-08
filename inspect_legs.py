import json

d = json.load(open("data/open/04a_covenants.json", encoding="utf-8"))
cs = d if isinstance(d, list) else d.get("covenants", d)

bad = {("B1","6.1"),("P1","6.1"),("P1","6.2"),("P10","6.1"),
       ("P2","6.1"),("P5","6.2"),("P6","6.1"),("P9","6.1")}

for c in cs:
    if (c.get("scenario_id"), c.get("slot")) not in bad:
        continue
    m = c.get("metric") or {}
    print(f"{c.get('scenario_id')}/{c.get('slot')}  "
          f"dir={c.get('direction')} thr={c.get('threshold')} "
          f"unit={c.get('threshold_unit')} kind={m.get('kind')}")
    for leg in ("numerator", "denominator", "category"):
        spec = m.get(leg)
        if spec:
            print(f"   {leg}: include={spec.get('include_keywords')} "
                  f"exclude={spec.get('exclude_keywords')} sign={spec.get('sign')}")
    print(f"   title: {c.get('title')}")
    print()