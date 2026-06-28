"""Recompute classical quality with correct labels; keep existing neural rows."""
import json, os
from compressors import CLASSICAL
from benchmark import build_multidoc_squad, evaluate_quality

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "results")

with open(os.path.join(R, "quality.json")) as f:
    old = json.load(f)
neural_rows = [r for r in old if r.get("family") == "neural"]

items = build_multidoc_squad(n_items=200, n_distractors=7, seed=13, gold_position="middle")
ratios = [0.1, 0.2, 0.33, 0.5, 0.7]
quality = []
for mname, factory in CLASSICAL.items():
    m = factory()
    m.name = mname
    for r in ratios:
        qs = evaluate_quality(m, items, keep_ratio=r)
        d = qs.__dict__.copy()
        d["family"] = "classical"
        quality.append(d)
    r033 = [q for q in quality if q["method"] == mname and abs(q["keep_ratio"]-0.33) < 1e-6][0]
    print(f"{mname:14} ret@0.33={r033['answer_retention']:.3f}  red={r033['mean_token_reduction']:.2f}")

quality.extend(neural_rows)
with open(os.path.join(R, "quality.json"), "w") as f:
    json.dump(quality, f, indent=2)
print("rewrote quality.json with", len(quality), "rows;", len(neural_rows), "neural reused")
