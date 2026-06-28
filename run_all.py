"""
Run the full benchmark and write results/*.json.

Usage:
    python run_all.py            # full run
    python run_all.py --quick    # smaller run for a fast smoke test
"""
from __future__ import annotations
import os, sys, json, time, platform

import numpy as np

from compressors import CLASSICAL, SelectiveContextCompressor
from benchmark import build_multidoc_squad, evaluate_quality, n_tokens
from latency import (build_sentence_corpus, measure_latency, peak_rss_mb,
                     model_param_footprint)

QUICK = "--quick" in sys.argv
RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


def save(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote results/{name}")


def main():
    t_start = time.time()
    n_items = 60 if QUICK else 200
    n_distractors = 7
    quality_ratios = [0.1, 0.2, 0.33, 0.5, 0.7]
    neural_ratios = [0.2, 0.33, 0.5]
    neural_quality_items = 60 if QUICK else 150
    lat_sizes = (256, 512, 1024, 2048) if QUICK else (256, 512, 1024, 2048, 4096)

    print("=" * 70)
    print("MESH ROUTING-LAYER COMPRESSION BENCHMARK")
    print("=" * 70)
    print(f"[1/5] Building multi-document SQuAD benchmark (n={n_items}, distractors={n_distractors}) ...")
    items = build_multidoc_squad(n_items=n_items, n_distractors=n_distractors,
                                 seed=13, gold_position="middle")
    src_tokens = [it.src_tokens for it in items]
    meta = {
        "n_items": len(items),
        "n_distractors": n_distractors,
        "mean_passages": float(np.mean([it.n_passages for it in items])),
        "mean_src_tokens": float(np.mean(src_tokens)),
        "median_src_tokens": float(np.median(src_tokens)),
        "min_src_tokens": int(np.min(src_tokens)),
        "max_src_tokens": int(np.max(src_tokens)),
        "machine": platform.platform(),
        "processor": platform.processor() or "Apple M3 Pro",
        "python": platform.python_version(),
        "gold_position": "middle",
    }
    print(f"      {len(items)} items, mean {meta['mean_src_tokens']:.0f} tokens, "
          f"{meta['mean_passages']:.0f} passages each.")
    save("meta.json", meta)

    # -------------------- QUALITY (classical) -------------------- #
    print(f"[2/5] Quality eval -- classical methods x {len(quality_ratios)} ratios ...")
    quality = []
    for mname, factory in CLASSICAL.items():
        method = factory()
        method.name = mname  # use the registry key as the canonical label
        for r in quality_ratios:
            qs = evaluate_quality(method, items, keep_ratio=r)
            d = qs.__dict__.copy()
            d["family"] = "classical"
            quality.append(d)
        last = quality[-1]
        print(f"      {mname:14} @keep0.33: retention="
              f"{[q['answer_retention'] for q in quality if q['method']==qs.method and abs(q['keep_ratio']-0.33)<1e-6][0]:.3f}")

    # -------------------- QUALITY (neural) -------------------- #
    print(f"[3/5] Quality eval -- neural SelectiveContext (distilgpt2, CPU) "
          f"x {len(neural_ratios)} ratios on {neural_quality_items} items ...")
    neural = SelectiveContextCompressor(model_name="distilgpt2", window=512, device="cpu")
    foot = model_param_footprint(neural.model)
    n_items_neural = items[:neural_quality_items]
    for r in neural_ratios:
        t0 = time.time()
        qs = evaluate_quality(neural, n_items_neural, keep_ratio=r)
        d = qs.__dict__.copy()
        d["family"] = "neural"
        quality.append(d)
        print(f"      keep={r:.2f}: retention={qs.answer_retention:.3f} "
              f"({time.time()-t0:.1f}s)")
    save("quality.json", quality)

    # -------------------- LATENCY + THROUGHPUT -------------------- #
    print(f"[4/5] Latency sweep over sizes {lat_sizes} tokens ...")
    corpus = build_sentence_corpus(n_paragraphs=400, seed=7)
    latency = {}
    for mname, factory in CLASSICAL.items():
        method = factory()
        latency[mname] = measure_latency(method, corpus, sizes=lat_sizes,
                                          n_per_size=(10 if QUICK else 25))
        ms = latency[mname][lat_sizes[2]]["ms_median"]
        print(f"      {mname:14} @1024tok: {ms:.3f} ms/call")
    # neural latency (fewer samples; it's the slow one)
    latency[neural.name] = measure_latency(neural, corpus, sizes=lat_sizes,
                                            n_per_size=(5 if QUICK else 12))
    print(f"      {neural.name:14} @1024tok: "
          f"{latency[neural.name][lat_sizes[2]]['ms_median']:.1f} ms/call")
    save("latency.json", latency)

    # -------------------- FOOTPRINT -------------------- #
    print("[5/5] Footprint ...")
    footprint = {
        "classical": {
            "n_params": 0,
            "param_mb_fp32": 0.0,
            "note": "no model weights; TF-IDF/graph state is transient per-request",
        },
        "neural_distilgpt2": foot,
        "process_peak_rss_mb": peak_rss_mb(),
        "machine": meta["machine"],
    }
    save("footprint.json", footprint)

    print("=" * 70)
    print(f"DONE in {time.time()-t_start:.1f}s. Results in {RESULTS}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
