"""
Export REAL data for the animated replays:
  - one genuine benchmark item where BM25+Q keeps the answer and the neural
    method drops it, with the actual per-sentence scores and keep/drop decisions
  - the real run-log lines (for the terminal replay)
  - the real chart numbers
Writes social/demo_data.json (self-contained, embedded into the HTML).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import build_multidoc_squad, normalize_answer, answer_present, n_tokens
from compressors import split_sentences, BM25Compressor, SelectiveContextCompressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEEP = 0.33


def select(scores, n, keep=KEEP):
    k = max(1, int(round(keep * n)))
    order = list(np.argsort(-scores))
    kept = sorted(order[:k])
    return kept, order


def main():
    items = build_multidoc_squad(n_items=150, n_distractors=4, seed=13, gold_position="middle")
    bm = BM25Compressor()
    print("loading neural model ...")
    neural = SelectiveContextCompressor("distilgpt2", window=512, device="cpu")

    chosen = None
    for it in items:
        sents = split_sentences(it.context)
        n = len(sents)
        if not (10 <= n <= 16):
            continue
        # gold sentence
        gold = next((i for i, s in enumerate(sents) if answer_present(s, it.answer)), None)
        if gold is None:
            continue
        bs, bscore = bm.sentence_scores(it.context, it.question)
        bkept, _ = select(np.array(bscore), n)
        bm_ok = gold in bkept
        if not bm_ok:
            continue
        # only now pay for the neural pass
        ns, nscore = neural.sentence_scores(it.context)
        nkept, _ = select(np.array(nscore), n)
        n_ok = gold in nkept
        if bm_ok and not n_ok:
            chosen = (it, sents, gold, bscore, bkept, nscore, nkept)
            break

    if chosen is None:
        print("no contrasting item found; relax constraints")
        return
    it, sents, gold, bscore, bkept, nscore, nkept = chosen

    def norm(v):
        v = np.array(v, dtype=float)
        if v.max() - v.min() < 1e-9:
            return [0.5] * len(v)
        return list((v - v.min()) / (v.max() - v.min()))

    data = {
        "item": {
            "question": it.question,
            "answer": it.answer,
            "gold_sentence_idx": gold,
            "n_sentences": len(sents),
            "src_tokens": it.src_tokens,
            "sentences": [{"i": i, "text": s.strip()} for i, s in enumerate(sents)],
        },
        "methods": {
            "BM25+Q": {
                "label": "BM25 (reads the question)",
                "query_aware": True,
                "scores": norm(bscore),
                "kept": bkept,
                "answer_kept": gold in bkept,
                "kept_tokens": n_tokens(" ".join(sents[i] for i in bkept)),
            },
            "Neural": {
                "label": "Selective Context (ignores the question)",
                "query_aware": False,
                "scores": norm(nscore),
                "kept": nkept,
                "answer_kept": gold in nkept,
                "kept_tokens": n_tokens(" ".join(sents[i] for i in nkept)),
            },
        },
    }

    # real run-log lines
    log = open(os.path.join(ROOT, "results", "run_full.log")).read().splitlines()
    log = [l for l in log if l.strip() and "examples/s" not in l and "Loading weights" not in l
           and "HF_TOKEN" not in l]
    data["run_log"] = log

    # chart numbers
    q = json.load(open(os.path.join(ROOT, "results", "quality.json")))
    lat = json.load(open(os.path.join(ROOT, "results", "latency.json")))
    methods = ["BM25+Q", "TF-IDF+Q", "TextRank+Q", "Centroid+MMR", "TF-IDF",
               "Random", "TextRank", "Lead", "SelectiveContext(neural)"]
    def ret(m):
        return [r for r in q if r["method"] == m and abs(r["keep_ratio"] - KEEP) < 1e-6][0]["answer_retention"] * 100
    data["charts"] = {
        "retention": [{"m": m, "v": round(ret(m), 1), "neural": m == "SelectiveContext(neural)"} for m in methods],
        "latency": [{"m": m, "v": round(lat[m]["1024"]["ms_median"], 3),
                     "neural": m == "SelectiveContext(neural)"}
                    for m in ["Lead", "BM25+Q", "TF-IDF", "TF-IDF+Q", "Centroid+MMR",
                              "TextRank", "TextRank+Q", "SelectiveContext(neural)"]],
    }

    def _ser(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    json.dump(data, open(os.path.join(HERE, "demo_data.json"), "w"), indent=1, default=_ser)
    print(f"chose item: q='{it.question}'  answer='{it.answer}'  n_sents={len(sents)}")
    print(f"  BM25 keeps answer: {gold in bkept}   Neural keeps answer: {gold in nkept}")
    print(f"  gold sentence #{gold}: {sents[gold][:90]}...")
    print("wrote demo_data.json")


if __name__ == "__main__":
    main()
