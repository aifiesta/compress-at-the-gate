"""
Latency, throughput and footprint measurement.

Latency is wall-clock per compression call, measured on the SAME machine (Apple
M3 Pro, CPU) for every method, across controlled context sizes. This is the
cost that matters at the gateway: it sits on the critical path of every request.

Footprint:
  * classical methods hold NO model weights -> ~0 resident parameters
  * the neural baseline's resident parameter memory is reported exactly from the
    loaded model (param count x dtype bytes), plus measured process peak RSS.
"""
from __future__ import annotations
import time
import random
import resource
import platform
from typing import List, Dict

import numpy as np

from benchmark import n_tokens, _ENC


def build_sentence_corpus(n_paragraphs: int = 400, seed: int = 7) -> List[str]:
    from datasets import load_dataset
    from compressors import split_sentences
    ds = load_dataset("squad", split="validation")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n_paragraphs, len(ds)))
    sents: List[str] = []
    for i in idxs:
        sents.extend(split_sentences(ds[i]["context"]))
    rng.shuffle(sents)
    return [s for s in sents if len(s) > 20]


def make_context(corpus: List[str], target_tokens: int, rng: random.Random) -> str:
    out, tot = [], 0
    while tot < target_tokens:
        s = corpus[rng.randrange(len(corpus))]
        out.append(s)
        tot += len(_ENC.encode(" " + s))
    return " ".join(out)


def measure_latency(method, corpus, sizes=(256, 512, 1024, 2048, 4096),
                    n_per_size: int = 25, warmup: int = 3, seed: int = 1) -> Dict:
    rng = random.Random(seed)
    results = {}
    for sz in sizes:
        contexts = [make_context(corpus, sz, rng) for _ in range(n_per_size)]
        q = "What is the main subject of the passage?"
        for c in contexts[:warmup]:
            method.compress(c, q, keep_ratio=0.33)
        times = []
        for c in contexts:
            t0 = time.perf_counter()
            method.compress(c, q, keep_ratio=0.33)
            times.append((time.perf_counter() - t0) * 1000.0)  # ms
        times = np.array(times)
        results[sz] = {
            "ms_mean": float(times.mean()),
            "ms_median": float(np.median(times)),
            "ms_p95": float(np.percentile(times, 95)),
            "ms_std": float(times.std()),
            "throughput_per_s": float(1000.0 / times.mean()),
        }
    return results


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    if platform.system() == "Darwin":
        return ru / (1024 * 1024)
    return ru / 1024


def model_param_footprint(model) -> Dict:
    import torch
    n_params = sum(p.numel() for p in model.parameters())
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "n_params": int(n_params),
        "param_mb_fp32": n_params * 4 / (1024 * 1024),
        "param_mb_loaded": bytes_total / (1024 * 1024),
    }
