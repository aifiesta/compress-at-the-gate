# Compress-at-the-Gate

Research artifact for the preprint **"Compress-at-the-Gate: Query-Aware Classical
Prompt Compression as a Zero-Footprint Context-Engineering Primitive for LLM
Gateways"** (June 2026).

**Thesis.** At the LLM *gateway* — the layer that touches every request — a prompt
compressor must first be cheap in *latency* and *accelerator footprint*, and only
then maximize retention. In that regime the classical-vs-neural trade-off inverts:
decades-old **query-aware extractive** methods (BM25, centroid+MMR) retain ~96% of
gold answers at ~3× compression while running **~100–1000× faster** than a neural
Selective-Context baseline and holding **zero** model weights.

## What's here

| File | Purpose |
|------|---------|
| `compressors.py` | Classical compressors (TF-IDF, TextRank, centroid+MMR, BM25) + faithful neural Selective-Context baseline (distilgpt2 self-information). |
| `benchmark.py`   | Multi-document "lost-in-the-middle" QA benchmark from SQuAD; intrinsic answer-retention metric (no LLM needed). |
| `latency.py`     | Latency / throughput / footprint measurement. |
| `run_all.py`     | Runs the whole benchmark → `results/*.json`. |
| `figures.py`     | Generates `figures/*.{pdf,png}`. |
| `fill_numbers.py`| Injects real numbers into the LaTeX (`numbers.tex`, `table_main.tex`). |
| `build_html.py`  | Renders a self-contained `paper.html` (→ PDF via headless Chrome). |
| `eval_e2e.py`    | **Optional** end-to-end QA accuracy through the Mesh API gateway. |
| `paper.tex`      | arXiv-ready LaTeX source. |

## Reproduce (no API key required)

```bash
pip install numpy scipy scikit-learn nltk networkx matplotlib transformers torch datasets tiktoken
python run_all.py        # ~10–15 min on a laptop CPU; writes results/*.json
python figures.py        # writes figures/
python fill_numbers.py   # writes numbers.tex + table_main.tex for the paper
python build_html.py     # writes paper.html
```

The headline results are fully reproducible offline (the only network access is the
one-time SQuAD and distilgpt2 download from the HuggingFace hub).

## Optional: end-to-end accuracy through Mesh API

```bash
export MESH_API_KEY=mesh_sk_...
python eval_e2e.py --model gpt-4o-mini --method BM25+Q --keep 0.33 --n 100
```

## Headline numbers (Apple M3 Pro, CPU)

See `results/` after a run. At ~3× compression: BM25+Q / Centroid+MMR ≈ 0.96 answer
retention at sub-millisecond–to–few-ms latency and 0 MB resident weights, vs the
neural Selective-Context baseline at ~0.4 retention and ~350 ms / 328 MB.

*Reproducibility note:* all randomness is seeded (`seed=13` benchmark, `seed=7`
latency corpus). Absolute latency depends on hardware; relative ratios are stable.
