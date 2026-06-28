"""
Read results/*.json and emit numbers.tex (LaTeX macros) + table_main.tex.
Also emits numbers.json for the HTML build.
"""
import os, json, math

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "results")


def load(n):
    with open(os.path.join(R, n)) as f:
        return json.load(f)


q = load("quality.json")
lat = load("latency.json")
foot = load("footprint.json")
meta = load("meta.json")

KEEP = 0.33
NEURAL = "SelectiveContext(neural)"


def ret(method):
    rows = [r for r in q if r["method"] == method and abs(r["keep_ratio"] - KEEP) < 1e-6]
    return rows[0]["answer_retention"] * 100 if rows else float("nan")


def tokred(method):
    rows = [r for r in q if r["method"] == method and abs(r["keep_ratio"] - KEEP) < 1e-6]
    return rows[0]["mean_token_reduction"] * 100 if rows else float("nan")


def lat1024(method):
    d = lat.get(method, {})
    return d.get("1024", {}).get("ms_median", float("nan"))


classical_methods = ["Lead", "Random", "TF-IDF", "TF-IDF+Q", "TextRank",
                     "TextRank+Q", "Centroid+MMR", "BM25+Q"]

ret_bmq = ret("BM25+Q")
ret_mmr = ret("Centroid+MMR")
ret_neural = ret(NEURAL)
lat_bmq = lat1024("BM25+Q")
lat_neural = lat1024(NEURAL)
classical_lats = [lat1024(m) for m in classical_methods]
lat_lo, lat_hi = min(classical_lats), max(classical_lats)
neural_params = foot["neural_distilgpt2"]["n_params"]
neural_mb = foot["neural_distilgpt2"]["param_mb_fp32"]
speedup = lat_neural / max(lat_bmq, 1e-9)
ret_gain = ret_bmq / max(ret_neural, 1e-9)
lat_neural_sec = lat_neural / 1000.0
neural_rps = 1.0 / lat_neural_sec

machine = "Apple M3 Pro"

macros = {
    "retBMQ": f"{ret_bmq:.1f}",
    "retMMR": f"{ret_mmr:.1f}",
    "retTFIDFQ": f"{ret('TF-IDF+Q'):.1f}",
    "retTRQ": f"{ret('TextRank+Q'):.1f}",
    "retTFIDF": f"{ret('TF-IDF'):.1f}",
    "retLead": f"{ret('Lead'):.1f}",
    "retNeural": f"{ret_neural:.1f}",
    "retGainNeuralX": f"{ret_gain:.1f}",
    "latBMQ": f"{lat_bmq:.2f}",
    "latNeural": f"{lat_neural:.0f}",
    "latSpeedup": f"{speedup:.0f}",
    "latRangeLo": f"{lat_lo:.2f}",
    "latRangeHi": f"{lat_hi:.2f}",
    "latNeuralSec": f"{lat_neural_sec:.3f}",
    "neuralRPS": f"{neural_rps:.1f}",
    "neuralParamMB": f"{neural_mb:.0f}",
    "neuralParams": f"{neural_params:,}",
    "nDistractors": f"{meta['n_distractors']}",
    "nItems": f"{meta['n_items']}",
    "meanSrcTokens": f"{meta['mean_src_tokens']:.0f}",
    "meanPassages": f"{meta['mean_passages']:.0f}",
    "machine": machine,
}

with open(os.path.join(HERE, "numbers.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

with open(os.path.join(HERE, "numbers.json"), "w") as f:
    json.dump(macros, f, indent=2)

# ---- main table ---- #
pretty_family = {"classical": "classical", "neural": "neural"}
rows_tex = []
display = classical_methods + [NEURAL]
labelmap = {NEURAL: "Selective Context (neural)"}
for m in display:
    fam = "neural" if m == NEURAL else "classical"
    r = ret(m); tr = tokred(m); l = lat1024(m)
    fp = f"{neural_mb:.0f}" if fam == "neural" else "$\\approx$0"
    name = labelmap.get(m, m).replace("+", "$+$")
    bold = m in ("BM25+Q", "Centroid+MMR")
    fmt = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
    rows_tex.append(
        f"{fmt(name)} & {fam} & {fmt(f'{r:.1f}')} & {tr:.1f} & {l:.2f} & {fp} \\\\")

table = r"""\begin{table}[t]
\centering
\caption{Head-to-head at $\sim$3$\times$ compression (keep 33\% of sentences) on the
multi-document QA benchmark (@NITEMS@ items, mean @MEANTOK@ tokens). Latency is median
ms/request for a 1024-token context on @MACHINE@ CPU. Footprint is resident model
weights. Query-aware lexical methods (\textbf{bold}) are both the most faithful and
among the cheapest; the neural baseline is query-agnostic.}
\label{tab:main}
\small
\begin{tabular}{llrrrr}
\toprule
Method & Family & Answer ret.\ (\%) & Token red.\ (\%) & Latency (ms) & Footprint (MB) \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
"""
table = (table.replace("@NITEMS@", str(meta["n_items"]))
              .replace("@MEANTOK@", f"{meta['mean_src_tokens']:.0f}")
              .replace("@MACHINE@", machine)
              .replace("@ROWS@", "\n".join(rows_tex)))

with open(os.path.join(HERE, "table_main.tex"), "w") as f:
    f.write(table)

print("wrote numbers.tex, numbers.json, table_main.tex")
print(json.dumps(macros, indent=2))
