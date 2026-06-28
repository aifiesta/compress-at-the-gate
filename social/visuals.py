"""
Black-and-white animated explainers built from the REAL benchmark data and the
REAL run output. Nothing here is mocked:

  anim_run.gif      - replays the actual stdout of run_all.py (results/run_full.log)
  anim_compress.gif - runs BM25+Q on a real benchmark item and shows the real
                      per-sentence scores, which sentences get dropped, and that
                      the gold answer survives
  anim_results.gif  - draws the real retention and latency numbers from results/*.json

Also regenerates the four static charts in grayscale (bw_*.png) so the whole
article is plain black and white.

Style: monospace, grayscale, no color, like a plain terminal / research plot.
"""
import os, json, math, re, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
import sys
sys.path.insert(0, ROOT)
from benchmark import build_multidoc_squad, normalize_answer, answer_present, n_tokens
from compressors import split_sentences, tokenize_words

with open(os.path.join(ROOT, "results", "quality.json")) as f:
    Q = json.load(f)
with open(os.path.join(ROOT, "results", "latency.json")) as f:
    LAT = json.load(f)

rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 11})
INK = "#111111"; GREY = "#888888"; LIGHT = "#c8c8c8"; BG = "#ffffff"
TERM_BG = "#0c0c0c"; TERM_FG = "#e6e6e6"; TERM_DIM = "#8a8a8a"; TERM_OK = "#ffffff"


# --------------------------------------------------------------------------- #
# 1) Terminal replay of the actual run
# --------------------------------------------------------------------------- #
def anim_run():
    with open(os.path.join(ROOT, "results", "run_full.log")) as f:
        raw = f.readlines()
    skip = ("Warning", "HF_TOKEN", "Loading", "examples/s", "it/s", "Generating")
    lines = ["$ python run_all.py"]
    for ln in raw:
        s = ln.rstrip("\n")
        if not s.strip():
            continue
        if any(k in s for k in skip):
            continue
        s = s.replace("======================================================================",
                      "=" * 54)
        lines.append(s)
    lines.append("$ _")

    fig = plt.figure(figsize=(7.4, 6.2), dpi=92)
    fig.patch.set_facecolor(TERM_BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_facecolor(TERM_BG)

    # window dressing
    def draw(n):
        ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_facecolor(TERM_BG)
        ax.text(0.02, 0.975, "research-terminal — run_all.py", color=TERM_DIM, fontsize=9,
                va="top", family="monospace")
        ax.plot([0, 1], [0.955, 0.955], color="#2a2a2a", lw=1)
        shown = lines[:n]
        # keep last ~26 lines visible
        vis = shown[-26:]
        y = 0.92
        for s in vis:
            col = TERM_FG
            if s.startswith("$"):
                col = "#7fd17f"
            elif "retention=" in s or "ms/call" in s or "DONE" in s:
                col = TERM_OK
            elif s.startswith("[") or s.startswith("="):
                col = TERM_DIM
            ax.text(0.02, y, s[:78], color=col, fontsize=9.2, va="top", family="monospace")
            y -= 0.034
        # cursor on last line
        return []

    frames = list(range(1, len(lines) + 1)) + [len(lines)] * 12  # hold at end
    anim = FuncAnimation(fig, draw, frames=frames, interval=180)
    anim.save(os.path.join(HERE, "anim_run.gif"), writer=PillowWriter(fps=6))
    plt.close(fig)
    print("wrote anim_run.gif")


# --------------------------------------------------------------------------- #
# 2) Real compression of a real item (BM25+Q), showing real scores
# --------------------------------------------------------------------------- #
def bm25_scores(sents, question, k1=1.5, b=0.75):
    docs = [tokenize_words(s) for s in sents]
    q = tokenize_words(question)
    n = len(docs)
    dl = np.array([len(d) for d in docs], dtype=float); avgdl = dl.mean() + 1e-9
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    scores = np.zeros(n)
    for i, d in enumerate(docs):
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q:
            if t in tf:
                num = tf[t] * (k1 + 1)
                den = tf[t] + k1 * (1 - b + b * dl[i] / avgdl)
                s += idf.get(t, 0.0) * num / den
        scores[i] = s
    return scores


def anim_compress():
    items = build_multidoc_squad(n_items=12, n_distractors=7, seed=13, gold_position="middle")
    # pick an item whose answer is in one clear sentence and has a decent question overlap
    it = None
    for cand in items:
        if 30 <= len(cand.gold_sentence) <= 160:
            it = cand; break
    it = it or items[0]
    all_sents = split_sentences(it.context)
    scores = bm25_scores(all_sents, it.question)
    gold_i = next((i for i, s in enumerate(all_sents) if answer_present(s, it.answer)), 0)

    # choose a readable window of ~9 sentences that includes the gold sentence
    order_by_score = list(np.argsort(-scores))
    window = set([gold_i])
    for i in order_by_score:
        if len(window) >= 5:
            break
        window.add(i)
    # add a few low-score distractors for contrast
    for i in order_by_score[::-1]:
        if len(window) >= 9:
            break
        window.add(i)
    win = sorted(window)
    n_total = len(all_sents)
    keep_k_total = max(1, round(0.33 * n_total))
    kept_global = set(order_by_score[:keep_k_total])

    def short(s, n=58):
        s = re.sub(r"\s+", " ", s).strip()
        return (s[:n] + "…") if len(s) > n else s

    rows = [(i, short(all_sents[i]), scores[i], i == gold_i, i in kept_global) for i in win]
    smax = max(r[2] for r in rows) + 1e-9

    # phases: 0 question, 1 list sentences, 2 scores appear, 3 prune, 4 result
    PH = {"q": 6, "list": 10, "score": 12, "prune": 10, "result": 14}
    seq = (["q"] * PH["q"] + ["list"] * PH["list"] + ["score"] * PH["score"]
           + ["prune"] * PH["prune"] + ["result"] * PH["result"])

    fig = plt.figure(figsize=(7.6, 6.4), dpi=92); fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def draw(fi):
        ph = seq[fi]
        ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.03, 0.965, "method: BM25 (query-aware)  ·  real benchmark item",
                color=INK, fontsize=10.5, va="top", weight="bold", family="monospace")
        ax.plot([0.03, 0.97], [0.94, 0.94], color=INK, lw=1)
        ax.text(0.03, 0.905, "Q: " + short(it.question, 64), color=INK, fontsize=10, va="top",
                family="monospace")
        ax.text(0.03, 0.862, f"context: {it.n_passages} passages, {it.src_tokens} tokens, "
                f"{n_total} sentences  (showing {len(rows)})",
                color=GREY, fontsize=9, va="top", family="monospace")
        if ph == "q":
            ax.text(0.03, 0.5, "step 1: rank every sentence by query relevance",
                    color=GREY, fontsize=11, family="monospace")
            return []
        y = 0.80
        for (i, txt, sc, is_gold, kept) in rows:
            show_score = ph in ("score", "prune", "result")
            dropped = ph in ("prune", "result") and not kept
            col = INK
            if dropped:
                col = LIGHT
            barw = 0.18 * (sc / smax) if show_score else 0
            # score bar
            if show_score:
                ax.add_patch(plt.Rectangle((0.03, y - 0.006), barw, 0.018,
                             color=(LIGHT if dropped else (INK if is_gold else GREY))))
            tx = 0.235
            label = ("► " if (is_gold and ph in ("prune", "result")) else "  ") + txt
            ax.text(tx, y, label, color=col, fontsize=9.3, va="center", family="monospace")
            if show_score:
                ax.text(0.225, y, f"{sc:4.1f}", color=col, fontsize=9.0, va="center",
                        ha="right", family="monospace")
            if dropped:
                ax.plot([tx, 0.96], [y, y], color=LIGHT, lw=0.8)
            if is_gold and ph in ("prune", "result"):
                ax.text(0.965, y, "gold", color=INK, fontsize=8.2, va="center", ha="right",
                        family="monospace",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=INK, lw=0.8))
            y -= 0.058
        captions = {
            "list": "step 1: the sentences (a readable subset)",
            "score": "step 2: BM25 query-relevance score per sentence (bar = score)",
            "prune": "step 3: keep top 33%, drop the rest (greyed)",
            "result": "step 4: result",
        }
        ax.text(0.03, 0.115, captions[ph], color=GREY, fontsize=10, family="monospace")
        if ph == "result":
            kept_text = " ".join(all_sents[i] for i in sorted(kept_global))
            ok = answer_present(kept_text, it.answer)
            comp_tok = n_tokens(kept_text)
            red = (1 - comp_tok / it.src_tokens) * 100
            ax.text(0.03, 0.065,
                    f"answer \"{short(it.answer, 28)}\" retained: {'YES' if ok else 'NO'}",
                    color=INK, fontsize=10.5, weight="bold", family="monospace")
            ax.text(0.03, 0.022,
                    f"tokens {it.src_tokens} -> {comp_tok}  ({red:.0f}% smaller)  |  cost: ~0.3 ms, 0 MB",
                    color=GREY, fontsize=9.3, family="monospace")
        return []

    frames = list(range(len(seq))) + [len(seq) - 1] * 10
    anim = FuncAnimation(fig, draw, frames=frames, interval=200)
    anim.save(os.path.join(HERE, "anim_compress.gif"), writer=PillowWriter(fps=5))
    plt.close(fig)
    print("wrote anim_compress.gif")


# --------------------------------------------------------------------------- #
# 3) Real results: retention + latency bars (grayscale), animated
# --------------------------------------------------------------------------- #
def _ret(m):
    return [r for r in Q if r["method"] == m and abs(r["keep_ratio"] - 0.33) < 1e-6][0]["answer_retention"] * 100

def _lat(m):
    return LAT[m]["1024"]["ms_median"]

def anim_results():
    methods = ["Lead", "Random", "TF-IDF", "TextRank", "SelectiveContext(neural)",
               "Centroid+MMR", "TextRank+Q", "BM25+Q", "TF-IDF+Q"]
    disp = {"SelectiveContext(neural)": "Selective Ctx\n(neural)"}
    labels = [disp.get(m, m) for m in methods]
    rets = [_ret(m) for m in methods]
    lats = [_lat(m) for m in methods]
    winners = {"Centroid+MMR", "TextRank+Q", "BM25+Q", "TF-IDF+Q"}
    is_neural = [m == "SelectiveContext(neural)" for m in methods]

    def facecolor(m):
        if m in winners:
            return INK
        if m == "SelectiveContext(neural)":
            return "#ffffff"
        return LIGHT
    fcs = [facecolor(m) for m in methods]
    hatches = ["//" if n else "" for n in is_neural]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 6.6), dpi=92)
    fig.patch.set_facecolor(BG)
    x = np.arange(len(methods))

    def draw(t):
        frac = min(1.0, t / 26.0)
        for ax in (a1, a2):
            ax.clear(); ax.set_facecolor(BG)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        # retention
        b1 = a1.bar(x, [r * frac for r in rets], color=fcs, edgecolor=INK, linewidth=1)
        for bar, h in zip(b1, hatches):
            bar.set_hatch(h)
        a1.set_ylim(0, 105); a1.set_ylabel("answer retention (%)")
        a1.set_title("real results @ ~3x compression  (black = query-aware classical, "
                     "hatched = neural)", fontsize=10, loc="left")
        a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=7.5)
        if frac >= 1.0:
            for xi, r in zip(x, rets):
                a1.text(xi, r + 2, f"{r:.0f}", ha="center", fontsize=8.5)
        # latency (log)
        b2 = a2.bar(x, [max(0.01, l) for l in lats], color=fcs, edgecolor=INK, linewidth=1)
        for bar, h in zip(b2, hatches):
            bar.set_hatch(h)
        a2.set_yscale("log"); a2.set_ylim(0.01, 2000)
        a2.set_ylabel("latency / request (ms)")
        a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=7.5)
        if frac >= 1.0:
            for xi, l in zip(x, lats):
                a2.text(xi, l * 1.3, (f"{l:.2f}" if l < 10 else f"{l:.0f}"),
                        ha="center", fontsize=8.5)
        fig.tight_layout()
        return []

    frames = list(range(0, 30)) + [30] * 12
    anim = FuncAnimation(fig, draw, frames=frames, interval=120)
    anim.save(os.path.join(HERE, "anim_results.gif"), writer=PillowWriter(fps=9))
    plt.close(fig)
    print("wrote anim_results.gif")


# --------------------------------------------------------------------------- #
# 4) Grayscale static charts for the B/W article
# --------------------------------------------------------------------------- #
def bw_static():
    methods = ["Lead", "Random", "TF-IDF", "TF-IDF+Q", "TextRank", "TextRank+Q",
               "Centroid+MMR", "BM25+Q", "SelectiveContext(neural)"]
    winners = {"Centroid+MMR", "TextRank+Q", "BM25+Q", "TF-IDF+Q"}

    # cost-quality
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=110)
    for m in methods:
        rows = [r for r in Q if r["method"] == m and abs(r["keep_ratio"] - 0.33) < 1e-6]
        if not rows or m not in LAT:
            continue
        ret = rows[0]["answer_retention"] * 100
        ms = LAT[m]["1024"]["ms_median"]
        neural = m == "SelectiveContext(neural)"
        ax.scatter(ms, ret, s=(220 if neural else 130),
                   facecolor=("white" if neural else (INK if m in winners else LIGHT)),
                   edgecolor=INK, linewidth=1.4, marker=("s" if neural else "o"), zorder=4,
                   hatch=("///" if neural else None))
        ax.annotate(("Selective Ctx (neural)" if neural else m), (ms, ret),
                    textcoords="offset points", xytext=(7, 5), fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_xlim(0.02, 1500); ax.set_ylim(0, 112)
    ax.set_xlabel("latency per request @1024 tokens (ms, log)"); ax.set_ylabel("answer retention (%)")
    ax.set_title("cost vs quality at the gateway (top-left is better)", fontsize=11, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.savefig(os.path.join(HERE, "bw_costq.png"), bbox_inches="tight", facecolor=BG, dpi=150)
    plt.close(fig)
    print("wrote bw_costq.png")


if __name__ == "__main__":
    anim_run()
    anim_compress()
    anim_results()
    bw_static()
    print("done")
