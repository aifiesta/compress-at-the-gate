"""Black-and-white charts for the plain article. No color; readable in grayscale."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
Q = json.load(open(os.path.join(ROOT, "results", "quality.json")))
LAT = json.load(open(os.path.join(ROOT, "results", "latency.json")))

rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": "#000", "axes.edgecolor": "#000", "axes.labelcolor": "#000",
    "xtick.color": "#000", "ytick.color": "#000",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
KEEP = 0.33
NEURAL = "SelectiveContext(neural)"
order = ["BM25+Q", "TF-IDF+Q", "TextRank+Q", "Centroid+MMR", "TF-IDF",
         "Random", "TextRank", "Lead", NEURAL]
lab = {NEURAL: "Selective Context\n(neural)"}


def ret(m):
    return [r for r in Q if r["method"] == m and abs(r["keep_ratio"] - KEEP) < 1e-6][0]["answer_retention"] * 100


def save(fig, stem):
    fig.savefig(os.path.join(HERE, stem + ".png"), bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", stem + ".png")


# 1) Answer retention bars
fig, ax = plt.subplots(figsize=(7.4, 4.6))
names = [lab.get(m, m) for m in order]
vals = [ret(m) for m in order]
colors = ["#000" if m != NEURAL else "white" for m in order]
bars = ax.barh(range(len(order)), vals, color=colors, edgecolor="#000",
               hatch=["" if m != NEURAL else "////" for m in order])
ax.set_yticks(range(len(order))); ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel("Answer retention at 3x compression (%)")
ax.set_xlim(0, 105)
for i, v in enumerate(vals):
    ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=11)
ax.set_title("Did the answer survive the compression?", loc="left", weight="bold")
save(fig, "bw_retention")

# 2) Latency bars (log)
fig, ax = plt.subplots(figsize=(7.4, 4.6))
order2 = ["Lead", "BM25+Q", "TF-IDF", "TF-IDF+Q", "Centroid+MMR", "TextRank",
          "TextRank+Q", NEURAL]
names2 = [lab.get(m, m) for m in order2]
ms = [LAT[m]["1024"]["ms_median"] for m in order2]
colors2 = ["#000" if m != NEURAL else "white" for m in order2]
ax.barh(range(len(order2)), ms, color=colors2, edgecolor="#000",
        hatch=["" if m != NEURAL else "////" for m in order2])
ax.set_yticks(range(len(order2))); ax.set_yticklabels(names2)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Time to compress one 1024-token prompt (ms, log scale)")
for i, v in enumerate(ms):
    ax.text(v * 1.15, i, f"{v:.2f} ms" if v < 10 else f"{v:.0f} ms", va="center", fontsize=10)
ax.set_xlim(0.02, 2000)
ax.set_title("How long the compressor takes (lower is better)", loc="left", weight="bold")
save(fig, "bw_latency")

# 3) Cost vs quality scatter, grayscale
fig, ax = plt.subplots(figsize=(7.4, 5.0))
pts = ["Lead", "Random", "TF-IDF", "TF-IDF+Q", "TextRank", "TextRank+Q",
       "Centroid+MMR", "BM25+Q", NEURAL]
off = {"BM25+Q": (6, 6), "Centroid+MMR": (6, -12), "TF-IDF+Q": (6, 6),
       "TextRank+Q": (6, 6), "TF-IDF": (6, 0), "TextRank": (6, -4),
       "Random": (6, 4), "Lead": (6, 2), NEURAL: (-12, -18)}
for m in pts:
    x = LAT[m]["1024"]["ms_median"]; y = ret(m)
    is_n = m == NEURAL
    ax.scatter(x, y, s=150 if not is_n else 200,
               facecolor="white" if is_n else "#000",
               edgecolor="#000", marker="s" if is_n else "o", linewidth=1.5, zorder=4,
               hatch="////" if is_n else "")
    name = "Selective Context (neural)" if is_n else m
    ax.annotate(name, (x, y), textcoords="offset points", xytext=off.get(m, (6, 4)),
                fontsize=9.5, ha="right" if is_n else "left")
ax.set_xscale("log"); ax.set_xlim(0.02, 1500); ax.set_ylim(0, 110)
ax.set_xlabel("Time per request (ms, log scale).  Left is cheaper.")
ax.set_ylabel("Answer retention (%).  Up is better.")
ax.set_title("Cheap and accurate is the top-left corner", loc="left", weight="bold")
save(fig, "bw_costquality")
