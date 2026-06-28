"""
Generate publication figures from results/*.json.
Writes both .pdf (for LaTeX) and .png (for HTML/preview) to figures/.
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "results")
F = os.path.join(HERE, "figures")
os.makedirs(F, exist_ok=True)

rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 140,
})

PURPLE = "#7c5cf0"
NEURAL = "#e2483d"
COLORS = {
    "Lead": "#9aa0a6", "Random": "#c7ccd1",
    "TF-IDF": "#3aa0c2", "TF-IDF+Q": "#1f7a99",
    "TextRank": "#9b6cf0", "TextRank+Q": "#6f4bd8",
    "Centroid+MMR": "#16a085", "BM25+Q": "#e08e0b",
    "SelectiveContext(neural)": NEURAL,
}


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def savefig(fig, stem):
    fig.savefig(os.path.join(F, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(F, stem + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  wrote figures/" + stem + ".{pdf,png}")


def fig_latency():
    lat = load("latency.json")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m, data in lat.items():
        sizes = sorted(int(s) for s in data.keys())
        ys = [data[str(s)]["ms_median"] for s in sizes]
        style = dict(marker="o", lw=2.2, ms=5, color=COLORS.get(m, "#444"))
        if m == "SelectiveContext(neural)":
            style.update(marker="s", lw=2.6, ms=6)
        ax.plot(sizes, ys, label=m, **style)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Compression latency (ms/request, median)")
    ax.set_title("Routing-layer compression latency (Apple M3 Pro, CPU)")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
    savefig(fig, "fig_latency")


def fig_retention_curve():
    q = load("quality.json")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    methods = [m for m in COLORS if any(r["method"] == m for r in q)]
    for m in methods:
        rows = sorted([r for r in q if r["method"] == m], key=lambda r: r["mean_token_reduction"])
        if not rows:
            continue
        x = [r["mean_token_reduction"] * 100 for r in rows]
        y = [r["answer_retention"] * 100 for r in rows]
        style = dict(marker="o", lw=2.0, ms=5, color=COLORS.get(m, "#444"))
        if m == "SelectiveContext(neural)":
            style.update(marker="s", lw=2.6, ms=7, zorder=5)
        ax.plot(x, y, label=m, **style)
    ax.set_xlabel("Token reduction (%)  —  higher = more compression")
    ax.set_ylabel("Answer retention (%)")
    ax.set_title("Answer retention vs. compression\n(multi-doc SQuAD, gold in middle)")
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower left")
    savefig(fig, "fig_retention")


def fig_cost_quality():
    q = load("quality.json")
    lat = load("latency.json")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    # manual label offsets (points) to avoid overlap in the dense top-left cluster
    off = {
        "BM25+Q": (-2, 12), "Centroid+MMR": (8, 9), "TF-IDF+Q": (-46, 12),
        "TextRank+Q": (8, -2), "TF-IDF": (8, 0), "TextRank": (8, -10),
        "Random": (-8, -14), "Lead": (8, -2),
        "SelectiveContext(neural)": (-30, -16),
    }
    ha = {"TF-IDF+Q": "right", "Random": "right", "SelectiveContext(neural)": "right"}
    for m in COLORS:
        rows = [r for r in q if r["method"] == m and abs(r["keep_ratio"] - 0.33) < 1e-6]
        if not rows or m not in lat:
            continue
        ret = rows[0]["answer_retention"] * 100
        l = lat[m].get("1024") or lat[m][list(lat[m])[len(lat[m]) // 2]]
        ms = l["ms_median"]
        is_neural = m == "SelectiveContext(neural)"
        ax.scatter(ms, ret, s=(360 if is_neural else 170), color=COLORS[m],
                   marker="s" if is_neural else "o",
                   edgecolor="white", linewidth=1.2, zorder=4)
        ax.annotate(m if not is_neural else "Selective Context\n(neural)", (ms, ret),
                    textcoords="offset points", xytext=off.get(m, (7, 5)),
                    fontsize=8.2, ha=ha.get(m, "left"), zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 1500)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Latency per request @1024 tokens (ms, log scale)")
    ax.set_ylabel("Answer retention @ ~3x compression (%)")
    ax.set_title("Cost–quality frontier at the routing layer\n(top-left is better: cheap + faithful)")
    ax.axvspan(0.02, 6, color=PURPLE, alpha=0.06)
    ax.text(0.13, 4, "classical, model-free zone", color=PURPLE, fontsize=8, alpha=0.8)
    savefig(fig, "fig_cost_quality")


def fig_footprint():
    foot = load("footprint.json")
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    labels = ["Classical\n(TF-IDF / TextRank /\nMMR / BM25)", "Neural\n(distilgpt2,\nSelective Context)"]
    vals = [max(0.5, foot["classical"]["param_mb_fp32"]),
            foot["neural_distilgpt2"]["param_mb_fp32"]]
    bars = ax.bar(labels, vals, color=[PURPLE, NEURAL], width=0.6)
    ax.set_ylabel("Resident model weights (MB)")
    ax.set_title("Accelerator/Memory footprint")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                f"{v:.0f} MB" if v > 1 else "~0 MB", ha="center", fontsize=10, weight="bold")
    savefig(fig, "fig_footprint")


if __name__ == "__main__":
    print("Generating figures ...")
    fig_latency()
    fig_retention_curve()
    fig_cost_quality()
    fig_footprint()
    print("Done.")
