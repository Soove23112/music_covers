"""Cosine similarity of same-song pairs against random pairs, raw and whitened embeddings.

python scripts/plots/similarity.py results/ docs/similarity.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import AQUA, BLUE, INK, INK2, MUTED, SURF, clean_axes


def panel(ax, pos, neg, title, r, xlim):
    clean_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    bins = np.linspace(xlim[0], xlim[1], 61)
    ax.hist(neg, bins=bins, density=True, color=BLUE, alpha=0.75, edgecolor=SURF, linewidth=0.4)
    ax.hist(pos, bins=bins, density=True, color=AQUA, alpha=0.75, edgecolor=SURF, linewidth=0.4)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)
    ax.set_xlabel("cosine similarity", fontsize=10.5, color=INK2)
    ax.text(
        0,
        -0.24,
        f"Recall@1 {r['recall_at_1']:.0%} · Recall@10 {r['recall_at_10']:.0%}",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0,
        -0.33,
        "original ranked first · in the top ten",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        va="top",
    )


def main(results, out):
    d = np.load(Path(results) / "retrieval_dists.npz")
    with open(Path(results) / "retrieval.json") as f:
        r = json.load(f)
    fig = plt.figure(figsize=(12, 6.75), dpi=200, facecolor=SURF)
    fig.text(
        0.04,
        0.945,
        "How close a cover's lyrics are to the original",
        fontsize=22,
        fontweight="bold",
        color=INK,
        va="center",
    )
    fig.text(
        0.04,
        0.895,
        f"Green: {r['n_pairs']} editor-linked cover/original pairs. Blue: 20,000 random track pairs.",
        fontsize=11,
        color=INK2,
        va="center",
    )
    panel(
        fig.add_axes([0.05, 0.33, 0.42, 0.47]),
        d["pos_raw"],
        d["neg_raw"],
        "Embeddings as they come",
        r["raw"],
        (0.4, 1.0),
    )
    panel(
        fig.add_axes([0.55, 0.33, 0.42, 0.47]),
        d["pos_white"],
        d["neg_white"],
        "After centering and whitening",
        r["whitened"],
        (-0.4, 1.0),
    )
    fig.patches.extend(
        [
            plt.Rectangle((0.05, 0.048), 0.014, 0.024, transform=fig.transFigure, color=AQUA),
            plt.Rectangle((0.27, 0.048), 0.014, 0.024, transform=fig.transFigure, color=BLUE),
        ]
    )
    fig.text(0.07, 0.06, "cover and its original", fontsize=10.5, color=INK2, va="center")
    fig.text(0.29, 0.06, "random pair", fontsize=10.5, color=INK2, va="center")
    fig.savefig(out, dpi=200, facecolor=SURF)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
