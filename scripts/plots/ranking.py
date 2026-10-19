"""Every track ranked by the model's cover score, with the truth strip and the threshold.

python scripts/plots/ranking.py data/hackathon.csv results/ docs/ranking.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import AQUA, BLUE, GRID, INK, INK2, MUTED, SURF, TINT, clean_axes


def main(csv, results, out):
    df = pd.read_csv(csv, low_memory=False, usecols=["track_id", "target"]).drop_duplicates(
        "track_id"
    )
    y = df["target"].to_numpy()
    oof = np.load(Path(results) / "oof.npy")
    n = len(y)
    with open(Path(results) / "final.json") as f:
        thr = json.load(f)["cost_threshold"]["threshold"]
    order = np.argsort(-oof)
    p = oof[order]
    y = y[order]
    cut = int((p >= thr).sum())
    n0 = int((y == 0).sum())
    fp = int((y[:cut] == 0).sum())
    fn = int((y[cut:] == 1).sum())
    share0 = pd.Series(1 - y).rolling(250, center=True, min_periods=50).mean().to_numpy()
    xs = np.arange(n)

    fig = plt.figure(figsize=(12, 6.75), dpi=200, facecolor=SURF)
    fig.text(
        0.04,
        0.945,
        "Model vs truth: tracks ranked by the 'this is a cover' score",
        fontsize=19,
        fontweight="bold",
        color=INK,
        va="center",
    )
    fig.text(
        0.04,
        0.895,
        "Each tick is one track. Top: editor labels. Bottom: grouped out-of-fold score; threshold at a 5:1 error cost.",
        fontsize=10.5,
        color=INK2,
        va="center",
    )
    axs = fig.add_axes([0.07, 0.70, 0.90, 0.09])
    axs.set_xlim(0, n)
    axs.set_ylim(0, 1)
    axs.axis("off")
    axs.vlines(xs[y == 1], 0, 1, colors=AQUA, lw=0.35)
    axs.vlines(xs[y == 0], 0, 1, colors=INK, lw=0.5)
    axs.axvline(cut, color=INK, lw=1.2, ls=(0, (4, 3)))
    fig.text(0.07, 0.80, "truth: cover", fontsize=10, color=AQUA, va="bottom", fontweight="bold")
    fig.text(0.155, 0.80, "original", fontsize=10, color=INK, va="bottom", fontweight="bold")

    ax = fig.add_axes([0.07, 0.17, 0.90, 0.50])
    clean_axes(ax)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.0)
    for yv in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.axhline(yv, color=GRID, lw=0.8, zorder=0)
    ax.fill_between(xs[:cut], 0, 1, color=TINT, zorder=0)
    ax.plot(xs, p, color=BLUE, lw=2.4, zorder=3)
    ax.plot(xs, share0, color=INK, lw=1.4, zorder=2, alpha=0.8)
    base = n0 / n
    ax.axhline(base, color=MUTED, lw=1.4, ls=(0, (2, 3)), zorder=2)
    ax.text(
        cut * 0.03,
        base + 0.025,
        f"baseline 'everything is a cover', random order: {base:.0%} originals in any window",
        fontsize=10,
        color=INK2,
        va="bottom",
    )
    ax.axvline(cut, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.text(
        cut - 60,
        0.97,
        f"threshold {thr:.2f}",
        fontsize=10.5,
        color=INK,
        ha="right",
        va="top",
        fontweight="bold",
    )
    ax.text(cut * 0.5, 0.88, "hide as cover", fontsize=11, color=INK2, ha="center", va="top")
    ax.text(cut + (n - cut) * 0.55, 0.97, "keep", fontsize=11, color=INK2, ha="center", va="top")
    ax.text(
        cut * 0.08,
        0.60,
        f"{fp:,} originals left of the threshold,\n{fp / n0:.0%} of all originals: the product\nwould hide them by mistake",
        fontsize=10.5,
        color=INK2,
        va="center",
        linespacing=1.35,
    )
    ax.text(
        cut + (n - cut) * 0.56,
        0.55,
        f"{fn:,} covers right of\nthe threshold: the product\nwould let them through",
        fontsize=10.5,
        color=INK2,
        va="center",
        linespacing=1.35,
    )
    ax.set_ylabel("model score", fontsize=10.5, color=INK2)
    ax.set_xlabel("tracks by descending score", fontsize=10.5, color=INK2)
    x = 0.07
    yl = 0.055
    for col, label in ((BLUE, "model score"), (INK, "share of originals in a 250-track window")):
        fig.lines.append(
            plt.Line2D([x, x + 0.02], [yl, yl], transform=fig.transFigure, color=col, lw=2.2)
        )
        fig.text(x + 0.026, yl, label, fontsize=10.5, color=INK2, va="center")
        x += 0.026 + 0.0068 * len(label) + 0.03
    fig.savefig(out, dpi=200, facecolor=SURF)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
