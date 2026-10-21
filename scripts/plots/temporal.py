"""How much the model leans on the date: trained on earlier years, tested on later ones.

python scripts/plots/temporal.py results/ docs/temporal.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import AQUA, BLUE, GRID, INK, INK2, MUTED, SURF, clean_axes

TEXT_EN = {
    "title": "Does the model lean on the date? Train on the past, test on later years",
    "subtitle": (
        "Test tracks whose song appears in training are dropped. Scores averaged over 5 fold "
        "models and 3 seeds; intervals are a song-level bootstrap."
    ),
    "left": "Ranking quality: trained on 2009–2021",
    "right": "Threshold life span: trained on 2009–2019",
    "sets": {
        "all": "all features",
        "no_date": "all except date",
        "lyrics_only": "lyrics only",
        "metadata": "date, duration, language",
        "date_only": "date only",
    },
    "periods": ("2009–2021, cross-validation", "2022", "2023"),
    "chance": "chance",
    "note_date": "constant by construction: no tree has seen these dates",
    "diff": "date adds {d22:+.3f} [{l22:.3f}; {h22:.3f}] in 2022\nand {d23:+.3f} [{l23:.3f}; {h23:.3f}] in 2023 (paired)",
    "recall": "share of covers caught at a fixed threshold",
    "lines": ("all features", "all except date"),
    "covers": "covers {:.0%}",
    "auc_note": "ROC-AUC stays {lo:.2f}–{hi:.2f}:\nthe ranking holds, the score\nscale drifts",
}


def _left(ax, t, text):
    sets = list(text["sets"])
    colors = (MUTED, BLUE, AQUA)
    clean_axes(ax)
    ax.set_title(text["left"], loc="left", fontsize=12, color=INK, fontweight="bold", pad=12)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(len(sets) - 0.4, -0.6)
    for xv in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        ax.axvline(xv, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0.5, color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=1)
    ax.text(0.505, -0.5, text["chance"], fontsize=9.5, color=INK2, va="top")
    ax.spines["left"].set_visible(False)
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([text["sets"][s] for s in sets], fontsize=10.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ROC-AUC", fontsize=10.5, color=INK2)
    for i, s in enumerate(sets):
        r = t["sets"][s]
        for k, (key, col) in enumerate(zip(("train_cv", "2022", "2023"), colors, strict=True)):
            e = r[key]
            y = i + (k - 1) * 0.22
            ax.plot([e["ci_low"], e["ci_high"]], [y, y], color=col, lw=2, zorder=2)
            face = SURF if k == 0 else col
            ax.plot(e["value"], y, "o", ms=8, mfc=face, mec=col, mew=2, zorder=3)
    d = t["all_minus_no_date"]
    ax.text(
        0.53,
        0.5,
        text["diff"].format(
            d22=d["2022"]["value"],
            l22=d["2022"]["ci_low"],
            h22=d["2022"]["ci_high"],
            d23=d["2023"]["value"],
            l23=d["2023"]["ci_low"],
            h23=d["2023"]["ci_high"],
        ),
        fontsize=9.5,
        color=INK2,
        va="center",
        linespacing=1.35,
    )
    ax.text(0.53, len(sets) - 1, text["note_date"], fontsize=9.5, color=INK2, va="center")


def _right(ax, rolling, text, cutoff=2020):
    clean_axes(ax)
    ax.set_title(text["right"], loc="left", fontsize=12, color=INK, fontweight="bold", pad=12)
    rows = [r for r in rolling if r["cutoff"] == cutoff]
    years = sorted({r["year"] for r in rows})
    for yv in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(yv, color=GRID, lw=0.8, zorder=0)
    styles = {"all": dict(color=INK, ls="-"), "no_date": dict(color=INK2, ls=(0, (4, 3)))}
    for (name, st), label in zip(styles.items(), text["lines"], strict=True):
        pts = sorted((r["year"], r["recall"]) for r in rows if r["set"] == name)
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, lw=2, marker="o", ms=8, mfc=SURF, mew=2, zorder=3, **st)
        ax.text(xs[-1] + 0.12, ys[-1], label, fontsize=10, color=INK, va="center")
        if name == "all":
            for x, v in pts:
                ax.text(x, v + 0.04, f"{v:.0%}", ha="center", fontsize=9.5, color=INK)
    aucs = [r["roc_auc"] for r in rows]
    ax.text(
        years[0] - 0.2,
        0.2,
        text["auc_note"].format(lo=min(aucs), hi=max(aucs)),
        fontsize=9.5,
        color=INK2,
        va="center",
        linespacing=1.35,
    )
    share = {r["year"]: r["cover_share"] for r in rows if r["set"] == "all"}
    ax.set_xticks(years)
    ax.set_xticklabels(
        [f"{y}\n" + text["covers"].format(share[y]) for y in years], fontsize=9.5, color=INK
    )
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(years[0] - 0.3, years[-1] + 1.1)
    ax.set_ylim(0, 1.0)
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(text["recall"], fontsize=10, color=INK2)


def draw(results, out, text=TEXT_EN):
    with open(Path(results) / "temporal.json") as f:
        t = json.load(f)
    with open(Path(results) / "rolling.json") as f:
        rolling = json.load(f)

    fig = plt.figure(figsize=(12, 6.75), dpi=200, facecolor=SURF)
    fig.text(0.04, 0.945, text["title"], fontsize=18, fontweight="bold", color=INK, va="center")
    fig.text(0.04, 0.895, text["subtitle"], fontsize=10.5, color=INK2, va="center")
    _left(fig.add_axes([0.19, 0.19, 0.39, 0.60]), t, text)
    _right(fig.add_axes([0.67, 0.19, 0.31, 0.60]), rolling, text)

    x = 0.19
    yl = 0.05
    for k, (col, label) in enumerate(zip((MUTED, BLUE, AQUA), text["periods"], strict=True)):
        face = SURF if k == 0 else col
        fig.lines.append(
            plt.Line2D(
                [x + 0.008],
                [yl],
                transform=fig.transFigure,
                marker="o",
                ms=8,
                mfc=face,
                mec=col,
                mew=2,
                lw=0,
            )
        )
        fig.text(x + 0.02, yl, label, fontsize=10.5, color=INK2, va="center")
        x += 0.02 + 0.0068 * len(label) + 0.035
    fig.savefig(out, dpi=200, facecolor=SURF)
    print("saved", out)


if __name__ == "__main__":
    draw(*sys.argv[1:3])
