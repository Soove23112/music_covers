import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, AQUA, INK, INK2, MUTED, SURF, GRID, TINT, BAND = (
    "#2a78d6",
    "#1baf7a",
    "#0b0b0b",
    "#52514e",
    "#8a8984",
    "#ffffff",
    "#e6e5e1",
    "#f3f4f2",
    "#c9ddf5",
)
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def clean_axes(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
