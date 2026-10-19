"""Metrics with confidence intervals, calibration and cost-based thresholds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class Estimate:
    value: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "ci_low": self.low, "ci_high": self.high}


def _metric(name: str, y: np.ndarray, p: np.ndarray) -> float:
    if name == "roc_auc":
        return float(roc_auc_score(y, p))
    if name == "pr_auc":
        return float(average_precision_score(y, p))
    if name == "brier":
        return float(brier_score_loss(y, p))
    raise ValueError(name)


def bootstrap(
    name: str,
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> Estimate:
    """Group bootstrap: resample whole songs, not rows.

    Rows of one song are not independent (a cover and its original share the
    text), so resampling rows would understate the interval.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    point = _metric(name, y, p)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(_metric(name, y[idx], p[idx]))
    low, high = np.percentile(vals, [2.5, 97.5])
    return Estimate(point, float(low), float(high))


def summary(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int = 1000
) -> dict[str, Estimate]:
    return {m: bootstrap(m, y, p, groups, n_boot=n_boot) for m in ("roc_auc", "pr_auc", "brier")}


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability table: for each score bin, predicted mean vs observed rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "n": int(m.sum()),
                "pred_mean": float(p[m].mean()),
                "observed_rate": float(y[m].mean()),
            }
        )
    return rows


def cost_threshold(
    y: np.ndarray, p: np.ndarray, cost_fp: float = 5.0, cost_fn: float = 1.0
) -> dict[str, float]:
    """Threshold minimising cost_fp * FP + cost_fn * FN.

    Product reading: a false positive hides an original from the listener, which
    is assumed ``cost_fp`` times worse than letting one cover through.
    """
    order = np.argsort(-p)
    y_sorted = y[order]
    p_sorted = p[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = y.sum() - tp
    cost = cost_fp * fp + cost_fn * fn
    best = int(np.argmin(cost))
    thr = float(p_sorted[best])
    pred = p >= thr
    tp_, fp_ = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn_ = int(((~pred) & (y == 1)).sum())
    prec = tp_ / max(tp_ + fp_, 1)
    rec = tp_ / max(tp_ + fn_, 1)
    return {
        "threshold": thr,
        "precision": prec,
        "recall": rec,
        "false_positives": fp_,
        "false_negatives": fn_,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
    }


def threshold_table(
    y: np.ndarray, p: np.ndarray, points: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
) -> list[dict[str, float]]:
    rows = []
    for thr in points:
        pred = p >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        rows.append(
            {
                "threshold": thr,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "false_positives": fp,
            }
        )
    return rows


def pr_curve(y: np.ndarray, p: np.ndarray) -> dict[str, list[float]]:
    prec, rec, _ = precision_recall_curve(y, p)
    return {"precision": prec.tolist(), "recall": rec.tolist()}
