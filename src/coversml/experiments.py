"""The experiments behind the results table. Each function returns plain dicts so the
scripts can dump them to JSON; nothing here touches the file system."""

from __future__ import annotations

import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .eval import calibration_table, cost_threshold, summary, threshold_table
from .hackathon import Hackathon

CATBOOST = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.1,
    "auto_class_weights": "Balanced",
    "verbose": False,
    "random_seed": 42,
}


def _oof(X: np.ndarray, y: np.ndarray, splits, importance: bool = False):
    oof = np.zeros(len(y))
    imp = np.zeros(X.shape[1])
    for tr, te in splits:
        m = CatBoostClassifier(**CATBOOST)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if importance:
            imp += m.get_feature_importance() / 5
    return oof, imp


def _scores(y, oof) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y, oof)), 3),
        "pr_auc": round(float(average_precision_score(y, oof)), 3),
    }


def grouped_splits(X, y, groups, seed=42):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups))


def compare_splits(h: Hackathon) -> list[dict]:
    """Same features, random folds against folds grouped by song."""
    X = np.hstack([h.emb, h.meta[:, :2]])
    random = StratifiedKFold(5, shuffle=True, random_state=42).split(X, h.y)
    out = []
    for name, splits in (
        ("random_5fold", random),
        ("grouped_5fold", grouped_splits(X, h.y, h.groups)),
    ):
        t0 = time.time()
        oof, _ = _oof(X, h.y, splits)
        out.append({"name": name, **_scores(h.y, oof), "seconds": round(time.time() - t0)})
    return out


def whiten(emb: np.ndarray) -> np.ndarray:
    centred = emb - emb.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ (vt.T / s * np.sqrt(len(centred) - 1))


def ablation(h: Hackathon) -> list[dict]:
    """Grouped CV on feature subsets: what each group of columns is worth."""
    meta = h.meta
    sets = {
        "date_only": meta[:, :1],
        "duration_only": meta[:, 1:2],
        "metadata": meta,
        "lyrics_only": h.emb,
        "lyrics_whitened": whiten(h.emb),
        "lyrics_and_date_duration": np.hstack([h.emb, meta[:, :2]]),
        "all": np.hstack([h.emb, meta]),
    }
    out = []
    for name, X in sets.items():
        t0 = time.time()
        oof, _ = _oof(X, h.y, grouped_splits(X, h.y, h.groups))
        out.append(
            {
                "name": name,
                "n_features": int(X.shape[1]),
                **_scores(h.y, oof),
                "seconds": round(time.time() - t0),
            }
        )
    return out


def final(h: Hackathon, cost_fp: float = 5.0, n_boot: int = 500) -> tuple[dict, np.ndarray]:
    """Full model on grouped folds: metrics with intervals, thresholds, calibration, importance."""
    X = np.hstack([h.emb, h.meta])
    names = [f"emb_{i}" for i in range(h.emb.shape[1])] + ["dttm", "duration", "language"]
    oof, imp = _oof(X, h.y, grouped_splits(X, h.y, h.groups), importance=True)
    y = h.y
    n0 = int((y == 0).sum())
    order = np.argsort(-oof)
    sorted_y = y[order]
    recall = np.cumsum(sorted_y) / sorted_y.sum()
    k = int(np.argmax(recall >= 0.9))
    hidden_at_90 = int((1 - sorted_y[: k + 1]).sum())
    report = {
        "n_tracks": int(len(y)),
        "n_covers": int(y.sum()),
        "n_originals": n0,
        "metrics": {m: e.as_dict() for m, e in summary(y, oof, h.groups, n_boot=n_boot).items()},
        "cost_threshold": cost_threshold(y, oof, cost_fp, 1.0),
        "thresholds": threshold_table(y, oof, (0.3, 0.5, 0.7, 0.8, 0.9)),
        "at_recall_0.9": {
            "threshold": float(oof[order][k]),
            "false_hidden_originals": hidden_at_90,
            "share_of_originals": round(hidden_at_90 / n0, 3),
        },
        "calibration": calibration_table(y, oof, 10),
        "top_features": [(names[i], round(float(imp[i]), 2)) for i in np.argsort(-imp)[:10]],
        "metadata_share_of_importance": round(float(imp[-3:].sum() / imp.sum()), 3),
    }
    return report, oof


def _rank_of_original(En: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    S = En[a] @ En.T
    S[np.arange(len(a)), a] = -9
    order = np.argsort(-S, axis=1)
    return np.array([int(np.where(o == t)[0][0]) + 1 for o, t in zip(order, b, strict=True)])


def retrieval(h: Hackathon, n_random: int = 20000, seed: int = 0) -> dict:
    """How well nearest-neighbour search finds a cover's original, raw and whitened."""
    a, b = h.linked_pairs()
    rng = np.random.default_rng(seed)
    ra = rng.integers(0, len(h.y), n_random)
    rb = rng.integers(0, len(h.y), n_random)
    keep = ra != rb

    def run(E):
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        rank = _rank_of_original(En, a, b)
        pos = np.einsum("ij,ij->i", En[a], En[b])
        neg = np.einsum("ij,ij->i", En[ra[keep]], En[rb[keep]])
        stats = {
            "recall_at_1": round(float((rank == 1).mean()), 3),
            "recall_at_5": round(float((rank <= 5).mean()), 3),
            "recall_at_10": round(float((rank <= 10).mean()), 3),
            "mrr": round(float((1 / rank).mean()), 3),
            "same_song_median_cosine": round(float(np.median(pos)), 3),
            "random_pair_median_cosine": round(float(np.median(neg)), 3),
            "share_same_song_above_0_95": round(float((pos > 0.95).mean()), 3),
        }
        return stats, pos, neg

    raw, pos_raw, neg_raw = run(h.emb)
    white, pos_white, neg_white = run(whiten(h.emb))
    return {
        "n_pairs": int(len(a)),
        "raw": raw,
        "whitened": white,
        "_dists": {
            "pos_raw": pos_raw,
            "neg_raw": neg_raw,
            "pos_white": pos_white,
            "neg_white": neg_white,
        },
    }
