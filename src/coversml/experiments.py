"""The experiments behind the results table. Each function returns plain dicts so the
scripts can dump them to JSON; nothing here touches the file system."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .eval import bootstrap, calibration_table, cost_threshold, summary, threshold_table
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


def _years(h: Hackathon) -> np.ndarray:
    return pd.to_datetime(h.frame["dttm"], unit="ms").dt.year.to_numpy()


def _feature_sets(h: Hackathon) -> dict[str, np.ndarray]:
    meta = h.meta
    return {
        "all": np.hstack([h.emb, meta]),
        "no_date": np.hstack([h.emb, meta[:, 1:]]),
        "lyrics_only": h.emb,
        "metadata": meta,
        "date_only": meta[:, :1],
    }


def _at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    return {
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(int(y.sum()), 1), 3),
        "hidden_originals": fp,
        "share_of_originals": round(fp / max(int((y == 0).sum()), 1), 3),
    }


def _past_and_future(X, y, groups, train, test, seed):
    """Grouped folds on the training period; the future is scored by the same fold models,
    so the threshold picked on their out-of-fold scores applies to the same score scale."""
    tr = np.flatnonzero(train)
    oof = np.zeros(len(tr))
    future = np.zeros(int(test.sum()))
    params = {**CATBOOST, "random_seed": seed}
    for a, b in grouped_splits(X[tr], y[tr], groups[tr], seed=seed):
        m = CatBoostClassifier(**params)
        m.fit(X[tr][a], y[tr][a])
        oof[b] = m.predict_proba(X[tr][b])[:, 1]
        future += m.predict_proba(X[test])[:, 1] / 5
    return oof, future


def _paired_diff(y, p1, p2, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Group bootstrap of ROC-AUC(p1) - ROC-AUC(p2) on the same resampled songs."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    low, high = np.percentile(vals, [2.5, 97.5])
    point = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    return {
        "value": round(float(point), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
    }


def temporal(h: Hackathon, cutoff_year: int = 2022, seeds=(42, 43, 44), n_boot: int = 500) -> dict:
    """Train on tracks added before ``cutoff_year``, test on each later year.

    Scores are averaged over fold models and seeds; per-seed ROC-AUC is kept to show
    the spread from training noise. The threshold is chosen on the training period's
    out-of-fold scores and carried forward unchanged. Test tracks whose song also
    appears in the training period are dropped.
    """
    years = _years(h)
    train = years < cutoff_year
    seen = np.isin(h.groups, np.unique(h.groups[train]))
    test = ~train & ~seen
    test_years = sorted(int(y) for y in np.unique(years[test]))
    out = {
        "cutoff_year": cutoff_year,
        "train_years": [int(years[train].min()), int(years[train].max())],
        "n_train": int(train.sum()),
        "cover_share_train": round(float(h.y[train].mean()), 3),
        "dropped_shared_songs": {y: int(((years == y) & seen).sum()) for y in test_years},
        "tests": {
            y: {
                "n": int((test & (years == y)).sum()),
                "cover_share": round(float(h.y[test & (years == y)].mean()), 3),
            }
            for y in test_years
        },
        "seeds": list(seeds),
        "sets": {},
    }
    yt, gt, yrs = h.y[test], h.groups[test], years[test]
    scores = {}
    for name, X in _feature_sets(h).items():
        t0 = time.time()
        runs = [_past_and_future(X, h.y, h.groups, train, test, s) for s in seeds]
        oof = np.mean([r[0] for r in runs], axis=0)
        fut = np.mean([r[1] for r in runs], axis=0)
        scores[name] = fut
        thr = cost_threshold(h.y[train], oof)["threshold"]
        res = {
            "threshold": round(thr, 3),
            "train_cv": {
                **bootstrap("roc_auc", h.y[train], oof, h.groups[train], n_boot=n_boot).as_dict(),
                **_at_threshold(h.y[train], oof, thr),
                "per_seed_auc": [round(float(roc_auc_score(h.y[train], r[0])), 3) for r in runs],
            },
        }
        for y in test_years:
            m = yrs == y
            res[y] = {
                **bootstrap("roc_auc", yt[m], fut[m], gt[m], n_boot=n_boot).as_dict(),
                **_at_threshold(yt[m], fut[m], thr),
                "per_seed_auc": [round(float(roc_auc_score(yt[m], r[1][m])), 3) for r in runs],
                "mean_score_covers": round(float(fut[m][yt[m] == 1].mean()), 3),
                "mean_score_originals": round(float(fut[m][yt[m] == 0].mean()), 3),
            }
        res["seconds"] = round(time.time() - t0)
        out["sets"][name] = res
    out["all_minus_no_date"] = {
        y: _paired_diff(
            yt[yrs == y], scores["all"][yrs == y], scores["no_date"][yrs == y], gt[yrs == y]
        )
        for y in test_years
    }
    return out


def rolling(h: Hackathon, cutoffs=(2020, 2021, 2022), sets=("all", "no_date")) -> list[dict]:
    """The threshold's life span: train before each cutoff, score every later year."""
    years = _years(h)
    feats = _feature_sets(h)
    rows = []
    for c in cutoffs:
        train = years < c
        test = ~train & ~np.isin(h.groups, np.unique(h.groups[train]))
        for name in sets:
            oof, fut = _past_and_future(feats[name], h.y, h.groups, train, test, 42)
            thr = cost_threshold(h.y[train], oof)["threshold"]
            yt, yrs = h.y[test], years[test]
            for y in sorted(int(v) for v in np.unique(yrs)):
                m = yrs == y
                rows.append(
                    {
                        "cutoff": int(c),
                        "set": name,
                        "year": y,
                        "years_after": y - int(c) + 1,
                        "n": int(m.sum()),
                        "cover_share": round(float(yt[m].mean()), 3),
                        "roc_auc": round(float(roc_auc_score(yt[m], fut[m])), 3),
                        **_at_threshold(yt[m], fut[m], thr),
                    }
                )
    return rows


def by_year(h: Hackathon, oof: np.ndarray, first: int = 2016, n_boot: int = 500) -> list[dict]:
    """ROC-AUC of grouped out-of-fold scores inside each year (earlier years pooled).

    Within one year the date carries little information, so this shows how well the
    model ranks tracks once the period is fixed.
    """
    years = np.maximum(_years(h), first - 1)
    X = _feature_sets(h)["no_date"]
    oof_no_date, _ = _oof(X, h.y, grouped_splits(X, h.y, h.groups))
    rows = []
    for y in np.unique(years):
        m = years == y
        rows.append(
            {
                "year": int(y),
                "pooled_before": bool(y == first - 1),
                "n": int(m.sum()),
                "cover_share": round(float(h.y[m].mean()), 3),
                "all": bootstrap("roc_auc", h.y[m], oof[m], h.groups[m], n_boot=n_boot).as_dict(),
                "no_date": bootstrap(
                    "roc_auc", h.y[m], oof_no_date[m], h.groups[m], n_boot=n_boot
                ).as_dict(),
            }
        )
    return rows


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
