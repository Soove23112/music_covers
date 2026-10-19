import numpy as np

from coversml.eval import bootstrap, cost_threshold, threshold_table


def test_cost_threshold_prefers_precision_when_false_positives_are_expensive():
    y = np.array([1] * 70 + [0] * 30)
    p = np.concatenate([np.linspace(0.5, 1.0, 70), np.linspace(0.0, 0.6, 30)])
    cheap = cost_threshold(y, p, cost_fp=1, cost_fn=1)
    strict = cost_threshold(y, p, cost_fp=10, cost_fn=1)
    assert strict["threshold"] >= cheap["threshold"]
    assert strict["false_positives"] <= cheap["false_positives"]


def test_threshold_table_is_monotone():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.5 + rng.normal(0.3, 0.2, 500), 0, 1)
    rows = threshold_table(y, p, (0.2, 0.5, 0.8))
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.25, 400), 0, 1)
    groups = rng.integers(0, 80, 400)
    e = bootstrap("roc_auc", y, p, groups, n_boot=200)
    assert e.low <= e.value <= e.high
