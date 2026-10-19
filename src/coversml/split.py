"""Leak-free cross-validation.

All tracks of one song (its original, its covers, anything that shares a lyric
record or an identical text) carry the same ``group`` id. ``StratifiedGroupKFold``
keeps every group inside one fold while keeping the cover/original ratio similar
across folds. ``assert_no_group_leak`` is the check the test-suite runs on every
split the pipeline produces.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from cv.split(np.zeros(len(y)), y, groups)


def assert_no_group_leak(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    shared = set(groups[train_idx]) & set(groups[test_idx])
    if shared:
        raise AssertionError(f"{len(shared)} groups appear on both sides of a split")
