import numpy as np
import pandas as pd

from coversml.hackathon import build_groups
from coversml.split import assert_no_group_leak, grouped_folds


def _frame():
    ids = [f"t{i}" for i in range(12)]
    links = ["Noo"] * 12
    links[1] = "t0"  # t1 is a cover of t0
    links[2] = "t0"  # t2 too
    links[5] = "t5"  # self link, as the editors write for originals
    emb = np.random.default_rng(0).normal(size=(12, 8)).astype(np.float32)
    emb[7] = emb[6]  # t6 and t7 share a lyric text
    target = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1]
    frame = pd.DataFrame({"track_id": ids, "original_track_id": links, "target": target})
    return frame, emb


def test_versions_of_one_song_share_a_group():
    frame, emb = _frame()
    g = build_groups(frame, emb)
    assert g[0] == g[1] == g[2]
    assert g[6] == g[7]
    assert g[0] != g[5] and g[3] != g[4]


def test_grouped_folds_never_split_a_song():
    frame, emb = _frame()
    g = build_groups(frame, emb)
    y = frame["target"].to_numpy()
    for tr, te in grouped_folds(y, g, n_splits=3):
        assert_no_group_leak(g, tr, te)
