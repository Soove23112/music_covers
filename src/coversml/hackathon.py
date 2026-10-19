"""The hackathon table: one row per track with editor label, metadata and a lyric embedding.

Columns: ``track_id``, ``original_track_id`` (``Noo`` when unknown), ``track_remake_type``,
``dttm`` (Unix ms), ``language`` (``Noo`` when unknown), ``duration`` (ms), ``target``
(1 = cover) and 768 embedding columns named ``0`` ... ``767``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _UnionFind

MISSING = "Noo"


@dataclass
class Hackathon:
    frame: pd.DataFrame  # metadata and label, one row per track
    emb: np.ndarray  # (n, 768) float32
    groups: np.ndarray  # song id per row; versions of one song share an id
    y: np.ndarray  # 1 = cover

    @property
    def meta(self) -> np.ndarray:
        """dttm, duration and a language code as a float matrix."""
        lang = pd.factorize(self.frame["language"])[0].astype(np.float32)[:, None]
        return np.hstack([self.frame[["dttm", "duration"]].to_numpy(np.float32), lang])

    def linked_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Row indices (cover, original) for covers whose original is present in the table."""
        f = self.frame
        idx = {t: i for i, t in enumerate(f["track_id"])}
        link = f["original_track_id"].where(f["original_track_id"] != MISSING)
        other = f[link.notna() & (link != f["track_id"])]
        pairs = [
            (idx[a], idx[b])
            for a, b in zip(other["track_id"], other["original_track_id"], strict=True)
            if b in idx
        ]
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def build_groups(frame: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
    """Union of two "same song" relations: the editors' link to the original and an
    identical embedding, which means an identical lyric text."""
    uf = _UnionFind()
    ids = frame["track_id"].tolist()
    pos = {t: i for i, t in enumerate(ids)}
    for t in ids:
        uf.find(t)
    for a, b in zip(ids, frame["original_track_id"], strict=True):
        if b != MISSING and b != a and b in pos:
            uf.union(a, b)
    key = pd.Series([hash(r.tobytes()) for r in np.round(emb, 4)])
    for members in key.groupby(key).indices.values():
        for j in members[1:]:
            uf.union(ids[int(members[0])], ids[int(j)])
    codes, _ = pd.factorize(pd.Series([uf.find(t) for t in ids]))
    return codes


def load(path: str | Path) -> Hackathon:
    df = pd.read_csv(path, low_memory=False).drop_duplicates("track_id").reset_index(drop=True)
    emb_cols = [c for c in df.columns if c.isdigit()]
    emb = df[emb_cols].to_numpy(np.float32)
    frame = df.drop(columns=emb_cols)
    return Hackathon(
        frame=frame, emb=emb, groups=build_groups(frame, emb), y=frame["target"].to_numpy()
    )
