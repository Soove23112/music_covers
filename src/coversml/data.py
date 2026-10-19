"""Loading the three catalogue files and building one track table.

Input files (JSON lines, one record per line), as provided by the streaming partner:

* ``music_covers.json``: ``track_id``, ``track_remake_type`` (ORIGINAL | COVER),
  ``original_track_id`` (for ORIGINAL rows it points to the row itself; for most
  COVER rows it is empty).
* ``music_lyrics.json``: ``lyricId``, ``text``, ``track_id``. A track may have several
  lyric records and a lyric record may be attached to several tracks.
* ``music_meta.json``: ``track_id``, ``dttm`` (first appearance in the catalogue,
  Unix milliseconds), ``title``, ``language``, ``isrc``, ``genres`` (list), ``duration`` (ms).

The output is one row per track that has lyrics, with a ``group`` column: tracks
that are known to be the same song (through an explicit original link, a shared
lyric record or an identical normalised text) share one group id. Validation must
never put two tracks of one group on different sides of a split.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COVERS_FILE = "music_covers.json"
LYRICS_FILE = "music_lyrics.json"
META_FILE = "music_meta.json"

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalise_text(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace.

    Two lyrics that differ only in casing, punctuation or line breaks map to the
    same string, which is what we need to detect duplicated texts across tracks.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


def text_key(text: str) -> str:
    """Short stable hash of the normalised text, used as a join key."""
    return hashlib.md5(normalise_text(text).encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, orient="records", lines=True)


@dataclass
class Catalogue:
    covers: pd.DataFrame
    lyrics: pd.DataFrame
    meta: pd.DataFrame


def load_catalogue(data_dir: str | Path) -> Catalogue:
    data_dir = Path(data_dir)
    covers = _read_jsonl(data_dir / COVERS_FILE)
    lyrics = _read_jsonl(data_dir / LYRICS_FILE)
    meta = _read_jsonl(data_dir / META_FILE)
    for df in (covers, lyrics, meta):
        df["track_id"] = df["track_id"].astype("string")
    if "original_track_id" in covers:
        covers["original_track_id"] = covers["original_track_id"].astype("string")
    lyrics["lyricId"] = lyrics["lyricId"].astype("string")
    return Catalogue(covers=covers, lyrics=lyrics, meta=meta)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(tracks: pd.DataFrame, covers: pd.DataFrame, lyrics: pd.DataFrame) -> pd.Series:
    """Connected components over three kinds of "same song" edges.

    Edges: track -> original_track_id (when it is another track), tracks sharing a
    lyric record, tracks with an identical normalised text. Returns a Series of group
    ids aligned with ``tracks.index``.
    """
    uf = _UnionFind()
    for tid in tracks["track_id"]:
        uf.find(tid)

    linked = covers.dropna(subset=["original_track_id"])
    linked = linked[linked["original_track_id"] != linked["track_id"]]
    for a, b in zip(linked["track_id"], linked["original_track_id"], strict=True):
        uf.union(a, b)

    for _, grp in lyrics.groupby("lyricId")["track_id"]:
        ids = grp.tolist()
        for other in ids[1:]:
            uf.union(ids[0], other)

    for _, grp in tracks.groupby("text_key")["track_id"]:
        ids = grp.tolist()
        for other in ids[1:]:
            uf.union(ids[0], other)

    roots = tracks["track_id"].map(uf.find)
    codes, _ = pd.factorize(roots)
    return pd.Series(codes, index=tracks.index, name="group")


def build_tracks(cat: Catalogue) -> pd.DataFrame:
    """One row per track with lyrics: label, metadata, normalised text, group id.

    ``label`` is 1 for COVER, 0 for ORIGINAL, NaN when the track is not in the
    editors' file. When a track has several lyric records we keep the longest text.
    """
    lyr = cat.lyrics.copy()
    lyr["text"] = lyr["text"].fillna("").astype(str)
    lyr["text_len"] = lyr["text"].str.len()
    lyr = lyr.sort_values(["track_id", "text_len"], ascending=[True, False])
    n_lyric_ids = lyr.groupby("track_id")["lyricId"].nunique().rename("n_lyric_ids")
    first_lyric = lyr.drop_duplicates("track_id")[["track_id", "lyricId", "text"]]

    tracks = first_lyric.merge(cat.meta, on="track_id", how="left")
    tracks = tracks.merge(
        cat.covers[["track_id", "track_remake_type", "original_track_id"]],
        on="track_id",
        how="left",
    )
    tracks = tracks.merge(n_lyric_ids, on="track_id", how="left")

    tracks["label"] = tracks["track_remake_type"].map({"COVER": 1.0, "ORIGINAL": 0.0})
    tracks["text_norm"] = tracks["text"].map(normalise_text)
    tracks["text_key"] = tracks["text"].map(text_key)
    tracks["dttm"] = pd.to_datetime(tracks["dttm"], unit="ms", errors="coerce")
    tracks["title"] = tracks["title"].fillna("").astype(str)
    tracks["language"] = tracks["language"].fillna("UNK").astype(str)
    if "genres" in tracks:
        tracks["genres"] = tracks["genres"].apply(
            lambda g: list(g) if isinstance(g, (list, np.ndarray)) else []
        )
    else:
        tracks["genres"] = [[] for _ in range(len(tracks))]
    tracks = tracks[tracks["text_norm"].str.len() > 0].reset_index(drop=True)
    tracks["group"] = build_groups(tracks, cat.covers, cat.lyrics)
    return tracks


def load_tracks(data_dir: str | Path) -> pd.DataFrame:
    return build_tracks(load_catalogue(data_dir))
