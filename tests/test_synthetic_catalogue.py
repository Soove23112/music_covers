from coversml.data import load_tracks
from coversml.synth import generate


def test_synthetic_catalogue_round_trips(tmp_path):
    counts = generate(tmp_path, n_songs=60, seed=1)
    assert counts["covers"] > 60
    tracks = load_tracks(tmp_path)
    assert {"track_id", "label", "text_norm", "group", "dttm"} <= set(tracks.columns)
    assert tracks["label"].notna().sum() > 0
    # a cover that shares its lyric record with the original must share its group
    shared = tracks.groupby("lyricId")["group"].nunique()
    assert (shared == 1).all()
