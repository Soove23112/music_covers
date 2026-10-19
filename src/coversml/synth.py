"""Synthetic catalogue in the partner's file format.

The real dataset cannot be redistributed. This generator produces three JSON-lines
files with the same columns so that the whole pipeline (audit, features, grouped
validation, model, report) runs end to end on a public fixture. The signal is
deliberately imperfect: covers keep most of the original's words, sometimes get a
"(Cover)"/"(Live)" suffix, usually but not always appear later, and a share of
covers is "translated" into a second pseudo-language.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

LANGS = ["EN", "RU", "ES", "DE", "FR"]
GENRES = ["POP", "ROCK", "RAP", "FOLK", "ELECTRONIC", "JAZZ"]
SUFFIXES = ["(Cover)", "(Live)", "(Acoustic)", "(Karaoke Version)", "- Remix", "(Tribute)"]


def _vocab(rng: random.Random, n: int, lang: str) -> list[str]:
    letters = "abcdefghijklmnopqrstuvwxyz" if lang != "RU" else "абвгдежзиклмнопрстуфхцчшэюя"
    words = set()
    while len(words) < n:
        words.add("".join(rng.choice(letters) for _ in range(rng.randint(3, 8))))
    return sorted(words)


def _song_text(rng: random.Random, vocab: list[str]) -> str:
    chorus = " ".join(rng.choice(vocab) for _ in range(rng.randint(6, 10)))
    lines = []
    for _ in range(rng.randint(2, 4)):
        for _ in range(rng.randint(3, 5)):
            lines.append(" ".join(rng.choice(vocab) for _ in range(rng.randint(5, 9))))
        lines.append(chorus)
        lines.append(chorus)
    return "\n".join(lines)


def _perturb(rng: random.Random, text: str, vocab: list[str], rate: float) -> str:
    out = []
    for line in text.split("\n"):
        words = line.split()
        kept = []
        for w in words:
            r = rng.random()
            if r < rate / 2:
                continue  # drop a word
            if r < rate:
                kept.append(rng.choice(vocab))  # replace a word
            else:
                kept.append(w)
        out.append(" ".join(kept))
    return "\n".join(out)


def _translate(rng: random.Random, text: str, mapping: dict[str, str]) -> str:
    return "\n".join(" ".join(mapping.get(w, w) for w in line.split()) for line in text.split("\n"))


def _isrc(lang: str, kind: str, dttm: float, base_ms: float, span_ms: float, song: int) -> str:
    year = int(2000 + (dttm - base_ms) / span_ms * 16) % 100
    return f"{lang}{kind}{year:02d}{song:05d}"


def generate(out_dir: str | Path, n_songs: int = 300, seed: int = 0) -> dict[str, int]:
    """Write music_covers.json, music_lyrics.json, music_meta.json into ``out_dir``."""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_by_lang = {lang: _vocab(rng, 400, lang) for lang in LANGS}
    # a fixed "dictionary" for pseudo-translation between any two languages
    translations = {
        (a, b): dict(
            zip(vocab_by_lang[a], rng.sample(vocab_by_lang[b], len(vocab_by_lang[b])), strict=True)
        )
        for a in LANGS
        for b in LANGS
        if a != b
    }

    covers_rows, lyrics_rows, meta_rows = [], [], []
    next_id = 0

    def new_id() -> str:
        nonlocal next_id
        next_id += 1
        return f"t{next_id:06d}"

    base_ms = 1_200_000_000_000  # ~2008
    span_ms = 500_000_000_000  # ~16 years

    for s in range(n_songs):
        lang = rng.choice(LANGS)
        vocab = vocab_by_lang[lang]
        title = " ".join(w.capitalize() for w in rng.sample(vocab, rng.randint(1, 3)))
        text = _song_text(rng, vocab)
        genre = rng.choice(GENRES)
        orig_id = new_id()
        orig_dttm = base_ms + rng.random() * span_ms * 0.7
        orig_lyric = f"l{orig_id}"
        covers_rows.append(
            {"track_id": orig_id, "track_remake_type": "ORIGINAL", "original_track_id": orig_id}
        )
        meta_rows.append(
            {
                "track_id": orig_id,
                "dttm": orig_dttm,
                "title": title,
                "language": lang if rng.random() < 0.35 else None,
                "isrc": _isrc(lang, "XX", orig_dttm, base_ms, span_ms, s),
                "genres": [genre],
                "duration": rng.randint(120_000, 300_000),
            }
        )
        if rng.random() < 0.85:  # most originals have lyrics
            lyrics_rows.append({"lyricId": orig_lyric, "text": text, "track_id": orig_id})

        n_covers = rng.choices([0, 1, 2, 3, 5], weights=[25, 40, 20, 10, 5])[0]
        for _ in range(n_covers):
            cid = new_id()
            c_lang = lang
            c_text = text
            if rng.random() < 0.25:  # translated cover
                c_lang = rng.choice([x for x in LANGS if x != lang])
                c_text = _translate(rng, text, translations[(lang, c_lang)])
            c_text = _perturb(
                rng, c_text, vocab_by_lang[c_lang], rate=rng.choice([0.0, 0.05, 0.15])
            )
            c_title = title + (" " + rng.choice(SUFFIXES) if rng.random() < 0.4 else "")
            later = rng.random() < 0.9
            c_dttm = (
                orig_dttm + rng.random() * (base_ms + span_ms - orig_dttm)
                if later
                else base_ms + rng.random() * (orig_dttm - base_ms)
            )
            shares_lyric = c_text == text and rng.random() < 0.3
            lyric_id = orig_lyric if shares_lyric else f"l{cid}"
            link = orig_id if rng.random() < 0.08 else None  # editors rarely fill the link
            covers_rows.append(
                {"track_id": cid, "track_remake_type": "COVER", "original_track_id": link}
            )
            meta_rows.append(
                {
                    "track_id": cid,
                    "dttm": c_dttm,
                    "title": c_title,
                    "language": c_lang if rng.random() < 0.3 else None,
                    "isrc": _isrc(c_lang, "YY", c_dttm, base_ms, span_ms, s),
                    "genres": [genre if rng.random() < 0.7 else rng.choice(GENRES)],
                    "duration": rng.randint(110_000, 320_000),
                }
            )
            if rng.random() < 0.6:  # fewer covers have lyrics
                lyrics_rows.append({"lyricId": lyric_id, "text": c_text, "track_id": cid})

    # unlabelled noise tracks with their own lyrics
    for _ in range(n_songs // 5):
        tid = new_id()
        lang = rng.choice(LANGS)
        meta_rows.append(
            {
                "track_id": tid,
                "dttm": base_ms + rng.random() * span_ms,
                "title": " ".join(rng.sample(vocab_by_lang[lang], 2)).title(),
                "language": lang if rng.random() < 0.3 else None,
                "isrc": None,
                "genres": [rng.choice(GENRES)],
                "duration": rng.randint(120_000, 300_000),
            }
        )
        lyrics_rows.append(
            {"lyricId": f"l{tid}", "text": _song_text(rng, vocab_by_lang[lang]), "track_id": tid}
        )

    for name, rows in (
        ("music_covers.json", covers_rows),
        ("music_lyrics.json", lyrics_rows),
        ("music_meta.json", meta_rows),
    ):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return {"covers": len(covers_rows), "lyrics": len(lyrics_rows), "meta": len(meta_rows)}
