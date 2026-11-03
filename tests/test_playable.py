"""Every tab position is playable: one note per string, reachable span, the right pitches, sane hand movement."""
from audio2tab.fretboard import parse_tuning, assign, candidates

STD = [40, 45, 50, 55, 59, 64]


def test_parse_tunings():
    assert parse_tuning("EADGBE") == STD
    assert parse_tuning("DADGBE") == [38, 45, 50, 55, 59, 64]
    assert parse_tuning("D2 A2 D3 G3 B3 E4") == [38, 45, 50, 55, 59, 64]
    assert parse_tuning("EbAbDbGbBbEb") == [39, 44, 49, 54, 58, 63]


def test_one_note_per_string_and_span():
    for shape in candidates([40, 47, 52, 55, 59, 64], STD):
        strings = [s for s, _ in shape.frets]
        assert len(set(strings)) == len(strings)
        fr = [f for _, f in shape.frets if f]
        assert not fr or max(fr) - min(fr) <= 4


def test_em_chord_shape():
    (res,) = assign([[40, 47, 52, 55, 59, 64]], STD)
    assert sorted(res) == [(0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)]


def test_pitches_map_back():
    ev = [[45], [48, 52, 55], [62], [67, 71]]
    for pitches, f in zip(ev, assign(ev, STD)):
        assert sorted(STD[s] + fr for s, fr in f) == sorted(pitches)


def test_low_hand_movement():
    # scale run: no jump bigger than 4 frets between consecutive fretted notes, stays low on the neck
    out = assign([[p] for p in [57, 60, 62, 64, 67, 69, 72, 69, 67, 64]], STD)
    frets = [f for ((_, f),) in out if f > 0]
    assert all(abs(a - b) <= 4 for a, b in zip(frets, frets[1:]))
    assert max(frets) <= 10


def test_capo_and_out_of_range():
    capo = [p + 2 for p in STD]
    (res,) = assign([[42]], capo)
    assert res == [(0, 0)]
    (res,) = assign([[30]], STD)  # below range -> dropped
    assert res == []


def test_too_dense_reduced():
    (res,) = assign([[40, 41, 42, 43, 44, 45, 46, 47]], STD)
    assert 1 <= len(res) <= 6


def test_wide_stretch_column_rejected():
    assert candidates([53, 80], STD) == []  # only E13 + e16 -> unplayable stretch


def test_unfingerable_shape_rejected():
    # 5 different fretted positions across 5 strings (no barre) needs 5 fingers
    for sh in candidates([41, 47, 52, 57, 62], STD):
        fr = [f for _, f in sh.frets if f]
        lo = min(fr) if fr else 0
        fingers = (1 if fr.count(lo) >= 2 else fr.count(lo)) + sum(1 for f in fr if f != lo)
        assert fingers <= 4 and (max(fr) - min(fr) if fr else 0) <= 4
