"""Regression tests for QA findings on a real-song run (v1 -> v2)."""
import numpy as np
from audio2tab import analysis as A
from audio2tab.fretboard import candidates
from audio2tab.tabfmt import render_tab, render_bar
from audio2tab.pipeline import choose_source
from audio2tab import export

STD = [40, 45, 50, 55, 59, 64]


def chroma_of(*pcs):
    v = np.zeros(12); v[list(pcs)] = 1.0; return v


# 1. chords: two per bar, key-aware, power-vs-triad
def test_chord_hmm_finds_mid_bar_change():
    segs = np.stack([chroma_of(6, 10, 1), chroma_of(11, 3, 6), chroma_of(1, 5, 8), chroma_of(1, 5, 8)], axis=1)
    lab = A.chord_hmm(segs, "F# major")
    assert [A.NOTE_NAMES[r] + q for r, q in lab] == ["F#", "B", "C#", "C#"]


def test_power_chord_label():
    assert A.power_or_triad(chroma_of(6, 1), (6, "")) == "F#5"
    assert A.power_or_triad(chroma_of(6, 10, 1), (6, "")) == "F#"


def test_key_tie_broken_by_diatonic_chords():
    # F#-major chords incl. B (not diatonic to C# major)
    segs = np.stack([chroma_of(6, 10, 1), chroma_of(11, 3, 6), chroma_of(1, 5, 8), chroma_of(3, 6, 10)] * 4, axis=1)
    key, _, _ = A.detect_key(segs.sum(axis=1), segs.mean(axis=1), segs)
    assert key == "F# major"
    assert A.key_fifths(key) == (6, "major")


# 2. grid: 16 columns per 4/4 bar, label says so
def test_bar_has_16_columns_and_count_line():
    bar = [None] * 16; bar[0] = [(0, 3)]; bar[4] = [(1, 5)]
    txt = render_tab([bar], STD, 80, [[(0, "G")]])
    count = [l for l in txt.splitlines() if l.strip().startswith("1 .")][0]
    assert count.split() == ["1", ".", ".", ".", "2", ".", ".", ".", "3", ".", ".", ".", "4", ".", ".", "."]
    e_line = [l for l in txt.splitlines() if l.startswith("E|")][0]
    assert len(e_line.split("|")[1]) == 32  # 16 columns x width 2


# 5. formatting: two-digit frets pad the whole bar; strings stay aligned
def test_two_digit_frets_align():
    bar = [None] * 16; bar[0] = [(5, 16), (0, 13)]; bar[3] = [(2, 2)]
    lines = render_bar(bar, 6)
    assert len({len(l) for l in lines}) == 1
    assert len(lines[0]) == 16 * 3  # every column padded to width 3
    assert lines[2][9:12] == "2--"


def test_unfingerable_shape_rejected():
    # 5 different fretted positions across 5 strings (no barre) needs 5 fingers
    for sh in candidates([41, 47, 52, 57, 62], STD):
        fr = [f for _, f in sh.frets if f]
        lo = min(fr) if fr else 0
        fingers = (1 if fr.count(lo) >= 2 else fr.count(lo)) + sum(1 for f in fr if f != lo)
        assert fingers <= 4 and (max(fr) - min(fr) if fr else 0) <= 4


# 4. over-segmentation
def test_clean_notes_merges_fragments_and_strums():
    onsets = np.array([0.0, 0.75, 1.5])
    notes = [(0.0, 0.3, 54, 0.8), (0.01, 0.3, 58, 0.7), (0.03, 0.3, 61, 0.7),   # strum
             (0.32, 0.7, 54, 0.6),                                            # retrigger, no audio onset
             (0.40, 0.5, 66, 0.35),                                           # weak, unsupported
             (0.75, 1.2, 59, 0.8), (0.80, 1.2, 63, 0.8),                      # strum, 50 ms spread
             (1.5, 2.0, 61, 0.9), (1.55, 1.9, 65, 0.9)]
    clean, st = A.clean_notes(notes, onsets, sixteenth_s=0.186)
    starts = sorted({round(n[0], 3) for n in clean})
    assert starts == [0.0, 0.75, 1.5]
    assert st["merged_retriggers"] == 1 and st["dropped_no_audio_onset"] == 1
    assert len(clean) == 7


# 7. downbeats: shifted bar line -> one short bar; pickup
def test_downbeats_insert_short_bar_when_phase_shifts():
    nb = 64
    chg = np.zeros(nb); low = np.zeros(nb)
    for b in list(range(0, 30, 4)) + list(range(30, nb, 4)):
        chg[b] = 1; low[b] = 1
    downs, conf, _ = A.downbeats(low, chg, np.zeros(nb), 0, nb, 4)
    gaps = np.diff(downs)
    assert sorted(set(gaps)) == [2, 4] and list(gaps).count(2) == 1


def test_downbeats_pickup():
    nb = 40
    chg = np.zeros(nb); low = np.zeros(nb)
    for b in range(2, nb, 4):
        chg[b] = 1; low[b] = 1
    downs, _, _ = A.downbeats(low, chg, np.zeros(nb), 0, nb, 4)
    assert downs[0] == 2 and set(np.diff(downs)) == {4}


# 9. separation choice
def test_choose_source():
    assert choose_source({"guitar": 0.07}, 0.12)[0] == "stem"
    assert choose_source({"guitar": 0.005}, 0.2)[0] == "mix"


# 3. key / tempo / time signature in exports
def test_exports_carry_key_tempo_meter(tmp_path):
    import guitarpro, pretty_midi
    meta = dict(title="k", tempo=81.0, tuning=STD, capo=0, beats_per_bar=4, sub=4, n_bars=2,
                bar_lengths=[4, 2], key_fifths=6, key_mode="major", first_bar_time=0.0)
    ev = [(0, 0, [(0, 2, 42, 4)]), (1, 0, [(1, 4, 49, 4)])]
    export.write_gp5(ev, meta, tmp_path / "a.gp5")
    song = guitarpro.parse(str(tmp_path / "a.gp5"))
    assert song.key == guitarpro.KeySignature.FMajorSharp and song.tempo == 81
    assert [m.header.timeSignature.numerator for m in song.tracks[0].measures] == [4, 2]
    export.write_musicxml(ev, meta, tmp_path / "a.musicxml")
    x = (tmp_path / "a.musicxml").read_text()
    assert "<fifths>6</fifths>" in x and "<per-minute>81</per-minute>" in x and "<beats>2</beats>" in x
    export.write_midi([(0.0, 0.5, 42, 0.8)], tmp_path / "a.mid", 81.0, meta)
    pm = pretty_midi.PrettyMIDI(str(tmp_path / "a.mid"))
    assert pm.key_signature_changes[0].key_number == 6  # F# major
    assert pm.time_signature_changes[0].numerator == 4
    assert abs(pm.get_tempo_changes()[1][0] - 81.0) < 0.1
