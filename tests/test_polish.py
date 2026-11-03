"""Regression tests for the v3 polish pass (phantom notes, octave errors, trailing bars, slash chords, MusicXML count)."""
import numpy as np
from audio2tab import analysis as A
from audio2tab.fretboard import candidates
from audio2tab.theory import voicing
from audio2tab import export

STD = [40, 45, 50, 55, 59, 64]
FPS = A.SR_A / A.HOP


def fake_audio(events, n_sec=4.0, noise=1e-3):
    """CQT/onset arrays: events = [(t0, t1, midi, level, onset_strength)]."""
    nf = int(n_sec * FPS)
    C = np.full((84, nf), noise); env = np.full(nf, 1.0)
    for t0, t1, p, lev, on in events:
        f0, f1 = int(t0 * FPS), int(t1 * FPS)
        C[p - A.CQT_LO, f0:f1] = lev
        env[f0] = max(env[f0], on)
    return C, env


def test_phantom_note_in_silence_dropped():
    C, env = fake_audio([(0.5, 1.0, 52, 1.0, 10.0)])
    notes = [(0.5, 1.0, 52, 0.8), (2.0, 2.4, 57, 0.8)]  # second note: no energy, no onset
    kept, st = A.verify_notes(notes, C, env)
    assert [n[2] for n in kept] == [52] and st["dropped_phantom"] == 1


def test_sustained_energy_without_onset_kept():
    # legato note: weak onset strength but its own pitch bin rises strongly
    C, env = fake_audio([(0.5, 1.5, 52, 1.0, 10.0), (1.0, 1.5, 55, 1.0, 1.2)])
    kept, _ = A.verify_notes([(0.5, 1.5, 52, 0.8), (1.0, 1.5, 55, 0.6)], C, env)
    assert {n[2] for n in kept} == {52, 55}


def test_octave_error_dropped_when_lower_sounding():
    C, env = fake_audio([(0.5, 1.0, 53, 1.0, 10.0), (0.5, 1.0, 65, 0.3, 10.0)])
    kept, st = A.verify_notes([(0.5, 1.0, 53, 0.9), (0.5, 1.0, 65, 0.6)], C, env)
    assert [n[2] for n in kept] == [53] and st["dropped_harmonic"] == 1


def test_octave_error_folded_when_lower_not_transcribed():
    C, env = fake_audio([(0.5, 1.0, 68, 1.0, 10.0), (0.5, 1.0, 80, 0.5, 10.0)])
    kept, st = A.verify_notes([(0.5, 1.0, 80, 0.6)], C, env)
    assert [n[2] for n in kept] == [68] and st["folded_octave"] == 1


def test_real_octave_doubling_kept():
    # G chord with doubled G: upper octave clearly present (ratio < 2) -> keep both
    C, env = fake_audio([(0.5, 1.0, 43, 1.0, 10.0), (0.5, 1.0, 55, 0.7, 10.0)])
    kept, _ = A.verify_notes([(0.5, 1.0, 43, 0.9), (0.5, 1.0, 55, 0.7)], C, env)
    assert {n[2] for n in kept} == {43, 55}


def test_bass_below_guitar_range_never_folds():
    C, env = fake_audio([(0.5, 1.0, 33, 1.0, 10.0), (0.5, 1.0, 45, 0.4, 10.0)])
    kept, _ = A.verify_notes([(0.5, 1.0, 45, 0.7)], C, env)
    assert [n[2] for n in kept] == [45]


def test_wide_stretch_column_rejected():
    assert candidates([53, 80], STD) == []  # only E13 + e16 -> unplayable stretch


def test_slash_chord_from_bass():
    ch = np.zeros(12); ch[[1, 5, 8]] = 1.0  # C# major
    bass = np.zeros(12); bass[5] = 1.0       # F in the bass
    assert A.apply_bass("C#", ch, bass, "F# major") == "C#/F"
    bass2 = np.zeros(12); bass2[1] = 1.0
    assert A.apply_bass("C#", ch, bass2, "F# major") == "C#"


def test_bass_not_in_chord_relabels_root():
    ch = np.zeros(12); ch[[10, 1, 5]] = 1.0; ch[8] = 0.4  # A#m with a little G#
    bass = np.zeros(12); bass[10] = 1.0
    assert A.apply_bass("C#", ch, bass, "F# major") == "A#m"


def test_bass_leakage_restricted_to_scale():
    assert 9 not in A.scale_pcs("F# major") and 10 in A.scale_pcs("F# major")


def test_slash_voicing_has_bass_lowest():
    v = voicing("C#/F", STD)
    low = next(STD[i] + f for i, f in enumerate(v) if f is not None)
    assert low % 12 == 5


def test_musicxml_pitch_count_matches_notes(tmp_path):
    import music21
    meta = dict(title="k", tempo=80.0, tuning=STD, capo=0, beats_per_bar=4, sub=4, n_bars=1,
                bar_lengths=[4], key_fifths=0, key_mode="major")
    ev = [(0, 0, [(0, 3, 43, 4), (1, 2, 47, 4), (2, 0, 50, 4)]), (0, 8, [(3, 0, 55, 8)])]
    export.write_musicxml(ev, meta, tmp_path / "a.musicxml")
    s = music21.converter.parse(str(tmp_path / "a.musicxml"))
    assert len(s.flatten().pitches) == 4
    # music21's recurse().pitches counts each pitch once per enclosing stream level (part, measure, note):
    # that is the 3x "2973 vs 991" in QA, not duplicated notes in the file.
    assert (tmp_path / "a.musicxml").read_text().count("<pitch>") == 4
