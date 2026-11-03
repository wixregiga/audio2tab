"""Exports open in other tools and carry the same notes, key, tempo and meter."""
from audio2tab import export

STD = [40, 45, 50, 55, 59, 64]


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
