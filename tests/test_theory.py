from audio2tab.theory import voicing, voicing_str, detect_tuning, TUNINGS
from audio2tab.pipeline import chord_name
import numpy as np

STD = TUNINGS["Standard (EADGBE)"]


def test_common_voicings():
    got = {c: voicing_str(voicing(c, STD)) for c in ["Em", "G", "C", "D", "A", "Am", "E", "F"]}
    assert got == {"Em": "022000", "G": "320003", "C": "x32010", "D": "xx0232", "A": "x02220",
                   "Am": "x02210", "E": "022100", "F": "133211"}


def test_voicing_other_tuning_has_right_notes():
    dd = TUNINGS["Drop D (DADGBE)"]
    v = voicing("D", dd)
    pcs = {(dd[i] + f) % 12 for i, f in enumerate(v) if f is not None}
    assert pcs == {2, 6, 9}


def test_chord_name():
    w = np.zeros(12); w[[4, 7, 11]] = 1
    assert chord_name(w) == "Em"


def test_detect_tuning():
    mk = lambda ps: [(i * 0.5, i * 0.5 + 0.3, p, 0.8) for i, p in enumerate(ps)]
    assert detect_tuning(mk([40, 45, 52, 55, 59] * 10))[0].startswith("Standard")
    assert detect_tuning(mk([38, 45, 50, 52, 59, 40] * 10))[0].startswith("Drop D")
    assert detect_tuning(mk([39, 44, 51, 54, 58] * 10))[0].startswith("Half-step")
