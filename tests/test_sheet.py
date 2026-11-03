"""The tab sheet: layout, bar grid, chord diagrams/voicings, tuning names."""
from audio2tab.tabfmt import render_tab, render_bar
from audio2tab.theory import voicing, voicing_str, detect_tuning, TUNINGS

STD = TUNINGS["Standard (EADGBE)"]
assert STD == [40, 45, 50, 55, 59, 64]


def test_full_tab_layout_wraps_at_80():
    bars = [[[(i % 6, i % 13)]] + [None] * 15 for i in range(12)]
    txt = render_tab(bars, STD, 80, ["Em"] * 12, ["Title: x"])
    lines = txt.splitlines()
    assert lines[0] == "Title: x"
    tab_lines = [l for l in lines if "|" in l]
    assert all(len(l) <= 80 for l in tab_lines)
    assert tab_lines[0].startswith("e|") and tab_lines[5].startswith("E|")
    assert all(l.endswith("|") for l in tab_lines)
    assert "Em" in txt and "12" in txt


def test_bar_has_16_columns_and_count_line():
    bar = [None] * 16; bar[0] = [(0, 3)]; bar[4] = [(1, 5)]
    txt = render_tab([bar], STD, 80, [[(0, "G")]])
    count = [l for l in txt.splitlines() if l.strip().startswith("1 .")][0]
    assert count.split() == ["1", ".", ".", ".", "2", ".", ".", ".", "3", ".", ".", ".", "4", ".", ".", "."]
    e_line = [l for l in txt.splitlines() if l.startswith("E|")][0]
    assert len(e_line.split("|")[1]) == 32  # 16 columns x width 2


def test_two_digit_frets_align():
    bar = [None] * 16; bar[0] = [(5, 16), (0, 13)]; bar[3] = [(2, 2)]
    lines = render_bar(bar, 6)
    assert len({len(l) for l in lines}) == 1
    assert len(lines[0]) == 16 * 3  # every column padded to width 3
    assert lines[2][9:12] == "2--"


def test_common_voicings():
    got = {c: voicing_str(voicing(c, STD)) for c in ["Em", "G", "C", "D", "A", "Am", "E", "F"]}
    assert got == {"Em": "022000", "G": "320003", "C": "x32010", "D": "xx0232", "A": "x02220",
                   "Am": "x02210", "E": "022100", "F": "133211"}


def test_voicing_other_tuning_has_right_notes():
    dd = TUNINGS["Drop D (DADGBE)"]
    v = voicing("D", dd)
    pcs = {(dd[i] + f) % 12 for i, f in enumerate(v) if f is not None}
    assert pcs == {2, 6, 9}


def test_slash_voicing_has_bass_lowest():
    v = voicing("C#/F", STD)
    low = next(STD[i] + f for i, f in enumerate(v) if f is not None)
    assert low % 12 == 5


def test_detect_tuning():
    mk = lambda ps: [(i * 0.5, i * 0.5 + 0.3, p, 0.8) for i, p in enumerate(ps)]
    assert detect_tuning(mk([40, 45, 52, 55, 59] * 10))[0].startswith("Standard")
    assert detect_tuning(mk([38, 45, 50, 52, 59, 40] * 10))[0].startswith("Drop D")
    assert detect_tuning(mk([39, 44, 51, 54, 58] * 10))[0].startswith("Half-step")
