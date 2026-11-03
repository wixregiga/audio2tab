from audio2tab.tabfmt import render_tab, render_bar, string_labels

STD = [40, 45, 50, 55, 59, 64]


def test_labels():
    assert [l.strip() for l in string_labels(STD)] == ["E", "A", "D", "G", "B", "e"]


def test_bar_render_and_alignment():
    bar = [None] * 16
    bar[0] = [(0, 0), (1, 2), (2, 2)]
    bar[8] = [(5, 12)]
    lines = render_bar(bar, 6)
    assert len({len(l) for l in lines}) == 1  # all strings same width
    assert lines[0].startswith("0") and lines[1].startswith("2")
    assert "12" in lines[5]


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
