"""ASCII tab formatter (6-line e|B|G|D|A|E, bars, count line, wrapping)."""
from __future__ import annotations
from .fretboard import NOTE_NAMES


def string_labels(open_strings):
    labs = [NOTE_NAMES[p % 12] for p in open_strings]
    if labs[-1] == labs[0] and labs[-1] == "E":
        labs[-1] = "e"
    w = max(len(l) for l in labs)
    return [l.ljust(w) for l in labs]


def bar_width(slots):
    """Every column in a bar gets the same width: widest fret number + 1 dash."""
    w = 1
    for slot in slots:
        if slot:
            w = max(w, max(len(str(f)) for _, f in slot))
    return w + 1


def render_bar(slots, n_strings, slot_width=None):
    """slots: list of None or list[(string, fret)]. Returns list of strings per string (0 = lowest)."""
    w = slot_width or bar_width(slots)
    lines = [[] for _ in range(n_strings)]
    for slot in slots:
        for s in range(n_strings):
            cell = "-" * w
            if slot:
                for st, f in slot:
                    if st == s:
                        t = str(f); cell = t + "-" * (w - len(t))
            lines[s].append(cell)
    return ["".join(l) for l in lines]


def count_line(n_slots, w, sub, offset=0):
    out = ""
    for i in range(n_slots):
        k = i + offset
        mark = str(k // sub + 1) if k % sub == 0 else "."
        out += mark.ljust(w)
    return out


def render_tab(bars, open_strings, width=80, chords=None, header_lines=(), bar_numbers=None,
               bar_offsets=None, sub=4):
    """bars: list of bars (each a full list of slots). chords: per bar a str or list[(slot, name)].
    bar_offsets: per bar, first slot to show (pickup bars show only their tail)."""
    n = len(open_strings)
    labs = string_labels(open_strings)
    bar_numbers = bar_numbers or list(range(1, len(bars) + 1))
    bar_offsets = bar_offsets or [0] * len(bars)
    rendered = []
    for bi, b in enumerate(bars):
        off = bar_offsets[bi]
        vis = b[off:]
        w = bar_width(vis)
        lines = render_bar(vis, n, w)
        cnt = count_line(len(vis), w, sub, off)
        ch = chords[bi] if chords else None
        if isinstance(ch, str):
            ch = [(0, ch)] if ch else []
        top = [" "] * len(lines[0])
        lab = f"{bar_numbers[bi]}" + (" (pickup)" if off else "")
        for i, c in enumerate(lab):
            if i < len(top): top[i] = c
        chord_row = [" "] * len(lines[0])
        for slot, name in (ch or []):
            pos = max(0, (slot - off) * w)
            for i, c in enumerate(name):
                if pos + i < len(chord_row): chord_row[pos + i] = c
        rendered.append(("".join(top), "".join(chord_row), cnt, lines))
    out = list(header_lines)
    i = 0
    prefix = len(labs[0]) + 1
    while i < len(rendered):
        line_len, group = prefix, []
        while i < len(rendered):
            bl = len(rendered[i][3][0]) + 1
            if group and line_len + bl > width:
                break
            group.append(i); line_len += bl; i += 1
        for row in range(3):
            line = " " * prefix + "".join(rendered[g][row] + " " for g in group)
            if line.strip():
                out.append(line.rstrip())
        for s in range(n - 1, -1, -1):
            out.append(labs[s] + "|" + "".join(rendered[g][3][s] + "|" for g in group))
        out.append("")
    return "\n".join(out)
