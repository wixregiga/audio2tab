"""Map MIDI pitches to (string, fret) with Viterbi over playable chord shapes."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
STD_OCTAVES = [2, 2, 3, 3, 3, 4]  # EADGBE low->high


def parse_tuning(spec: str = "EADGBE") -> list[int]:
    """'EADGBE', 'DADGBE', 'E2 A2 D3 G3 B3 E4', or 'D#2,G#2,...' -> MIDI pitches low->high."""
    spec = spec.strip()
    toks = spec.replace(",", " ").split()
    if len(toks) == 1:  # compact letters e.g. EADGBE / DADGAD / Eb... ; parse letter(+#/b)
        s, toks = toks[0], []
        i = 0
        while i < len(s):
            t = s[i].upper(); i += 1
            if i < len(s) and s[i] in "#b":
                t += s[i]; i += 1
            toks.append(t)
    pitches = []
    for k, t in enumerate(toks):
        name = t.rstrip("0123456789")
        octv = t[len(name):]
        base = name[0].upper() + name[1:]
        if base.endswith("b") and len(base) == 2:
            pc = (NOTE_NAMES.index(base[0]) - 1) % 12
        else:
            pc = NOTE_NAMES.index(base)
        if octv:
            pitches.append(12 * (int(octv) + 1) + pc)
        else:  # infer octave: nearest to standard string k
            ref = 12 * (STD_OCTAVES[min(k, 5)] + 1) + NOTE_NAMES.index("EADGBE"[min(k, 5)])
            cand = [12 * (o + 1) + pc for o in range(1, 6)]
            pitches.append(min(cand, key=lambda p: (abs(p - ref), -p)))
    return pitches


@dataclass
class Shape:
    frets: tuple  # per note in event order: (string_idx, fret)
    pos: float
    static: float


def candidates(pitches, open_strings, max_fret=22, max_span=4, limit=60):
    """All assignments of pitches to distinct strings with playable span."""
    opts = []
    for p in pitches:
        o = [(s, p - op) for s, op in enumerate(open_strings) if 0 <= p - op <= max_fret]
        if not o:
            return []
        opts.append(o)
    out = []
    for combo in product(*opts):
        strings = [s for s, _ in combo]
        if len(set(strings)) != len(strings):
            continue
        fretted = [f for _, f in combo if f > 0]
        span = (max(fretted) - min(fretted)) if fretted else 0
        if span > max_span:
            continue
        # fingerability: at most 4 fretting fingers; a barre (lowest fret used on >=2 strings) counts as one
        if fretted:
            lo = min(fretted)
            fingers = (1 if fretted.count(lo) >= 2 else fretted.count(lo)) + sum(1 for f in fretted if f != lo)
            if fingers > 4:
                continue
        pos = sum(fretted) / len(fretted) if fretted else 0.0
        static = 0.08 * pos + 0.3 * span + (2.0 if pos > 12 else 0.0) + 0.3 * max(0, pos - 9)
        # mild penalty for a low pitch on a high string (non-idiomatic)
        static += 0.02 * sum(f for _, f in combo)
        out.append(Shape(tuple(combo), pos, static))
    out.sort(key=lambda s: s.static)
    return out[:limit]


def assign(events, open_strings, max_fret=22, max_span=4, move_weight=1.0):
    """events: list of pitch lists. Returns per event list of (string, fret) aligned with
    (possibly reduced) pitch list. Too-dense or unplayable events drop notes."""
    cand_list, kept = [], []
    for pitches in events:
        ps = sorted(set(pitches))
        # drop out-of-range pitches
        ps = [p for p in ps if open_strings[0] <= p <= open_strings[-1] + max_fret]
        ps = ps[:6] if len(ps) <= 6 else ps[:1] + ps[-5:]
        c = candidates(ps, open_strings, max_fret, max_span)
        while not c and len(ps) > 1:  # drop inner notes until playable
            ps = ps[:-2] + ps[-1:] if len(ps) > 2 else ps[:1]
            c = candidates(ps, open_strings, max_fret, max_span)
        cand_list.append(c); kept.append(ps)
    # Viterbi
    n = len(events)
    if n == 0:
        return []
    INF = float("inf")
    cost, back, eff = [], [], []  # eff: hand position carried through open-string shapes
    prev_pos = None
    for i, c in enumerate(cand_list):
        if not c:
            cost.append([]); back.append([]); eff.append([]); continue
        if prev_pos is None:
            cost.append([s.static for s in c]); back.append([-1] * len(c)); eff.append([s.pos for s in c])
        else:
            j0 = prev_pos
            pc = cand_list[j0]
            row, br, er = [], [], []
            for s in c:
                best, bj = INF, -1
                for j, t in enumerate(pc):
                    tp = eff[j0][j]
                    # open-only shapes don't force hand movement; hand stays where it was
                    d = 0.0 if (s.pos == 0 or tp == 0) else abs(s.pos - tp)
                    mv = d + 2.0 * max(0, d - 4)
                    v = cost[j0][j] + move_weight * mv
                    if v < best:
                        best, bj = v, j
                row.append(best + s.static); br.append(bj)
                er.append(s.pos if s.pos > 0 else eff[j0][bj])
            cost.append(row); back.append(br); eff.append(er)
        prev_pos = i
    # backtrack
    result = [[] for _ in range(n)]
    idxs = [i for i in range(n) if cand_list[i]]
    if not idxs:
        return result
    last = idxs[-1]
    k = min(range(len(cost[last])), key=lambda q: cost[last][q])
    for pos_i in range(len(idxs) - 1, -1, -1):
        i = idxs[pos_i]
        result[i] = list(cand_list[i][k].frets)
        k = back[i][k]
    return result
