"""Tuning, time-signature estimation and chord-diagram voicings."""
from __future__ import annotations
from itertools import product
import numpy as np
from .fretboard import NOTE_NAMES

# name -> MIDI low->high
TUNINGS = {
    "Standard (EADGBE)": [40, 45, 50, 55, 59, 64],
    "Drop D (DADGBE)": [38, 45, 50, 55, 59, 64],
    "Half-step down (Eb Ab Db Gb Bb Eb)": [39, 44, 49, 54, 58, 63],
    "Drop C# (C#G#C#F#A#D#)": [37, 44, 49, 54, 58, 63],
    "D standard (DGCFAD)": [38, 43, 48, 53, 57, 62],
    "Drop C (CGCFAD)": [36, 43, 48, 53, 57, 62],
}


def detect_tuning(notes):
    """Estimate tuning from the lowest confident pitches. Returns (name, midi list, confidence 0..1, reason)."""
    ps = np.array([p for s, e, p, a in notes if a >= 0.35 and (e - s) >= 0.08])
    if len(ps) < 20:
        return "Standard (EADGBE)", TUNINGS["Standard (EADGBE)"], 0.3, "too few confident notes; assumed standard"
    low = ps[ps < 46]
    cnt = {p: int((low == p).sum()) for p in range(34, 46)}
    lowest = None
    for p in range(34, 46):  # robust lowest: needs >=3 occurrences and >=1% of notes
        if cnt[p] >= max(3, 0.01 * len(ps)):
            lowest = p; break
    if lowest is None or lowest >= 40:
        conf = 0.6 + 0.3 * min(1, cnt.get(40, 0) / 10)
        return "Standard (EADGBE)", TUNINGS["Standard (EADGBE)"], round(conf, 2), f"no confident notes below E2 (E2 count {cnt.get(40,0)})"
    pcs = np.bincount(ps % 12, minlength=12) / len(ps)
    if lowest == 39:
        return "Half-step down (Eb Ab Db Gb Bb Eb)", TUNINGS["Half-step down (Eb Ab Db Gb Bb Eb)"], 0.6, "lowest confident note Eb2"
    if lowest == 38:
        # Drop D keeps the E(4)/A(9)/B(11) strings; D standard emphasises C/F/G(0,5,7)
        e_use = pcs[4] + pcs[11]; dstd = pcs[0] + pcs[5]
        if e_use >= dstd:
            return "Drop D (DADGBE)", TUNINGS["Drop D (DADGBE)"], 0.6, f"lowest D2 (x{cnt[38]}), E/B pitch classes common"
        return "D standard (DGCFAD)", TUNINGS["D standard (DGCFAD)"], 0.5, f"lowest D2 (x{cnt[38]}), C/F pitch classes common"
    if lowest == 37:
        return "Drop C# (C#G#C#F#A#D#)", TUNINGS["Drop C# (C#G#C#F#A#D#)"], 0.45, "lowest confident note C#2"
    return "Drop C (CGCFAD)", TUNINGS["Drop C (CGCFAD)"], 0.45, f"lowest confident note {NOTE_NAMES[lowest%12]}{lowest//12-1}"


def detect_meter(wav, beats):
    """Compare 3- vs 4-beat periodicity of beat-level onset strength. Returns (beats_per_bar, label, conf)."""
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    t = librosa.times_like(env, sr=sr)
    bs = np.interp(beats[(beats >= 0) & (beats <= t[-1])], t, env)
    # low-frequency (kick/bass) accent tends to mark downbeats
    S = np.abs(librosa.stft(y, n_fft=2048))
    low = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[:40]), sr=sr)
    bl = np.interp(beats[(beats >= 0) & (beats <= t[-1])], t, low)
    x = (bs / (bs.mean() + 1e-9)) + (bl / (bl.mean() + 1e-9))
    if len(x) < 12:
        return 4, "4/4", 0.3
    def strength(m):
        best = 0
        for ph in range(m):
            acc = x[ph::m].mean() - np.delete(x, np.arange(ph, len(x), m)).mean()
            best = max(best, acc)
        return best
    s3, s4 = strength(3), strength(4)
    s2 = strength(2)
    if s3 > s4 * 1.15 and s3 > s2:
        return 3, "3/4", round(min(0.9, 0.5 + (s3 - s4)), 2)
    return 4, "4/4", round(min(0.9, 0.5 + max(0, s4 - s3)), 2)


CHORD_IV = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "5": (0, 7)}


def parse_chord(name):
    root = name[:2] if len(name) > 1 and name[1] == "#" else name[:1]
    return NOTE_NAMES.index(root), name[len(root):]


STD_SHAPES = {"F": "133211", "B": "x24442", "F#m": "244222", "Bm": "x24432", "F#": "244322", "G#m": "466444",
              "Fm": "133111", "Cm": "x35543", "C#m": "x46654", "C#": "x46664", "Eb": "x68886", "D#": "x68886",
              "G#": "466544", "A#": "x13331", "Gm": "355333", "D#m": "x68876", "A#m": "x13321", "B7": "x21202", "C#7": "x46464", "D#7": "x68686", "F#7": "242322", "A#7": "x13131", "G#7": "464544"}


def voicing(name, open_strings, max_fret=12):
    """Best playable voicing: list per string low->high of fret or None (muted)."""
    r0, q0 = parse_chord(name)
    if q0 == "5":  # power chord: root on lowest two strings, root-fifth-octave shape
        best = None
        for i in (0, 1):
            f = (r0 - open_strings[i]) % 12
            fifth = (r0 + 7 - open_strings[i + 1]) % 12
            octv = (r0 - open_strings[i + 2]) % 12
            if max(f, fifth, octv) - min(f, fifth, octv) <= 3 and (best is None or f < best[0]):
                v = [None] * len(open_strings)
                v[i], v[i + 1], v[i + 2] = f, fifth, octv
                best = (f, v)
        if best:
            return best[1]
    if open_strings == [40, 45, 50, 55, 59, 64] and name in STD_SHAPES:
        return [None if c == "x" else int(c) for c in STD_SHAPES[name]]
    r, q = parse_chord(name)
    pcs = {(r + i) % 12 for i in CHORD_IV[q]}
    best, bs = None, -1e9
    for pos in range(0, 10):
        opts = []
        for op in open_strings:
            o = [None] + [f for f in ([0] + list(range(max(1, pos), pos + 4))) if (op + f) % 12 in pcs]
            opts.append(sorted(set(o), key=lambda v: (v is None, v)))
        for combo in product(*opts):
            played = [(i, f) for i, f in enumerate(combo) if f is not None]
            if len(played) < min(3, len(pcs) + 1) and q != "5" or len(played) < 2:
                continue
            first = played[0][0]
            if any(combo[i] is None for i in range(first, len(combo))):
                continue  # muted strings only at the bass side (strummable)
            notes = [open_strings[i] + f for i, f in played]
            if notes[0] % 12 != r:
                continue
            if {n % 12 for n in notes} != pcs:
                continue
            fr = [f for _, f in played if f > 0]
            span = max(fr) - min(fr) if fr else 0
            if span > 3:
                continue
            fingers = len(fr)
            if fingers > 4 and len(set(fr)) > 1 and fr.count(min(fr)) < 2:
                continue
            opens = sum(1 for _, f in played if f == 0)
            sc = 0.6 * len(played) - 0.45 * (max(fr) if fr else 0) - 0.2 * fingers + 0.3 * opens
            if opens and fr and min(fr) > 4:
                sc -= 2
            if sc > bs:
                bs, best = sc, list(combo)
    return best


def voicing_str(v):
    return "".join("x" if f is None else (str(f) if f < 10 else f"({f})") for f in v)


def ascii_diagram(name, v, open_strings):
    fr = [f for f in v if f]
    base = min(fr) if fr and max(fr) > 4 else 1
    lines = [f"{name} ({voicing_str(v)})", "  " + " ".join("x" if f is None else ("o" if f == 0 else " ") for f in v)]
    for row in range(base, base + 4):
        lab = f"{row}" if row == base and base > 1 else " "
        lines.append(lab.rjust(2) + "|".join(["", *["●" if f == row else "-" for f in v], ""])[1:-1].replace("", "") )
    return lines


def svg_diagram(name, v):
    fr = [f for f in v if f]
    base = min(fr) if fr and max(fr) > 4 else 1
    W, H, x0, y0, dx, dy = 110, 140, 20, 34, 14, 20
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">',
         f'<text x="{W/2}" y="14" text-anchor="middle" font-size="14" font-weight="bold">{name}</text>']
    n = len(v)
    for i in range(n):
        s.append(f'<line x1="{x0+i*dx}" y1="{y0}" x2="{x0+i*dx}" y2="{y0+4*dy}" stroke="#333"/>')
    for j in range(5):
        s.append(f'<line x1="{x0}" y1="{y0+j*dy}" x2="{x0+(n-1)*dx}" y2="{y0+j*dy}" stroke="#333" stroke-width="{4 if j==0 and base==1 else 1}"/>')
    if base > 1:
        s.append(f'<text x="{x0-14}" y="{y0+dy*0.7}" font-size="10">{base}fr</text>')
    for i, f in enumerate(v):
        x = x0 + i * dx
        if f is None:
            s.append(f'<text x="{x}" y="{y0-5}" text-anchor="middle" font-size="11">×</text>')
        elif f == 0:
            s.append(f'<circle cx="{x}" cy="{y0-9}" r="4" fill="none" stroke="#333"/>')
        else:
            s.append(f'<circle cx="{x}" cy="{y0+(f-base+0.5)*dy}" r="6" fill="#222"/>')
    s.append(f'<text x="{W/2}" y="{H-8}" text-anchor="middle" font-size="11">{voicing_str(v)}</text></svg>')
    return "".join(s)
