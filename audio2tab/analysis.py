"""Audio-side analysis: onsets, note cleanup, beat-synchronous chroma chords (HMM), key, downbeat."""
from __future__ import annotations
import numpy as np
from .fretboard import NOTE_NAMES

SR_A = 22050
HOP = 512

MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7, 5: -1, 10: -2, 3: -3, 8: -4}


def key_fifths(key: str):
    """'F# major' -> (6, 'major'); 'D# minor' -> (6, 'minor')."""
    name, mode = key.split()
    pc = NOTE_NAMES.index(name)
    rel = pc if mode == "major" else (pc + 3) % 12
    return MAJOR_FIFTHS[rel], mode


def load_mono(path):
    import librosa
    y, _ = librosa.load(str(path), sr=SR_A, mono=True)
    return y


def audio_onsets(y):
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=SR_A, hop_length=HOP)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR_A, hop_length=HOP, units="time", backtrack=False)
    return np.asarray(on, float), env


def clean_notes(notes, onsets, sixteenth_s, strum_s=0.05, snap_s=0.07, min_ioi_frac=0.6, weak_amp=0.5):
    """Collapse basic-pitch fragments into playable events.
    1) same-pitch retriggers with no audio onset -> merged into one sustained note
    2) notes starting within strum_s -> one strum event
    3) events snapped to nearest audio onset (<= snap_s); weak events with no audio onset dropped
    4) events closer than min_ioi_frac * 16th to the previous event are merged into it
    Returns (clean_notes, stats)."""
    stats = {"raw": len(notes)}
    def near(t, tol):
        if len(onsets) == 0:
            return None
        i = int(np.argmin(np.abs(onsets - t)))
        return float(onsets[i]) if abs(onsets[i] - t) <= tol else None
    # 1 retrigger merge
    by_pitch, merged = {}, []
    for s, e, p, a in sorted(notes):
        prev = by_pitch.get(p)
        if prev is not None and s - prev[1] < 0.06 and near(s, snap_s) is None:
            prev[1] = max(prev[1], e); prev[3] = max(prev[3], a); continue
        n = [s, e, p, a]; by_pitch[p] = n; merged.append(n)
    stats["merged_retriggers"] = len(notes) - len(merged)
    # 2 strum clusters
    merged.sort()
    clusters = []
    for n in merged:
        if clusters and n[0] - clusters[-1][0][0] <= strum_s:
            clusters[-1].append(n)
        else:
            clusters.append([n])
    # 3 snap / drop unsupported weak events
    events, dropped_weak = [], 0
    for c in clusters:
        t0 = min(n[0] for n in c)
        t = near(t0, snap_s)
        if t is None:
            if max(n[3] for n in c) < weak_amp:
                dropped_weak += len(c); continue
            t = t0
        events.append([t, c])
    stats["dropped_no_audio_onset"] = dropped_weak
    # 4 min IOI
    min_ioi = min_ioi_frac * sixteenth_s
    out_ev, merged_ioi = [], 0
    for t, c in events:
        if out_ev and t - out_ev[-1][0] < min_ioi:
            have = {n[2] for n in out_ev[-1][1]}
            for n in c:
                if n[2] not in have:
                    out_ev[-1][1].append(n)
                else:
                    merged_ioi += 1
            continue
        out_ev.append([t, list(c)])
    stats["merged_min_ioi"] = merged_ioi
    clean = []
    for t, c in out_ev:
        seen = set()
        for s, e, p, a in c:
            if p in seen:
                continue
            seen.add(p); clean.append((t, max(e, t + 0.05), p, a))
    clean.sort()
    stats["after_cleanup"] = len(clean)
    return clean, stats


def beat_chroma(y, beats):
    """Beat-synchronous CQT chroma (12 x n_beats-1), each column L2-normalised."""
    import librosa
    yh = librosa.effects.harmonic(y)
    C = librosa.feature.chroma_cqt(y=yh, sr=SR_A, hop_length=HOP)
    fr = librosa.time_to_frames(beats, sr=SR_A, hop_length=HOP)
    fr = np.clip(fr, 0, C.shape[1])
    out = np.zeros((12, len(beats) - 1))
    for i in range(len(beats) - 1):
        a, b = fr[i], max(fr[i] + 1, fr[i + 1])
        if a < C.shape[1]:
            out[:, i] = C[:, a:b].mean(axis=1)
    return out


QUAL = {"": (0, 4, 7), "m": (0, 3, 7)}


def _templates():
    names, T = [], []
    for r in range(12):
        for q, ivs in QUAL.items():
            v = np.zeros(12)
            v[[(r + i) % 12 for i in ivs]] = [1.0, 0.8, 0.9]
            names.append((r, q)); T.append(v / np.linalg.norm(v))
    return names, np.array(T)


CH_NAMES, CH_T = _templates()


def diatonic(key):
    name, mode = key.split()
    pc = NOTE_NAMES.index(name)
    if mode == "minor":
        pc = (pc + 3) % 12
    return {(pc, ""), ((pc + 5) % 12, ""), ((pc + 7) % 12, ""), ((pc + 2) % 12, "m"), ((pc + 4) % 12, "m"), ((pc + 9) % 12, "m")}


def chord_hmm(seg_chroma, key=None, change_penalty=0.12, key_bonus=0.06):
    """Viterbi over chord labels for each segment column. Returns list of (root, q) or None (silence)."""
    n = seg_chroma.shape[1]
    norms = np.linalg.norm(seg_chroma, axis=0)
    X = seg_chroma / (norms + 1e-9)
    E = CH_T @ X  # (K, n)
    if key:
        dia = diatonic(key)
        bonus = np.array([key_bonus if c in dia else 0.0 for c in CH_NAMES])
        E = E + bonus[:, None]
    K = len(CH_NAMES)
    score = E[:, 0].copy(); back = np.zeros((K, n), int)
    for t in range(1, n):
        stay = score
        best_prev = score.max() - change_penalty
        bp = int(score.argmax())
        new = np.where(stay >= best_prev, stay, best_prev)
        back[:, t] = np.where(stay >= best_prev, np.arange(K), bp)
        score = new + E[:, t]
    k = int(score.argmax()); path = [k]
    for t in range(n - 1, 0, -1):
        k = back[k, t]; path.append(k)
    path = path[::-1]
    silent = norms < 0.05 * (np.median(norms[norms > 0]) if (norms > 0).any() else 1)
    return [None if silent[i] else CH_NAMES[path[i]] for i in range(n)]


def power_or_triad(seg, rq):
    """Label as a power chord when the third is weak relative to root and fifth."""
    r, q = rq
    third = seg[(r + (3 if q == "m" else 4)) % 12]
    rf = (seg[r] + seg[(r + 7) % 12]) / 2
    if third < 0.35 * rf:
        return NOTE_NAMES[r] + "5"
    return NOTE_NAMES[r] + q


def raw_reference(seg_chroma):
    """Independent-ish reference: best maj/min template per segment, no key prior, no smoothing."""
    X = seg_chroma / (np.linalg.norm(seg_chroma, axis=0) + 1e-9)
    return [CH_NAMES[int(np.argmax(CH_T @ X[:, i]))] for i in range(X.shape[1])]


def root_of(label):
    if not label:
        return None
    r = label[:2] if len(label) > 1 and label[1] == "#" else label[:1]
    return NOTE_NAMES.index(r)


def chord_match(label, ref_rq):
    """Root match, and quality match when label is a triad (power chords match either quality)."""
    if not label or ref_rq is None:
        return False
    r = root_of(label)
    if r != ref_rq[0]:
        return False
    if label.endswith("5"):
        return True
    q = "m" if label[len(NOTE_NAMES[r]):].startswith("m") else ""
    return q == ref_rq[1]


KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def ks_scores(h):
    out = []
    for r in range(12):
        for prof, nm in ((KS_MAJ, "major"), (KS_MIN, "minor")):
            out.append((float(np.corrcoef(h, np.roll(prof, r))[0, 1]), f"{NOTE_NAMES[r]} {nm}"))
    return sorted(out, reverse=True)


def detect_key(note_hist, chroma_mean, seg_chroma):
    """KS on notes+chroma; near-ties (<0.03) broken by how much chord mass is diatonic."""
    h = note_hist / (note_hist.sum() + 1e-9) + chroma_mean / (chroma_mean.sum() + 1e-9)
    sc = ks_scores(h)
    raw = raw_reference(seg_chroma)
    w = np.linalg.norm(seg_chroma, axis=0)
    def dia_cover(k):
        d = diatonic(k)
        return float(sum(wi for rq, wi in zip(raw, w) if rq in d) / (w.sum() + 1e-9))
    top = [s for s in sc if s[0] >= sc[0][0] - 0.03][:3]
    best = max(top, key=lambda s: (dia_cover(s[1]), s[0]))
    note = ""
    if len(top) > 1:
        note = "near tie with " + ", ".join(f"{k} ({r:.2f})" for r, k in top if k != best[1]) + f"; chose by diatonic chord coverage {dia_cover(best[1]):.0%}"
    return best[1], best[0], note


def choose_phase(beat_strength_low, novelty, note_on_beat, first_beat, bpb):
    """Downbeat phase (0..bpb-1) maximising low-band accent + chord-change novelty + note onsets."""
    def z(x):
        x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-9)
    L = min(len(beat_strength_low), len(novelty), len(note_on_beat))
    feats = z(beat_strength_low[:L]) + 1.5 * z(novelty[:L]) + 0.5 * z(note_on_beat[:L])
    # prior: songs usually start on a downbeat (first_beat is index 0 of the inputs); a pickup must win clearly
    scores = [float(feats[ph::bpb].mean()) + (1.2 if ph == 0 else 0.0) for ph in range(bpb)]
    ph = int(np.argmax(scores))
    srt = sorted(scores, reverse=True)
    conf = float(min(0.95, 0.5 + (srt[0] - srt[1]) / 2)) if bpb > 1 else 1.0
    return ph, conf, scores


def downbeats(low_b, chg, note_b, first_beat, nb, bpb, allow_short=True, short_cost=3.0, start_bonus=1.2,
              force_first=False):
    """Viterbi over beats. State = beats since the last downbeat. A downbeat scores
    z(low-band accent) + 1.5 z(chord change) + 0.5 z(note onsets). Bars are bpb beats; a bpb/2 bar
    costs short_cost (handles 2/4 bars or a dropped beat that shifts the bar line).
    The first downbeat falls within bpb beats of the first note (a pickup), with a bonus for the
    first note itself. Returns (downbeat beat indices, confidence, info)."""
    def z(x):
        x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-9)
    f = z(low_b[:nb]) + 1.5 * z(chg[:nb]) + 0.5 * z(note_b[:nb])
    NEG = -1e18
    W = bpb  # waiting (pre-first-downbeat) state index
    S = bpb + 1
    fb = min(first_beat, nb - 1)
    score = np.full(S, NEG)
    score[0] = f[fb] + start_bonus
    if not force_first:
        score[W] = 0.0
    back = []
    half = bpb // 2
    for b in range(fb + 1, nb):
        new = np.full(S, NEG); bp = np.zeros(S, int)
        cands = [(score[bpb - 1], bpb - 1)]
        if allow_short and half >= 1:
            cands.append((score[half - 1] - short_cost, half - 1))
        if b - fb < bpb:
            cands.append((score[W], W))
        v, src = max(cands)
        new[0] = v + f[b]; bp[0] = src
        for s_ in range(1, bpb):
            new[s_] = score[s_ - 1]; bp[s_] = s_ - 1
        if b - fb < bpb - 1:
            new[W] = score[W]; bp[W] = W
        back.append(bp); score = new
    st = int(np.argmax(score[:bpb]))
    downs = []
    for i in range(len(back) - 1, -1, -1):
        b = fb + 1 + i
        if st == 0:
            downs.append(b)
        st = back[i][st]
    if st == 0:
        downs.append(fb)
    downs = sorted(downs)
    # confidence: margin of chosen phase over best alternative constant phase in the same span
    alt = [float(np.mean(f[[d for d in range(fb + ph, nb, bpb)]])) for ph in range(bpb)]
    chosen = float(np.mean(f[downs])) if downs else 0.0
    srt = sorted(alt, reverse=True)
    conf = float(np.clip(0.5 + (chosen - (srt[1] if len(srt) > 1 else 0)) / 3, 0.05, 0.95))
    return downs, conf, {"constant_phase_scores": [round(a, 3) for a in alt], "chosen_mean": round(chosen, 3)}


CQT_LO = 24  # MIDI of CQT bin 0


def note_evidence(y):
    """CQT magnitude (1 bin/semitone from MIDI 24) and onset-strength envelope of the source audio."""
    import librosa
    C = np.abs(librosa.cqt(y, sr=SR_A, hop_length=HOP, fmin=librosa.midi_to_hz(CQT_LO), n_bins=84))
    env = librosa.onset.onset_strength(y=y, sr=SR_A, hop_length=HOP)
    return C, env


def verify_notes(notes, C, env, onset_min=2.5, rise_min=2.0, abs_min=0.15, silence=0.05, rel_min=0.1, oct_ratio=2.0, oct_rel=0.7,
                 fifth_ratio=3.0, fifth_rel=0.1, fold_ratio=1.5, low_limit=40):
    """Per-note check against the source audio. notes: (start, end, pitch, amp).
    - phantom: pitch energy near silence (< silence x that pitch's 95th-percentile level in the song), or weak onset strength
      at the note start AND (no energy rise in the note's own CQT bin, or low absolute energy) -> drop
    - weak salience: the pitch bin is < rel_min of the strongest bin in that frame and does not rise -> drop
      (frame-relative salience alone is unreliable in a full mix, where drums/bass/keys own the loudest bin)
    - octave/harmonic: the pitch is weak relative to a strongly present pitch an octave (12, 24),
      an octave+fifth (19; 3rd harmonic) or a fifth (7) below -> drop if that pitch is already
      sounding, else fold the note down to it.
    Returns (kept_notes, stats)."""
    med = float(np.median(env)) + 1e-9
    nf = C.shape[1]
    gb = np.percentile(C, 95, axis=1) + 1e-9  # per-pitch loud level over the song (robust to a loud bass/drum mix)
    out, st = [], {"dropped_phantom": 0, "dropped_weak_salience": 0, "dropped_harmonic": 0, "folded_octave": 0}
    def fr(t):
        return int(min(max(0, round(t * SR_A / HOP)), nf - 1))
    def sounding(p, t):
        return any(n[2] == p and n[0] - 0.03 <= t < n[1] for n in notes)
    for s, e, p, a in notes:
        f0 = fr(s); f1 = max(f0 + 1, min(nf, f0 + 6))
        seg = C[:, f0:f1].mean(axis=1)
        b = p - CQT_LO
        if not 0 <= b < C.shape[0]:
            out.append((s, e, p, a)); continue
        sal = float(seg[b]); rel = sal / (float(seg.max()) + 1e-9)
        on = float(env[max(0, f0 - 3):f0 + 3].max()) / med
        pre = float(C[b, max(0, f0 - 8):max(1, f0 - 2)].mean())
        rise = sal / (pre + 1e-9)
        ab = sal / gb[b]
        if ab < silence or (on < onset_min and (rise < rise_min or ab < abs_min)):
            st["dropped_phantom"] += 1; continue
        if rel < rel_min and rise < rise_min:
            st["dropped_weak_salience"] += 1; continue
        low = None
        for d in (12, 19, 24, 7):
            lp = p - d
            if b - d < 0 or lp < low_limit or rel >= (fifth_rel if d == 7 else oct_rel):
                continue  # the lower pitch must be a guitar note (not bass) and this pitch must be weak
            need = fifth_ratio if d == 7 else (oct_ratio if sounding(lp, s) else fold_ratio)
            if seg[b - d] >= need * sal and seg[b - d] >= 0.5 * seg.max():
                low = lp; break
        if low is not None:
            if sounding(low, s):
                st["dropped_harmonic"] += 1; continue
            st["folded_octave"] += 1; p = low
        out.append((s, e, p, a))
    # folding can create duplicates at the same start
    seen, ded = set(), []
    for n in sorted(out):
        k = (round(n[0], 3), n[2])
        if k in seen:
            st["dropped_harmonic"] += 1; continue
        seen.add(k); ded.append(n)
    return ded, st


def bass_pcs(C, beats, lo_midi=28, hi_midi=52, strong=0.6, allowed=None):
    """Per beat: pitch class of the lowest strong CQT bin in [lo_midi, hi_midi), plus a low-band energy weight.
    C: CQT from note_evidence (bin 0 = MIDI CQT_LO)."""
    import librosa
    fb = np.clip(librosa.time_to_frames(beats, sr=SR_A, hop_length=HOP), 0, C.shape[1])
    L = C[lo_midi - CQT_LO:hi_midi - CQT_LO]
    out = np.zeros((12, len(beats) - 1))
    for i in range(len(beats) - 1):
        a, b = fb[i], max(fb[i] + 1, fb[i + 1])
        if a >= C.shape[1]:
            continue
        v = L[:, a:b].mean(axis=1)
        if allowed is not None:  # semitone leakage (slides, vibrato): only scale tones can be the bass
            v = v * np.array([1.0 if (lo_midi + k) % 12 in allowed else 0.0 for k in range(len(v))])
        if v.max() <= 0:
            continue
        k = int(np.argmax(v >= strong * v.max()))
        out[(lo_midi + k) % 12, i] += v.max()
    return out


def apply_bass(label, seg_chroma, bass_vec, key=None, alt_ratio=0.85, min_share=0.5):
    """Use the segment's bass note to (a) relabel when the bass is not a chord tone but a triad rooted on
    the bass fits the chroma nearly as well (e.g. C# -> A#m), or (b) write an inversion as a slash chord (C#/F).
    bass_vec: 12-vector of bass weight over the segment. Power chords are never slashed."""
    if not label or bass_vec.sum() <= 0:
        return label
    bp = int(np.argmax(bass_vec))
    if bass_vec[bp] < min_share * bass_vec.sum():
        return label  # no dominant bass note in this segment
    r = root_of(label)
    if bp == r:
        return label
    x = seg_chroma / (np.linalg.norm(seg_chroma) + 1e-9)
    sc = dict(zip(CH_NAMES, CH_T @ x))
    q = "m" if label[len(NOTE_NAMES[r]):].startswith("m") else ""
    cur = sc.get((r, q), max(sc.get((r, ""), 0), sc.get((r, "m"), 0)))
    tones = {r, (r + 7) % 12} if label.endswith("5") else {r, (r + (3 if q == "m" else 4)) % 12, (r + 7) % 12}
    dia = diatonic(key) if key else None
    alts = [(sc[(bp, qq)], qq) for qq in ("", "m") if (bp, qq) in sc and (dia is None or (bp, qq) in dia)]
    if alts:
        s_alt, qq = max(alts)
        if s_alt >= alt_ratio * cur and (bp not in tones or s_alt >= cur):
            return power_or_triad(x, (bp, qq))
    if bp in tones and not label.endswith("5"):
        return f"{label}/{NOTE_NAMES[bp]}"
    return label


def scale_pcs(key):
    name, mode = key.split()
    r = NOTE_NAMES.index(name)
    return {(r + i) % 12 for i in ((0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10))}
