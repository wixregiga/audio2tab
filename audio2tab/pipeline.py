"""audio -> separated guitar -> notes -> beat grid -> fretboard -> outputs."""
from __future__ import annotations
import json, os, subprocess, sys, time, warnings, logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

SR = 44100


@dataclass
class Params:
    tuning: str = "auto"
    capo: int = 0
    separate: bool = True
    model: str = "htdemucs_6s"
    start: float | None = None
    end: float | None = None
    title: str | None = None
    onset_threshold: float = 0.6
    frame_threshold: float = 0.4
    min_note_ms: float = 80.0
    min_amp: float = 0.3
    subdivision: int = 4  # slots per beat (4 = 16th notes)
    beats_per_bar: int | None = None  # None = auto-detect
    source: str = "auto"  # auto | stem | mix
    max_fret: int = 22
    max_span: int = 4
    pickup: str = "auto"  # auto | none
    bpm: float | None = None


def log(*a):
    print("[audio2tab]", *a, file=sys.stderr, flush=True)


def load_audio(path, out_wav, start=None, end=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)] if start is None else ["-t", str(end - start)]
    cmd += ["-i", str(path), "-vn", "-map", "0:a:0", "-ac", "2", "-ar", str(SR), "-af", "loudnorm=I=-16:TP=-1.5", str(out_wav)]
    subprocess.run(cmd, check=True)
    return out_wav


def separate(wav, workdir, model):
    """Run Demucs; return ({stem: path}, {stem: rms})."""
    import soundfile as sf
    sep_dir = Path(workdir) / "demucs"
    subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-d", "cpu", "-j", "2", "--segment", "7",
                    "-o", str(sep_dir), str(wav)], check=True, stdout=subprocess.DEVNULL)
    stems_dir = sep_dir / model / Path(wav).stem
    paths, rms = {}, {}
    for f in stems_dir.glob("*.wav"):
        if f.stem.startswith("_"):
            continue
        y, _ = sf.read(f)
        paths[f.stem] = f; rms[f.stem] = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    return paths, rms


def choose_source(rms, mix_rms, min_ratio=0.25):
    """Use the Demucs guitar stem only when it carries a real share of the mix energy.
    Otherwise transcribe the full mix: a weak guitar stem means Demucs didn't recognise the
    guitar (it went to bass/other), and transcribing that loses notes (synthetic test: F1
    0.34 with the weak stem vs 0.40 on the mix)."""
    g = rms.get("guitar", 0.0)
    ratio = g / (mix_rms + 1e-12)
    if ratio >= min_ratio:
        return "stem", ratio
    return "mix", ratio


def transcribe(wav, p: Params):
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    import pretty_midi
    _, midi, events = predict(str(wav), ICASSP_2022_MODEL_PATH,
                              onset_threshold=p.onset_threshold, frame_threshold=p.frame_threshold,
                              minimum_note_length=p.min_note_ms, minimum_frequency=70.0,
                              maximum_frequency=1400.0, multiple_pitch_bends=False,
                              melodia_trick=True)
    notes = [(float(s), float(e), int(pt), float(a)) for s, e, pt, a, *_ in events if a >= p.min_amp]
    notes.sort()
    return notes


def beat_grid(wav, p: Params):
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    if p.bpm:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=p.bpm, units="time", tightness=400)
    else:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    dur = len(y) / sr
    if len(beats) < 4 or tempo <= 0:
        tempo = tempo if tempo > 0 else 120.0
        beats = np.arange(0, dur, 60 / tempo)
    beats = np.asarray(beats, float)
    # extrapolate grid to cover whole file
    ib = float(np.median(np.diff(beats)))
    pre = np.arange(beats[0] - ib, -ib, -ib)[::-1]
    post = np.arange(beats[-1] + ib, dur + ib, ib)
    beats = np.concatenate([pre, beats, post])
    tempo = 60.0 / ib
    return tempo, beats, dur


def time_to_slot(t, beats, sub):
    b = np.interp(t, beats, np.arange(len(beats)))
    return int(round(b * sub))


TEMPLATES = {}
for root in range(12):
    for name, ivs in (("", (0, 4, 7)), ("m", (0, 3, 7)), ("7", (0, 4, 7, 10)), ("5", (0, 7))):
        v = np.zeros(12)
        for i in ivs:
            v[(root + i) % 12] = 1
        TEMPLATES[(root, name)] = v / np.linalg.norm(v)


def chord_name(weights):
    from .fretboard import NOTE_NAMES
    if weights.sum() <= 0:
        return ""
    w = weights / np.linalg.norm(weights)
    best = max(TEMPLATES, key=lambda k: float(w @ TEMPLATES[k]) - (0.02 if k[1] == "5" else 0) - (0.03 if k[1] == "7" else 0))
    if float(w @ TEMPLATES[best]) < 0.6:
        return ""
    return NOTE_NAMES[best[0]] + best[1]


def run(path, out_dir, p: Params):
    import shutil, soundfile as sf, librosa
    from .fretboard import parse_tuning, assign, NOTE_NAMES
    from .tabfmt import render_tab
    from .theory import detect_tuning, detect_meter, TUNINGS, voicing, voicing_str
    from . import export, analysis as A
    t0 = time.time(); timings = {}
    path = Path(path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    work = out / "work"; work.mkdir(exist_ok=True)
    title = p.title or path.stem
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
    wav = load_audio(path, work / f"{stem}.wav", p.start, p.end)
    ymix_st, _ = sf.read(wav)
    mix_rms = float(np.sqrt(np.mean(ymix_st ** 2)))
    timings["load"] = time.time() - t0

    # ---- separation + source choice
    stems, rms, ratio, source = {}, {}, None, "mix"
    if p.separate:
        t = time.time(); log("separating with Demucs", p.model, "...")
        stems, rms = separate(wav, work, p.model)
        timings["separate"] = time.time() - t
        source, ratio = choose_source(rms, mix_rms)
        if p.source in ("stem", "mix"):
            source = p.source
        log(f"guitar stem / mix RMS = {ratio:.2f} -> transcribing {source}")
    src_wav = stems["guitar"] if source == "stem" else wav
    source_desc = (f"Demucs {p.model} guitar stem (stem/mix RMS {ratio:.2f})" if source == "stem" else
                   (f"full mix (guitar stem too weak: stem/mix RMS {ratio:.2f})" if p.separate else "full mix (--no-separate)"))
    guitar_out = out / f"{stem}.guitar.wav"
    shutil.copy(src_wav, guitar_out)

    t = time.time(); log("transcribing notes (basic-pitch) ...")
    raw = transcribe(src_wav, p)
    timings["transcribe"] = time.time() - t
    t = time.time()
    tempo, beats, dur = beat_grid(wav, p)
    ysrc = A.load_mono(src_wav)
    onsets, _ = A.audio_onsets(ysrc)
    sixteenth = 60.0 / tempo / 4
    notes, cstats = A.clean_notes(raw, onsets, sixteenth)
    Cq, oenv = A.note_evidence(ysrc)
    notes, vstats = A.verify_notes(notes, Cq, oenv)
    cstats.update(vstats); cstats["after_cleanup"] = len(notes)

    # ---- tuning
    lows = sorted(n[2] for n in notes if n[2] < 40)
    if p.tuning == "auto":
        tun_label, base_tuning, tun_conf, tun_reason = detect_tuning(notes)
    else:
        base_tuning = parse_tuning(p.tuning)
        tun_label = next((k for k, v in TUNINGS.items() if v == base_tuning), " ".join(NOTE_NAMES[x % 12] for x in base_tuning))
        tun_conf, tun_reason = 1.0, "set by --tuning"
    if lows and p.tuning == "auto":
        from collections import Counter
        cnt = Counter(lows)
        desc = ", ".join(f"{NOTE_NAMES[q % 12]}{q // 12 - 1} x{c}" for q, c in sorted(cnt.items()))
        tun_reason += f"; notes below E2 after cleanup: {desc} ({len(lows) / max(1, len(notes)):.1%} of notes, below the 1%/3-note threshold so treated as bleed/octave errors)" if base_tuning[0] == 40 else ""
    elif p.tuning == "auto":
        tun_reason = tun_reason.replace("no confident notes below E2", "no notes below E2")
    try:
        off = float(librosa.estimate_tuning(y=ysrc, sr=A.SR_A)) * 1.0  # fraction of a semitone
    except Exception:
        off = 0.0
    tun_ambig = ""
    if base_tuning == TUNINGS["Standard (EADGBE)"]:
        tun_ambig = ("Ambiguous: half-step-down tuning (Eb) with every shape one fret higher sounds identical; "
                     "audio can't tell them apart. Ask the player.")
    timings["onsets_cleanup"] = time.time() - t

    if p.beats_per_bar is None:
        p.beats_per_bar, ts_label, ts_conf = detect_meter(wav, beats)
    else:
        ts_label, ts_conf = f"{p.beats_per_bar}/4", 1.0
    bpb, sub = p.beats_per_bar, p.subdivision
    spb = bpb * sub

    # ---- chroma (stem + half-weight bass if separated, else mix)
    t = time.time()
    if source == "stem":
        ych = ysrc + 0.5 * A.load_mono(stems["bass"]) if "bass" in stems else ysrc
    else:
        ych = A.load_mono(wav)
    bch = A.beat_chroma(ych, beats)  # 12 x (nb-1)
    timings["chroma"] = time.time() - t

    # ---- downbeats: Viterbi over beats; bars are bpb beats, a short (bpb/2) bar allowed at a cost
    ymono = A.load_mono(wav)
    S = np.abs(librosa.stft(ymono, n_fft=2048, hop_length=A.HOP))
    low_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[:40]), sr=A.SR_A, hop_length=A.HOP)
    tt = librosa.times_like(low_env, sr=A.SR_A, hop_length=A.HOP)
    nb = len(beats) - 1
    low_b = np.interp(beats[:nb], tt, low_env)
    bidx = np.interp([n[0] for n in notes], beats, np.arange(len(beats))) if notes else np.array([0.0])
    note_b = np.zeros(nb)
    for bi, n in zip(bidx, notes):
        k = int(round(bi))
        if 0 <= k < nb and abs(bi - k) < 0.15:
            note_b[k] += n[3]
    beat_lab = A.chord_hmm(bch, None, change_penalty=0.3)
    chg = np.array([0.0] + [1.0 if beat_lab[i] != beat_lab[i - 1] else 0.0 for i in range(1, nb)])
    first_beat = max(0, int(np.floor(bidx.min() + 0.15)))
    starts, ds_conf, ds_info = A.downbeats(low_b, chg, note_b, first_beat, nb, bpb, allow_short=(bpb % 2 == 0),
                                           force_first=(p.pickup == "none"))
    pickup_beats = starts[0] - first_beat  # notes before the first downbeat -> pickup bar 0
    if pickup_beats > 0:
        starts = [starts[0] - bpb] + starts
    has_pickup = pickup_beats > 0
    lengths = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)] + [bpb]
    # trim trailing bars after the last note; the final bar only as long as the music lasts (whole beats)
    trimmed_bars, final_bar_beats = 0, None
    if notes:
        last_b = float(np.interp(max(n[1] for n in notes), beats, np.arange(len(beats))))
        last_on = float(np.interp(max(n[0] for n in notes), beats, np.arange(len(beats))))
        keep = int(np.searchsorted(starts, last_on, side="right"))
        trimmed_bars = len(starts) - keep
        starts, lengths = starts[:keep], lengths[:keep]
        need = int(np.ceil(min(last_b, len(beats) - 1) - starts[-1] - 1e-6))
        need = max(1, min(lengths[-1], need))
        if need < lengths[-1]:
            final_bar_beats = need
            lengths[-1] = need
    n_bars = len(starts)
    st_slots = np.array(starts) * sub

    # ---- grid: (bar, pos) for every note
    def locate(tm):
        g = int(round(np.interp(tm, beats, np.arange(len(beats))) * sub))
        b = int(np.searchsorted(st_slots, g, side="right") - 1)
        b = min(max(b, 0), n_bars - 1)
        return b, g - st_slots[b], g
    grid, grid_dupes = {}, 0
    for n in notes:
        b, pos, g = locate(n[0])
        pos = min(max(pos, 0), lengths[b] * sub - 1)
        d = max(1, int(round(np.interp(n[1], beats, np.arange(len(beats))) * sub)) - g)
        lst = grid.setdefault((b, pos), [])
        if all(x[0] != n[2] for x in lst):
            lst.append((n[2], n[3], d, n))
        else:
            grid_dupes += 1
    keys = sorted(grid)
    open_strings = [x + p.capo for x in base_tuning]
    events = [[x[0] for x in grid[k]] for k in keys]
    t = time.time()
    fing = assign(events, open_strings, p.max_fret - p.capo, p.max_span)
    timings["fretboard"] = time.time() - t
    bars = [[None] * (lengths[b] * sub) for b in range(n_bars)]
    ev_list, final_notes, dropped_unplayable = [], [], 0
    for (b, pos), ev, f in zip(keys, events, fing):
        if not f:
            dropped_unplayable += len(ev); continue
        bars[b][pos] = f
        byp = {x[0]: x for x in grid[(b, pos)]}
        ns = []
        for s_, fr in f:
            pt = open_strings[s_] + fr
            x = byp[pt]
            ns.append((s_, fr, pt, x[2]))
            final_notes.append((x[3][0], x[3][1], pt, x[3][3]))
        ev_list.append((b, pos, ns))
        dropped_unplayable += len(ev) - len(f)
    final_notes.sort()
    pickup_slots = (bpb - pickup_beats) * sub if has_pickup else 0
    n_short = sum(1 for b in range(1 if has_pickup else 0, n_bars - 1) if lengths[b] != bpb)

    # ---- key + chords (per half bar; one segment for short bars)
    t = time.time()
    seg_list = []  # (bar, slot_in_bar, beat_lo, beat_hi)
    for b in range(n_bars):
        half = bpb // 2 if (bpb % 2 == 0 and lengths[b] == bpb) else lengths[b]
        lo0 = starts[b] + (bpb - pickup_beats if (b == 0 and has_pickup) else 0)
        for lo in range(starts[b], starts[b] + lengths[b], half):
            lo_eff = max(lo, lo0)
            if lo_eff < lo + half:
                seg_list.append((b, (lo_eff - starts[b]) * sub, lo_eff, lo + half))
    def seg_mat(C):
        M = np.zeros((12, len(seg_list)))
        for j, (_, _, lo, hi) in enumerate(seg_list):
            lo_, hi_ = max(0, lo), min(C.shape[1], hi)
            if hi_ > lo_:
                M[:, j] = C[:, lo_:hi_].sum(axis=1)
        return M
    SC = seg_mat(bch)
    hist = np.zeros(12)
    for s_, e_, pt, a in final_notes:
        hist[pt % 12] += (e_ - s_) * a
    key, kconf, key_note = A.detect_key(hist, bch.mean(axis=1), SC)
    lab = A.chord_hmm(SC, key)
    seg_names = [A.power_or_triad(SC[:, j] / (np.linalg.norm(SC[:, j]) + 1e-9), rq) if rq else "" for j, rq in enumerate(lab)]
    # bass note per segment (bass stem + chosen source, or the mix) -> inversions / slash chords
    ybass = (ysrc + A.load_mono(stems["bass"])) if (source == "stem" and "bass" in stems) else A.load_mono(wav)
    Cb, _ = A.note_evidence(ybass) if ybass is not ysrc else (Cq, None)
    BS = seg_mat(A.bass_pcs(Cb, beats, allowed=A.scale_pcs(key)))
    seg_names_nobass = list(seg_names)
    seg_names = [A.apply_bass(nm, SC[:, j], BS[:, j], key) for j, nm in enumerate(seg_names)]
    chords = [[] for _ in range(n_bars)]
    for j, (b, slot, _, _) in enumerate(seg_list):
        nm = seg_names[j]
        if not nm:
            continue
        if chords[b] and A.root_of(chords[b][-1][1]) == A.root_of(nm):
            if chords[b][-1][1].endswith("5") and not nm.endswith("5"):
                chords[b][-1] = (chords[b][-1][0], nm)  # same root: power chord + triad -> triad
            continue
        chords[b].append((slot, nm))
    Cref = librosa.feature.chroma_stft(y=librosa.effects.harmonic(ysrc), sr=A.SR_A, hop_length=A.HOP)
    frb = np.clip(librosa.time_to_frames(beats, sr=A.SR_A, hop_length=A.HOP), 0, Cref.shape[1])
    bref = np.zeros((12, nb))
    for i in range(nb):
        a_, b_ = frb[i], max(frb[i] + 1, frb[i + 1])
        if a_ < Cref.shape[1]:
            bref[:, i] = Cref[:, a_:b_].mean(axis=1)
    ref = A.raw_reference(seg_mat(bref))
    v1 = []  # v1 method: one chord per bar from transcribed notes
    for b in range(n_bars):
        w = np.zeros(12)
        for bb, _, ns in ev_list:
            if bb == b:
                for _, _, pt, d in ns:
                    w[pt % 12] += d
        v1.append(chord_name(w) if (w > 0).sum() >= 2 else "")
    valid = [j for j in range(len(seg_list)) if SC[:, j].sum() > 0]
    agree_v2 = sum(A.chord_match(seg_names[j], ref[j]) for j in valid) / max(1, len(valid))
    agree_v2_nobass = sum(A.chord_match(seg_names_nobass[j], ref[j]) for j in valid) / max(1, len(valid))
    bass_root = [NOTE_NAMES[int(np.argmax(BS[:, j]))] if BS[:, j].sum() > 0 else "" for j in range(len(seg_list))]
    agree_bassroot = {k: round(sum(A.root_of(names[j]) is not None and NOTE_NAMES[A.root_of(names[j])] == bass_root[j] or
                                    ("/" in names[j] and names[j].split("/")[1] == bass_root[j]) for j in valid) / max(1, len(valid)), 3)
                      for k, names in (("v2", seg_names_nobass), ("v3", seg_names))}
    agree_v1 = sum(A.chord_match(v1[seg_list[j][0]], ref[j]) for j in valid) / max(1, len(valid))
    timings["chords_key"] = time.time() - t

    # ---- onset report: tab onsets vs audio onsets per bar
    bt = lambda x: float(np.interp(x, np.arange(len(beats)), beats))
    disp = lambda b: b + (0 if has_pickup else 1)
    onset_report = {}
    for b in range(n_bars):
        a_ = bt(starts[b] + (bpb - pickup_beats if (b == 0 and has_pickup) else 0)); z_ = bt(starts[b] + lengths[b])
        onset_report[disp(b)] = {"window_s": [round(a_, 2), round(z_, 2)], "beats": lengths[b],
                                 "tab_onsets": sum(1 for bb, _, _ in ev_list if bb == b),
                                 "audio_onsets": int(((onsets >= a_ - 0.03) & (onsets < z_ - 0.03)).sum())}
    ds_scores = ds_info

    # ---- texts
    used = []
    for cl in chords:
        for _, c in cl:
            if c not in used:
                used.append(c)
    chart = [(c, v) for c in used for v in [voicing(c, base_tuning)] if v]
    tun_names = " ".join(NOTE_NAMES[x % 12] for x in base_tuning)
    fifths, mode = A.key_fifths(key)
    ksig = f"{abs(fifths)} {'sharp' if fifths > 0 else 'flat'}{'s' if abs(fifths) != 1 else ''}" if fifths else "no sharps/flats"
    info = {"Title": title,
            "Tuning": f"{tun_label}  [{tun_names}, low->high]  (confidence {tun_conf:.0%}; {tun_reason})" + (f" {tun_ambig}" if tun_ambig else ""),
            "Capo": f"{p.capo}" + (" (tab frets are relative to the capo)" if p.capo else " (none detected; capo can't be detected from audio)"),
            "Key": f"{key} ({ksig}; estimate, r={kconf:.2f}{'; ' + key_note if key_note else ''})",
            "Tempo": f"{tempo:.0f} BPM",
            "Time signature": f"{ts_label} (confidence {ts_conf:.0%})" + (f"; starts with a {pickup_beats}-beat pickup (bar 0)" if has_pickup else "")
                              + (f"; {n_short} short {bpb // 2}-beat bar(s) inserted where the downbeat shifts" if n_short else "")
                              + (f"; final bar is {final_bar_beats}/4 (song ends mid-bar)" if final_bar_beats else "")
                              + f"; downbeat confidence {ds_conf:.0%}"}
    header = ["=" * 78] + [f"{k + ':':16s}{v}" for k, v in info.items()] + ["=" * 78, ""]
    header += ["CHORDS (fret per string, low E -> high e; x = muted, 0 = open)"]
    row = ""
    for c, v in chart:
        cell = f"{c}: {voicing_str(v)}"
        if len(row) + len(cell) + 4 > 78:
            header.append("  " + row.rstrip()); row = ""
        row += cell.ljust(18)
    if row:
        header.append("  " + row.rstrip())
    header += ["", f"TAB  Rhythm: {spb} columns per full {ts_label} bar, one column = one 1/{sub * 4} note; the count line",
               "     ('1 . . . 2 ...') marks the beats. Numbers = fret, - = sustain/rest, | = bar line.",
               "     Chord names sit where they start (up to 2 per bar). h/p/b/slides not detected; verify by ear.", ""]
    bar_offsets = [pickup_slots if (b == 0 and has_pickup) else 0 for b in range(n_bars)]
    bar_numbers = [disp(b) for b in range(n_bars)]
    tab_kw = dict(chords=chords, bar_numbers=bar_numbers, bar_offsets=bar_offsets, sub=sub)
    txt = render_tab(bars, open_strings, 80, header_lines=header, **tab_kw)
    files = {}
    files["tab_txt"] = out / f"{stem}.tab.txt"; files["tab_txt"].write_text(txt)
    files["tab_html"] = out / f"{stem}.tab.html"
    export.write_html(info, chart, render_tab(bars, open_strings, 100, **tab_kw), files["tab_html"])
    pdf = export.html_to_pdf(files["tab_html"], out / f"{stem}.tab.pdf")
    if pdf:
        files["tab_pdf"] = pdf
    meta = dict(title=title, tempo=tempo, tuning=open_strings, capo=p.capo, beats_per_bar=bpb, sub=sub,
                n_bars=n_bars, bar_lengths=lengths, first_bar_time=bt(starts[0]), key_fifths=fifths, key_mode=mode, pickup_slots=pickup_slots)
    files["midi"] = out / f"{stem}.mid"; export.write_midi(final_notes, files["midi"], tempo, meta)
    files["musicxml"] = out / f"{stem}.musicxml"; export.write_musicxml(ev_list, meta, files["musicxml"])
    files["gp5"] = out / f"{stem}.gp5"
    export.write_gp5(ev_list, meta, files["gp5"])
    files["alphatex"] = out / f"{stem}.alphatex"; export.write_alphatex(ev_list, meta, files["alphatex"])
    files["guitar_stem"] = guitar_out
    timings["total"] = time.time() - t0
    in_tab = sum(len(x[2]) for x in ev_list)
    counts = {"transcribed_raw": cstats["raw"], "merged_retriggers": cstats["merged_retriggers"],
              "dropped_no_audio_onset": cstats["dropped_no_audio_onset"], "merged_min_ioi": cstats["merged_min_ioi"],
              **{k: cstats[k] for k in ("dropped_phantom", "dropped_weak_salience", "dropped_harmonic", "folded_octave")},
              "after_cleanup": cstats["after_cleanup"], "merged_same_grid_slot": grid_dupes, "dropped_unplayable_or_out_of_range": dropped_unplayable,
              "in_tab": in_tab, "in_midi": len(final_notes)}
    assert counts["in_tab"] == counts["in_midi"] == counts["after_cleanup"] - grid_dupes - dropped_unplayable, counts
    amps = [n[3] for n in final_notes]
    summary = {
        "title": title, "source": str(path), "duration_s": round(dur, 2), "tempo_bpm": round(tempo, 1),
        "key_estimate": key, "key_fifths": fifths, "key_confidence_r": round(kconf, 2), "key_note": key_note,
        "tuning_detected": tun_label, "tuning_confidence": tun_conf, "tuning_reason": tun_reason, "tuning_ambiguity": tun_ambig,
        "tuning_offset_semitones": round(off, 3), "time_signature": ts_label, "time_signature_confidence": ts_conf,
        "short_bars": n_short, "trailing_bars_trimmed": trimmed_bars, "final_bar_beats": final_bar_beats, "bar_start_beats": starts, "downbeat_confidence": round(ds_conf, 2), "downbeat_scores": ds_scores, "pickup_beats": int(pickup_beats),
        "chord_chart": {c: voicing_str(v) for c, v in chart}, "tuning": tun_names, "capo": p.capo,
        "transcription_source": source_desc, "stem_rms": rms, "mix_rms": mix_rms,
        "note_counts": counts, "notes_in_tab": in_tab,
        "mean_note_confidence": round(float(np.mean(amps)), 2) if amps else 0,
        "bars": n_bars, "chords_by_bar": [" / ".join(c for _, c in cl) for cl in chords],
        "chord_agreement_vs_chroma_ref": {"segments": len(valid), "v1_notes_one_per_bar": round(agree_v1, 3), "v2_chroma_hmm_half_bar": round(agree_v2_nobass, 3), "v3_with_bass": round(agree_v2, 3)},
        "chord_root_vs_bass_note": agree_bassroot, "bass_root_half_bar": bass_root, "chords_half_bar_v2": seg_names_nobass,
        "chords_half_bar": seg_names, "chroma_reference_half_bar": [NOTE_NAMES[r] + q for r, q in ref],
        "onsets_per_bar": onset_report,
        "chord_segments_v2": [[round(bt(lo), 3), round(bt(hi), 3), seg_names[j]] for j, (_, _, lo, hi) in enumerate(seg_list)],
        "chord_segments_v1": [[round(bt(lo), 3), round(bt(hi), 3), v1[b]] for j, (b, _, lo, hi) in enumerate(seg_list)],
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "outputs": {k: str(v) for k, v in files.items()},
    }
    sm = out / f"{stem}.summary.md"
    warn = []
    if summary["mean_note_confidence"] < 0.4: warn.append("low average note confidence; expect errors")
    if source != "stem" and p.separate: warn.append("transcribed the full mix; bass/vocals may leak into the tab")
    if tun_ambig: warn.append(tun_ambig)
    if ds_conf < 0.6: warn.append("downbeat placement uncertain; bar lines may be shifted")
    tot_tab = sum(v["tab_onsets"] for v in onset_report.values()); tot_aud = sum(v["audio_onsets"] for v in onset_report.values())
    sm.write_text(f"# {title}: audio2tab summary\n\n" + "\n".join(f"- {k}: {v}" for k, v in info.items()) +
                  f"\n- Source: {source_desc}\n- Notes: {counts}\n- Tab onsets {tot_tab} vs audio onsets {tot_aud}\n"
                  f"- Chords by bar: {' | '.join(summary['chords_by_bar'])}\n"
                  f"- Chord agreement with a chroma reference (half bars, not ground truth): v1-method {agree_v1:.0%}, v2 {agree_v2:.0%}\n"
                  f"- Confidence notes: {'; '.join(warn) or 'none flagged'}\n- Runtime: {timings['total']:.1f}s\n\n## Files\n"
                  + "\n".join(f"- {k}: {v}" for k, v in files.items()) + "\n")
    files["summary"] = sm
    (out / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2, default=str))
    summary["outputs"]["summary"] = str(sm)
    return summary, final_notes, txt
