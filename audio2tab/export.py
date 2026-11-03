"""MIDI, MusicXML (TAB staff), Guitar Pro 5, alphaTex writers."""
from __future__ import annotations
from xml.sax.saxutils import escape
from .fretboard import NOTE_NAMES


def write_midi(notes, path, tempo, meta=None):
    """Writes the FINAL note list (same notes as the tab), plus key and time signature meta events."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    if meta:
        lengths = meta.get("bar_lengths") or [meta["beats_per_bar"]]
        spq = 60.0 / tempo; tcur = meta.get("first_bar_time", 0.0); prev = None
        for L in lengths:
            if L != prev:
                pm.time_signature_changes.append(pretty_midi.TimeSignature(L, 4, max(0.0, tcur)))
                prev = L
            tcur += L * spq
        tonic = {v: k for k, v in __import__("audio2tab.analysis", fromlist=["x"]).MAJOR_FIFTHS.items()}[meta["key_fifths"]]
        if meta["key_mode"] == "minor":
            pm.key_signature_changes.append(pretty_midi.KeySignature(12 + (tonic - 3) % 12, 0.0))
        else:
            pm.key_signature_changes.append(pretty_midi.KeySignature(tonic, 0.0))
    inst = pretty_midi.Instrument(program=25, name="Guitar")  # steel acoustic
    for s, e, p, a in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(max(1, min(127, 30 + a * 97))), pitch=int(p), start=s, end=e))
    pm.instruments.append(inst)
    pm.write(str(path))


def bar_sequences(ev_list, meta):
    """-> list of bars; each bar list of (length_slots, dotted, base_slots, notes|None).
    ev_list items: (bar, pos, notes). Bar lengths in beats come from meta['bar_lengths']."""
    sub, nb = meta["sub"], meta["n_bars"]
    lengths = meta.get("bar_lengths") or [meta["beats_per_bar"]] * nb
    plain = sorted(set([4 * sub, 2 * sub, sub] + [sub // d for d in (2, 4, 8) if sub % d == 0]), reverse=True)
    allowed = sorted({(p * 3 // 2, True, p) for p in plain if p % 2 == 0 and p * 3 // 2 <= 4 * sub} | {(p, False, p) for p in plain}, reverse=True)

    def split(n):
        out = []
        while n > 0:
            for L, d, b in allowed:
                if L <= n:
                    out.append((L, d, b)); n -= L; break
        return out

    by_bar = {}
    for b, pos, ns in ev_list:
        by_bar.setdefault(b, []).append((pos, ns))
    bars = []
    for b in range(nb):
        spb = lengths[b] * sub
        evs = sorted(by_bar.get(b, []), key=lambda x: x[0])
        cur, seq = 0, []
        for i, (pos, ns) in enumerate(evs):
            if pos > cur:
                seq += [(L, d, bs, None) for L, d, bs in split(pos - cur)]
            nxt = evs[i + 1][0] if i + 1 < len(evs) else spb
            want = max(1, min(max(n[3] for n in ns), nxt - pos))
            L, d, bs = split(want)[0]
            seq.append((L, d, bs, ns))
            cur = pos + L
        if cur < spb:
            seq += [(L, d, bs, None) for L, d, bs in split(spb - cur)]
        bars.append(seq)
    return bars


def _type(base_slots, sub):
    q = base_slots / sub  # in quarters
    return {4: "whole", 2: "half", 1: "quarter", 0.5: "eighth", 0.25: "16th", 0.125: "32nd"}[q]


def write_musicxml(ev_list, meta, path):
    sub, tuning = meta["sub"], meta["tuning"]
    ns = len(tuning)
    capo = meta.get("capo", 0)
    open_ = [t - capo for t in tuning]
    x = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
         '<score-partwise version="4.0">', f'<work><work-title>{escape(meta["title"])}</work-title></work>',
         '<part-list><score-part id="P1"><part-name>Guitar</part-name>'
         '<midi-instrument id="P1-I1"><midi-program>26</midi-program></midi-instrument></score-part></part-list>', '<part id="P1">']
    lengths = meta.get("bar_lengths") or [meta["beats_per_bar"]] * meta["n_bars"]
    for bi, seq in enumerate(bar_sequences(ev_list, meta)):
        x.append(f'<measure number="{bi + 1}">')
        if bi > 0 and lengths[bi] != lengths[bi - 1]:
            x.append(f'<attributes><time><beats>{lengths[bi]}</beats><beat-type>4</beat-type></time></attributes>')
        if bi == 0:
            x.append(f'<attributes><divisions>{sub}</divisions><key><fifths>{meta.get("key_fifths", 0)}</fifths><mode>{meta.get("key_mode", "major")}</mode></key><time><beats>{lengths[0]}</beats><beat-type>4</beat-type></time>'
                     '<clef><sign>TAB</sign><line>5</line></clef>'
                     f'<staff-details><staff-lines>{ns}</staff-lines>')
            for i, t in enumerate(reversed(open_)):
                x.append(f'<staff-tuning line="{ns - i}"><tuning-step>{NOTE_NAMES[t % 12][0]}</tuning-step>'
                         + ('<tuning-alter>1</tuning-alter>' if "#" in NOTE_NAMES[t % 12] else '')
                         + f'<tuning-octave>{t // 12 - 1}</tuning-octave></staff-tuning>')
            x.append((f'<capo>{capo}</capo>' if capo else '') + '</staff-details></attributes>')
            x.append(f'<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                     f'<per-minute>{round(meta["tempo"])}</per-minute></metronome></direction-type><sound tempo="{round(meta["tempo"])}"/></direction>')
        for L, dot, base, notes in seq:
            ty = _type(base, sub)
            dt = '<dot/>' if dot else ''
            if notes is None:
                x.append(f'<note><rest/><duration>{L}</duration><type>{ty}</type>{dt}</note>')
                continue
            for j, (s, fr, pt, _) in enumerate(sorted(notes, key=lambda n: n[2])):
                nm = NOTE_NAMES[pt % 12]
                x.append('<note>' + ('<chord/>' if j else '') +
                         f'<pitch><step>{nm[0]}</step>' + ('<alter>1</alter>' if "#" in nm else '') +
                         f'<octave>{pt // 12 - 1}</octave></pitch><duration>{L}</duration><type>{ty}</type>{dt}'
                         f'<notations><technical><string>{ns - s}</string><fret>{fr}</fret></technical></notations></note>')
        x.append('</measure>')
    x.append('</part></score-partwise>')
    open(path, "w").write("\n".join(x))


def write_gp5(ev_list, meta, path):
    import guitarpro as gp
    sub = meta["sub"]
    song = gp.Song()
    song.title = meta["title"]; song.tempo = int(round(meta["tempo"]))
    track = song.tracks[0]
    track.name = "Guitar"
    track.channel.instrument = 25
    track.offset = meta.get("capo", 0)
    ks = {(f, m): k for k in gp.KeySignature for f, m in [k.value]}
    key = ks.get((meta.get("key_fifths", 0), 1 if meta.get("key_mode") == "minor" else 0), gp.KeySignature.CMajor)
    song.key = key
    open_ = [t - meta.get("capo", 0) for t in meta["tuning"]]
    track.strings = [gp.GuitarString(i + 1, v) for i, v in enumerate(reversed(open_))]
    seqs = bar_sequences(ev_list, meta) or [[(4 * sub, False, 4 * sub, None)]]
    for bi, seq in enumerate(seqs):
        if bi == 0:
            header, measure = song.measureHeaders[0], track.measures[0]
        else:
            header = gp.MeasureHeader(); song.addMeasureHeader(header)
            measure = gp.Measure(track, header); track.measures.append(measure)
        header.timeSignature.numerator = (meta.get("bar_lengths") or [meta["beats_per_bar"]] * len(seqs))[bi]
        header.keySignature = key
        voice = measure.voices[0]
        for L, dot, base, notes in seq:
            beat = gp.Beat(voice)
            beat.duration = gp.Duration(value=int(4 * sub // base), isDotted=dot)
            if notes is None:
                beat.status = gp.BeatStatus.rest
            else:
                beat.status = gp.BeatStatus.normal
                for s, fr, pt, _ in notes:
                    n = gp.Note(beat, value=fr, string=len(open_) - s, type=gp.NoteType.normal)
                    beat.notes.append(n)
            voice.beats.append(beat)
    gp.write(song, str(path), version=(5, 1, 0))


def write_alphatex(ev_list, meta, path):
    sub = meta["sub"]
    capo = meta.get("capo", 0)
    open_ = [t - capo for t in meta["tuning"]]
    tun = " ".join(f"{NOTE_NAMES[t % 12].lower()}{t // 12 - 1}" for t in reversed(open_))
    out = [f'\\title "{meta["title"]}"', f'\\tempo {round(meta["tempo"])}', '.',
           f'\\track "Guitar"', '\\staff {tabs}', f'\\tuning {tun}'] + ([f'\\capo {capo}'] if capo else [])
    bars = []
    lengths = meta.get("bar_lengths") or [meta["beats_per_bar"]] * meta["n_bars"]
    for bi, seq in enumerate(bar_sequences(ev_list, meta)):
        toks = [f"\\ts {lengths[bi]} 4"] if bi == 0 or lengths[bi] != lengths[bi - 1] else []
        for L, dot, base, notes in seq:
            d = int(4 * sub // base); eff = " {d}" if dot else ""
            if notes is None:
                toks.append(f"r.{d}{eff}")
            elif len(notes) == 1:
                s, fr, _, _ = notes[0]; toks.append(f"{fr}.{len(open_) - s}.{d}{eff}")
            else:
                toks.append("(" + " ".join(f"{fr}.{len(open_) - s}" for s, fr, _, _ in notes) + f").{d}{eff}")
        bars.append(" ".join(toks))
    open(path, "w").write("\n".join(out) + "\n" + " |\n".join(bars) + "\n")


def write_html(info, chart, tab_text, path):
    from .theory import svg_diagram
    rows = "".join(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in info.items())
    diags = "".join(f'<div class="d">{svg_diagram(escape(c), v)}</div>' for c, v in chart) or "<p>No chords detected.</p>"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(info['Title'])}</title>
<style>body{{font-family:-apple-system,Segoe UI,Helvetica,sans-serif;margin:24px;color:#111}}
h1{{margin:0 0 8px}}table{{border-collapse:collapse;margin-bottom:16px}}th{{text-align:left;padding:2px 14px 2px 0;color:#555}}
.chart{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:16px}}pre{{font:12px/1.25 Menlo,Consolas,monospace;white-space:pre}}
h2{{font-size:16px;border-bottom:1px solid #ccc;padding-bottom:2px}}.d{{break-inside:avoid}}@page{{size:A4;margin:12mm}}</style></head>
<body><h1>{escape(info['Title'])}</h1><table>{rows}</table><h2>Chords</h2><div class="chart">{diags}</div>
<h2>Tab</h2><pre>{escape(tab_text)}</pre><p style="color:#777;font-size:11px">Auto-transcribed by audio2tab. Verify by ear.</p></body></html>"""
    open(path, "w").write(html)


def html_to_pdf(html_path, pdf_path):
    try:
        import weasyprint
        weasyprint.HTML(str(html_path)).write_pdf(str(pdf_path))
        return pdf_path
    except Exception as e:  # pragma: no cover
        print("[audio2tab] PDF render skipped:", e)
        return None
