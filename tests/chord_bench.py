"""Chord benchmark: synthetic songs with known chords (random keys, diatonic progressions incl. inversions,
tempos, strum patterns), guitar-only and full mix (drums + bass + lead). Runs the whole pipeline with
--chords chroma vs model and scores mir_eval majmin / root WCSR on the emitted chord segments.
Usage: python -m tests.chord_bench [n_songs] [out_json]"""
import json, sys, time, tempfile, numpy as np, soundfile as sf, mir_eval
from audio2tab.pipeline import Params, run
from audio2tab.analysis import NOTE_NAMES

SR = 22050
PCS = NOTE_NAMES


def tone(f, dur, sr=SR, rng=None, decay=3.0):
    t = np.arange(int(dur * sr)) / sr
    y = sum(np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6)) * (0.7 ** k) * np.exp(-t * decay * (1 + 0.4 * k))
            for k in range(1, 8) if f * k < sr / 2)
    a = min(len(t), int(0.004 * sr)); y[:a] *= np.linspace(0, 1, a)
    return y


def voicing(root, minor, rng, inv):
    ivs = [0, 3 if minor else 4, 7]
    bass_pc = (root + ivs[inv]) % 12
    lo = 40 + (bass_pc - 4) % 12  # lowest note in E2..D#3
    pcs = [(root + i) % 12 for i in ivs]
    notes, p = [lo], lo
    while len(notes) < rng.integers(4, 7):
        p += 1
        while p % 12 not in pcs:
            p += 1
        notes.append(p)
    return notes


def song(seed, mixed):
    rng = np.random.default_rng(seed)
    key = int(rng.integers(12)); minor_key = bool(rng.integers(2))
    deg = [(0, 0), (2, 1), (4, 1), (5, 0), (7, 0), (9, 1)]  # major-key diatonic triads (offset, minor?)
    if minor_key:
        key = (key + 3) % 12
    bpm = float(rng.uniform(80, 140)); q = 60 / bpm
    n_ch = 16; per = int(rng.choice([2, 4]))
    prog = [deg[int(rng.integers(len(deg)))] for _ in range(4)]
    seq = [prog[i % 4] for i in range(n_ch)]
    dur = n_ch * per * q + 1.5
    N = int(dur * SR); gtr = np.zeros(N); bass = np.zeros(N); lead = np.zeros(N); drums = np.zeros(N)
    truth = []
    pattern = rng.choice(["quarters", "eighths", "whole"])
    for i, (off, mn) in enumerate(seq):
        root = (key + off) % 12
        inv = int(rng.choice([0, 0, 0, 1, 2]))
        vs = voicing(root, mn, rng, inv)
        t0 = 0.5 + i * per * q
        truth.append((t0, t0 + per * q, f"{PCS[root]}:{'min' if mn else 'maj'}"))
        step = {"quarters": q, "eighths": q / 2, "whole": per * q}[pattern]
        for s in np.arange(t0, t0 + per * q - 1e-6, step):
            for k, p in enumerate(vs):
                x = tone(440 * 2 ** ((p - 69) / 12), min(step * 1.1, 2.0), rng=rng) * 0.15
                a = int((s + k * 0.01) * SR); gtr[a:a + len(x)] += x[:max(0, N - a)]
        if mixed:
            x = tone(440 * 2 ** ((28 + (root - 4) % 12 - 69) / 12), per * q, rng=rng, decay=1.0)
            a = int(t0 * SR); bass[a:a + len(x)] += x[:max(0, N - a)]
            for b in range(per):
                p = 64 + (root + int(rng.choice([0, 3 if mn else 4, 7, 2, 9]))) % 12
                x = tone(440 * 2 ** ((p - 69) / 12), q, rng=rng, decay=0.8)
                a = int((t0 + b * q) * SR); lead[a:a + len(x)] += x[:max(0, N - a)]
    if mixed:
        for k, bt in enumerate(np.arange(0.5, dur - 1, q)):
            a = int(bt * SR); L = min(int(0.15 * SR), N - a); tt = np.arange(L) / SR
            drums[a:a + L] += (np.sin(2 * np.pi * (50 + 100 * np.exp(-tt * 30)) * tt) if k % 2 == 0
                               else rng.normal(0, 0.6, L)) * np.exp(-tt * 22)
    nz = lambda v: v / (np.abs(v).max() + 1e-9)
    y = nz(gtr) * 0.7 + (nz(bass) * 0.4 + nz(lead) * 0.3 + nz(drums) * 0.5 if mixed else 0)
    return nz(y) * 0.9, truth


def to_mirex(lab):
    if not lab:
        return "N"
    lab = lab.split("/")[0]
    r = lab[:2] if len(lab) > 1 and lab[1] == "#" else lab[:1]
    q = lab[len(r):]
    return f"{r}:" + {"m": "min", "5": "(1,5)", "": "maj", "7": "7"}.get(q, "maj")


def score(truth, segs):
    ri = np.array([[a, b] for a, b, _ in truth]); rl = [l for *_, l in truth]
    ei = np.array([[a, b] for a, b, _ in segs]); el = [to_mirex(l) for *_, l in segs]
    ei, el = mir_eval.util.adjust_intervals(ei, el, ri.min(), ri.max(), "N", "N")
    ints, r_, e_ = mir_eval.util.merge_labeled_intervals(ri, rl, ei, el)
    w = mir_eval.util.intervals_to_durations(ints)
    out = {m: float(mir_eval.chord.weighted_accuracy(getattr(mir_eval.chord, m)(r_, e_), w)) for m in ("root", "majmin")}
    # repo convention (analysis.chord_match): a power chord X5 counts for X:maj or X:min
    e2 = [r if (e.endswith(":(1,5)") and e.split(":")[0] == r.split(":")[0]) else e for r, e in zip(r_, e_)]
    out["majmin_pow"] = float(mir_eval.chord.weighted_accuracy(mir_eval.chord.majmin(r_, e2), w))
    return out


def oracle(y, truth):
    """Detector-only accuracy on ground-truth segment boundaries (no beat tracking): chroma template vs BTC."""
    import librosa
    from audio2tab import analysis as A, btc
    C = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512); tc = librosa.times_like(C, sr=SR, hop_length=512)
    P, tp = btc.frame_probs(y.astype(np.float32))
    res = {}
    for name, lab in (("chroma", lambda a, b: A.CH_NAMES[int(np.argmax(A.CH_T @ C[:, (tc >= a) & (tc < b)].mean(1)))]),
                      ("model:btc", lambda a, b: A.CH_NAMES[int(P[(tp >= a) & (tp < b), :24].mean(0).argmax())])):
        segs = [(a, b, A.NOTE_NAMES[lab(a, b)[0]] + lab(a, b)[1]) for a, b, _ in truth]
        res[name] = score(truth, segs)
    return res


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "oracle":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    acc = {}
    for i in range(n):
        for mixed in (False, True):
            y, truth = song(100 + i, mixed)
            for m, r in oracle(y, truth).items():
                acc.setdefault(f"{'mixed' if mixed else 'guitar'}/{m}", []).append(r["majmin"])
    print(json.dumps({k: round(float(np.mean(v)), 3) for k, v in acc.items()}, indent=1))
    sys.exit()

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for i in range(n):
            for mixed in (False, True):
                y, truth = song(100 + i, mixed); f = f"{d}/s{i}_{int(mixed)}.wav"; sf.write(f, y, SR)
                for method in ("chroma", "model"):
                    t = time.time()
                    s = run(f, f"{d}/o{i}{int(mixed)}{method}", Params(separate=False, chords=method, title="bench"))[0]
                    rows.append({"song": i, "mixed": mixed, "method": s["chord_method"], "s": time.time() - t,
                                 **score(truth, s["chord_segments_v2"])})
                    print(rows[-1], flush=True)
    agg = {}
    for mixed in (False, True):
        for m in ("chroma", "model:btc"):
            r = [x for x in rows if x["mixed"] == mixed and x["method"] == m]
            if r:
                agg[f"{'mixed' if mixed else 'guitar'}/{m}"] = {k: round(float(np.mean([x[k] for x in r])), 3) for k in ("root", "majmin", "majmin_pow", "s")}
    print(json.dumps(agg, indent=1))
    if len(sys.argv) > 2:
        json.dump({"agg": agg, "rows": rows}, open(sys.argv[2], "w"), indent=1)
