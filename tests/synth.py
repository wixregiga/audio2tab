"""Synthesize a known guitar-like test piece (Karplus-Strong) + optional mix layers."""
import numpy as np, soundfile as sf

SR = 44100


def ks(freq, dur, sr=SR, decay=0.996, bright=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sr / freq); N = int(dur * sr)
    buf = rng.uniform(-1, 1, n)
    buf = np.convolve(buf, [bright, 1 - bright], "same")
    out = np.zeros(N)
    for i in range(N):
        out[i] = buf[i % n]
        buf[i % n] = decay * 0.5 * (buf[i % n] + buf[(i + 1) % n])
    env = np.ones(N); r = min(N, int(0.02 * sr)); env[-r:] = np.linspace(1, 0, r)
    return out * env


def midi_hz(p):
    return 440 * 2 ** ((p - 69) / 12)


def piece(bpm=100):
    """Returns list of (start, end, pitch). 8 bars: 4 bars riff (8ths), 4 bars strummed chords."""
    q = 60 / bpm; e = q / 2
    notes = []
    riff = [40, 43, 45, 47, 45, 43, 40, 43,  52, 50, 47, 45, 47, 45, 43, 40]  # E minor pentatonic
    t = 0.0
    for bar in range(2):
        for p in riff:
            notes.append((t, t + e * 0.95, p)); t += e
    # chords: Em G D C, half notes, strums slightly spread
    shapes = {"Em": [40, 47, 52, 55, 59, 64], "G": [43, 47, 50, 55, 59, 67], "D": [50, 57, 62, 66], "C": [48, 52, 55, 60, 64]}
    for name in ["Em", "G", "D", "C", "Em", "G", "D", "C"]:
        for i, p in enumerate(shapes[name]):
            st = t + i * 0.012
            notes.append((st, t + 2 * q * 0.97, p))
        t += 2 * q
    return notes


def render(notes, sr=SR, tail=1.0):
    dur = max(e for _, e, _ in notes) + tail
    y = np.zeros(int(dur * sr))
    for i, (s, e, p) in enumerate(notes):
        x = ks(midi_hz(p), e - s, sr, seed=i) * 0.3
        a = int(s * sr); y[a:a + len(x)] += x
    return y / (np.abs(y).max() + 1e-9) * 0.8


def mix_layers(n_samples, bpm=100, sr=SR, seed=1):
    """Synthetic drums + 'vocal' (vibrato sawtooth with formant-ish filtering) + sub bass."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sr
    q = 60 / bpm
    drums = np.zeros(n_samples)
    for k, bt in enumerate(np.arange(0, t[-1], q)):
        a = int(bt * sr)
        L = min(int(0.15 * sr), n_samples - a)
        tt = np.arange(L) / sr
        if k % 2 == 0:  # kick
            drums[a:a + L] += np.sin(2 * np.pi * (50 + 100 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 20)
        else:  # snare
            drums[a:a + L] += rng.normal(0, 0.5, L) * np.exp(-tt * 25)
        h = int((bt + q / 2) * sr)
        Lh = min(int(0.04 * sr), n_samples - h)
        if Lh > 0:
            drums[h:h + Lh] += rng.normal(0, 0.2, Lh) * np.exp(-np.arange(Lh) / sr * 80)
    # vocal melody: sustained notes w/ vibrato, harmonics
    mel = [64, 67, 69, 67, 64, 62, 64, 59]
    f = np.zeros(n_samples)
    seg = int(2 * q * sr)
    for i in range(n_samples // seg + 1):
        f[i * seg:(i + 1) * seg] = midi_hz(mel[i % len(mel)] + 12 * 0)
    f = f * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / sr
    vocal = sum(np.sin(k * ph) * (1 / k) * (1.5 if k in (3, 4) else 1) for k in range(1, 10))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * t / (2 * q) - np.pi / 2) ** 2
    vocal *= env
    bass_notes = [28, 31, 26, 24]
    bf = np.array([midi_hz(bass_notes[int(x / (2 * q)) % 4]) for x in t[::441]])
    bf = np.repeat(bf, 441)[:n_samples]
    bass = np.sin(2 * np.pi * np.cumsum(bf) / sr)
    n = lambda x: x / (np.abs(x).max() + 1e-9)
    return n(drums) * 0.6, n(vocal) * 0.35, n(bass) * 0.35


def write_all(dirpath):
    import os, json
    os.makedirs(dirpath, exist_ok=True)
    notes = piece()
    g = render(notes)
    sf.write(f"{dirpath}/clean_guitar.wav", g, SR)
    d, v, b = mix_layers(len(g))
    mix = g * 0.7 + d + v + b
    sf.write(f"{dirpath}/mixed.wav", mix / np.abs(mix).max() * 0.9, SR)
    json.dump(notes, open(f"{dirpath}/ground_truth.json", "w"))
    return notes


if __name__ == "__main__":
    import sys
    write_all(sys.argv[1] if len(sys.argv) > 1 else "testdata")


def _truth(bpm=100):
    q = 60 / bpm; t = 32 * q / 2
    out = []
    names = ["Em", "G", "D", "C", "Em", "G", "D", "C"]
    rq = {"Em": (4, "m"), "G": (7, ""), "D": (2, ""), "C": (0, "")}
    for n in names:
        out.append((t, t + 2 * q, n, rq[n])); t += 2 * q
    return out


CHORD_TRUTH = _truth()
