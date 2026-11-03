"""BTC chord recognizer (Park et al., ISMIR 2019; MIT, github.com/jayg996/BTC-ISMIR19), vendored model code.

Weights (12 MB, MIT) are not committed: they are downloaded once to $AUDIO2TAB_CACHE (default
~/.cache/audio2tab) from a pinned upstream commit and SHA-256 checked. $AUDIO2TAB_BTC_WEIGHTS overrides the path.
Output classes: idx = root*2 + (0 major | 1 minor), root C..B; idx 24 = no chord. Same order as analysis.CH_NAMES."""
import hashlib, os, urllib.request
from pathlib import Path
import numpy as np

COMMIT = "2682317be668032e6e4b269ded36adaa2ad57df0"
URL = f"https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/{COMMIT}/test/btc_model.pt"
SHA256 = "71c2c5db17e8c43b8a9a9da5db36ef2d667158c07a214eba16344c154c00bf54"
SR, N_BINS, BPO, HOP, TIMESTEP = 22050, 144, 24, 2048, 108
CONFIG = dict(feature_size=144, timestep=TIMESTEP, num_chords=25, input_dropout=0.2, layer_dropout=0.2,
              attention_dropout=0.2, relu_dropout=0.2, num_layers=8, num_heads=4, hidden_size=128,
              total_key_depth=128, total_value_depth=128, filter_size=128, loss="ce", probs_out=True)
_MODEL = None


def weights_path():
    p = os.environ.get("AUDIO2TAB_BTC_WEIGHTS")
    if p:
        return Path(p)
    d = Path(os.environ.get("AUDIO2TAB_CACHE", Path.home() / ".cache" / "audio2tab"))
    f = d / "btc_model.pt"
    if not f.exists():
        d.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != SHA256:
            tmp.unlink(); raise RuntimeError("BTC weights checksum mismatch")
        tmp.rename(f)
    return f


def load():
    global _MODEL
    if _MODEL is None:
        import torch
        from .btc_model import BTC_model
        ck = torch.load(weights_path(), map_location="cpu", weights_only=False)
        m = BTC_model(config=CONFIG)
        m.load_state_dict(ck["model"]); m.eval()
        _MODEL = (m, np.asarray(ck["mean"]), np.asarray(ck["std"]))
    return _MODEL


def frame_probs(y):
    """y: mono float audio at 22050 Hz. Returns (probs (T, 25), frame_times (T,))."""
    import librosa, torch
    m, mean, std = load()
    F = np.log(np.abs(librosa.cqt(y, sr=SR, n_bins=N_BINS, bins_per_octave=BPO, hop_length=HOP)) + 1e-6).T
    F = (F - mean) / std
    T = len(F)
    pad = (-T) % TIMESTEP
    X = torch.tensor(np.pad(F, ((0, pad), (0, 0))), dtype=torch.float32).view(-1, TIMESTEP, N_BINS)
    with torch.no_grad():
        logits = m.self_attn_layers(X)[0]
        P = torch.softmax(m.output_layer(logits), -1).reshape(-1, 25).numpy()[:T]
    return P, librosa.frames_to_time(np.arange(T), sr=SR, hop_length=HOP) + HOP / (2 * SR)


def beat_probs(y, beats):
    """Mean class probabilities between consecutive beats: (25, len(beats)-1)."""
    P, ft = frame_probs(y)
    out = np.zeros((25, len(beats) - 1))
    for i in range(len(beats) - 1):
        sel = (ft >= beats[i]) & (ft < beats[i + 1])
        out[:, i] = P[sel].mean(0) if sel.any() else P[min(len(P) - 1, int(np.searchsorted(ft, beats[i])))]
    return out
