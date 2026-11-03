"""BTC chord-model path: emission HMM, model inference, end-to-end use, and chroma fallback."""
import numpy as np, pytest
from pathlib import Path
from audio2tab import analysis as A
from audio2tab.pipeline import Params, run
from tests.chord_bench import song, score, SR
from tests.synth import write_all


def test_hmm_uses_emission_and_no_chord():
    SC = np.ones((12, 4))  # flat chroma: emission must decide
    E = np.full((24, 4), 0.01)
    E[A.CH_NAMES.index((0, "")), :2] = 0.9   # C, C
    E[A.CH_NAMES.index((9, "m")), 2:] = 0.9  # Am, Am
    lab = A.chord_hmm(SC, "C major", emission=E, no_chord=np.array([False, False, False, True]))
    assert lab == [(0, ""), (0, ""), (9, "m"), None]


def test_btc_recognizes_synthetic_progression():
    from audio2tab import btc
    y, truth = song(7, mixed=True)
    P, ft = btc.frame_probs(y.astype(np.float32))
    assert P.shape[1] == 25 and np.allclose(P.sum(1), 1, atol=1e-4)
    segs = []
    for a, b, _ in truth:
        sel = (ft >= a) & (ft < b)
        k = int(P[sel].mean(0)[:24].argmax())
        r, q = A.CH_NAMES[k]
        segs.append((a, b, A.NOTE_NAMES[r] + q))
    assert score(truth, segs)["majmin"] >= 0.8


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthc"); write_all(str(d)); return d / "clean_guitar.wav"


def test_pipeline_model_path(clean, tmp_path):
    from tests.eval_e2e import chord_acc
    s = run(clean, tmp_path, Params(separate=False, title="t", chords="model"))[0]
    assert s["chord_method"] == "model:btc" and s["chord_model_fallback"] is None
    assert chord_acc(s, "chord_segments_v2") >= 0.8
    assert s["chord_chart"] and all(Path(s["outputs"][k]).stat().st_size > 0 for k in ("tab_pdf", "musicxml", "gp5"))


def test_pipeline_falls_back_to_chroma(clean, tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO2TAB_BTC_WEIGHTS", str(tmp_path / "missing.pt"))
    import audio2tab.btc as btc
    monkeypatch.setattr(btc, "_MODEL", None)
    s = run(clean, tmp_path, Params(separate=False, title="t", chords="model"))[0]
    assert s["chord_method"] == "chroma" and s["chord_model_fallback"]
    assert Path(s["outputs"]["gp5"]).stat().st_size > 0


def test_cli_chords_flag():
    from audio2tab import cli
    import argparse
    with pytest.raises(SystemExit):
        cli.main(["x.wav", "--chords", "bogus"])
