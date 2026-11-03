"""End-to-end, as a user sees it: inputs (WAV/MP3/M4A/MP4), the tab sheet header, exports, CLI options,
chord accuracy gates (BTC model vs chroma) and the fallback when the chord model is missing."""
import json, subprocess, sys, numpy as np, mir_eval, pytest
from pathlib import Path
from tests.synth import write_all
from audio2tab.pipeline import Params, run


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    ref = write_all(str(d))
    return d, ref


def f1(ref, notes):
    ri = np.array([[s, e] for s, e, _ in ref]); rp = mir_eval.util.midi_to_hz(np.array([p for *_, p in ref]))
    ei = np.array([[s, e] for s, e, *_ in notes]); ep = mir_eval.util.midi_to_hz(np.array([n[2] for n in notes]))
    return mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)[2]


def test_clean_accuracy(data, tmp_path):
    d, ref = data
    s, notes, txt = run(d / "clean_guitar.wav", tmp_path, Params(separate=False, title="t"))
    assert f1(ref, notes) >= 0.83  # v2 baseline 0.836
    # counts consistent: MIDI and tab come from the same final note list
    import pretty_midi
    c = s["note_counts"]
    assert c["in_tab"] == c["in_midi"] == len(pretty_midi.PrettyMIDI(s["outputs"]["midi"]).instruments[0].notes)
    from tests.eval_e2e import chord_acc
    assert chord_acc(s, "chord_segments_v2") >= 0.8
    assert "16 columns per full 4/4 bar" in txt
    # no trailing empty bars; last bar holds notes
    ob = s["onsets_per_bar"]
    assert ob[list(ob)[-1]]["tab_onsets"] > 0
    assert abs(s["tempo_bpm"] - 100) < 3
    assert s["time_signature"] == "4/4"
    assert s["tuning_detected"].startswith("Standard")
    for k in ("tab_txt", "tab_html", "tab_pdf", "musicxml", "gp5", "alphatex", "midi", "guitar_stem"):
        assert Path(s["outputs"][k]).stat().st_size > 0
    for field in ("Title:", "Tuning:", "Capo:", "Key:", "Tempo:", "Time signature:", "CHORDS", "Ambiguous"):
        assert field in txt
    import guitarpro
    assert len(guitarpro.parse(s["outputs"]["gp5"]).tracks[0].measures) == s["bars"]


import pytest as _pt


@_pt.mark.parametrize("ext,codec", [("mp3", ["-c:a", "libmp3lame"]), ("m4a", ["-c:a", "aac"]), ("mp4", None)])
def test_container_inputs(data, tmp_path, ext, codec):
    d, ref = data
    src = tmp_path / f"clip.{ext}"
    if codec is None:  # video with an audio track
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10",
               "-i", str(d / "clean_guitar.wav"), "-shortest", "-c:v", "libx264", "-c:a", "aac", str(src)]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "clean_guitar.wav"), *codec, str(src)]
    subprocess.run(cmd, check=True)
    out = tmp_path / "o"
    r = subprocess.run([str(Path(__file__).parents[1] / "run.sh"), str(src), "--out", str(out), "--no-separate"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.loads(next(out.glob("*.summary.json")).read_text())
    assert s["notes_in_tab"] > 40 and abs(s["tempo_bpm"] - 100) < 3
    assert Path(s["outputs"]["tab_pdf"]).stat().st_size > 0
from audio2tab import analysis as A
from tests.chord_bench import song, score, SR


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


def test_pipeline_model_path(data, tmp_path):
    from tests.eval_e2e import chord_acc
    s = run(data[0] / "clean_guitar.wav", tmp_path, Params(separate=False, title="t", chords="model"))[0]
    assert s["chord_method"] == "model:btc" and s["chord_model_fallback"] is None
    assert chord_acc(s, "chord_segments_v2") >= 0.8
    assert s["chord_chart"] and all(Path(s["outputs"][k]).stat().st_size > 0 for k in ("tab_pdf", "musicxml", "gp5"))


def test_pipeline_falls_back_to_chroma(data, tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO2TAB_BTC_WEIGHTS", str(tmp_path / "missing.pt"))
    import audio2tab.btc as btc
    monkeypatch.setattr(btc, "_MODEL", None)
    s = run(data[0] / "clean_guitar.wav", tmp_path, Params(separate=False, title="t", chords="model"))[0]
    assert s["chord_method"] == "chroma" and s["chord_model_fallback"]
    assert Path(s["outputs"]["gp5"]).stat().st_size > 0





def test_chord_benchmark_gate(tmp_path):
    """Full pipeline on 3 synthetic full-mix songs: the model must beat chroma and stay >= 0.75 majmin."""
    import soundfile as sf
    acc = {"chroma": [], "model": []}
    for i in range(3):
        y, truth = song(100 + i, mixed=True); f = tmp_path / f"s{i}.wav"; sf.write(f, y, SR)
        for m in acc:
            s = run(f, tmp_path / f"o{i}{m}", Params(separate=False, chords=m, title="b"))[0]
            acc[m].append(score(truth, s["chord_segments_v2"])["majmin"])
    assert np.mean(acc["model"]) >= max(0.75, np.mean(acc["chroma"]))


def test_cli_user_options(data, tmp_path):
    out = tmp_path / "cli"
    r = subprocess.run([str(Path(__file__).parents[1] / "run.sh"), str(data[0] / "clean_guitar.wav"), "--out", str(out),
                        "--no-separate", "--tuning", "DADGBE", "--capo", "2", "--chords", "chroma", "--title", "My Song",
                        "--beats-per-bar", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    txt = next(out.glob("*.tab.txt")).read_text()
    import re
    assert "My Song" in txt and re.search(r"Capo:\s+2", txt) and "Drop D" in txt and re.search(r"Time signature:\s+3/4", txt)
    s = json.loads(next(out.glob("*.summary.json")).read_text())
    assert s["chord_method"] == "chroma"
    bad = subprocess.run([str(Path(__file__).parents[1] / "run.sh"), "x.wav", "--chords", "bogus"], capture_output=True)
    assert bad.returncode != 0
