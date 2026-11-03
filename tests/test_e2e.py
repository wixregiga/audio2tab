"""End-to-end smoke + accuracy gate on synthesized Karplus-Strong guitar; MP4 container input."""
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
    assert f1(ref, notes) > 0.75
    # counts consistent: MIDI and tab come from the same final note list
    import pretty_midi
    c = s["note_counts"]
    assert c["in_tab"] == c["in_midi"] == len(pretty_midi.PrettyMIDI(s["outputs"]["midi"]).instruments[0].notes)
    from tests.eval_e2e import chord_acc
    assert chord_acc(s, "chord_segments_v2") >= 0.8
    assert "16 columns per full 4/4 bar" in txt
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
