"""End-to-end: run pipeline on synthesized audio; score notes (mir_eval) and chords against ground truth."""
import json, sys, numpy as np, mir_eval
sys.path.insert(0, ".")
from tests.synth import write_all, piece, CHORD_TRUTH
from audio2tab.pipeline import Params, run
from audio2tab import analysis as A


def score(ref, est):
    ri = np.array([[s, e] for s, e, _ in ref]); rp = mir_eval.util.midi_to_hz(np.array([p for *_, p in ref]))
    ei = np.array([[s, e] for s, e, _, _ in est]); ep = mir_eval.util.midi_to_hz(np.array([p for _, _, p, _ in est]))
    on = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    onoff = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep)
    ons = mir_eval.transcription.onset_precision_recall_f1(ri, ei)
    return dict(note_onset_pitch_PRF=[round(x, 3) for x in on[:3]], note_onset_offset_PRF=[round(x, 3) for x in onoff[:3]],
                onset_only_PRF=[round(x, 3) for x in ons])


def chord_acc(summary, key):
    """Fraction of ground-truth chord time (strummed section) whose label matches (root, and quality unless power chord)."""
    segs = summary[key]
    hit = tot = 0.0
    for t0, t1, name, rq in CHORD_TRUTH:
        for a, b, lab in segs:
            ov = max(0.0, min(b, t1) - max(a, t0))
            if ov > 0:
                tot += ov; hit += ov * A.chord_match(lab, rq)
    return round(hit / tot, 3) if tot else None


if __name__ == "__main__":
    d = "testdata"
    ref = write_all(d)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    def add(name, s, notes):
        res[name] = score(ref, notes) | {"tempo": s["tempo_bpm"], "source": s["transcription_source"],
                                         "chord_acc_v2_hmm": chord_acc(s, "chord_segments_v2"),
                                         "chord_acc_v1_notes": chord_acc(s, "chord_segments_v1"),
                                         "runtime_s": s["timings_s"]["total"]}
    if which in ("clean", "all"):
        add("clean_no_separation", *run(f"{d}/clean_guitar.wav", "out/e2e_clean", Params(separate=False, title="Test piece (clean)"))[:2])
    if which in ("mixed", "all"):
        add("mixed_auto", *run(f"{d}/mixed.wav", "out/e2e_mixed", Params(title="Test piece (mixed)"))[:2])
        add("mixed_forced_stem", *run(f"{d}/mixed.wav", "out/e2e_mixed_stem", Params(title="Test piece (mixed, stem)", source="stem"))[:2])
        add("mixed_no_separation", *run(f"{d}/mixed.wav", "out/e2e_mixed_nosep", Params(separate=False, title="Test piece (mixed, no sep)"))[:2])
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"out/e2e_{which}_metrics.json", "w"), indent=1)
