import argparse, json, sys
from .pipeline import Params, run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="audio2tab", description="Audio/video file (mp3 m4a mp4 wav flac ogg ...) -> guitar tab (txt, MusicXML, GP5, alphaTex, MIDI)")
    ap.add_argument("file")
    ap.add_argument("--out", default=None, help="output dir (default ./out/<name>)")
    ap.add_argument("--no-separate", action="store_true", help="skip Demucs (solo guitar recording)")
    ap.add_argument("--model", default="htdemucs_6s")
    ap.add_argument("--source", default="auto", choices=["auto", "stem", "mix"],
                    help="what to transcribe after separation: auto picks the guitar stem only if it holds >=25%% of mix RMS")
    ap.add_argument("--no-pickup", action="store_true", help="never treat the opening as a pickup bar")
    ap.add_argument("--tuning", default="auto", help="auto (detect) or low->high e.g. EADGBE, DADGBE, 'D2 A2 D3 G3 B3 E4'")
    ap.add_argument("--capo", type=int, default=0)
    ap.add_argument("--start", type=float); ap.add_argument("--end", type=float)
    ap.add_argument("--title")
    ap.add_argument("--bpm", type=float, help="force tempo hint")
    ap.add_argument("--beats-per-bar", type=int, default=None, help="default: auto-detect 4/4 vs 3/4")
    ap.add_argument("--subdivision", type=int, default=4, help="grid slots per beat (4=16ths, 2=8ths)")
    ap.add_argument("--onset-threshold", type=float, default=0.6)
    ap.add_argument("--frame-threshold", type=float, default=0.4)
    ap.add_argument("--min-note-ms", type=float, default=80)
    ap.add_argument("--min-amp", type=float, default=0.3)
    ap.add_argument("--max-span", type=int, default=4)
    a = ap.parse_args(argv)
    from pathlib import Path
    out = a.out or str(Path("out") / Path(a.file).stem)
    p = Params(tuning=a.tuning, capo=a.capo, separate=not a.no_separate, model=a.model, start=a.start, end=a.end,
               title=a.title, onset_threshold=a.onset_threshold, frame_threshold=a.frame_threshold,
               min_note_ms=a.min_note_ms, min_amp=a.min_amp, subdivision=a.subdivision,
               beats_per_bar=a.beats_per_bar, max_span=a.max_span, bpm=a.bpm, source=a.source, pickup="none" if a.no_pickup else "auto")
    summary, _, _ = run(a.file, out, p)
    print("Outputs:")
    for k, v in summary["outputs"].items():
        print(f"  {k:12s} {Path(v).resolve()}")
    print(f"Tempo ~{summary['tempo_bpm']} BPM | key {summary['key_estimate']} | {summary['notes_in_tab']} notes | {summary['timings_s']['total']}s")


if __name__ == "__main__":
    main()
