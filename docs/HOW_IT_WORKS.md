# How audio2tab works (details)

## Pipeline
1. ffmpeg: first audio stream only (`-vn -map 0:a:0`, so video and cover art are ignored), 44.1 kHz stereo, EBU loudnorm.
2. Demucs `htdemucs_6s` separation. Uses the guitar stem. If guitar RMS < 0.35 x `other`, it uses guitar+other (AI/synth mixes often put guitar in `other`).
3. librosa beat tracking. The grid is interpolated between tracked beats, so tempo drift is handled. The meter (4/4 vs 3/4) comes from 3- vs 4-beat accent periodicity (full-band and low-band onset strength). The downbeat phase comes from accent weighting.
4. Spotify basic-pitch (ONNX, polyphonic) for notes. Filters: onset 0.6, frame 0.4, min amplitude 0.3, min length 80 ms. Notes are quantized to 16ths.
5. Tuning detection: the lowest confidently repeated pitch (Standard / Drop D / half-step down / D standard / Drop C# / Drop C), with pitch-class evidence for Drop D vs D standard. Reported with confidence. `--tuning` overrides it.
6. Fretboard: Viterbi over every playable shape per onset (one note per string, fret span ≤ 4, frets ≤ 22). Cost covers position height, span, and hand movement (carried through open strings), with extra cost for jumps > 4 frets.
7. Chords per bar: template match (maj, min, 7, 5) on duration-weighted pitch classes. Key: Krumhansl-Schmuckler. Voicings: a curated standard-tuning table, root/fifth/octave power chords, and a fretboard search for other tunings.

## Tuning thresholds
`--onset-threshold` / `--frame-threshold`: higher means fewer, surer notes. Lower them for quiet fingerpicking and raise them for dense distorted parts. `--min-amp` drops weak notes. `--min-note-ms` drops blips. `--subdivision 2` gives an 8th-note grid (cleaner for strummed songs). `--max-span` sets the allowed fret stretch. `--bpm` sets a tempo hint (use it if the tempo comes out double or half).

## Tests
`.venv/bin/python -m pytest -q` runs the unit tests (fretboard, formatter, theory) and end to end on a synthetic Karplus-Strong piece (accuracy gate, all outputs, MP4 input).
`.venv/bin/python tests/eval_e2e.py all` prints mir_eval metrics for clean, mixed+separated, and mixed-unseparated runs.

## Limitations
- Rhythm is quantized to a 16th grid from tracked beats. Swing, triplets, rubato and pickups come out approximate, and the downbeat or time signature can be wrong (6/8 isn't detected yet).
- No bend, slide, hammer-on or pull-off, vibrato, or palm-mute detection. Bends often show up as two adjacent pitches.
- Dense strums lose inner notes, and octave errors are common. Heavy distortion adds false harmonics. Two guitars get merged into one part.
- Chord names are one per bar (two chords in a bar shows the dominant one). Only maj/min/7/power templates are used.
- Tuning detection only notices tunings lower than standard when low notes are actually played. A capo can't be detected.
- Separation quality depends on the mix. On synthetic test mixes it did worse than no separation (see metrics). On real recordings Demucs is generally the better choice.

## Note verification (v3)
After basic-pitch and cleanup, each note is checked against the chosen source audio (CQT, 1 bin/semitone, plus onset strength):
- **Phantom notes** are dropped when the pitch's energy is near silence for that pitch (< 5% of its 95th-percentile level), or when there is a weak onset (< 2.5x the median onset strength) together with no energy rise in the note's own bin or low absolute energy.
- **Octave/harmonic errors**: a note that is weak in its frame (< 70% of the loudest bin) while the pitch an octave, octave+fifth or two octaves below is much stronger is dropped if that lower note is already transcribed (lower >= 2x), or folded down to it otherwise (>= 1.5x). The lower pitch must be in guitar range (>= E2), so bass notes never trigger a fold.
- **Unplayable stretches** (fretted notes 4+ strings apart with a 3+ fret span, e.g. E13 + e16) are rejected by the fretboard search.
- Trailing bars after the last note are trimmed, and a final bar shorter than the meter is written with its real length.
- **Slash chords**: the lowest strong bass note per half bar (bass stem + guitar, restricted to scale tones to absorb semitone leakage) relabels a chord when the bass is a better root (C# -> A#m), or writes inversions as C#/F.
- MusicXML: music21's `score.recurse().pitches` counts each pitch once per enclosing stream (3x). Use `score.flatten().pitches`, which matches the note count.

## v4: chord model (BTC)

Chord detection now uses BTC, the Bi-directional Transformer for Chord recognition (Park et al., ISMIR 2019, MIT, vendored in `audio2tab/btc/`). It runs on the full mix and produces per-frame maj/min/N.C. probabilities. These are averaged per half bar and passed to the existing key-aware Viterbi in place of the chroma-template scores. Beat/bar alignment, power-chord labels (used only when the model can't tell major from minor), bass-aware slash chords, diagrams, and GP5/MusicXML output are unchanged.

- `--chords model` (default) or `--chords chroma` (the v3 method). If the model fails to load for any reason, the run falls back to chroma; `summary.json` then reports `chord_method` and `chord_model_fallback`.
- Weights (12 MB, MIT) are downloaded on first use from a pinned upstream commit to `~/.cache/audio2tab/btc_model.pt` and checked by SHA-256. You can override this with `AUDIO2TAB_CACHE` or `AUDIO2TAB_BTC_WEIGHTS`. Weights are never committed.
- Cost: about 1 s extra per 3-min song on CPU and about 100 MB extra RAM.
- Benchmark: `python -m tests.chord_bench 10 out/chord_bench.json` runs the full pipeline. `python -m tests.chord_bench oracle 30` scores the detector alone on true segment boundaries. Both use synthetic songs with random keys and progressions, inversions, and guitar-only vs. full-mix arrangements, scored with mir_eval.

### Stem vs mix
The guitar stem is used only when its RMS is at least 25% of the mix (`--source auto`). On the synthetic band mix,
Demucs heavily attenuates the Karplus-Strong guitar (stem/mix RMS 0.04) and transcribing the stem scores worse
(onset+pitch F1 0.48 vs 0.62 on the mix), so auto picks the mix. On real recordings with a prominent guitar the stem
wins. Force with `--source stem|mix`.

