# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-03

### Added
- Audio/video input (MP3, M4A, MP4, WAV, FLAC, OGG) decoded by ffmpeg.
- Guitar isolation with Demucs `htdemucs_6s`. The stem is used only when it is strong enough; otherwise the full mix is transcribed.
- Note transcription with Spotify basic-pitch (ONNX). Each note is checked against the audio (phantom-note, octave and harmonic filtering).
- Fret assignment by Viterbi over playable shapes (at most 4 fingers, span of 4 frets or less, capo and alternate tunings).
- Tuning detection (Standard, Drop D, half-step down, and more), key estimate, tempo, 4/4 vs 3/4, pickup bars and downbeats.
- **Chords from the BTC transformer** (Park et al., ISMIR 2019), with key-aware smoothing per half bar, bass-aware slash chords and power chords. The chroma-template HMM remains as a fallback (`--chords chroma`).
- Outputs: tab `.txt`/`.html`/`.pdf` with chord diagrams, Guitar Pro 5, MusicXML (TAB staff), alphaTex, MIDI, and a JSON/Markdown summary.
- Docker image (CPU-only) and a GHCR publishing workflow.
- mir_eval benchmarks for notes and chords on synthetic songs, plus end-to-end tests.

[0.1.0]: https://github.com/wixregiga/audio2tab/releases/tag/v0.1.0
