# Contributing to audio2tab

Thanks for helping! Bug reports, test clips, and pull requests are all welcome.

## Dev setup

```bash
git clone https://github.com/wixregiga/audio2tab && cd audio2tab
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[test]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
sudo apt install ffmpeg            # or: brew install ffmpeg
.venv/bin/python -m pytest -q      # about 2 minutes on CPU
```

## Guidelines

- **Tests cover what a user can see.** Write tests for inputs that are accepted, what the tab sheet shows, exports that open, playable frets, and chord accuracy. Don't write tests for private helpers. New features need an end-to-end or sheet-level test.
- **Accuracy changes need numbers.** Run `python -m tests.chord_bench 10` (chords) or `python tests/eval_e2e.py` (notes) and paste the before/after numbers in the PR.
- **No copyrighted audio.** Fixtures are synthesized at test time (`tests/synth.py`, `tests/chord_bench.py`). Never commit audio files, outputs, or model weights.
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`).
- Keep the CPU path fast. Check the runtime on a 3-minute song before adding a model.

## Reporting bugs

Open an issue with the bug template. Include the command you ran, the `*.summary.json` file, and if you can, a short clip you have the rights to share.
