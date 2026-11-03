#!/usr/bin/env bash
# One-command wrapper: ./run.sh song.mp3 [--out DIR] [--no-separate] [--tuning EADGBE] [--capo N] ...
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
exec "$DIR/.venv/bin/python" -m audio2tab.cli "$@" 2> >(grep -vE "Coremltools|tflite-runtime|Tensorflow is not installed|onnxruntime|^WARNING:root" >&2)
