# syntax=docker/dockerfile:1
# audio2tab: audio/video -> guitar tab. CPU-only, multi-stage.
# Baked: BTC chord weights (12 MB) and basic-pitch ONNX (ships in the wheel).
# Demucs htdemucs_6s (~53 MB) downloads on first run into the /cache volume
# (build with --build-arg BAKE_DEMUCS=1 to bake it in for offline use).
ARG PYTHON=3.11

FROM python:${PYTHON}-slim AS build
ARG BAKE_DEMUCS=0
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchaudio==2.11.1
WORKDIR /src
COPY docker/requirements.txt ./
RUN pip install --no-deps -r requirements.txt
COPY pyproject.toml README.md ./
COPY audio2tab ./audio2tab
RUN pip install --no-deps --no-build-isolation . 2>/dev/null || (pip install setuptools && pip install --no-deps --no-build-isolation .) \
 && python -c "import audio2tab.pipeline, basic_pitch.inference, demucs.separate, weasyprint, librosa, music21, guitarpro" \
 && pip uninstall -y pip setuptools wheel || true
# bake BTC weights (MIT, SHA-256 checked) and optionally Demucs
RUN AUDIO2TAB_CACHE=/opt/models python -c "from audio2tab import btc; print(btc.weights_path())" \
 && if [ "$BAKE_DEMUCS" = 1 ]; then TORCH_HOME=/opt/models/torch python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"; fi
# strip: bytecode caches, bundled test suites, basic-pitch non-ONNX model formats, headers, static libs
RUN SP=/opt/venv/lib/python*/site-packages \
 && find /opt/venv -name __pycache__ -type d -prune -exec rm -rf {} + \
 && find $SP -mindepth 2 -type d -name tests -prune -exec rm -rf {} + \
 && rm -rf $SP/basic_pitch/saved_models/icassp_2022/nmp $SP/basic_pitch/saved_models/icassp_2022/nmp.mlpackage $SP/basic_pitch/saved_models/icassp_2022/nmp.tflite \
 && rm -rf $SP/torch/include $SP/torch/share/cmake $SP/torch/bin/test_* \
 && find $SP -name "*.a" -delete

# static ffmpeg (decodes MP3/M4A/MP4; ~80 MB) instead of Debian's ffmpeg, which drags in ~300 MB of LLVM/Mesa/codec libs
FROM python:${PYTHON}-slim AS ffmpeg
RUN python - <<'PY'
import hashlib, tarfile, urllib.request, io
u = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
data = urllib.request.urlopen(u).read()
assert hashlib.md5(data).hexdigest() == urllib.request.urlopen(u + ".md5").read().split()[0].decode()
with tarfile.open(fileobj=io.BytesIO(data)) as t:
    m = next(x for x in t.getmembers() if x.name.endswith("/ffmpeg"))
    open("/ffmpeg", "wb").write(t.extractfile(m).read())
PY
RUN chmod 755 /ffmpeg && /ffmpeg -version | head -1

FROM python:${PYTHON}-slim
RUN for i in 1 2 3 4 5; do apt-get -o Acquire::Retries=5 update && break; sleep 3; done \
 && apt-get install -y --no-install-recommends libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/* /usr/share/doc/* /usr/share/man/*
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/models /opt/models
COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg
RUN useradd -m -u 1000 a2t && mkdir -p /cache /data && chown a2t /cache /data
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 \
    AUDIO2TAB_BTC_WEIGHTS=/opt/models/btc_model.pt TORCH_HOME=/cache/torch \
    NUMBA_CACHE_DIR=/cache/numba MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/cache
USER a2t
WORKDIR /data
VOLUME ["/cache"]
ENTRYPOINT ["audio2tab"]
CMD ["--help"]
