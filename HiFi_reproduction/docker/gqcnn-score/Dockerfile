FROM --platform=linux/amd64 python:3.7.17-slim-bullseye

ARG GQCNN_COMMIT=499a609fe9dfb074bdfb6c4e6e33667ea50f4c21

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    CUDA_VISIBLE_DEVICES="" \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libspatialindex-dev \
      libxext6 \
      libxrender1 \
      unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
      pip==23.3.2 setuptools==68.0.0 wheel==0.42.0

COPY docker/gqcnn-score/requirements-py37.txt /tmp/requirements-py37.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-py37.txt

# Install the official ecosystem packages without letting their open-ended
# dependencies replace the pinned Python 3.7 / TensorFlow 1.15 stack.
RUN python -m pip install --no-cache-dir --no-deps \
      autolab-core==1.1.1 \
      autolab-perception==1.0.0 \
      visualization==1.0.0

COPY docker/gqcnn-score/visualization_headless_init.py \
     /usr/local/lib/python3.7/site-packages/visualization/__init__.py

# Fetch a clean official tree inside the image. The host checkout contains two
# deliberate macOS import guards and is therefore not copied into this image.
RUN git clone https://github.com/BerkeleyAutomation/gqcnn.git /opt/gqcnn \
    && cd /opt/gqcnn \
    && git checkout --detach "${GQCNN_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${GQCNN_COMMIT}" \
    && printf '%s\n' /opt/gqcnn > /usr/local/lib/python3.7/site-packages/gqcnn-official.pth

RUN python - <<'PY'
import platform
import tensorflow as tf
import gqcnn
from gqcnn.version import __version__ as gqcnn_version
assert platform.machine() == "x86_64", platform.machine()
assert tf.__version__ == "1.15.0", tf.__version__
assert not tf.test.is_gpu_available()
assert gqcnn_version == "1.3.0", gqcnn_version
print("GQCNN_DOCKER_IMPORT_OK machine=%s tensorflow=%s gqcnn=%s gpu=%s" % (
    platform.machine(), tf.__version__, gqcnn_version, tf.test.is_gpu_available()))
PY

WORKDIR /workspace/HiFi_reproduction
