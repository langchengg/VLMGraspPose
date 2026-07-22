FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/HiFi_reproduction
COPY docker/sam3-refinement/requirements.txt /tmp/sam3-requirements.txt
RUN python3 -m pip install --break-system-packages \
      torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install --break-system-packages -r /tmp/sam3-requirements.txt

CMD ["python3", "scripts/run_sam3_refinement.py", "--help"]

