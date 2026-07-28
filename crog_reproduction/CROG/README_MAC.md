# CROG single-device training on Apple Silicon

This branch keeps the official CROG model, losses, OCID-VLG dataset, and metrics while adding a single-process `mps`/CUDA/CPU path. CUDA DDP remains available through the original files.

## Setup

```bash
git clone https://github.com/HilbertXu/CROG.git
cd CROG
git switch mac-mps-single-device

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements_mac.txt
```

Download the official OpenAI CLIP RN50 checkpoint expected by CROG:

```bash
mkdir -p exp/pretrain_clip
curl -L https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt -o exp/pretrain_clip/RN50.pt
shasum -a 256 exp/pretrain_clip/RN50.pt
```

Expected SHA-256:

```text
afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762
```

Set `root_path` in both Mac YAML files. In the reproduction workspace it is `../OCID-VLG`.

## Checks and debug run

```bash
python scripts/check_mps.py

python scripts/inspect_ocid_vlg.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml

python scripts/visualize_ocid_vlg_sample.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml \
  --num_samples 10

python train_crog_mac.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml

python test_crog_mac.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml \
  --checkpoint exp/OCID-VLG_multiple_mac/CROG_mac_mps_debug/last_model.pth \
  --split val
```

Optional prediction visualization:

```bash
python scripts/visualize_ocid_vlg_sample.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml \
  --num_samples 3 \
  --checkpoint exp/OCID-VLG_multiple_mac/CROG_mac_mps_debug/last_model.pth
```

## Full training command

Do not run this until the debug path and memory use have been reviewed:

```bash
python train_crog_mac.py \
  --config config/OCID-VLG/CROG_mac_mps.yaml
```

The full configuration retains 416×416 input, 50 epochs, batch size 1, and eight-step accumulation. Training in FP32 on a MacBook can take many hours or days and will not match the official two-RTX-4090 runtime. Accumulation increases effective batch size but does not reduce the activation memory of one sample.

The full Mac config enables a rolling mid-epoch recovery checkpoint:

```yaml
checkpoint_interval: 1000
```

This writes `mid_epoch_model.pth` after the optimizer step at roughly every 1000 training mini-batches. It is overwritten atomically, so it does not accumulate hundreds of 1.6 GB checkpoint files. Epoch-end checkpoints are still `last_model.pth`, `best_iou_model.pth`, and `best_jindex_model.pth`.

To resume from the latest completed epoch:

```yaml
resume: exp/OCID-VLG_multiple_mac/CROG_mac_mps/last_model.pth
```

To recover from an interruption inside an epoch:

```yaml
resume: exp/OCID-VLG_multiple_mac/CROG_mac_mps/mid_epoch_model.pth
```

Mid-epoch recovery restores model, optimizer, scheduler, and scaler state. It restarts the interrupted epoch from the beginning with the saved weights; it is a practical recovery checkpoint, not an exact data-loader-position resume.

## Known limitations

- MPS AMP is intentionally disabled; MPS and CPU train in FP32.
- `BatchNorm1d` running statistics are frozen for batch size 1; affine parameters remain trainable.
- Mid-epoch checkpoints are rolling recovery checkpoints. They prevent losing model/optimizer progress but do not resume at the exact shuffled mini-batch position inside the epoch.
- No global `PYTORCH_ENABLE_MPS_FALLBACK=1` is enabled. Record and isolate any unsupported operation instead.
- Python 3.12 requires modern NumPy/PyArrow. Legacy PyArrow-serialized LMDB helpers are not part of the OCID-VLG path and remain unsupported.
- `numpy<2` is used for compatibility with the official grasp-processing code.

## Return to official CUDA/DDP

```bash
git switch main
python -u train_crog.py --config config/OCID-VLG/crog_multiple_r50.yaml
```

Use the official Linux/CUDA environment for that path. Mac checkpoints are saved without a `module.` prefix; the Mac loaders accept both wrapped and unwrapped checkpoints.
