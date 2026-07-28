# LAVT-RIS on OCID-VLG

This adaptation keeps the official LAVT-One network and changes only the data,
runtime, loss/evaluation, and experiment plumbing needed for OCID-VLG visual
grounding:

```text
RGB + referring expression -> foreground probability -> binary target mask
```

Depth, grasp labels, ground-truth masks, boxes, SAM-family models, and HiFi-CS
predictions are never model inputs or post-processing inputs.

## Reproduction status

The audited `unique` protocol has 26,295 train, 3,778 validation, and 7,675 test
expressions. Dataset audit, 33 tests, a real Swin-Base CPU forward/backward, a
480×480 MPS forward/backward, an 8-sample overfit gate, and a checkpoint
save/resume/export smoke run have passed. The current truthful status is
`SUCCESS_SMOKE_ONLY`: no 40-epoch full-dataset checkpoint or full-test LAVT
result is claimed.

The architecture source is the [official LAVT-RIS repository](https://github.com/yz93/LAVT-RIS).
The split metadata comes from the local installation of the
[official OCID-VLG API](https://github.com/gtziafas/OCID-VLG). Because that
repository has no declared license, its code is dynamically imported rather
than copied here.

## Environment and weights

Create a Python 3.11 environment and install:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements_macos.txt
```

The tested local weight layout is:

```text
pretrained_weights/
  swin_base_patch4_window12_384_22k.pth
  bert-base-uncased/
    config.json
    pytorch_model.bin
    vocab.txt
```

The Swin file is the [official ImageNet-22K Swin-Base checkpoint](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth).
Its tested SHA256 is
`70812ab6b0a7a38712409d13976df9431632466eaacf991d5e90d9a1e91f3ab1`.
Training refuses to start without the requested Swin initialization.

## One-command pipeline

With the data/API and weights available, this command performs discovery,
audit, tests, overfit, smoke/resume, 40-epoch training, full evaluation,
prediction export, and the guarded HiFi-CS comparison:

```bash
bash scripts/run_ocid_vlg_pipeline.sh
```

It is intentionally not a quick demo. The measured MPS smoke throughput
projects roughly 242 hours for 40 full train/validation epochs on the tested
machine. Every stage fails closed: full training starts only after audit,
tests, overfit, and smoke succeed.

## Individual entry points

```bash
# Discover the canonical OCID-VLG root and reproduce the HiFi frozen train/test IDs.
.venv/bin/python scripts/discover_data.py --version unique

# Full integrity and token audit.
eval "$(.venv/bin/python scripts/paths_to_env.py)"
.venv/bin/python scripts/audit_ocid_vlg.py \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --tokenizer pretrained_weights/bert-base-uncased \
  --max-tokens 20 --image-size 480 --seed 42 \
  --json-output outputs/dataset_audit.json \
  --token-output outputs/token_length_audit.json \
  --markdown-output docs/OCID_VLG_DATASET_AUDIT.md \
  --visualization-dir outputs/audit_visualizations

.venv/bin/python -m pytest -q
bash scripts/run_overfit_test.sh
bash scripts/run_smoke_test.sh
bash scripts/train_ocid_vlg.sh
```

Evaluation, export, and comparison accept the resulting run:

```bash
bash scripts/evaluate_ocid_vlg.sh \
  outputs/ocid_vlg/RUN/checkpoints/checkpoint_best_miou.pth \
  outputs/ocid_vlg/RUN
bash scripts/export_ocid_predictions.sh \
  outputs/ocid_vlg/RUN/checkpoints/checkpoint_best_miou.pth \
  outputs/ocid_vlg/RUN
bash scripts/compare_with_hifics.sh outputs/ocid_vlg/RUN
```

The primary prediction is two-class `argmax`. Original-resolution metrics are
computed by bilinearly resizing logits to 480×640 before softmax/argmax and
comparing against the untouched instance-derived binary target.

## Outputs and resume

Each run records resolved configuration, environment, training history,
pretrained-load audit, full-state checkpoints, both evaluation resolutions,
per-sample CSV/Parquet, prediction manifest, probability arrays, binary masks,
metadata, and qualitative figures.

The first smoke invocation uses `--stop_after_epochs 1` while retaining a
two-epoch scheduler horizon. The second invocation restores model, optimizer,
scheduler, Python/NumPy/PyTorch RNG state, and continues at epoch 1. The same
`--resume` path supports interrupted full training.

See [the implementation audit](docs/IMPLEMENTATION_AUDIT.md),
[the dataset audit](docs/OCID_VLG_DATASET_AUDIT.md), and
[the current results](docs/LAVT_OCID_VLG_RESULTS.md).
