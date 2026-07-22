# Official SAM 3 refinement runbook

This runbook separates CUDA SAM 3 from both `.venv-gqcnn` and the TensorFlow 1.15 GQ-CNN
container. Commands assume the repository is `HiFi_reproduction` and a Linux x86-64 host
with an NVIDIA GPU and working Docker daemon.

## 1. Obtain gated access

1. Sign in at <https://huggingface.co/facebook/sam3> and accept the official conditions.
2. Confirm the authorized account with `hf auth whoami`.
3. Supply `HF_TOKEN` through the cloud secret manager or run `hf auth login`. Never put the
   token in a CLI argument, config file, shell history, or experiment log.

The pinned Transformers revision used by this experiment is:

```text
3c879f39826c281e95690f02c7821c4de09afae7
```

## 2. Prepare portable inputs on the Mac

The current ten-sample bundle is already at `outputs/sam3_refinement_inputs`. To recreate
it under a fresh path, pass `--output-root`; the script refuses overwrite.

```bash
python scripts/prepare_sam3_refinement_inputs.py \
  --output-root outputs/sam3_refinement_inputs_ten_rebuilt
```

## 3. Build the isolated CUDA image

```bash
docker build \
  -f docker/sam3-refinement/Dockerfile \
  -t vlmgrasp/sam3-refinement:1.0.0 \
  .
```

The image pins Python 3.12 from Ubuntu 24.04, CUDA 12.8 runtime, PyTorch 2.10.0 cu128,
torchvision 0.25.0, and Transformers 5.14.1. This is independent of HiFi-CS training and
GQ-CNN scoring environments.

## 4. Download and verify official Transformers weights

```bash
export SAM3_MODEL_CACHE="$PWD/models/sam3-huggingface"

docker/sam3-refinement/run.sh \
  python3 scripts/sam3/download_sam3_model.py \
  --repo-id facebook/sam3 \
  --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
  --local-dir /models/huggingface/facebook-sam3
```

The resulting `facebook-sam3.download_manifest.json` records the resolved commit, file
sizes, and SHA-256 values, but never the token. A repeat offline validation is:

```bash
docker/sam3-refinement/run.sh \
  python3 scripts/sam3/download_sam3_model.py \
  --repo-id facebook/sam3 \
  --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
  --local-dir /models/huggingface/facebook-sam3 \
  --local-files-only \
  --manifest /models/huggingface/offline_verification_manifest.json
```

## 5. One real sample, then the frozen ten

Run one sample into a distinct root first:

```bash
docker/sam3-refinement/run.sh \
  python3 scripts/run_sam3_refinement.py \
  --input-root outputs/sam3_refinement_inputs \
  --output-root outputs/sam3_refined_masks_one_sample \
  --model-id-or-path /models/huggingface/facebook-sam3 \
  --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
  --local-files-only \
  --sample-id q0000000_b32eb3299dcd3ae9
```

Inspect `prompt_visualization.png`, all candidate masks, selection metrics, selected mask,
and fallback metadata. If valid, run the same pinned model on all ten:

```bash
docker/sam3-refinement/run.sh \
  python3 scripts/run_sam3_refinement.py \
  --input-root outputs/sam3_refinement_inputs \
  --output-root outputs/sam3_refined_masks \
  --model-id-or-path /models/huggingface/facebook-sam3 \
  --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
  --local-files-only
```

The runner requests multimask output, restores each hypothesis to 480 × 640, retains every
hypothesis and quality score, selects without GT, and writes an explicit HiFi fallback if
inference or validation fails.

## 6. Import and evaluate on Mac

If the GPU wrote to a different filesystem, copy the complete output directory and verify
it before import:

```bash
python scripts/import_sam3_refinement_outputs.py \
  /path/to/transferred/sam3_refined_masks \
  --destination outputs/sam3_refined_masks

python scripts/evaluate_sam3_mask_refinement.py
```

## 7. Regenerate and rank grasps

The orchestrator refuses to run unless `run_metadata.json` proves at least one real SAM 3
inference completed. It prints commands by default:

```bash
python scripts/run_coarse_to_fine_grasp_pipeline.py
python scripts/run_coarse_to_fine_grasp_pipeline.py --execute
```

It creates only new roots, in order:

1. `outputs/sam3_dexnet_input_bundles`
2. `outputs/dexnet_candidates_sam3_ten_samples`
3. GQCNN-2.1 fixed-candidate q values in the new candidate root
4. frozen geometric ranking in the new candidate root
5. `outputs/gqcnn_sam3_ranking_evaluation`
6. `outputs/sam3_grasp_comparison`

## 8. Larger held-out cohorts after the ten-sample smoke test

First 100 frozen manifest rows:

```bash
python scripts/prepare_sam3_refinement_inputs.py \
  --sample-limit 100 \
  --output-root outputs/sam3_refinement_inputs_first100
```

Full 7,675-row frozen test manifest:

```bash
python scripts/prepare_sam3_refinement_inputs.py \
  --sample-limit 7675 \
  --output-root outputs/sam3_refinement_inputs_full
```

Use new refinement, adapter, candidate, scoring, ranking, and evaluation roots for each
cohort. Do not tune prompt, selector, sampler, consistency, or ranking thresholds on the
final test outputs.

## 9. Final integrity checks

```bash
python scripts/audit_sam3_protected_outputs.py verify
python -m pytest -q tests/test_sam3_refinement.py
.venv-gqcnn/bin/python -m pytest -q tests \
  --ignore tests/test_sam3_refinement.py \
  --ignore tests/test_vgn_integration.py
```

Do not collect tests recursively from the repository root: historical run/artifact source
snapshots contain duplicate test modules. VGN tests belong to their separate environment and
are not dependencies of this SAM 3 → Dex-Net experiment.
