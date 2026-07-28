# HiFi-CS predicted masks to official VGN candidates

This module implements only the offline inference boundary:

```text
OCID-VLG raw RGB-D + HiFi-CS predicted target mask
  -> target-centred local TSDF
  -> pretrained official VGN
  -> official 6-DoF candidates
  -> target-mask filtering
  -> deterministic official-quality top-K / top-1
```

It does not train HiFi-CS or VGN, add a reranker, mix semantic/mask/collision
scores, perform IK, or control a robot. Candidate count is a count, not a grasp
success rate. `vgn_quality` is the official processed quality value; it is not
reported as a calibrated probability.

## Pinned upstream and model

- Repository: [ethz-asl/vgn](https://github.com/ethz-asl/vgn)
- Branch: `corl2020`
- Commit: `d7af0622433f52ae88ebe81533f12b46b33e951a`
- License: [BSD-3-Clause](https://github.com/ethz-asl/vgn/blob/d7af0622433f52ae88ebe81533f12b46b33e951a/LICENSE)
- Expected checkpoint: `third_party/vgn/data/models/vgn_conv.pth`
- Checkpoint in this workspace: SHA256
  `ba3391d0805e9c9b178cd18106866313cee808ff2b654f689663e92a814cec4b`

The checkpoint came from the data bundle linked by the
[official README](https://github.com/ethz-asl/vgn/blob/d7af0622433f52ae88ebe81533f12b46b33e951a/README.md).
The upstream project does not publish a reference checkpoint SHA256, so the
local hash is a reproducibility identifier, not independent proof of origin.
If the file is missing, inference stops with instructions to obtain this bundle;
random weights are never used.

The checkout is used without ROS. The production adapter imports the official
network, perception, grasp and transform modules. It mirrors the official
`predict/process/select` path because upstream `vgn.detection` imports ROS
visualization at module import time. The mirrored BSD-licensed code is
attributed in `src/grasping/vgn_adapter.py` and regression-tested directly
against upstream.

## Discovered local data contract

The verified prediction-only handoff is:

```text
runs/hifics_ocidvlg_20260711_112921/anygrasp_input_predicted_mask/
```

Its `manifest.jsonl` has 7,675 rows. The detected mapping is printed at startup:

```text
sample_id      <- sample_id
dataset_index  <- sample_index
instruction    <- query
scene_id       <- scene_id
bundle_dir     <- output_dir
rgb/depth      <- output_dir/metadata.json: source_rgb/source_depth
predicted mask <- output_dir/target_mask.png
```

The raw root is `../crog_reproduction/OCID-VLG`. RGB, depth and mask are
640×480; depth is raw 16-bit millimetres and the predicted mask is uint8 0/255.
Do not use `hifics/datasets/ocidvlg_final_dataset/...`: those 352×352 prepared
assets contain an 8-bit transformed depth and a GT mask field.

No factory calibration exists in the raw dataset. Each bundle contains a
scene-specific `intrinsics.json` fitted from its organized PCD. The runner can
use it only through the explicit `--intrinsics per-sample-bundle` setting and
records `source=derived_from_organized_pcd`, `factory_calibration=false`, and
the fit diagnostics. It must not be described as manufacturer calibration. A
strict JSON/YAML configuration is also supported; start with
`configs/ocid_intrinsics.example.yaml` and replace every placeholder with a
real, provenance-backed value.

## Environment and preflight

The upstream README reports Python 3.8 / Ubuntu 20.04 testing. This adapter was
validated on macOS arm64 with Python 3.12.12, PyTorch 2.6.0, SciPy 1.17.1 and
Open3D 0.19.0. Create an isolated environment and install:

```bash
python -m pip install -r requirements-vgn.txt
```

Minimal real-checkpoint CPU check:

```bash
python -m scripts.check_vgn_environment \
  --vgn-root third_party/vgn \
  --vgn-weights third_party/vgn/data/models/vgn_conv.pth \
  --device cpu
```

`auto` chooses CUDA when available and otherwise CPU; it never selects MPS on
macOS. Explicit `--device mps` first runs a Conv3d smoke test, and an MPS
inference failure is logged before falling back to CPU.

Inspect the real geometry/provenance before inference:

```bash
python -m scripts.inspect_ocid_geometry \
  --ocid-root ../crog_reproduction/OCID-VLG \
  --manifest runs/hifics_ocidvlg_20260711_112921/anygrasp_input_predicted_mask/manifest.jsonl \
  --max-samples 10
```

## Official model and post-processing contract

The `official` preset fixes:

- workspace `0.30 m`, resolution `40`, voxel `0.0075 m`;
- finger depth `0.05 m`, support plane at task `z=0.05 m`;
- input `[B,1,40,40,40]`;
- sigmoid quality, normalized `xyzw` quaternion and scalar width volumes;
- Gaussian quality smoothing `sigma=1.0`;
- accepted width interval `1.33..9.33` voxels;
- quality threshold `0.90` and maximum filter size `4`.

The adapter preserves `[i,j,k] * voxel_size` with no half-voxel offset. Changing
workspace size or resolution emits:

```text
changing physical scale invalidates strict pretrained-model comparability
```

The TSDF integrates the full local scene depth, not masked-only depth. The
target mask is used for target centroid/workspace construction and candidate
filtering. A single OCID frame is explicitly recorded as
`tsdf_mode=single_view_adaptation`; official simulations and the Panda example
integrate multiple views, so these results are not a reproduction of the
official multi-view robot pipeline. Uncalibrated top/bottom fusion is rejected.
`--multi-view-manifest` is reserved but deliberately fails closed in this
release: calibrated common-frame multi-view integration remains future work and
is not claimed by the present single-view results.

## Run

Single-sample CPU smoke test:

```bash
python -m scripts.run_vgn_on_hifics \
  --ocid-root ../crog_reproduction/OCID-VLG \
  --manifest runs/hifics_ocidvlg_20260711_112921/anygrasp_input_predicted_mask/manifest.jsonl \
  --hifi-root . \
  --vgn-root third_party/vgn \
  --vgn-weights third_party/vgn/data/models/vgn_conv.pth \
  --intrinsics per-sample-bundle \
  --output outputs/hifics_vgn \
  --device cpu \
  --vgn-preset official \
  --selection-policy highest_vgn_quality \
  --sample-id q0000000_b32eb3299dcd3ae9 \
  --top-k 50 --seed 42 \
  --save-tsdf --save-pointclouds --visualize --overwrite
```

Ten-sample resumable run:

```bash
python -m scripts.run_vgn_on_hifics \
  --ocid-root ../crog_reproduction/OCID-VLG \
  --manifest runs/hifics_ocidvlg_20260711_112921/anygrasp_input_predicted_mask/manifest.jsonl \
  --hifi-root . \
  --vgn-root third_party/vgn \
  --vgn-weights third_party/vgn/data/models/vgn_conv.pth \
  --intrinsics per-sample-bundle \
  --output outputs/hifics_vgn \
  --device auto --vgn-preset official \
  --selection-policy highest_vgn_quality \
  --top-k 50 --max-samples 10 --seed 42 \
  --save-tsdf --save-pointclouds --visualize
```

Successful samples are skipped. Existing failures require `--retry-failures`;
`--overwrite` reprocesses either. A SHA256 run signature covers the manifest,
checkpoint, calibration config, inference options, and saved-artifact options;
resuming with an incompatible signature fails closed. Explicit `--overwrite`
starts a clean generated run on a signature change and clears a selected sample
directory before rewriting it, preventing stale optional artifacts. JSON, CSV
and NPZ files use temporary files and atomic rename. An atomic per-sample
`_SUCCESS.json` is written only after all configured diagnostics and
visualizations complete, so an interrupted partial sample is not mistaken for
a resumable success.

## Candidate filtering and selection

Every official local maximum is saved in original `(i,j,k)` scan order. Its
camera-frame center is projected with the configured intrinsics. Default target
acceptance is only `inside_dilated_target_mask`, with a 3-pixel dilation.
Nearest target distance and projected depth difference are diagnostics and do
not modify quality. If official candidates exist but none project into the
target mask, status is `no_target_grasp`; a grasp on another object is never
substituted.

`highest_vgn_quality` stably sorts by
`(-vgn_quality, official_selection_index)` and selects the first candidate.
`official_sim_random` is a fixed-seed simulation of the official random
execution choice. `official_panda_highest_z` simulates the Panda example's
highest task-z choice. These are execution selection policies; they are not
three different VGN scoring functions.

## Tests and outputs

Run the focused suite:

```bash
PYTHONPATH=. pytest -q \
  tests/test_vgn_integration.py \
  tests/test_full_vgn_experiment.py \
  tests/test_full_vgn_runner.py \
  tests/test_ocid_annotations.py \
  tests/test_vgn_report_gallery.py \
  tests/test_vgn_representatives.py \
  tests/test_vgn_oracle_metrics.py
```

It contains all required unit tests, including
`test_official_postprocessing_matches_upstream`. The completed predicted-mask
run contains all 7,675 manifest rows in `outputs/hifics_vgn_full/summary.csv`:
3,263 `ok`, 2,938 `no_target_grasp`, and 1,474 `no_official_grasp`, with zero
technical failures. Official-candidate coverage is 6,201/7,675 (80.79%) and
target-candidate coverage is 3,263/7,675 (42.51%). These are offline candidate
coverage metrics, not physical grasp success rates.

The separate GT-mask oracle also contains all 7,675 rows and differs only in
the target-mask source. Recompute the paired scene-cluster comparison with:

```bash
python -m scripts.evaluate_full_ocid_vgn \
  --output outputs/hifics_vgn_full \
  --oracle-output outputs/hifics_vgn_gt_oracle_full \
  --bootstrap-replicates 10000 --cluster-key scene_id --seed 42
```

The frozen paired output is
`outputs/hifics_vgn_full/metrics/oracle_delta.json`. GT-oracle minus predicted
target-candidate coverage is -0.07062 (scene-cluster 95% CI
[-0.08994, -0.05102]). This result is specific to center-projection filtering
in the single-view adaptation; it is not 6-DoF grasp accuracy.

Sixty deterministic representative 3-D cases were selected across statuses,
query types, target categories, mask-IoU/VGN-quality quantiles, and candidate
counts. To reproduce the selection and synchronize rendered PLY diagnostics:

```bash
python -m scripts.select_vgn_representatives \
  --ocid-output outputs/hifics_vgn_full \
  --manifest runs/hifics_ocidvlg_20260711_112921/anygrasp_input_predicted_mask/manifest.jsonl \
  --count 60 \
  --rendered-output outputs/hifics_vgn_full/representative_3d
```

The first real CPU smoke sample produced three official candidates, one
target-filtered candidate, and target top-1 quality `0.9471205473` with
`custom_reranking=false`.

Each sample records candidate NPZ/JSON, `top1.json`, workspace and support plane,
2-D overlays, quality projections, and optional TSDF/point clouds. Every
`top1.json` states:

```json
{
  "selection_policy": "highest_vgn_quality",
  "score_source": "official_vgn_processed_quality",
  "custom_reranking": false
}
```

Known scientific limitations are stored in `run_config.json` and every
`top1.json`: single-view TSDF adaptation, no OCID-VLG 6-DoF ground truth, and no
robot execution validation.
