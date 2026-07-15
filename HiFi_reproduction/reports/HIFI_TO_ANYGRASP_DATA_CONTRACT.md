# HiFi-to-AnyGrasp data contract

## Scope and boundary

This contract defines the hand-off from the frozen HiFi-CS OCID-VLG evaluation to a separate AnyGrasp candidate-generation environment. The main export is prediction-only: `target_mask.png` comes from the HiFi predicted foreground mask. Ground-truth instance masks, answers converted into masks, and other oracle target regions must not enter the main AnyGrasp input or candidate-ranking path.

`anygrasp_verified_subset/` is a separately named diagnostic subset for format and geometry checks. It must never be concatenated with the main inference set or used to replace predicted masks. Any diagnostic reference labels in that subset are evaluation-only.

No external code is copied into this package. The official [AnyGrasp SDK demo](https://github.com/graspnet/anygrasp_sdk/blob/main/grasp_detection/demo.py) is used only as an interface reference: it converts RGB-D pixels to metric XYZ, calls `AnyGrasp.get_grasp(points, colors, ...)`, then performs NMS and score sorting.

## Package structure

```text
hifi_anygrasp_inputs_<run>/
├── anygrasp_input_predicted_mask/
│   └── <sample_id>/
│       ├── color.png
│       ├── depth.png
│       ├── target_mask.png
│       ├── target_probability.npy
│       ├── language.txt
│       ├── intrinsics.json
│       ├── metadata.json
│       └── checksums.sha256
├── manifests/
│   ├── manifest.ready.jsonl
│   ├── manifest.ready.csv
│   └── package_manifest.json
├── checksums/
│   └── PACKAGE_CONTENTS.sha256
├── docs/
│   └── HIFI_TO_ANYGRASP_DATA_CONTRACT.md
└── anygrasp_verified_subset/
```

Only rows marked `ready: true` with no blockers are packaged. `.venv`, Git metadata, source code, logs, the broader HiFi `predictions/` workspace, model-resolution probability/logit arrays, checkpoints, and unrelated data are excluded. `target_probability.npy` is not an intermediate in this contract: it is the required original-resolution foreground probability paired with the predicted target mask.

## Per-sample inputs

All image-aligned arrays use the original OCID-VLG resolution and pixel coordinates.

| File | Required format | Meaning |
|---|---|---|
| `color.png` | 640×480 RGB, `uint8` | Original RGB frame. |
| `depth.png` | 640×480 single-channel `uint16` | Original raw depth in millimetres. Zero is invalid/missing depth. |
| `target_mask.png` | 640×480 single-channel `uint8`, values exactly `0` or `255` | HiFi predicted target region; `255` is selected foreground. |
| `target_probability.npy` | shape `(480, 640)`, little/native-endian `float32`, values in `[0, 1]` | HiFi foreground probability at original resolution, retained for threshold auditing or downstream soft weighting. It is not a logit. |
| `language.txt` | UTF-8 | Exact referring expression, retained for provenance and downstream semantic scoring. |
| `intrinsics.json` | JSON | Per-sample effective pinhole intrinsics and fit diagnostics. |
| `metadata.json` | JSON | Stable IDs, source paths/hashes, HiFi checkpoint/evaluation provenance, and explicit predicted-mask source. |
| `checksums.sha256` | SHA-256 list | Checksums for the packaged files in that sample directory. |

`depth_scale` is `1000.0`: metric depth is `z_m = depth_uint16 / 1000.0`. Do not reinterpret `depth.png` as normalized 8-bit depth.

## Intrinsics provenance

Each `intrinsics.json` stores `fx`, `fy`, `cx`, `cy`, `depth_scale`, fit point count, RMSE, and p95 reprojection residual. Its `source` must be `derived_from_organized_pcd`.

These are effective pinhole intrinsics fitted from the supplied organized PCD pixel-to-XYZ correspondences. They are not factory camera calibration. Lens distortion coefficients and the original distortion model are unknown. Consumers must preserve that distinction in reports and must not label the fitted matrix as manufacturer calibration.

## Point-wise target-region construction

For pixel `(u, v)` with depth `d`:

```text
z = d / 1000.0
x = (u - cx) * z / fx
y = (v - cy) * z / fy
valid = (d > 0) and finite(x, y, z)
target = valid and (target_mask[v, u] == 255)
```

Build scene points/colors from `valid`. Build the language-conditioned region mask point-wise using the same flattened pixel order: `region_mask = target[valid]`. A runner may crop to `points[region_mask]`, or pass the corresponding object mask through an SDK adapter, but it must not substitute a ground-truth mask. Preserve the unmasked scene cloud when collision checking requires surrounding geometry.

## Expected AnyGrasp Top-M candidate file

The downstream runner must save one canonical file per sample:

```text
grasp_candidates_top_m.json
```

Project schema:

```json
{
  "schema_version": 1,
  "sample_id": "...",
  "coordinate_frame": "camera",
  "length_unit": "metres",
  "generator": "AnyGrasp",
  "top_m": 20,
  "candidates": [
    {
      "candidate_id": "g000",
      "rank_original": 1,
      "score_original": 0.0,
      "translation_m": [0.0, 0.0, 0.0],
      "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      "width_m": 0.0,
      "height_m": 0.0,
      "depth_m": 0.0,
      "object_id": -1
    }
  ]
}
```

Candidate matrices and translations are camera-frame values. The runner must document any coordinate transform it applies. Reject non-finite poses, malformed rotation matrices, invalid widths, and duplicate `candidate_id` values before ranking.

Save this Top-M candidate set once. Original ranking and reranking must reference the same `candidate_id` values rather than duplicating or regenerating candidates.

## Original ranking versus reranking

1. Run AnyGrasp once on the predicted-mask input.
2. Apply the declared collision/NMS settings once.
3. Freeze the resulting Top-M set in `grasp_candidates_top_m.json` in descending original AnyGrasp score order.
4. Treat `rank_original` and `score_original` as immutable baseline fields.
5. Compute semantic-geometric features for those same candidates.
6. Write reranking results separately, keyed by `candidate_id`, with reranker score and `rank_reranked`.
7. Compare original top-1/top-k and reranked top-1/top-k on the identical sample population and identical candidate set.

A reranker may reorder candidates only. Candidate regeneration, different collision filtering, or target-mask replacement creates a different experiment and must not be reported as reranking.

## Remaining blockers

- The licensed AnyGrasp SDK and model checkpoint are not present in this workspace.
- AnyGrasp execution requires a compatible CUDA environment; the current MPS evaluation environment cannot run the SDK path.
- Camera-to-robot/world extrinsics are unavailable, so exported candidates remain in the camera frame and cannot yet be executed by a robot.
- Factory intrinsics and lens distortion are unavailable; only PCD-derived effective intrinsics are supplied.
- No full six-degree-of-freedom AnyGrasp candidate files have been generated yet.
- Until those candidates exist, original-versus-reranked grasp metrics cannot be computed.
