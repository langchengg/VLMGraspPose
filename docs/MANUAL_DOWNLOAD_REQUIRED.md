# Official GraspNet compact-data execution and recovery

The old 250.63 GB simultaneous-footprint estimate is retained only as
historical audit evidence. It is not an execution gate. The active route
budgets one official archive, its measured selective extraction, one temporary
member, the next cache increment, and a fixed 20 GB reserve. The current
decision and history are written to:

- `artifacts/graspnet6d/audit/staged_disk_budget.json`
- `artifacts/graspnet6d/audit/staged_disk_budget.md`
- `artifacts/graspnet6d/audit/staged_disk_budget_history.jsonl`

## Official sources and compact scope

The source of truth is <https://graspnet.net/datasets.html>. The downloader
uses only the Google Drive IDs and JBox links published there, alternating the
two sources for up to 12 resumable attempts. Every response must pass size,
ZIP magic, `file`, `unzip -t`, Python CRC, and SHA-256 checks. HTML/login
responses are quarantined and never accepted as data.

The route also expects a local object-name catalogue at
`configs/graspnet6d/graspnet_object_catalog_v1.json`. This dataset-derived
metadata is ignored by Git and must be retained or reconstructed locally from
the official GraspNet object catalogue after accepting its current terms.

The paper-lite route downloads `train_4.zip` and `train_3.zip`, adding
`train_2.zip` only if fewer than 35 unique training scenes are available. It
then downloads `grasp_label.zip`, `collision_label.zip`, and `models.zip`.
`dex_models.zip` is attempted only as a non-blocking evaluator acceleration.
Official test packages, rect labels, and `train_1.zip` are excluded.

The locked configuration retains the repository's existing `kinect` camera
(not both cameras), 16 deterministic frames per selected scene, and a
scene-disjoint 20/5/10 train/validation/test split. Training ZIPs are listed
before extraction. Only selected scene metadata and RGB/depth/label/meta/XML
files are materialised. Grasp labels and models are complete; collision labels
are limited to selected scenes.

## Automatic command

From the repository root:

```bash
PYTHONPATH=src .venv-graspnet6d/bin/python \
  -m graspnet6d.cli all --profile paper-lite --run-id <NEW_RUN_ID>
```

The resolved paper-lite config records the blocked predecessor
`20260816_022747_graspnet6d_vgn_lambdamart` as `parent_run_id`. Resume the same
immutable run after interruption:

```bash
PYTHONPATH=src .venv-graspnet6d/bin/python \
  -m graspnet6d.cli all --profile paper-lite --run-id <NEW_RUN_ID> --resume
```

Partial downloads are retained. A deleted archive is skipped on resume only
when its official download SHA, complete extraction marker, every selected
file's size/CRC/loader check, and a successful cleanup-audit row all agree.

## AI-assisted geometry review

Formal conversion renders and hashes at least 20 real geometry figures. The
review evidence uses schema
`graspnet6d_ai_assisted_geometry_visual_review_v2` and explicitly records:

- `review_kind: ai_assisted_geometry_review`
- a non-empty `reviewer_system`
- `independent_human_review: false`
- the fixed disclaimer that this is not an independent human review
- all rendered group IDs and exact figure SHA-256 values
- explicit approach-axis, table-orientation, and overlay checks per group

The executing agent inspects the hashed montage with visual capability and
then resumes automatically. The route never represents this as independent
human review. Programmatic reprojection, round-trip, SO(3), unit, workspace,
and evaluator-parity checks remain mandatory and fail closed.

## Audit and licence

Downloads are recorded in `downloads/graspnet/download_manifest.json`,
`downloads/graspnet/archive_checksums.sha256`, and
`artifacts/graspnet6d/audit/download_log.jsonl`. Archive removals happen only
after verified extraction and downstream loading, and every removal is added
to `disk_cleanup_log.csv` and `disk_cleanup_report.md`.

Read `docs/DATA_LICENSES.md` and the official dataset terms before redistribution.
The compact subset is derived from GraspNet training scenes; it is not the
complete benchmark and not an official test-server result.
