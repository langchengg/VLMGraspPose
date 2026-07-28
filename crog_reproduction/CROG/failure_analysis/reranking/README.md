# Frozen CROG candidate reranking

This package asks one narrow question: with CROG's original Top-5 candidates
held fixed, can a post-hoc scorer choose a better Top-1 grasp? It never trains
or modifies CROG and refuses feature configurations that change `K=5`, peak
threshold `0.4`, minimum peak distance `2`, or predicted-mask threshold `0.35`.

## Outputs

`build-features` writes a new directory containing:

- `features.jsonl`: inference-time data only. It contains predicted maps' per-
  candidate aggregates, frozen geometry, reliabilities, and missing reasons.
- `labels.jsonl`: candidate keys and official CROG/Jacquard validity labels.
- `predictions.jsonl`: a backward-compatible combined view retaining all old
  failure-analysis fields. Rankers never read this file.
- `metadata.json`: checkpoint SHA256, source commit, dataset/split, runtime,
  post-processing provenance, config hashes, and a cache fingerprint.
- `commit_journal.jsonl`: ordered transaction markers used to trim only an
  uncommitted JSONL tail after normal process interruption.

The test exporter independently joins the immutable historical prediction
JSONL and fails on any old/new J@1 or J@Any mismatch. Fingerprints include
dirty/untracked behavior source, model/util sources, inputs, configs, runtime,
effective batch size, and workers. Resume never silently accepts a different
ranker, input file, seed, model, calibration, or tuning file.

Dense maps are off by default. `--save-dense-maps` is intended only for small
debug runs and writes compressed float16 NPZ files.

## Commands

```bash
.venv/bin/python -m failure_analysis.reranking.cli build-features \
  --split test --limit 10 --device mps --batch-size 2 --workers 0 \
  --output failure_analysis/reranking_outputs/test_10

.venv/bin/python -m failure_analysis.reranking.cli build-labels \
  --features <dir>/features.jsonl --predictions <dir>/predictions.jsonl \
  --regression-reference failure_analysis/predictions/test_predictions.jsonl \
  --output <dir>/labels_rebuilt.jsonl --limit 10

.venv/bin/python -m failure_analysis.reranking.cli evaluate \
  --features <dir>/features.jsonl --labels <dir>/labels.jsonl \
  --ranker rule_fixed_v1 --output <dir>/eval_rule_fixed_v1 --limit 10

.venv/bin/python -m failure_analysis.reranking.cli train-mlp \
  --features <train>/features.jsonl --labels <train>/labels.jsonl \
  --output <train>/mlp_ranker.pt --device cpu --seed 17

.venv/bin/python -m failure_analysis.reranking.cli evaluate \
  --features <test>/features.jsonl --labels <test>/labels.jsonl \
  --ranker mlp --mlp-model <train>/mlp_ranker.pt \
  --output <test>/eval_mlp --device cpu
```

Every command supports `--help`, `--limit`, `--resume`, and explicit
`--overwrite`. Existing outputs are not overwritten by default.

Calibration files must use
`{"calibration": {...}, "provenance": {"source_split": "train"}}`.
Validation-tuned weight files must use
`{"weights": {...}, "provenance": {"source_split": "val"}}`.

## Leakage boundary

All scoring goes through `INFERENCE_FEATURE_ALLOWLIST`. Candidate ids, original
ranks, image ids, absolute coordinates, GT grasps/masks, evaluator results, and
PCD `label` are not model inputs. Missing inference geometry is represented by
a value, a reliability in `[0,1]`, and a reason. Rule scores use
`E(f,r)=r*f+(1-r)*0.5`.

The metric 3D collision path is disabled because the repository does not define
the physical gripper or approach frame. The implemented safety feature is named
`relative_2p5d_obstacle_proxy`; it is not a collision-free guarantee.

Main Jacquard metrics intentionally retain CROG's legacy angle gate for
baseline comparability. It is not the general 180-degree symmetric angle
formula; the implementation report records a corrected-angle sensitivity
analysis separately.
