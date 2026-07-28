# CROG Re-ranking V2 implementation

Status: implementation, development experiment, immutable lock, and one-time formal test complete  
Run root: `failure_analysis/reranking_outputs/v2_20260727T174412+0100`

## What V2 is allowed to change

CROG still creates exactly five grasp candidates. V2 is a judge that may
reorder those five cards; it may not draw a sixth card or move any rectangle.
Every inference and evaluation path checks candidate IDs and geometry
checksums. Oracle@5 must therefore remain constant within each evaluator track.

Ground truth is kept in separate label artifacts. The inference loaders read
only RGB/depth, language, predicted maps, frozen candidate geometry, CROG
latents, and learned model outputs. In particular, gate inference has a
label-free code path and a test that fails if it tries to access a label.

## Sources of truth

- Frozen V1 candidates:
  `failure_analysis/reranking_outputs/full_test_17749_v1`
- CROG checkpoint:
  `exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth`
- Split protocol:
  `failure_analysis/reranking_outputs/v2_20260727T174412+0100/split_manifest.json`
- Evaluation geometry: `utils/grasp_metrics.py`
- V2 package: `failure_analysis/reranking_v2`

The design was checked against the official
[CROG paper](https://proceedings.mlr.press/v229/tziafas23a.html) and
[MIT-licensed implementation](https://github.com/HilbertXu/CROG). No external
code was copied. The local OCID-VLG data repository was used as a split and
format reference; its repository does not state a reusable software license.

## Data protocol

Official train, validation, and test frames have zero RGB-D content-hash
overlap. V2 uses split-qualified IDs such as
`multiple:train:00000007`, because the raw integer question index restarts in
each split.

The official train split is divided by capture sequence into:

| Partition | Expressions | Frames | Capture sequences | Purpose |
|---|---:|---:|---:|---|
| train | 53,431 | 1,035 | 108 | model fitting and grouped OOF |
| calibration | 9,790 | 166 | 19 | reserved train-internal calibration |
| validation | 8,669 | 172 | 99 | architecture and threshold selection |
| test | 17,749 | 344 | 115 | one locked final evaluation |

The split audit checks sample ID, frame ID, scene ID, separate RGB and depth
SHA-256 hashes, and a combined RGB-D content hash. All required intersections
are zero. Capture-sequence names can occur in
different official splits because the official dataset's scene boundary is the
individual RGB-D frame; this is reported rather than hidden.

Three grouped OOF folds are used to fit the stacked system. Three folds were
chosen because a full CROG export on this M5 Pro takes hours; in-sample stacking
is still forbidden. Every OOF sample records the checkpoint and fit folds that
produced its prediction. A validator rejects a checkpoint whose fit-fold list
contains the sample's held-out fold.

## Two isolated evaluator tracks

- `legacy_official`: the CROG-compatible rectangle/angle definition used for
  primary model selection and the frozen 83.2046876% baseline.
- `corrected`: a separately labelled geometric sensitivity analysis.

Both tracks have their own JSONL labels with an explicit evaluator version.
Corrected test results cannot be used to retrain or retune the primary method.

## Shared artifact DAG

```text
frozen CROG checkpoint + official data
                |
                v
       frozen Top-5 features
                |
       +--------+---------+
       |                  |
       v                  v
dual label artifacts   enhanced feature shards
(evaluation only)      (crop + latent ROI)
                          |
                grouped OOF base models
                          |
                grouped OOF SetRank
                          |
                pairwise R/H/N gate
                          |
              ensemble + perturbation check
                          |
                 original five IDs only
```

Every long artifact has input hashes, config, checkpoint hash, evaluator/code
fingerprint, seed, device, progress manifest, output hashes, unique-ID count,
and resume checks. Completed artifacts are idempotent; incompatible resume
attempts fail.

## Method 1: conservative pairwise gain gate

For original Top-1 versus each challenger, the training label is:

- R: Top-1 wrong and challenger correct;
- H: Top-1 correct and challenger wrong;
- N: the two have the same outcome.

The MLP predicts `p(R), p(H), p(N)` and computes
`gain = p(R) - harm_cost * p(H)`. A challenger replaces Top-1 only above the
validation-selected threshold. Harm cost and threshold are selected by the
lower bound of a frame-cluster bootstrap. If that lower bound is not positive,
the finite no-switch threshold is 2.0, which is above the maximum possible
gain. Ties use gain, original Q rank, then stable candidate ID.

The first model uses only existing scalar and pair-relation features and is
trained as a three-seed ensemble. The full gate also receives OOF critic,
latent, and SetRank summaries; ensemble and perturbation uncertainty are
applied as the final safety check after the gate.

## Method 2: aligned RGB-D critic

Each original candidate is sampled into a candidate-aligned crop. The axial
angle is canonicalized modulo 180 degrees. RGB and predicted soft maps use
linear interpolation; depth and validity use nearest-neighbour interpolation.
Depth is verified to be metres after the dataset loader, centered by the local
valid median, and accompanied by a validity channel. Missing depth produces
finite neutral values and never removes the sample.

The 14 channels are RGB, relative depth, depth validity, predicted mask,
quality, `sin(2θ)`, `cos(2θ)`, width, left/right finger templates, contact
template, and full gripper template. A small MPS/CPU-compatible CNN uses
candidate BCE plus hard within-list pair loss.

Declared ablations are RGB; RGB+mask+Q; RGB+depth; all channels; and all
channels without the gripper template.

## Method 3: frozen CROG latent ROI residual

Read-only hooks capture two verified tensors:

- neck output before the decoder: `[B,512,26,26]`;
- decoder output after language-conditioned cross attention:
  `[B,512,676]`, reshaped to `[B,512,26,26]`.

The code compares a forward pass with hooks removed against one with hooks
installed. Exact output equality is required. Rotated `grid_sample` ROIs are
average- and max-pooled, so only a `[5,1024]` vector per layer is cached.

The residual score is the clipped Q logit plus
`alpha * tanh(residual)`. `alpha=0` is tested to be exactly Q-only.
Projector/neck, decoder, scalar, and latent alternatives are validation
ablations.

## Method 4: listwise SetRank

A query remains one five-candidate set. A small Transformer has no positional
encoding and is tested for permutation equivariance. The original Q rank is an
explicit feature rather than an implicit token position. The model includes:

- listwise loss with a uniform target over all correct candidates;
- a candidate-correctness sigmoid head for Brier/NLL/ECE;
- an any-positive head so all-five-wrong queries remain in training;
- a bounded residual skip over Q-only ordering.

OOF critic scores/embeddings and OOF latent scores are used for train tokens.
Validation and test use the three train-fitted seed ensemble.

## Method 5: perturbation and ensemble uncertainty

For each original candidate, 17 internal views are scored:

- unchanged base;
- x and y centre shifts of ±2 and ±4 pixels;
- angle shifts of ±5 and ±10 degrees;
- width changes of ±5% and ±10%.

These views are never emitted as candidates. Only mean, standard deviation,
minimum, valid fraction, stability penalty, stable score, and seed
disagreement are saved. Final replacement needs a gate gain lower bound,
declared seed consensus, and no perturbation stability regression; otherwise
it falls back to original Top-1.

## Method 6: marked-candidate VLM reviewer

The offline implementation contains:

- a whole RGB view with neutral candidate markings;
- five independently marked local crops;
- a predicted-mask panel explicitly labelled “not GT”;
- an optional relative depth heatmap;
- deterministic shuffled display IDs and reverse mapping;
- strict JSON schema/parser, provider protocol, cache, retry, replay, and
  Q-only fallback.

Five GT-free train panels were rendered and visually inspected. No VLM
credential is configured and no paid call is authorized, so live VLM scoring
is marked `blocked`; no score is fabricated. VLM is excluded from the primary
combination.

## Primary V2 candidate

```text
Frozen Top-5
 -> aligned RGB-D critic + CROG latent ROI + scalar features
 -> residual listwise SetRank
 -> pairwise R/H/N expected-gain gate
 -> three-seed consensus + perturbation lower-bound check
 -> switch only with sufficient evidence
```

The validation result may legally select Q-only if the clustered lower bound
is not positive. That is a scientific result, not an implementation failure.

## Main CLI

All long commands expose `--resume`, `--dry-run`, `--max-samples`,
`--device auto|mps|cpu`, `--num-workers`, and `--seed`.

```bash
.venv/bin/python -m failure_analysis.reranking_v2.cli audit ...
.venv/bin/python -m failure_analysis.reranking_v2.cli export-dev ...
.venv/bin/python -m failure_analysis.reranking_v2.cli build-labels ...
.venv/bin/python -m failure_analysis.reranking_v2.cli extract-crops ...
.venv/bin/python -m failure_analysis.reranking_v2.cli extract-latents ...
.venv/bin/python -m failure_analysis.reranking_v2.cli train-critic ...
.venv/bin/python -m failure_analysis.reranking_v2.cli train-latent ...
.venv/bin/python -m failure_analysis.reranking_v2.cli train-setrank ...
.venv/bin/python -m failure_analysis.reranking_v2.cli train-oof-base ...
.venv/bin/python -m failure_analysis.reranking_v2.cli train-oof-primary ...
.venv/bin/python -m failure_analysis.reranking_v2.cli export-rankings ...
.venv/bin/python -m failure_analysis.reranking_v2.cli train-gate ...
.venv/bin/python -m failure_analysis.reranking_v2.cli extract-stability ...
.venv/bin/python -m failure_analysis.reranking_v2.cli prepare-vlm ...
.venv/bin/python -m failure_analysis.reranking_v2.cli run-vlm ...
.venv/bin/python -m failure_analysis.reranking_v2.cli evaluate-validation ...
.venv/bin/python -m failure_analysis.reranking_v2.cli lock-experiment ...
.venv/bin/python -m failure_analysis.reranking_v2.cli run-test ...
.venv/bin/python -m failure_analysis.reranking_v2.cli evaluate-test ...
.venv/bin/python -m failure_analysis.reranking_v2.cli build-report ...
```

The separate `failure_analysis.reranking_v2.independent_evaluator` recomputes
cohort counts, J@1, Oracle@5, recovered/harmful/net, switch coverage, and exact
McNemar p-values without calling the primary metric implementation.

## Integrity stops

Formal execution stops rather than silently dropping samples when any of the
following occurs:

- checkpoint/config/hash mismatch;
- candidate peak coordinate, order, value, or checksum mismatch;
- train/validation/test ID or content overlap;
- inference record contains evaluation-only fields;
- missing or duplicated sample;
- OOF checkpoint saw the held-out fold;
- Oracle@5 changes after re-ranking;
- incomplete crop/latent/stability artifact;
- formal test claim or frozen manifest already exists.

The V1 aggregate test result was known before V2. The final report must
therefore say that V2 is evaluated after aggregate test exposure, with all V2
choices locked on validation before the single formal V2 test run.
