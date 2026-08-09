# SAM 3 Proposal-Bank P@90 Audit

## Scope and chronology

This audit was written before implementation of the
`sam3_proposal_bank_p90_v1` experiment. The new experiment is additive and must
not overwrite the frozen repeated-FiLM HiFi-CS predictions, the previously
locked selective-SAM outputs, or their manifests. The 7,675-row split has
already informed earlier aggregate and failure analyses; it is therefore a
locked **post-hoc benchmark evaluation**, not a pristine confirmatory test.

## Protected authoritative inputs

| Input | Authoritative path | Verified identity |
|---|---|---|
| repeated-FiLM checkpoint | `runs/hifics_ocidvlg_hierfilm_20260727_214615/checkpoints/best.pth` | SHA-256 `b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601` |
| train manifest | `artifacts/data_audit/frozen_manifests/ocidvlg_unique_train.json` | 26,295 rows; SHA-256 `a986bcce3e1961be816a295c3ae0942e64e61275524a85c0a8957563e7f920c1` |
| validation manifest | `artifacts/data_audit/frozen_manifests/ocidvlg_unique_val.json` | 3,778 rows; SHA-256 `573c6ecd9ed9963eda525162279836b7649d163d83c57f164598604579b8b84a` |
| post-hoc benchmark manifest | `artifacts/data_audit/frozen_manifests/ocidvlg_unique_test.json` | 7,675 rows; SHA-256 `915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409` |
| compact repeated-FiLM inputs | `runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/` | per-row source and output checksums are embedded in each manifest |
| locked selective-SAM outputs | `outputs/selective_sam3_vg/formal_test_masks/` | output manifest SHA-256 `d104981b062a961af98f54c11b10491dfc60112e87ed64e1b633deca6da7a9f0` |
| locked selective-SAM canonical root | `runs/hifics_selective_sam3_LOCKED/` | 7,675 exported rows; preserved read-only by experiment policy |

The new checksum manifest will reference every protected per-sample checksum
already present in the compact and selective-SAM manifests and will independently
hash the controlling checkpoint, manifests, configuration, and lock files.

## Data and split facts

- Train/validation/post-hoc benchmark contain 26,295/3,778/7,675 expressions.
- They contain 1,104/165/325 scene-frame groups respectively.
- The prior split audit reports zero scene and processed-RGB overlap for every
  pair of splits. The new experiment additionally checks RGB SHA-256 and parsed
  frame-ID overlap.
- All expressions sharing one `scene_id`, source RGB checksum, or frame ID stay
  in one group. Row-wise random splitting is forbidden.
- No genuinely untouched official holdout has been established. Grouped nested
  cross-validation on validation scene-frames is therefore the primary
  development evidence unless a legitimate untouched group set is discovered.

## Baseline and evaluator contract

- Repeated-FiLM architecture:
  `models.hifics.HierarchicalCLIPDensePredT`, ViT-B/16, five hierarchical
  projection levels, `extended_film=true`, `hierarchical_film=true`.
- Stored foreground probability is `sigmoid(-background_logit)` at 352×352.
- The model mask is `probability >= 0.5`; the native 480×640 mask is a
  nearest-neighbour resize of that binary mask.
- Per-sample IoU is computed in float32 to match the authoritative evaluator.
- P@50 through P@90 use strict `IoU > threshold`, not `>=`.
- Authoritative post-hoc baseline: mIoU `0.8074137115168442`, P@90
  `3363/7675 = 0.43817589576547233`.
- Previously locked selective SAM 3: mIoU `0.8148962625524094`, P@90
  `3598/7675 = 0.46879478827361565`, mean boundary F-score
  `0.8391754785087902`.
- Previous limited validation oracle used one selected Tracker prompt family
  with four model hypotheses plus the HiFi fallback; it reached mIoU
  `0.862600505158432` and P@90 `2227/3778 = 0.5894653255690842`. It is not the
  expanded proposal-bank oracle requested here.

## Official SAM 3 runtime contract

- Official model ID: `facebook/sam3`.
- Pinned local revision: `3c879f39826c281e95690f02c7821c4de09afae7`.
- Local checkpoint: `models/huggingface/facebook-sam3/3c879f39826c281e95690f02c7821c4de09afae7/`.
- `model.safetensors` SHA-256:
  `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`.
- Environment: Python 3.12.12, PyTorch 2.13.0, Transformers 5.14.1,
  Apple M5 Pro, 24 GiB unified memory.
- Execution is CPU/float32 only, one RGB image per inference batch, inference
  mode, no CUDA, MPS, quantisation, automatic device mapping, or SAM
  fine-tuning. PCS reuses one cached image representation for a pilot-validated
  micro-batch of four independent text prompts; it never batches images.
- Installed public signatures expose `Sam3Model.get_vision_features()` and
  `forward(vision_embeds=...)`, plus
  `Sam3TrackerModel.get_image_embeddings()` and
  `forward(image_embeddings=..., input_points=..., input_boxes=...,
  input_masks=...)`.
- `AutoModelForMaskGeneration` maps `sam3_tracker` to the official Tracker
  model. The official mask-generation pipeline supplies a deterministic point
  grid, caches image embeddings within an image call, and filters by predicted
  IoU, stability, and NMS.

## Scientific information boundary

Prediction-time code may read only RGB, depth, intrinsics, query text,
query-derived deterministic semantics, frozen HiFi probability/mask, official
SAM 3 outputs, frozen image/text features, and candidate-to-candidate
relationships. It must fail closed on any attempt to read GT, answer instance,
test IoU, candidate correctness, threshold labels, or target annotations.

Ground truth is restricted to grouped training/development labels, oracle
diagnostics, and evaluation after outputs have been checksummed and marked
`LOCKED_BEFORE_GT_EVALUATION`.

## Query-source audit

The public OCID-VLG unique annotations expose template families (`name`,
`attribute`, `relation`, `location`) and a symbolic program. Those annotations
also contain answer and target-instance fields, so the locked inference parser
will not load them. Parser vocabulary and lexical rules are fixed from public
template/category/attribute terminology and demonstrated against train and
validation questions; inference receives query text only.

## Existing-code reuse decision

The experiment will reuse the verified repeated-FiLM compact-manifest adapter,
float32 evaluator, Tracker image-embedding cache pattern, atomic output helpers,
and leakage guards. New code is required for PCS text proposals, automatic
proposal provenance, deterministic query semantics, candidate deduplication,
relation/depth features, grouped P@90 selection, Stage-2 refinement, locking,
and the new report namespace.

## Protected output policy

All writes are confined to:

- `outputs/sam3_proposal_bank_p90_v1/`
- `artifacts/sam3_proposal_bank_p90_v1/`
- `configs/sam3_proposal_bank_p90_v1/`
- `runs/hifics_sam3_proposal_selector_LOCKED/` only after safeguards pass
- new source, test, script, and documentation files required by this experiment

Dex-Net, GQ-CNN, VGN, grasp candidate generation, and grasp re-ranking are out
of scope and must not be executed.

## External references inspected

- Hugging Face Transformers SAM 3 model documentation:
  <https://huggingface.co/docs/transformers/model_doc/sam3>
- Hugging Face Transformers SAM 3 Tracker documentation:
  <https://huggingface.co/docs/transformers/model_doc/sam3_tracker>
- Official Transformers SAM 3 source/documentation repository:
  <https://github.com/huggingface/transformers>
- Official HiFi-CS repository:
  <https://github.com/vineet2104/hifics>

No external implementation code is copied. Official documentation and installed
public APIs are used as behavioural references; repository-local verified code
is preferred for adaptation.
