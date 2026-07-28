# LAVT OCID-VLG implementation audit

## Source and scope

- Upstream: `yz93/LAVT-RIS`, commit
  `1da0af9f21b637c0cae9ea1363d2dd9b40e19628`, GPL-3.0.
- Primary model: `lavt_one`, Swin-Base window 12, BERT-base-uncased.
- Task boundary: RGB/text to a two-class segmentation mask only.
- Parameter count verified by real forward: 227,739,514.
- No depth, grasp, box, GT-guided inference, SAM, CROG, or HiFi prediction is
  consumed by the model.

## Upstream files changed

| File | Reason | Network computation changed? |
|---|---|---|
| `args.py` | Add config, OCID paths, device, loss, limits, accumulation, resume, and run arguments. | No |
| `train.py` | Remove unconditional CUDA/DDP assumptions so the original entry point remains device-safe. | No |
| `utils.py` | Select NCCL only for CUDA; make synchronization and logging single-process safe. | No |
| `lib/backbone.py` | Replace legacy dependency imports, make checkpoint semantics explicit, and use explicit `meshgrid(indexing="ij")`. | No |
| `lib/segmentation.py` | Thread activation-checkpoint configuration and clarify full-checkpoint construction logs. | No |

`lib/compat.py` replaces only `timm` tensor helpers and MMCV checkpoint loading
with current public PyTorch equivalents. LAVT's Swin blocks, PWAM fusion, BERT
encoder, decoder, tensor shapes, and parameter groups are retained.

## New implementation

- `data/dataset_ocid_vlg_bert.py`: dynamically instantiates the local official
  OCID-VLG API, validates a frozen manifest against its metadata, and reads only
  RGB, instance mask, text, object ID, and stable identifiers. Its grasp-aware
  `__getitem__` is deliberately bypassed because it fails when grasp masks are
  disabled.
- `scripts/discover_data.py` and `scripts/audit_ocid_vlg.py`: canonical-root
  selection, exact HiFi train/test alignment, source-resolution manifests,
  integrity/leakage/token checks, and deterministic visualizations.
- `ocid_vlg/device.py`, `losses.py`, `metrics.py`, `checkpoint.py`,
  `engine.py`: device selection, explicit Dice or weighted CE, dual-resolution
  metrics, strict `P@X` comparison (`IoU > threshold`), full-state atomic
  checkpoints, training/evaluation/export.
- `train_ocid_vlg.py`, `evaluate_ocid_vlg.py`,
  `compare_with_hifics.py`: auditable experiment entry points and fair-comparison
  guards.

## Training semantics

| Setting | Official LAVT | OCID-VLG primary |
|---|---|---|
| Network | LAVT-One + Swin-Base + BERT | Same |
| Image size | 480×480 | Same |
| Token limit | 20 | Same |
| Optimizer | AdamW, 5e-5, wd 0.01 | Same |
| Parameter groups | Swin decay/no-decay, decoder, BERT layers 0–9 | Same |
| Schedule | polynomial power 0.9, 40 epochs | Same |
| Loss | released code uses weighted CE; README reports Dice results | Dice primary; device-safe weighted CE retained |
| Runtime | multi-GPU CUDA/DDP | single-process MPS; CUDA/CPU supported |
| Batch | distributed training | batch 1, accumulation 8 in full config |
| Dataset | RefCOCO-family | OCID-VLG `unique` |

The local multi-class Dice loss averages foreground and background soft Dice,
uses epsilon `1e-6`, and is not represented as unreleased upstream source.
RGB uses bilinear antialiased resize; masks use nearest-neighbour resize and are
re-binarized. BERT tokenization follows the bundled upstream tokenizer and
20-token truncation.

## Pretrained initialization

- Swin checkpoint size: 450,809,979 bytes.
- SHA256:
  `70812ab6b0a7a38712409d13976df9431632466eaacf991d5e90d9a1e91f3ab1`.
- Checkpoint keys: 364; shape-compatible keys loaded into LAVT: 349.
- Missing: 64 LAVT-only fusion, residual-gate, and output-normalization keys
  absent from a classification checkpoint.
- Unexpected: 15 classification head/norm and precomputed attention-mask keys.
- Result: successful Swin classification-backbone initialization; this is not a
  random-backbone fallback.

## MPS and fallback status

`PYTORCH_ENABLE_MPS_FALLBACK=1` was set and recorded. The real 480×480
Swin-Base backward completed on MPS, so no Small/Tiny fallback backbone was
used. Activation checkpointing was enabled. CUDA AMP, SyncBatchNorm, DDP, and
CUDA cache/synchronization calls are not used on MPS.

## Data and HiFi-CS differences

The HiFi frozen train/test identities exactly equal the official OCID-VLG
`unique` train/test identities. The 3,778-item validation set is the official
`unique` validation split; HiFi did not freeze a separate validation manifest.
HiFi's processed 352×352 masks are not used as ground truth. Both evaluators
instead use the original 480×640 instance mask and object ID.

## Unresolved items

- The 40-epoch full training and 7,675-sample LAVT test export have not run.
- Consequently, the available full HiFi prediction set cannot yet be compared
  to an identically complete LAVT prediction set.
- Small/Tiny fallback configurations are templates only; no fallback run or
  matching fallback pretrained checkpoint was used.
