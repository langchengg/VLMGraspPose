# Repeated-FiLM source audit

Audit time: 2026-07-29T19:44:06Z.  
Decision: `COMPLETE_AND_REUSABLE`.

This experiment uses only the retained hierarchical repeated-FiLM source run:

`runs/hifics_ocidvlg_hierfilm_20260727_214615`

The deleted single-FiLM lineage is intentionally excluded. Its checkpoint,
predictions, candidates, metrics and historical comparisons are not recovered,
regenerated, mixed, or used as a baseline.

## Checkpoint and strict load

- `checkpoints/best.pth` exists, is 24,075,091 bytes, and has SHA-256
  `b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601`.
- Format: `hifics_hierfilm_trainable_only_v1`.
- Best step/epoch: 19,728 / 12.
- Recorded best validation mean IoU: 0.8207412727974825.
- Its 92 trainable keys exactly match the instantiated model: 92 loaded,
  zero missing, zero unexpected.
- The independently recomputed trainable-state digest is
  `378134d0fa668cd24f4ea89d3d274dd909ea0b1c792497a35502728cfb89b657`.
- The exact frozen CLIP dependency exists at `~/.cache/clip/ViT-B-16.pt`
  with SHA-256
  `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.

Strict loading was performed with `hifics/.venv/bin/python` and PyTorch 2.12.1.
The general Anaconda interpreter is not the source-of-truth environment because
it lacks the required `clip` package.

## Architecture identity

The frozen source snapshot defines
`models.hifics.HierarchicalCLIPDensePredT`. It requires visual layers
`[1,3,5,7,9]`, reverses them to decoder order `[9,7,5,3,1]`, and constructs five
independent projection, FiLM and decoder stages.

Each level executes:

```text
visual projection -> language-conditioned FiLM -> decoder block
```

The runtime trace is exactly:

```text
film_0, decoder_0, film_1, decoder_1, ... film_4, decoder_4
```

The checkpoint contains tensors for five `film_stages`, five `reduces`, five
decoder `blocks`, and the output transposed convolution. It contains no legacy
`film_mul`, `film_add`, or single shared `reduce` state. The instantiated model
also lacks these old single-FiLM attributes.

Training started fresh: `resume=false`, no checkpoint was supplied, automatic
latest-checkpoint loading was disabled, and the recorded weight-source manifest
says the decoder, FiLM and head were deterministically initialized rather than
loaded from the deleted lineage.

## Data protocol

The official OCID-VLG unique manifests were independently counted and hashed:

| Split | Samples | SHA-256 |
|---|---:|---|
| train | 26,295 | `a986bcce3e1961be816a295c3ae0942e64e61275524a85c0a8957563e7f920c1` |
| val | 3,778 | `573c6ecd9ed9963eda525162279836b7649d163d83c57f164598604579b8b84a` |
| test | 7,675 | `915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409` |

Manifest-set SHA-256:
`fa36db4ff548f2ca06abadefb30cacc741165478dff8691203375a311b86c0c8`.
The independent split audit reports zero overlap in sample ID, scene-frame ID,
RGB SHA-256, depth SHA-256, and RGB-D-pair SHA-256 between every split pair.
The broader OCID sequence paths are not disjoint: train/validation share 23
sequence paths (and each split pair shares sequence paths). This is a
diagnostic grouping fact rather than frame leakage; the new internal
development/calibration partition is therefore grouped by sequence/scene
instead of randomly splitting expression rows.

## Provenance and caveat

The frozen source snapshot aggregate SHA-256 is
`4e05df9f909e966e4531d9c5e36d6582e077d61400ff2d574c4114ccc2584f84`.
It is the source of truth because the original training run recorded a dirty
working tree.

`REPORTS_COMPLETE.json` references an old comparison CSV that was removed in
the authorized cleanup. That file is documentation-only and is not required to
load the checkpoint or reproduce repeated-FiLM inference.

The machine-readable record is:

`runs/modular_reranking_repeatedfilm_v1_20260729_203147/manifests/repeatedfilm_source_manifest.json`
