# Repeated-FiLM compact-storage audit

Status: active run in progress.  
Run: `runs/modular_reranking_repeatedfilm_v1_20260729_203147`.

## Budgets and deletion boundary

- persistent reranking artifacts: target at most 8 GiB, excluding local VLM
  weights and the retained read-only source runs;
- regenerated train/validation pipeline: target at most 18 GiB;
- ordinary VLM inputs are on-demand recipes, not persistent PNG trees;
- only artifacts below this run's `tmp/` may be deleted automatically, and
  only after hashes, row counts, candidate identity, downstream feature
  consumption, and final compact publication have all passed;
- anything outside this run's `tmp/` receives a dry-run deletion plan only.

The source repeated-FiLM checkpoint and the retained test pipeline are
read-only and excluded from the new-artifact budget.

## Persistent compact inputs

The hierarchical repeated-FiLM train and validation predictions are stored as
lossless float32 compressed probability arrays. Repeated RGB, depth and
intrinsics are referenced rather than copied per expression.

| Stage | Samples | Allocated bytes | Verified |
|---|---:|---:|---|
| train HiFi compact predictions | 26,295 | 12,213,858,304 | yes |
| validation HiFi compact predictions | 3,778 | 1,755,312,128 | yes |
| train GT-free query metadata | 26,295 | 5,701,632 | yes |
| validation GT-free query metadata | 3,778 | 815,104 | yes |

The source probabilities remain float32: no float16 or other lossy
quantization was introduced.

## Streaming candidate/scoring design

Dex-Net generation is partitioned by
`int(sha256(scene_id), 16) mod 8`. This keeps every referring expression for a
scene in one shard. Each completed shard records:

- scorer-compatible candidate geometry;
- raw, mask-validated and NMS ZSTD Parquet tables;
- separate GT-only labels;
- protocol/config/checkpoint hashes;
- zero-failure accounting.

To avoid simultaneously retaining a full verbose candidate tree and a full
verbose GQ-CNN tree, each completed scene shard follows this order:

1. GQ-CNN score and independently verify the immutable candidate IDs;
2. compact the score vector to a temporary scene-shard Parquet;
3. extract inference features, joining labels only after the GT-free feature
   hash is frozen;
4. verify the compact score and feature shard;
5. publish only canonical merged candidate/score/feature Parquets;
6. remove consumed verbose data only from this run's `tmp/`.

The full test candidate/GQ-CNN source remains referenced read-only and is not
copied into the new run.

## Low-peak train-shard merge transaction

The train merge must not call the ordinary all-at-once candidate merge while
the eight source stage tables are still present.  The implemented compatible
path keeps the final contract as one Parquet per artifact, but handles
candidate stages in this order:

1. preflight every frozen shard config, compact-table SHA-256, row count,
   schema, deterministic scene partition and completed verbose-prune receipt;
2. estimate the next stage peak against the 18 GiB run budget (current
   allocated bytes plus 110% of source logical bytes plus 64 MiB headroom);
3. stream one stage (`raw`, then `mask_validated`, then `nms`) through
   `ParquetWriter`;
4. close and `fsync` the output, then independently re-read it and prove exact
   row count, schema, ZSTD compression, contiguous per-sample PK uniqueness
   and ordered `(sample_id, candidate_id)` plus full logical-row digest
   equality;
5. durably journal output SHA-256, PK digest, source hashes and protocol
   lineage below `manifests/low_peak_merge/`;
6. only with explicit `--execute`, unlink the exact registered source-stage
   Parquet files one at a time.  Keep each source `run_config.json`,
   `run_manifest.jsonl`, `summary.csv`, `funnel_labels.jsonl`, cleanup receipt
   and all oracle-audit metadata.

The new path defaults to a zero-write dry-run:

```bash
python tools/modular_reranking/merge_compact_candidate_shards.py \
  --prediction-manifest <train_prediction_manifest.jsonl> \
  --output-root <run/tmp/train/candidates/compact_merged> \
  --tmp-root <run/tmp> \
  --compact-only \
  --low-peak-release-inputs \
  --shard-root <shard_0> ... --shard-root <shard_7>
```

After checking the printed plan and budget, rerun the exact command with
`--execute`.  If interrupted, rerun the exact same command; any path,
budget, config hash, source hash, checkpoint/protocol lineage or output
identity drift is rejected.

GQ-CNN and feature merges are much smaller, but their merge tools now apply
the same 18 GiB conservative preflight.  After the final merged output is
verified, use the default-dry-run release command:

```bash
python tools/modular_reranking/release_compact_shard_parquets.py \
  --family gqcnn \
  --output <run/compact_inputs/train/gqcnn_scores.parquet> \
  --tmp-root <run/tmp> \
  --shard <run/tmp/train/gqcnn_score_shards/shard_0.parquet> ...

python tools/modular_reranking/release_compact_shard_parquets.py \
  --family features \
  --output <run/features/train> \
  --tmp-root <run/tmp> \
  --shard <run/tmp/train/features_scene/shard_0> ...
```

Add `--execute` only after the dry-run evidence is accepted.  These releases
verify exact source/output PK sets, rows, SHA-256 and manifest lineage, then
unlink only `*.parquet` inputs below the active run's `tmp/`.  They never
delete directories or the shard manifests needed for later provenance audit.

Implementation reference: Apache Arrow documents
[`ParquetWriter`](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html)
as an incremental row-group writer and
[`ParquetFile.iter_batches`](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html)
as a streaming reader.  Arrow does not establish application PK ordering or
uniqueness, so the transaction performs those checks explicitly.  The
official interfaces were used as reference only; no external code was copied.

## Local VLM storage

The exact system prompt and JSON schema are frozen in
`runs/modular_reranking_repeatedfilm_v1_20260729_203147/local_vlm/contract`.
Full validation/formal input preparation stores a GT-free, source-hashed
visualization recipe. At inference time one sample is rendered into a unique
directory below the current run's `tmp/`; its image hashes, request hash,
parser result and SQLite cache record are made durable before ordinary PNGs
are removed. Only selected paper/gallery cases persist as PNG.

## Live high-water evidence

At the first completed train candidate shard, the complete run occupied
16,590,452 KiB of allocated filesystem space and 245,863,984 KiB remained
available. This includes 13,641,768 KiB of train/validation compact
predictions plus active temporary shards. The final high-water mark and
per-stage rows will be taken from `storage_usage_by_stage.csv` after all
downstream verification and authorized `tmp/` cleanup.
