#!/usr/bin/env python3
"""Cache frozen dense CLIP semantics for all generated proposal candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.clip_candidate_features import FrozenClipCandidateEncoder  # noqa: E402
from src.segmentation.proposal_types import load_candidate_masks_npz  # noqa: E402
from src.segmentation.query_semantics import parse_query  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--proposal-stage", choices=("stage1", "stage2"), default="stage1"
    )
    parser.add_argument("--sample-list", type=Path)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1",
    )
    return parser.parse_args()


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    experiment = args.experiment_root.expanduser().resolve()
    compact_path = PROJECT_ROOT / (
        "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/"
        f"{args.split}/manifest.jsonl"
    )
    rows = [json.loads(line) for line in compact_path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {str(row["sample_id"]): row for row in rows}
    if args.sample_list:
        sample_ids = [value for value in args.sample_list.read_text(encoding="utf-8").splitlines() if value]
    else:
        bank_name = "proposals" if args.proposal_stage == "stage1" else "stage2"
        sample_ids = list(by_id)
        missing = [
            sample_id
            for sample_id in sample_ids
            if not (experiment / bank_name / sample_id / "terminal_status.json").is_file()
        ]
        if missing and args.sample_limit is None:
            raise RuntimeError(
                f"CLIP extraction requires the complete {args.proposal_stage} "
                f"{args.split} bank; missing {len(missing)} samples"
            )
        missing_set = set(missing)
        sample_ids = [
            sample_id for sample_id in sample_ids if sample_id not in missing_set
        ]
    if args.sample_limit is not None:
        sample_ids = sample_ids[: int(args.sample_limit)]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    namespace = (
        "dense_candidates"
        if args.proposal_stage == "stage1"
        else "dense_candidates_stage2"
    )
    manifest_root = experiment / "cache/clip_embeddings"
    canonical_manifest = (
        manifest_root / f"manifest_{args.proposal_stage}_{args.split}.json"
    )
    if args.rebuild_manifest:
        if args.num_shards != 1 or args.shard_index != 0:
            raise ValueError("manifest rebuilding must run without sharding")
        shard_paths = sorted(
            manifest_root.glob(
                f"manifest_{args.proposal_stage}_{args.split}.shard-*-of-*.json"
            )
        )
        if not shard_paths:
            raise FileNotFoundError("no CLIP shard manifests were found")
        records: dict[str, dict] = {}
        for shard_path in shard_paths:
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            if shard.get("status") != "COMPLETE":
                raise ValueError(f"incomplete CLIP shard manifest: {shard_path}")
            for record in shard["records"]:
                sample_id = str(record["sample_id"])
                if sample_id in records:
                    raise ValueError(f"duplicate CLIP shard sample: {sample_id}")
                records[sample_id] = record
        if set(records) != set(sample_ids):
            raise ValueError(
                f"CLIP shards cover {len(records)}/{len(sample_ids)} selected samples"
            )
        ordered_records = []
        for sample_id in sample_ids:
            record = records[sample_id]
            cache_path = manifest_root / namespace / f"{sample_id}.npz"
            if not cache_path.is_file():
                raise FileNotFoundError(f"missing CLIP candidate cache: {cache_path}")
            archive = np.load(cache_path, allow_pickle=False)
            try:
                candidate_count = len(archive["candidate_id"])
            finally:
                archive.close()
            if candidate_count != int(record["candidate_count"]):
                raise ValueError(f"CLIP candidate-count drift for {sample_id}")
            ordered_records.append(record)
        manifest = {
            "status": "COMPLETE",
            "split": args.split,
            "proposal_stage": args.proposal_stage,
            "samples": len(ordered_records),
            "candidate_rows": int(
                sum(row["candidate_count"] for row in ordered_records)
            ),
            "runtime_seconds": float(
                sum(row["runtime_seconds"] for row in ordered_records)
            ),
            "backend": "frozen OpenAI CLIP ViT-B/16 dense 14x14 mask/box pooling",
            "fine_tuned": False,
            "records": ordered_records,
            "rebuilt_from_shards": [path.name for path in shard_paths],
        }
        _write_manifest(canonical_manifest, manifest)
        print(
            json.dumps(
                {key: value for key, value in manifest.items() if key != "records"},
                indent=2,
            )
        )
        return 0
    if args.num_shards > 1:
        frame_indices: dict[str, int] = {}
        for sample_id in sample_ids:
            scene_id = str(by_id[sample_id]["scene_id"])
            frame_indices.setdefault(scene_id, len(frame_indices))
        sample_ids = [
            sample_id
            for sample_id in sample_ids
            if frame_indices[str(by_id[sample_id]["scene_id"])] % args.num_shards
            == args.shard_index
        ]
        if not sample_ids:
            raise ValueError("selected CLIP extraction shard is empty")
    encoder = FrozenClipCandidateEncoder(
        experiment / "cache/clip_embeddings",
        batch_size=int(args.batch_size),
        num_threads=int(args.num_threads),
    )
    runtimes = []
    for number, sample_id in enumerate(sample_ids, start=1):
        started = time.perf_counter()
        row = by_id[sample_id]
        bank_name = "proposals" if args.proposal_stage == "stage1" else "stage2"
        proposal = experiment / bank_name / sample_id
        masks = load_candidate_masks_npz(proposal / "candidate_masks.npz")
        candidate_ids = sorted(masks)
        rgb = np.asarray(Image.open(row["source_rgb_path"]).convert("RGB"), dtype=np.uint8)
        semantics = parse_query(str(row["query"]))
        features = encoder.dense_candidate_features(
            rgb,
            str(row["source_rgb_sha256"]),
            masks,
            candidate_ids,
            full_query=semantics.query,
            target_category=semantics.target_category_prompt,
            target_attribute=semantics.target_attribute_prompt,
            sample_id=sample_id,
            namespace=namespace,
        )
        runtimes.append(
            {
                "sample_id": sample_id,
                "candidate_count": len(features["candidate_id"]),
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        if number % 25 == 0:
            print(f"CLIP features: {number}/{len(sample_ids)}", flush=True)
    manifest = {
        "status": "COMPLETE",
        "split": args.split,
        "proposal_stage": args.proposal_stage,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "samples": len(runtimes),
        "candidate_rows": int(sum(row["candidate_count"] for row in runtimes)),
        "runtime_seconds": float(sum(row["runtime_seconds"] for row in runtimes)),
        "backend": "frozen OpenAI CLIP ViT-B/16 dense 14x14 mask/box pooling",
        "fine_tuned": False,
        "records": runtimes,
    }
    path = (
        manifest_root
        / (
            f"manifest_{args.proposal_stage}_{args.split}."
            f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
        )
        if args.num_shards > 1
        else canonical_manifest
    )
    _write_manifest(path, manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
