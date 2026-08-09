#!/usr/bin/env python3
"""Build GT-free 4-DoF deployment Parquets and separate label Parquets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.data_contract import (  # noqa: E402
    SPLITS,
    build_split_records,
    write_parquet_bundle,
)


DEFAULT_COMPACT_ROOT = (
    PROJECT_ROOT
    / "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs"
)
DEFAULT_SOURCE_RUN = PROJECT_ROOT / "runs/hifics_ocidvlg_hierfilm_20260727_214615"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--frozen-manifests-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/data_audit/frozen_manifests",
    )
    parser.add_argument(
        "--compact-root", type=Path, default=DEFAULT_COMPACT_ROOT
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=PROJECT_ROOT / "OCID-VLG/refer/unique",
    )
    parser.add_argument(
        "--hifics-root", type=Path, default=PROJECT_ROOT / "hifics"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_RUN / "checkpoints/best.pth",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_SOURCE_RUN / "config.yaml"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    manifests_dir = run_dir / "manifests"
    if manifests_dir.exists() and manifests_dir.is_symlink():
        raise ValueError(f"refusing symlink manifests directory: {manifests_dir}")

    deployment_by_split = {}
    labels_by_split = {}
    hash_cache = {}
    for split in SPLITS:
        deployment, labels = build_split_records(
            split=split,
            frozen_manifest_path=(
                args.frozen_manifests_dir / f"ocidvlg_unique_{split}.json"
            ),
            compact_manifest_path=args.compact_root / split / "manifest.jsonl",
            annotations_path=args.annotations_root / f"{split}_expressions.json",
            hifics_root=args.hifics_root,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            hash_cache=hash_cache,
        )
        deployment_by_split[split] = deployment
        labels_by_split[split] = labels
    outputs = write_parquet_bundle(
        output_dir=manifests_dir,
        deployment_by_split=deployment_by_split,
        labels_by_split=labels_by_split,
    )
    print(
        json.dumps(
            {
                "status": "MANIFESTS_COMPLETE",
                "row_counts": {
                    split: len(deployment_by_split[split]) for split in SPLITS
                },
                "outputs": outputs,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
