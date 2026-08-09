#!/usr/bin/env python3
"""Hash and verify the official unique train/val/test split identities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.split_audit import build_split_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hifics-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--hifi-checkpoint", type=Path, required=True)
    parser.add_argument("--dexnet-config", type=Path, required=True)
    parser.add_argument("--gqcnn-verification-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    gqcnn = json.loads(
        args.gqcnn_verification_report.read_text(encoding="utf-8")
    )
    model_hash = (
        gqcnn.get("model_file_manifest_hash")
        or gqcnn.get("model", {}).get("model_file_manifest_hash")
        or gqcnn.get("expected_identity", {}).get("model_file_manifest_hash")
    )
    if not model_hash:
        raise ValueError("GQ-CNN verification report has no model manifest hash")
    provenance = {
        "hifi_checkpoint_path": str(args.hifi_checkpoint.resolve()),
        "hifi_checkpoint_sha256": sha256_file(args.hifi_checkpoint),
        "dexnet_config_path": str(args.dexnet_config.resolve()),
        "dexnet_config_sha256": sha256_file(args.dexnet_config),
        "gqcnn_verification_report_path": str(
            args.gqcnn_verification_report.resolve()
        ),
        "gqcnn_verification_report_sha256": sha256_file(
            args.gqcnn_verification_report
        ),
        "gqcnn_model_manifest_sha256": str(model_hash),
    }
    summary, frame = build_split_audit(
        manifest_paths={
            "train": args.train_manifest,
            "val": args.val_manifest,
            "test": args.test_manifest,
        },
        hifics_root=args.hifics_root.expanduser().resolve(),
        provenance=provenance,
        workers=args.workers,
    )
    (output / "split_audit.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    frame.to_parquet(
        output / "split_sample_manifest.parquet",
        index=False,
        compression="zstd",
    )
    (output / "run_command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
