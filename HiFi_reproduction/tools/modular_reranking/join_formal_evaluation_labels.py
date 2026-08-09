#!/usr/bin/env python3
"""Join GT labels only after the locked formal inference has completed."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.features import (  # noqa: E402
    FORBIDDEN_GT_COLUMNS,
    join_candidate_labels,
)
from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.experiment_lock import (  # noqa: E402
    verify_completed_formal_stage,
)
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
    validate_artifact_identity,
    validate_matching_artifact_identity,
)


def assert_locked(lock: dict, path: Path) -> None:
    resolved = path.expanduser().resolve()
    matches = [
        item
        for item in lock["artifacts"].values()
        if Path(item["path"]).resolve() == resolved
    ]
    if len(matches) != 1 or matches[0]["sha256"] != sha256_file(resolved):
        raise ValueError(f"evaluation artifact absent from or changed since lock: {resolved}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-per-candidate", type=Path, required=True)
    parser.add_argument("--labels-parquet", type=Path, required=True)
    parser.add_argument("--formal-inference-manifest", type=Path, required=True)
    parser.add_argument("--experiment-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    inference_path = args.inference_per_candidate.expanduser().resolve()
    labels_path = args.labels_parquet.expanduser().resolve()
    formal_manifest_path = args.formal_inference_manifest.expanduser().resolve()
    lock_path = args.experiment_lock.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite evaluation join: {output_root}")
    completed = verify_completed_formal_stage(
        lock_path,
        stage="TEST",
        manifest_path=formal_manifest_path,
    )
    lock = completed["lock"]
    assert_locked(lock, inference_path)
    assert_locked(lock, labels_path)
    formal = completed["manifest"]
    validate_artifact_identity(
        formal, context="formal reranker inference manifest"
    )
    validate_matching_artifact_identity(
        formal,
        lock,
        context="formal reranker inference manifest",
    )
    if formal.get("lock_content_sha256") != lock["manifest_content_sha256"]:
        raise ValueError("formal inference manifest belongs to another experiment lock")
    if Path(formal["per_candidate"]).resolve() != inference_path:
        raise ValueError("formal inference manifest references another candidate table")
    if formal["per_candidate_sha256"] != sha256_file(inference_path):
        raise ValueError("formal inference candidate table hash changed")
    predictions_path = Path(formal["predictions"])
    if (
        not predictions_path.is_file()
        or sha256_file(predictions_path) != formal["predictions_sha256"]
    ):
        raise ValueError("formal prediction artifact is missing or changed")

    inference = pd.read_parquet(inference_path)
    forbidden_present = sorted(set(inference.columns) & set(FORBIDDEN_GT_COLUMNS))
    if forbidden_present:
        raise ValueError(
            "formal inference table already contains GT columns: "
            f"{forbidden_present}"
        )
    labels = pd.read_parquet(labels_path)
    labels = labels.loc[
        labels["sample_id"].astype(str).isin(
            set(inference["sample_id"].astype(str))
        )
    ].copy()
    joined = pd.DataFrame(
        join_candidate_labels(
            inference.to_dict("records"),
            labels.to_dict("records"),
        )
    )
    identity_columns = [
        "sample_id",
        "candidate_id",
        "candidate_identity_sha256",
    ]
    if not joined.loc[:, identity_columns].equals(
        inference.loc[:, identity_columns]
    ):
        raise AssertionError("candidate identity changed during GT label join")
    if "candidate_positive" not in joined.columns:
        raise ValueError("candidate_positive was not joined")
    output_root.mkdir(parents=True)
    output_path = output_root / "per_candidate_evaluation.parquet"
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    joined.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output_path)
    manifest = {
        "schema_version": 1,
        **identity_payload(),
        "post_formal_inference_gt_join": True,
        "experiment_lock": str(lock_path),
        "lock_content_sha256": lock["manifest_content_sha256"],
        "formal_inference_manifest": str(formal_manifest_path),
        "formal_inference_manifest_sha256": sha256_file(formal_manifest_path),
        "inference_per_candidate": str(inference_path),
        "inference_per_candidate_sha256": sha256_file(inference_path),
        "labels_parquet": str(labels_path),
        "labels_parquet_sha256": sha256_file(labels_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "candidate_count": len(joined),
        "candidate_identity_invariant": True,
        "formal_predictions_unchanged": True,
    }
    (output_root / "join_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
