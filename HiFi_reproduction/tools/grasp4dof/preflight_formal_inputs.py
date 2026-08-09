#!/usr/bin/env python3
"""Hash every formal data reference and the two direct R0 source tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


DEPLOYMENT_REFERENCES = (
    ("source_rgb", "source_rgb_path", "source_rgb_sha256"),
    ("source_depth", "source_depth_path", "source_depth_sha256"),
    ("predicted_mask", "predicted_mask_path", "predicted_mask_sha256"),
    (
        "predicted_probability",
        "predicted_probability_path",
        "predicted_probability_sha256",
    ),
    ("repeatedfilm_checkpoint", "checkpoint_path", "checkpoint_sha256"),
    ("repeatedfilm_config", "config_path", "config_sha256"),
    ("frozen_source_manifest", "frozen_manifest_path", "frozen_manifest_sha256"),
    ("compact_source_manifest", "compact_manifest_path", "compact_manifest_sha256"),
)
LABEL_REFERENCES = (
    ("prepared_gt_mask", "prepared_gt_mask_path", "prepared_gt_mask_sha256"),
    (
        "official_annotations",
        "official_annotations_path",
        "official_annotations_sha256",
    ),
)
EXPECTED_COUNTS = {"train": 26_295, "validation": 3_778, "test": 7_675}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify(
    path_value: Any,
    expected_value: Any,
    *,
    cache: dict[str, str],
    label: str,
) -> tuple[str, str]:
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty {label}: {path}")
    expected = str(expected_value).lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"invalid declared SHA-256 for {label}")
    observed = cache.get(str(path))
    if observed is None:
        observed = _sha256(path)
        cache[str(path)] = observed
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}: {path}"
        )
    return str(path), observed


def verify_split(
    samples_path: Path, labels_path: Path, *, split: str, cache: dict[str, str]
) -> dict[str, Any]:
    samples = pd.read_parquet(samples_path)
    labels = pd.read_parquet(labels_path)
    expected_count = EXPECTED_COUNTS[split]
    if len(samples) != expected_count or len(labels) != expected_count:
        raise ValueError(f"{split}: manifest count mismatch")
    if samples.sample_id.duplicated().any() or labels.sample_id.duplicated().any():
        raise ValueError(f"{split}: duplicate sample identity")
    if set(samples.sample_id.astype(str)) != set(labels.sample_id.astype(str)):
        raise ValueError(f"{split}: deployment/label coverage mismatch")
    label_by_id = labels.set_index(labels.sample_id.astype(str), drop=False)
    declarations: list[dict[str, str]] = []
    for sample in samples.itertuples(index=False):
        sample_id = str(sample.sample_id)
        language = str(sample.language)
        language_sha = hashlib.sha256(language.encode("utf-8")).hexdigest()
        if language_sha != str(sample.language_sha256):
            raise ValueError(f"{split}/{sample_id}: language SHA-256 mismatch")
        sample_map = sample._asdict()
        for logical_name, path_field, sha_field in DEPLOYMENT_REFERENCES:
            path, digest = _verify(
                sample_map[path_field],
                sample_map[sha_field],
                cache=cache,
                label=f"{split}/{sample_id}/{logical_name}",
            )
            declarations.append(
                {"sample_id": sample_id, "field": logical_name, "path": path, "sha256": digest}
            )
        intrinsics_path = sample_map.get("intrinsics_path")
        if intrinsics_path is not None and bool(pd.notna(intrinsics_path)):
            path, digest = _verify(
                sample_map["intrinsics_path"],
                sample_map["intrinsics_sha256"],
                cache=cache,
                label=f"{split}/{sample_id}/intrinsics",
            )
            declarations.append(
                {"sample_id": sample_id, "field": "intrinsics", "path": path, "sha256": digest}
            )
        provenance = json.loads(str(sample_map["intrinsics_provenance"]))
        if provenance.get("kind") == "derived_from_organized_pcd":
            path, digest = _verify(
                provenance["source_pcd_path"],
                provenance["source_pcd_sha256"],
                cache=cache,
                label=f"{split}/{sample_id}/intrinsics_pcd",
            )
            declarations.append(
                {"sample_id": sample_id, "field": "intrinsics_pcd", "path": path, "sha256": digest}
            )
        label = label_by_id.loc[sample_id].to_dict()
        for logical_name, path_field, sha_field in LABEL_REFERENCES:
            path, digest = _verify(
                label[path_field],
                label[sha_field],
                cache=cache,
                label=f"{split}/{sample_id}/{logical_name}",
            )
            declarations.append(
                {"sample_id": sample_id, "field": logical_name, "path": path, "sha256": digest}
            )
    return {
        "sample_count": expected_count,
        "samples_manifest": str(samples_path),
        "samples_manifest_sha256": _sha256(samples_path),
        "labels_manifest": str(labels_path),
        "labels_manifest_sha256": _sha256(labels_path),
        "reference_declaration_count": len(declarations),
        "reference_set_sha256": _canonical_sha(declarations),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    if (run / "manifests/experiment_lock.json").exists():
        raise RuntimeError("formal input preflight is forbidden after lock")
    output = run / "audit/formal_input_reference_preflight.json"
    if output.exists():
        raise FileExistsError(output)
    cache: dict[str, str] = {}
    split_records = {
        split: verify_split(
            run / f"manifests/{split}_samples.parquet",
            run / f"manifests/{split}_labels.parquet",
            split=split,
            cache=cache,
        )
        for split in EXPECTED_COUNTS
    }
    reference_inventory_path = run / "audit/reference_baseline_inventory.json"
    inventory = json.loads(reference_inventory_path.read_text(encoding="utf-8"))
    if inventory.get("reference_run_reusable") is not True:
        raise ValueError("reference run is not approved by the source audit")
    reference = Path(str(inventory["reference_modular_run"])).expanduser().resolve()
    direct_paths = {
        "per_candidate": reference / "evaluation/hierfilm_gqcnn_per_candidate.parquet",
        "per_sample": reference / "evaluation/hierfilm_per_sample_pipeline_metrics.csv",
    }
    direct = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _verify(path, _sha256(path), cache=cache, label=f"R0/{name}")[1],
        }
        for name, path in direct_paths.items()
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "checked_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": run.name,
        "split_references": split_records,
        "unique_referenced_file_count": len(cache),
        "loader_contract": "row_declared_sha256_verified_fail_closed",
        "reference_inventory": {
            "path": str(reference_inventory_path),
            "sha256": _sha256(reference_inventory_path),
        },
        "r0": {
            "reference_run": str(reference),
            "sample_count": int(inventory["sample_count"]),
            "direct_inputs": direct,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
