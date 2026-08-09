#!/usr/bin/env python3
"""Release verified GQ-CNN or feature shard Parquets after their final merge.

This is the second half of the low-peak merge path for the comparatively
small GQ-CNN and feature tables.  The existing merge tools first create their
single-file canonical outputs.  This tool then proves exact row/primary-key,
hash, and manifest-lineage equivalence.  It defaults to a zero-write dry-run;
``--execute`` durably journals the proof before unlinking only the registered
input Parquet files below the current active run's ``tmp/`` directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.modular_reranking import low_peak_transaction as low_peak  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("gqcnn", "features"), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Merged GQ-CNN Parquet for family=gqcnn, or merged feature "
            "directory for family=features."
        ),
    )
    parser.add_argument(
        "--shard",
        type=Path,
        action="append",
        required=True,
        help=(
            "Input score Parquet for family=gqcnn, or input feature directory "
            "for family=features."
        ),
    )
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--max-run-bytes",
        type=int,
        default=low_peak.DEFAULT_MAX_RUN_BYTES,
        help="Recorded run budget; release remains allowed when cleanup lowers it.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def gqcnn_manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def exact_source_entry(
    entries: Sequence[Mapping[str, Any]],
    *,
    key: str,
    path: Path,
) -> Mapping[str, Any]:
    matches = [
        entry
        for entry in entries
        if Path(str(entry.get(key, ""))).resolve() == path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError(f"merged manifest has no unique lineage for {path}")
    return matches[0]


def inspect_gqcnn(
    *,
    output: Path,
    shard_paths: list[Path],
    prior_stage: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    output_manifest_path = gqcnn_manifest_path(output)
    output_manifest = read_json(output_manifest_path)
    if (
        output_manifest.get("status") != "COMPLETED"
        or output_manifest.get("primary_key")
        != ["sample_id", "candidate_id"]
        or Path(
            str(output_manifest.get("gqcnn_scores_parquet", ""))
        ).resolve()
        != output
        or output_manifest.get("gqcnn_scores_parquet_sha256")
        != low_peak.sha256_file(output)
    ):
        raise ValueError("merged GQ-CNN output identity is invalid")
    source_entries = output_manifest.get("source_shards")
    if (
        not isinstance(source_entries, list)
        or len(source_entries) != len(shard_paths)
    ):
        raise ValueError("merged GQ-CNN manifest has no source_shards")

    inputs: list[dict[str, Any]] = []
    releases: list[tuple[Path, str]] = []
    all_sources_exist = True
    for shard_path in shard_paths:
        manifest_path = gqcnn_manifest_path(shard_path)
        manifest = read_json(manifest_path)
        digest = str(manifest.get("gqcnn_scores_parquet_sha256", ""))
        entry = exact_source_entry(
            source_entries, key="score_path", path=shard_path
        )
        if (
            Path(str(manifest.get("gqcnn_scores_parquet", ""))).resolve()
            != shard_path
            or entry.get("score_sha256") != digest
            or Path(str(entry.get("manifest_path", ""))).resolve()
            != manifest_path
            or entry.get("manifest_sha256")
            != low_peak.sha256_file(manifest_path)
            or int(entry.get("rows", -1)) != int(manifest.get("rows", -2))
        ):
            raise ValueError(f"GQ-CNN shard lineage differs: {shard_path}")
        if shard_path.exists():
            if shard_path.is_symlink() or low_peak.sha256_file(
                shard_path
            ) != digest:
                raise ValueError(f"GQ-CNN shard changed: {shard_path}")
        else:
            all_sources_exist = False
        inputs.append(
            {
                "path": str(shard_path),
                "sha256": digest,
                "rows": int(manifest["rows"]),
                "manifest_path": str(manifest_path),
                "manifest_sha256": low_peak.sha256_file(manifest_path),
                "candidate_protocol_identity_sha256": manifest.get(
                    "candidate_protocol_identity_sha256"
                ),
            }
        )
        releases.append((shard_path, digest))

    output_contract = low_peak.parquet_key_set_contract(
        [output],
        key_columns=("sample_id", "candidate_id"),
        include_all_columns=True,
    )
    if int(output_contract["rows"]) != int(output_manifest.get("rows", -1)):
        raise ValueError("merged GQ-CNN output row count differs")
    if all_sources_exist:
        source_contract = low_peak.parquet_key_set_contract(
            shard_paths,
            key_columns=("sample_id", "candidate_id"),
            include_all_columns=True,
        )
    elif prior_stage is not None:
        source_contract = prior_stage.get("source_key_contract")
    else:
        raise FileNotFoundError(
            "a GQ-CNN shard is missing without a matching resume journal"
        )
    if (
        source_contract != output_contract
        or sum(int(item["rows"]) for item in inputs)
        != int(output_contract["rows"])
    ):
        raise ValueError("GQ-CNN source/output key universe differs")
    return (
        {
            "inputs": inputs,
            "outputs": [
                {
                    "path": str(output),
                    "sha256": low_peak.sha256_file(output),
                    "rows": int(output_contract["rows"]),
                },
                {
                    "path": str(output_manifest_path),
                    "sha256": low_peak.sha256_file(output_manifest_path),
                },
            ],
            "source_key_contract": source_contract,
            "output_key_contract": output_contract,
            "lineage_sha256": low_peak.canonical_json_sha256(
                output_manifest["source_shards"]
            ),
        },
        releases,
    )


def inspect_features(
    *,
    output_root: Path,
    shard_roots: list[Path],
    prior_stage: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    output_manifest_path = output_root / "dataset_manifest.json"
    output_manifest = read_json(output_manifest_path)
    output_candidate = output_root / "per_candidate.parquet"
    output_sample = output_root / "per_sample.parquet"
    if (
        output_manifest.get("status") != "COMPLETED"
        or output_manifest.get("streaming_scene_shards") is not True
        or output_manifest.get("per_candidate_sha256")
        != low_peak.sha256_file(output_candidate)
        or output_manifest.get("per_sample_sha256")
        != low_peak.sha256_file(output_sample)
    ):
        raise ValueError("merged feature output identity is invalid")
    source_entries = output_manifest.get("source_shards")
    if (
        not isinstance(source_entries, list)
        or len(source_entries) != len(shard_roots)
    ):
        raise ValueError("merged feature manifest has no source_shards")

    inputs: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    sample_paths: list[Path] = []
    releases: list[tuple[Path, str]] = []
    all_sources_exist = True
    for root in shard_roots:
        manifest_path = root / "dataset_manifest.json"
        manifest = read_json(manifest_path)
        entry = exact_source_entry(source_entries, key="path", path=root)
        if (
            entry.get("dataset_manifest_sha256")
            != low_peak.sha256_file(manifest_path)
            or int(entry.get("sample_count", -1))
            != int(manifest.get("sample_count", -2))
            or int(entry.get("candidate_count", -1))
            != int(manifest.get("candidate_count", -2))
        ):
            raise ValueError(f"feature shard lineage differs: {root}")
        for filename, hash_key, paths in (
            ("per_candidate.parquet", "per_candidate_sha256", candidate_paths),
            ("per_sample.parquet", "per_sample_sha256", sample_paths),
        ):
            path = root / filename
            digest = str(manifest.get(hash_key, ""))
            if path.exists():
                if path.is_symlink() or low_peak.sha256_file(path) != digest:
                    raise ValueError(f"feature shard changed: {path}")
            else:
                all_sources_exist = False
            paths.append(path)
            releases.append((path, digest))
            inputs.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "rows": int(
                        manifest[
                            "candidate_count"
                            if filename == "per_candidate.parquet"
                            else "sample_count"
                        ]
                    ),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": low_peak.sha256_file(manifest_path),
                    "shard_index": int(manifest["shard_index"]),
                }
            )

    output_candidate_contract = low_peak.parquet_key_set_contract(
        [output_candidate],
        key_columns=("sample_id", "candidate_id"),
        include_all_columns=True,
    )
    output_sample_contract = low_peak.parquet_key_set_contract(
        [output_sample],
        key_columns=("sample_id",),
        include_all_columns=True,
    )
    if (
        int(output_candidate_contract["rows"])
        != int(output_manifest.get("candidate_count", -1))
        or int(output_sample_contract["rows"])
        != int(output_manifest.get("sample_count", -1))
    ):
        raise ValueError("merged feature output row count differs")
    if all_sources_exist:
        source_contracts: Mapping[str, Any] = {
            "per_candidate": low_peak.parquet_key_set_contract(
                candidate_paths,
                key_columns=("sample_id", "candidate_id"),
                include_all_columns=True,
            ),
            "per_sample": low_peak.parquet_key_set_contract(
                sample_paths,
                key_columns=("sample_id",),
                include_all_columns=True,
            ),
        }
    elif prior_stage is not None:
        source_contracts = prior_stage.get("source_key_contracts", {})
    else:
        raise FileNotFoundError(
            "a feature shard is missing without a matching resume journal"
        )
    output_contracts = {
        "per_candidate": output_candidate_contract,
        "per_sample": output_sample_contract,
    }
    if source_contracts != output_contracts:
        raise ValueError("feature source/output key universes differ")
    return (
        {
            "inputs": inputs,
            "outputs": [
                {
                    "path": str(output_candidate),
                    "sha256": low_peak.sha256_file(output_candidate),
                    "rows": int(output_candidate_contract["rows"]),
                },
                {
                    "path": str(output_sample),
                    "sha256": low_peak.sha256_file(output_sample),
                    "rows": int(output_sample_contract["rows"]),
                },
                {
                    "path": str(output_manifest_path),
                    "sha256": low_peak.sha256_file(output_manifest_path),
                },
            ],
            "source_key_contracts": source_contracts,
            "output_key_contracts": output_contracts,
            "lineage_sha256": low_peak.canonical_json_sha256(
                output_manifest["source_shards"]
            ),
        },
        releases,
    )


def main() -> int:
    args = parse_args()
    raw_output = args.output.expanduser().resolve()
    raw_shards = [path.expanduser().resolve() for path in args.shard]
    input_parquets = (
        raw_shards
        if args.family == "gqcnn"
        else [
            root / filename
            for root in raw_shards
            for filename in ("per_candidate.parquet", "per_sample.parquet")
        ]
    )
    output, tmp_root, run_root, resolved_inputs = (
        low_peak.validate_release_scope(
            output=raw_output,
            tmp_root=args.tmp_root,
            input_parquets=input_parquets,
        )
    )
    if args.family == "features":
        resolved_shards = [path.expanduser().resolve() for path in raw_shards]
        if any(
            not low_peak.is_below(root, tmp_root) for root in resolved_shards
        ):
            raise ValueError("feature shard roots must be below run tmp/")
    else:
        resolved_shards = resolved_inputs

    receipt_path = (
        args.receipt_path.expanduser().resolve()
        if args.receipt_path is not None
        else low_peak.default_receipt_path(
            run_root, family=args.family, output=output
        )
    )
    receipt_path = low_peak.validate_receipt_path(
        receipt_path, run_root=run_root
    )
    existing = low_peak.load_receipt(receipt_path)
    prior_stage = (
        existing.get("stages", {}).get("release")
        if existing is not None
        else None
    )
    if args.family == "gqcnn":
        evidence, releases = inspect_gqcnn(
            output=output,
            shard_paths=resolved_shards,
            prior_stage=prior_stage,
        )
    else:
        evidence, releases = inspect_features(
            output_root=output,
            shard_roots=resolved_shards,
            prior_stage=prior_stage,
        )
    plan = {
        "family": args.family,
        "output": str(output),
        "tmp_root": str(tmp_root),
        "max_run_bytes": int(args.max_run_bytes),
        "inputs": evidence["inputs"],
        "outputs": evidence["outputs"],
        "lineage_sha256": evidence["lineage_sha256"],
    }
    plan_sha256 = low_peak.canonical_json_sha256(plan)
    current_allocated = low_peak.allocated_bytes(run_root)
    dry_run = {
        "schema_version": 1,
        "status": "VERIFIED_DRY_RUN",
        "plan": plan,
        "plan_sha256": plan_sha256,
        "receipt_path": str(receipt_path),
        "writes_performed": False,
        "storage": {
            "current_allocated_bytes": current_allocated,
            "max_run_bytes": int(args.max_run_bytes),
            "within_budget_before_release": (
                current_allocated <= int(args.max_run_bytes)
            ),
            "release_logical_bytes": sum(
                path.stat().st_size for path, _ in releases if path.exists()
            ),
        },
    }
    if not args.execute:
        if existing is not None and (
            existing.get("plan_sha256") != plan_sha256
            or existing.get("plan") != plan
        ):
            raise ValueError(
                "low-peak resume parameters or source identities changed"
            )
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        return 0

    output_directories: set[Path] = set()
    for item in evidence["outputs"]:
        path = Path(str(item["path"]))
        low_peak.fsync_file(path)
        output_directories.add(path.parent)
    for directory in output_directories:
        low_peak.fsync_directory(directory)
    receipt = low_peak.bind_or_resume_receipt(
        receipt_path=receipt_path,
        plan=plan,
        tmp_root=tmp_root,
    )
    stage = receipt.get("stages", {}).get("release")
    expected_stage = {
        **evidence,
        "status": "VERIFIED_READY_TO_RELEASE",
    }
    if stage is None:
        receipt.setdefault("stages", {})["release"] = expected_stage
        low_peak.durable_atomic_json(
            receipt_path, receipt, tmp_root=tmp_root
        )
        low_peak.fault_point(f"{args.family}_release:after_verified")
    else:
        for key, value in expected_stage.items():
            if key != "status" and stage.get(key) != value:
                raise ValueError(
                    f"{args.family} resume evidence changed for {key}"
                )
        if stage.get("status") not in {
            "VERIFIED_READY_TO_RELEASE",
            "RELEASED",
        }:
            raise ValueError(f"invalid {args.family} release journal state")
    low_peak.release_verified_parquets(
        paths_and_hashes=releases,
        tmp_root=tmp_root,
        receipt_path=receipt_path,
        receipt=receipt,
        stage_name="release",
    )
    receipt["stages"]["release"]["status"] = "RELEASED"
    receipt["status"] = "COMPLETED"
    receipt["storage_after_release_allocated_bytes"] = (
        low_peak.allocated_bytes(run_root)
    )
    low_peak.durable_atomic_json(
        receipt_path, receipt, tmp_root=tmp_root
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
