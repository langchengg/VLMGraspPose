#!/usr/bin/env python3
"""Merge scene-grouped Dex-Net shards with full artifact audits.

The default compatibility mode hard-links scorer inputs exactly as before.
``--compact-only`` merges only the three verified stage Parquets and small
root manifests, so source shards may already have had their verbose per-sample
trees pruned after scoring and feature extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.modular_reranking import low_peak_transaction as low_peak  # noqa: E402


STAGE_FILES = {
    "raw": "raw_candidates.parquet",
    "mask_validated": "mask_validated_candidates.parquet",
    "nms": "nms_candidates.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, action="append", required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="Do not require or retain per-sample scorer bundles or label files.",
    )
    parser.add_argument(
        "--low-peak-release-inputs",
        action="store_true",
        help=(
            "Use a resumable stagewise merge. Without --execute this is a "
            "zero-write dry-run; with --execute, only verified compact input "
            "Parquets below the active run tmp/ are unlinked stage by stage."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute a requested low-peak release transaction.",
    )
    parser.add_argument(
        "--max-run-bytes",
        type=int,
        default=low_peak.DEFAULT_MAX_RUN_BYTES,
        help="Maximum allocated bytes permitted by each stage preflight.",
    )
    parser.add_argument(
        "--release-receipt",
        type=Path,
        help=(
            "Persistent transaction receipt; defaults below "
            "run/manifests/low_peak_merge/."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def validate_tmp_scope(
    output_root: Path, tmp_root: Path, *, create_tmp: bool = True
) -> tuple[Path, Path]:
    output = output_root.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    run_root = protected_run_root(output)
    if protected_run_root(temporary) != run_root:
        raise ValueError("output-root and tmp-root belong to different active runs")
    configured_tmp = (run_root / "tmp").resolve()
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    if output == temporary or temporary not in output.parents:
        raise ValueError(
            "merged candidate output must be stored below tmp-root"
        )
    if create_tmp:
        temporary.mkdir(parents=True, exist_ok=True)
    elif not temporary.is_dir():
        raise FileNotFoundError(
            f"low-peak dry-run requires an existing tmp-root: {temporary}"
        )
    return output, temporary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=True,
        ).encode("utf-8")
    ).hexdigest()


def atomic_text(path: Path, text: str, tmp_root: Path) -> None:
    temporary = tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any, tmp_root: Path) -> None:
    atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        tmp_root,
    )


def link_directory(source: Path, destination: Path, tmp_root: Path) -> None:
    staging = tmp_root / f"{destination.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    for path in sorted(source.iterdir()):
        if not path.is_file():
            raise ValueError(f"unexpected nested candidate artifact: {path}")
        os.link(path, staging / path.name)
    os.replace(staging, destination)


def parquet_compressions(path: Path) -> set[str]:
    parquet = pq.ParquetFile(path)
    return {
        str(
            parquet.metadata.row_group(row_group)
            .column(column)
            .compression
        ).lower()
        for row_group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    }


def require_pruned_compact_shard(
    *,
    root: Path,
    config: dict[str, Any],
    run_root: Path,
) -> None:
    """Reject candidate roots that still contain active verbose directories."""

    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if directories:
        raise ValueError(
            "low-peak release refuses an active verbose candidate root: "
            f"{directories[0]}"
        )
    shard_index = int(config["selection"]["shard_index"])
    receipt_path = (
        run_root
        / "manifests"
        / "streaming_cleanup"
        / str(config["split"])
        / f"shard_{shard_index}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifacts = receipt.get("artifacts", {})
    expected_stage_hashes = {
        stage: str(config["candidate_stage_artifacts"][stage]["sha256"])
        for stage in STAGE_FILES
    }
    if (
        receipt.get("status") != "COMPLETED"
        or receipt.get("executed") is not True
        or Path(str(receipt.get("candidate_root", ""))).resolve() != root
        or artifacts.get("candidate_run_config_sha256")
        != sha256_file(root / "run_config.json")
        or artifacts.get("summary_sha256")
        != sha256_file(root / "summary.csv")
        or artifacts.get("funnel_labels_sha256")
        != sha256_file(root / "funnel_labels.jsonl")
        or artifacts.get("candidate_stage_sha256")
        != expected_stage_hashes
        or receipt.get("remaining_eligible_directories", []) != []
    ):
        raise ValueError(
            f"candidate shard lacks an exact completed verbose-prune receipt: {root}"
        )


def candidate_low_peak_plan(
    *,
    prediction_manifest: Path,
    prediction_manifest_sha256: str,
    published_root: Path,
    staging_root: Path,
    shard_roots: list[Path],
    configs: list[dict[str, Any]],
    max_run_bytes: int,
) -> dict[str, Any]:
    return {
        "family": "candidate_stage_tables",
        "prediction_manifest": str(prediction_manifest),
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "output_root": str(published_root),
        "staging_root": str(staging_root),
        "max_run_bytes": int(max_run_bytes),
        "source_shards": [
            {
                "root": str(root),
                "run_config_sha256": sha256_file(root / "run_config.json"),
                "protocol_identity_sha256": config[
                    "protocol_identity_sha256"
                ],
                "shard_index": int(config["selection"]["shard_index"]),
                "artifacts": {
                    stage: {
                        "path": str((root / filename).resolve()),
                        "sha256": str(
                            config["candidate_stage_artifacts"][stage][
                                "sha256"
                            ]
                        ),
                        "rows": int(
                            config["candidate_stage_artifacts"][stage]["rows"]
                        ),
                    }
                    for stage, filename in STAGE_FILES.items()
                },
            }
            for root, config in zip(shard_roots, configs, strict=True)
        ],
    }


def validate_completed_low_peak_output(
    *,
    output_root: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    config_path = output_root / "run_config.json"
    if not config_path.is_file():
        raise ValueError("completed low-peak output has no run_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "COMPLETED":
        raise ValueError("completed low-peak output config is not COMPLETED")
    stages = receipt.get("stages", {})
    for stage, filename in STAGE_FILES.items():
        state = stages.get(stage, {})
        path = output_root / filename
        artifact = config.get("candidate_stage_artifacts", {}).get(stage, {})
        if (
            state.get("status") != "RELEASED"
            or not path.is_file()
            or sha256_file(path) != state.get("output_sha256")
            or artifact.get("sha256") != state.get("output_sha256")
            or int(artifact.get("rows", -1)) != int(state.get("rows", -2))
        ):
            raise ValueError(
                f"completed low-peak output differs for stage {stage}"
            )
    return config


def merge_stage_parquets(
    shard_roots: list[Path],
    configs: list[dict[str, Any]],
    output_root: Path,
    tmp_root: Path,
    *,
    manifest_root: Path | None = None,
    low_peak_context: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    published_root = output_root if manifest_root is None else manifest_root
    merged: dict[str, dict[str, Any]] = {}
    for stage, filename in STAGE_FILES.items():
        if low_peak_context is not None:
            source_paths = [root / filename for root in shard_roots]
            inputs = [
                {
                    "path": str(path.resolve()),
                    "sha256": str(
                        config["candidate_stage_artifacts"][stage]["sha256"]
                    ),
                    "rows": int(
                        config["candidate_stage_artifacts"][stage]["rows"]
                    ),
                    "protocol_identity_sha256": config[
                        "protocol_identity_sha256"
                    ],
                }
                for path, config in zip(source_paths, configs, strict=True)
            ]
            receipt = low_peak_context["receipt"]
            receipt_path = low_peak_context["receipt_path"]
            stage_state = receipt.get("stages", {}).get(stage)
            destination = output_root / filename
            if stage_state is not None:
                if (
                    stage_state.get("inputs") != inputs
                    or stage_state.get("status")
                    not in {"VERIFIED_READY_TO_RELEASE", "RELEASED"}
                    or not destination.is_file()
                    or sha256_file(destination)
                    != stage_state.get("output_sha256")
                ):
                    raise ValueError(
                        f"low-peak resume state differs for stage {stage}"
                    )
                output_contract = low_peak.parquet_key_contract(
                    destination,
                    key_columns=("sample_id", "candidate_id"),
                    require_sample_grouping=True,
                    include_all_columns=True,
                )
                if (
                    output_contract
                    != stage_state.get("output_key_contract")
                    or int(stage_state.get("rows", -1))
                    != int(output_contract["rows"])
                    or parquet_compressions(destination) != {"zstd"}
                ):
                    raise ValueError(
                        f"low-peak resumed output differs for stage {stage}"
                    )
            else:
                for item, path in zip(inputs, source_paths, strict=True):
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or sha256_file(path) != item["sha256"]
                    ):
                        raise ValueError(
                            f"low-peak source differs for stage {stage}: {path}"
                        )
                temporary = output_root / f".{filename}.building.tmp"
                if temporary.exists():
                    if temporary.is_symlink() or not temporary.is_file():
                        raise ValueError(
                            f"invalid interrupted stage temporary: {temporary}"
                        )
                    temporary.unlink()
                    low_peak.fsync_directory(temporary.parent)
                budget = low_peak.enforce_budget_preflight(
                    run_root=low_peak_context["run_root"],
                    input_paths=source_paths,
                    max_run_bytes=int(low_peak_context["max_run_bytes"]),
                )
                source_contract = low_peak.ordered_key_contract_for_paths(
                    source_paths,
                    key_columns=("sample_id", "candidate_id"),
                    require_sample_grouping=True,
                    include_all_columns=True,
                )
                expected_rows = sum(int(item["rows"]) for item in inputs)
                if int(source_contract["rows"]) != expected_rows:
                    raise ValueError(
                        f"{stage} source key/row accounting differs"
                    )
                schema: pa.Schema | None = None
                row_count = 0
                writer: pq.ParquetWriter | None = None
                try:
                    for path in source_paths:
                        parquet = pq.ParquetFile(path)
                        if schema is None:
                            schema = parquet.schema_arrow
                            writer = pq.ParquetWriter(
                                temporary,
                                schema,
                                compression="zstd",
                                use_dictionary=True,
                            )
                        elif parquet.schema_arrow != schema:
                            raise ValueError(
                                f"{stage} shard Parquet schemas differ"
                            )
                        for batch in parquet.iter_batches(batch_size=50_000):
                            assert writer is not None
                            writer.write_batch(batch)
                            row_count += batch.num_rows
                            low_peak.fault_point(
                                f"candidate_{stage}:after_batch"
                            )
                finally:
                    if writer is not None:
                        writer.close()
                if schema is None or row_count != expected_rows:
                    raise ValueError(
                        f"low-peak {stage} merge row verification failed"
                    )
                low_peak.fsync_file(temporary)
                temporary_contract = low_peak.parquet_key_contract(
                    temporary,
                    key_columns=("sample_id", "candidate_id"),
                    require_sample_grouping=True,
                    include_all_columns=True,
                )
                if (
                    temporary_contract != source_contract
                    or parquet_compressions(temporary) != {"zstd"}
                ):
                    raise ValueError(
                        f"low-peak {stage} key/schema verification failed"
                    )
                os.replace(temporary, destination)
                low_peak.fsync_directory(destination.parent)
                verified_peak = low_peak.allocated_bytes(
                    low_peak_context["run_root"]
                )
                if verified_peak > int(low_peak_context["max_run_bytes"]):
                    destination.unlink()
                    low_peak.fsync_directory(destination.parent)
                    raise ValueError(
                        "low-peak stage exceeded the storage budget despite "
                        f"preflight: stage={stage} allocated={verified_peak} "
                        f"max={low_peak_context['max_run_bytes']}"
                    )
                output_sha256 = sha256_file(destination)
                stage_state = {
                    "status": "VERIFIED_READY_TO_RELEASE",
                    "inputs": inputs,
                    "rows": row_count,
                    "source_key_contract": source_contract,
                    "output_key_contract": temporary_contract,
                    "output_path": str(destination.resolve()),
                    "output_sha256": output_sha256,
                    "output_size_bytes": destination.stat().st_size,
                    "compression": "zstd",
                    "budget_preflight": budget,
                    "verified_peak_allocated_bytes": verified_peak,
                    "lineage_sha256": low_peak.canonical_json_sha256(inputs),
                }
                receipt.setdefault("stages", {})[stage] = stage_state
                low_peak.durable_atomic_json(
                    receipt_path, receipt, tmp_root=tmp_root
                )
                low_peak.fault_point(f"candidate_{stage}:after_verified")
            low_peak.release_verified_parquets(
                paths_and_hashes=[
                    (path, str(item["sha256"]))
                    for path, item in zip(source_paths, inputs, strict=True)
                ],
                tmp_root=tmp_root,
                receipt_path=receipt_path,
                receipt=receipt,
                stage_name=stage,
            )
            stage_state = receipt["stages"][stage]
            stage_state["status"] = "RELEASED"
            low_peak.durable_atomic_json(
                receipt_path, receipt, tmp_root=tmp_root
            )
            merged[stage] = {
                "path": str(published_root / filename),
                "sha256": stage_state["output_sha256"],
                "rows": int(stage_state["rows"]),
                "primary_key": ["sample_id", "candidate_id"],
                "compression": "zstd",
                "row_order": "shard_index_then_source_manifest_order",
                "ordered_primary_key_sha256": stage_state[
                    "output_key_contract"
                ]["ordered_primary_key_sha256"],
                "source_lineage_sha256": stage_state["lineage_sha256"],
            }
            continue
        schema: pa.Schema | None = None
        row_count = 0
        temporary = tmp_root / f"{filename}.{uuid.uuid4().hex}.tmp"
        writer: pq.ParquetWriter | None = None
        try:
            for root, config in zip(shard_roots, configs, strict=True):
                artifact = config["candidate_stage_artifacts"][stage]
                path = root / filename
                if (
                    Path(str(artifact["path"])).resolve() != path.resolve()
                    or sha256_file(path) != artifact["sha256"]
                ):
                    raise ValueError(f"{stage} shard artifact identity mismatch")
                parquet = pq.ParquetFile(path)
                if schema is None:
                    schema = parquet.schema_arrow
                    writer = pq.ParquetWriter(
                        temporary,
                        schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                elif parquet.schema_arrow != schema:
                    raise ValueError(f"{stage} shard Parquet schemas differ")
                for batch in parquet.iter_batches(batch_size=50_000):
                    assert writer is not None
                    writer.write_batch(batch)
                    row_count += batch.num_rows
        finally:
            if writer is not None:
                writer.close()
        if schema is None:
            raise ValueError(f"no {stage} candidate-stage schema")
        expected = sum(
            int(config["candidate_stage_artifacts"][stage]["rows"])
            for config in configs
        )
        parquet = pq.ParquetFile(temporary)
        if parquet.schema_arrow != schema or row_count != expected:
            raise ValueError(f"merged {stage} Parquet verification failed")
        destination = output_root / filename
        os.replace(temporary, destination)
        merged[stage] = {
            "path": str(published_root / filename),
            "sha256": sha256_file(destination),
            "rows": row_count,
            "primary_key": ["sample_id", "candidate_id"],
            "compression": "zstd",
            "row_order": "shard_index_then_source_manifest_order",
        }
    return merged


def preflight_shard_artifacts(
    shard_roots: list[Path],
    configs: list[dict[str, Any]],
    *,
    released_paths: set[str] | None = None,
) -> None:
    """Reject incomplete or changed shards before creating any merged output."""

    expected_schema: dict[str, pa.Schema] = {}
    for root, config in zip(shard_roots, configs, strict=True):
        if config.get("status") != "COMPLETED":
            raise ValueError(f"candidate shard is not COMPLETED: {root}")
        counts = config.get("counts")
        if not isinstance(counts, dict):
            raise ValueError(f"candidate shard has no counts object: {root}")
        if int(counts.get("execution_failures", -1)) != 0:
            raise ValueError(f"candidate shard records execution failures: {root}")
        artifacts = config.get("candidate_stage_artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"candidate shard has no stage artifacts: {root}")
        count_keys = {
            "raw": "raw_candidates",
            "mask_validated": "mask_validated_candidates",
            "nms": "nms_candidates",
        }
        for stage, filename in STAGE_FILES.items():
            artifact = artifacts.get(stage)
            if not isinstance(artifact, dict):
                raise ValueError(f"candidate shard lacks {stage} artifact: {root}")
            path = root / filename
            if not path.is_file():
                if (
                    released_paths is not None
                    and str(path.resolve()) in released_paths
                ):
                    continue
                raise FileNotFoundError(
                    f"candidate shard artifact is missing: {path}"
                )
            if Path(str(artifact.get("path"))).resolve() != path.resolve():
                raise ValueError(f"{stage} shard artifact path mismatch: {root}")
            if sha256_file(path) != artifact.get("sha256"):
                raise ValueError(f"{stage} shard artifact hash mismatch: {root}")
            parquet = pq.ParquetFile(path)
            artifact_rows = int(artifact.get("rows", -1))
            if parquet.metadata.num_rows != artifact_rows:
                raise ValueError(f"{stage} shard artifact row mismatch: {root}")
            if artifact_rows != int(counts.get(count_keys[stage], -1)):
                raise ValueError(
                    f"{stage} shard artifact rows disagree with counts: {root}"
                )
            if stage not in expected_schema:
                expected_schema[stage] = parquet.schema_arrow
            elif parquet.schema_arrow != expected_schema[stage]:
                raise ValueError(f"{stage} shard Parquet schemas differ")


def main() -> int:
    args = parse_args()
    if args.execute and not args.low_peak_release_inputs:
        raise ValueError("--execute requires --low-peak-release-inputs")
    if args.low_peak_release_inputs and not args.compact_only:
        raise ValueError(
            "low-peak release is supported only with --compact-only"
        )
    prediction_manifest = args.prediction_manifest.expanduser().resolve()
    published_root, tmp_root = validate_tmp_scope(
        args.output_root,
        args.tmp_root,
        create_tmp=not args.low_peak_release_inputs,
    )
    shard_roots = [path.expanduser().resolve() for path in args.shard_root]
    if published_root.exists() and not args.low_peak_release_inputs:
        raise FileExistsError(f"merged output already exists: {published_root}")
    if len(shard_roots) < 2:
        raise ValueError("at least two shard roots are required")
    prediction_rows = read_jsonl(prediction_manifest)
    prediction_manifest_sha256 = sha256_file(prediction_manifest)
    expected_ids = [str(row["sample_id"]) for row in prediction_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("prediction manifest has duplicate sample IDs")

    configs = [
        json.loads((root / "run_config.json").read_text(encoding="utf-8"))
        for root in shard_roots
    ]
    for root, config in zip(shard_roots, configs, strict=True):
        if config.get("prediction_manifest_sha256") != prediction_manifest_sha256:
            raise ValueError(
                f"shard prediction manifest hash mismatch: {root}"
            )
        if (
            Path(str(config.get("prediction_manifest"))).resolve()
            != prediction_manifest
        ):
            raise ValueError(
                f"shard prediction manifest path mismatch: {root}"
            )
    identity_keys = (
        "split",
        "annotations_sha256",
        "config_sha256",
        "configuration_hash",
        "evaluation_config_sha256",
        "sample_seed_mode",
        "seed_namespace",
        "protocol_family_identity_sha256",
        "candidate_stage_schema_sha256",
    )
    for key in identity_keys:
        values = {json.dumps(config[key], sort_keys=True) for config in configs}
        if len(values) != 1:
            raise ValueError(f"shard identity mismatch for {key}: {values}")
    indices = sorted(
        int(config["selection"]["shard_index"]) for config in configs
    )
    num_shards = {int(config["selection"]["num_shards"]) for config in configs}
    if len(num_shards) != 1 or indices != list(range(len(shard_roots))):
        raise ValueError("shard set is incomplete or duplicated")
    if next(iter(num_shards)) != len(shard_roots):
        raise ValueError("declared num_shards disagrees with shard roots")
    ordered_pairs = sorted(
        zip(shard_roots, configs, strict=True),
        key=lambda pair: int(pair[1]["selection"]["shard_index"]),
    )
    shard_roots = [pair[0] for pair in ordered_pairs]
    configs = [pair[1] for pair in ordered_pairs]
    low_peak_context: dict[str, Any] | None = None
    if args.low_peak_release_inputs:
        all_stage_paths = [
            root / filename
            for root in shard_roots
            for filename in STAGE_FILES.values()
        ]
        (
            published_root,
            tmp_root,
            run_root,
            _,
        ) = low_peak.validate_release_scope(
            output=published_root,
            tmp_root=tmp_root,
            input_parquets=all_stage_paths,
        )
        for root in shard_roots:
            if (
                published_root == root
                or root in published_root.parents
                or published_root in root.parents
            ):
                raise ValueError(
                    "low-peak output and source shard roots must be disjoint"
                )
        staging_id = low_peak.canonical_json_sha256(
            str(published_root)
        )[:16]
        staging_root = (
            tmp_root
            / ".low_peak_candidate_merge"
            / f"{staging_id}.staging"
        )
        receipt_path = (
            args.release_receipt.expanduser().resolve()
            if args.release_receipt is not None
            else low_peak.default_receipt_path(
                run_root,
                family="candidate",
                output=published_root,
            )
        )
        receipt_path = low_peak.validate_receipt_path(
            receipt_path, run_root=run_root
        )
        plan = candidate_low_peak_plan(
            prediction_manifest=prediction_manifest,
            prediction_manifest_sha256=prediction_manifest_sha256,
            published_root=published_root,
            staging_root=staging_root,
            shard_roots=shard_roots,
            configs=configs,
            max_run_bytes=args.max_run_bytes,
        )
        existing_receipt = low_peak.load_receipt(receipt_path)
        if existing_receipt is not None and (
            existing_receipt.get("plan_sha256")
            != low_peak.canonical_json_sha256(plan)
            or existing_receipt.get("plan") != plan
        ):
            raise ValueError(
                "low-peak resume parameters or source identities changed"
            )
        released_paths = set(
            map(
                str,
                (existing_receipt or {}).get("released_inputs", []),
            )
        )
        for state in (existing_receipt or {}).get("stages", {}).values():
            if state.get("status") in {
                "VERIFIED_READY_TO_RELEASE",
                "RELEASED",
            }:
                released_paths.update(
                    str(item["path"]) for item in state.get("inputs", [])
                )
        preflight_shard_artifacts(
            shard_roots,
            configs,
            released_paths=released_paths,
        )
        for root, config in zip(shard_roots, configs, strict=True):
            require_pruned_compact_shard(
                root=root, config=config, run_root=run_root
            )
        budget_preflights: dict[str, dict[str, int] | str] = {}
        for stage, filename in STAGE_FILES.items():
            paths = [root / filename for root in shard_roots]
            if all(path.is_file() for path in paths):
                budget_preflights[stage] = low_peak.enforce_budget_preflight(
                    run_root=run_root,
                    input_paths=paths,
                    max_run_bytes=int(args.max_run_bytes),
                )
            else:
                budget_preflights[stage] = "already durably merged or released"
        dry_run = {
            "schema_version": 1,
            "status": "VERIFIED_DRY_RUN",
            "plan_sha256": low_peak.canonical_json_sha256(plan),
            "plan": plan,
            "receipt_path": str(receipt_path),
            "resume_detected": existing_receipt is not None,
            "writes_performed": False,
            "budget_preflights": budget_preflights,
        }
        if not args.execute:
            print(json.dumps(dry_run, indent=2, sort_keys=True))
            return 0
        receipt = low_peak.bind_or_resume_receipt(
            receipt_path=receipt_path,
            plan=plan,
            tmp_root=tmp_root,
        )
        if published_root.exists():
            config = validate_completed_low_peak_output(
                output_root=published_root,
                receipt=receipt,
            )
            receipt.update(
                {
                    "status": "COMPLETED",
                    "output_run_config_sha256": sha256_file(
                        published_root / "run_config.json"
                    ),
                }
            )
            low_peak.durable_atomic_json(
                receipt_path, receipt, tmp_root=tmp_root
            )
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        if staging_root.exists() and not existing_receipt:
            raise ValueError(
                f"untracked low-peak staging root exists: {staging_root}"
            )
        staging_root.mkdir(parents=True, exist_ok=True)
        low_peak_context = {
            "run_root": run_root,
            "receipt_path": receipt_path,
            "receipt": receipt,
            "max_run_bytes": int(args.max_run_bytes),
        }
    else:
        preflight_shard_artifacts(shard_roots, configs)

    sample_to_root: dict[str, Path] = {}
    summaries: dict[str, dict[str, str]] = {}
    labels: dict[str, dict[str, Any]] = {}
    summary_fields: list[str] | None = None
    for root, config in zip(shard_roots, configs, strict=True):
        local_summaries: list[dict[str, str]] = []
        with (root / "summary.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            if summary_fields is None:
                summary_fields = fields
            elif fields != summary_fields:
                raise ValueError("shard summary schemas differ")
            for row in reader:
                sample_id = str(row["sample_id"])
                if sample_id in sample_to_root:
                    raise ValueError(f"sample appears in multiple shards: {sample_id}")
                sample_to_root[sample_id] = root
                summaries[sample_id] = dict(row)
                local_summaries.append(dict(row))
        local_labels = read_jsonl(root / "funnel_labels.jsonl")
        for row in local_labels:
            sample_id = str(row["sample_id"])
            if sample_id in labels:
                raise ValueError(f"duplicate funnel label: {sample_id}")
            labels[sample_id] = row
        local_summary_by_id = {
            str(row["sample_id"]): row for row in local_summaries
        }
        local_label_by_id = {
            str(row["sample_id"]): row for row in local_labels
        }
        if (
            len(local_summary_by_id) != len(local_summaries)
            or len(local_label_by_id) != len(local_labels)
            or set(local_summary_by_id) != set(local_label_by_id)
        ):
            raise ValueError(f"shard summary/funnel universe differs: {root}")
        for sample_id, summary in local_summary_by_id.items():
            funnel = local_label_by_id[sample_id]
            for summary_key, funnel_key in (
                ("raw_candidate_count", "raw_candidate_count"),
                ("mask_validated_count", "mask_validated_candidate_count"),
                ("post_nms_count", "nms_candidate_count"),
            ):
                if int(summary[summary_key]) != int(funnel[funnel_key]):
                    raise ValueError(
                        f"shard summary/funnel counts differ: {sample_id}"
                    )
            expected_status = (
                "success_empty"
                if bool(funnel["valid_empty"])
                else "success_nonempty"
            )
            if summary["status"] != expected_status:
                raise ValueError(
                    f"shard summary/funnel status differs: {sample_id}"
                )
        local_counts = {
            "samples": len(local_summaries),
            "nonempty_samples": sum(
                row["status"] == "success_nonempty" for row in local_summaries
            ),
            "empty_samples": sum(
                row["status"] == "success_empty" for row in local_summaries
            ),
            "raw_candidates": sum(
                int(row["raw_candidate_count"]) for row in local_summaries
            ),
            "mask_validated_candidates": sum(
                int(row["mask_validated_count"]) for row in local_summaries
            ),
            "nms_candidates": sum(
                int(row["post_nms_count"]) for row in local_summaries
            ),
            "raw_oracle": sum(bool(row["raw_oracle"]) for row in local_labels),
            "mask_validated_oracle": sum(
                bool(row["mask_validated_oracle"]) for row in local_labels
            ),
            "nms_oracle": sum(bool(row["nms_oracle"]) for row in local_labels),
        }
        if any(
            int(config["counts"].get(key, -1)) != value
            for key, value in local_counts.items()
        ):
            raise ValueError(f"shard summary/funnel counts differ from config: {root}")
        if args.compact_only and not (root / "_labels").exists():
            shard_index = int(config["selection"]["shard_index"])
            receipt_path = (
                protected_run_root(root)
                / "manifests"
                / "streaming_cleanup"
                / str(config["split"])
                / f"shard_{shard_index}.json"
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_artifacts = receipt.get("artifacts", {})
            if (
                receipt.get("status") != "COMPLETED"
                or receipt.get("executed") is not True
                or Path(str(receipt.get("candidate_root", ""))).resolve() != root
                or receipt_artifacts.get("candidate_run_config_sha256")
                != sha256_file(root / "run_config.json")
                or receipt_artifacts.get("summary_sha256")
                != sha256_file(root / "summary.csv")
                or receipt_artifacts.get("funnel_labels_sha256")
                != sha256_file(root / "funnel_labels.jsonl")
                or receipt_artifacts.get("candidate_stage_sha256")
                != {
                    stage: config["candidate_stage_artifacts"][stage]["sha256"]
                    for stage in STAGE_FILES
                }
            ):
                raise ValueError(
                    f"pruned shard lacks a matching persistent cleanup receipt: {root}"
                )
    if set(sample_to_root) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(sample_to_root))
        extra = sorted(set(sample_to_root) - set(expected_ids))
        raise ValueError(
            f"shard coverage mismatch missing={missing[:5]} extra={extra[:5]}"
        )
    if set(labels) != set(expected_ids):
        raise ValueError("funnel label coverage does not match prediction manifest")

    if not args.compact_only:
        for sample_id in expected_ids:
            source_root = sample_to_root[sample_id]
            sample_root = source_root / sample_id
            label_path = source_root / "_labels" / f"{sample_id}.json"
            if not sample_root.is_dir():
                raise FileNotFoundError(
                    f"candidate sample directory is missing: {sample_root}"
                )
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"candidate label is missing: {label_path}"
                )
            unexpected = [
                path for path in sample_root.iterdir() if not path.is_file()
            ]
            if unexpected:
                raise ValueError(
                    "candidate sample directory contains nested artifacts: "
                    f"{unexpected[0]}"
                )

    if low_peak_context is not None:
        output_root = staging_root
    else:
        staging_root = (
            tmp_root / f".{published_root.name}.merge-{uuid.uuid4().hex}"
        )
        staging_root.mkdir()
        output_root = staging_root
    if not args.compact_only:
        labels_root = output_root / "_labels"
        labels_root.mkdir()
        for position, row in enumerate(prediction_rows, start=1):
            sample_id = str(row["sample_id"])
            source_root = sample_to_root[sample_id]
            link_directory(
                source_root / sample_id,
                output_root / sample_id,
                tmp_root,
            )
            os.link(
                source_root / "_labels" / f"{sample_id}.json",
                labels_root / f"{sample_id}.json",
            )
            if position % 500 == 0 or position == len(expected_ids):
                print(f"merge {position}/{len(expected_ids)}", flush=True)

    if summary_fields is None:
        raise ValueError("no shard summary rows")
    summary_path = output_root / "summary.csv"
    summary_temporary = tmp_root / f"summary.csv.{uuid.uuid4().hex}.tmp"
    with summary_temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries[sample_id] for sample_id in expected_ids)
    os.replace(summary_temporary, summary_path)
    atomic_text(
        output_root / "funnel_labels.jsonl",
        "".join(
            json.dumps(labels[sample_id], sort_keys=True, allow_nan=False) + "\n"
            for sample_id in expected_ids
        ),
        tmp_root,
    )
    atomic_text(
        output_root / "run_manifest.jsonl",
        "".join(
            json.dumps(
                {
                    "sample_index": int(row["sample_index"]),
                    "sample_id": str(row["sample_id"]),
                    "question_index": int(row["question_index"]),
                    "scene_id": str(row["scene_id"]),
                    "query": str(row["query"]),
                    "source_status": summaries[str(row["sample_id"])]["status"],
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in prediction_rows
        ),
        tmp_root,
    )
    ordered_summaries = [summaries[sample_id] for sample_id in expected_ids]
    ordered_labels = [labels[sample_id] for sample_id in expected_ids]
    counts = {
        "samples": len(expected_ids),
        "nonempty_samples": sum(
            row["status"] == "success_nonempty" for row in ordered_summaries
        ),
        "empty_samples": sum(
            row["status"] == "success_empty" for row in ordered_summaries
        ),
        "raw_candidates": sum(
            int(row["raw_candidate_count"]) for row in ordered_summaries
        ),
        "mask_validated_candidates": sum(
            int(row["mask_validated_count"]) for row in ordered_summaries
        ),
        "nms_candidates": sum(
            int(row["post_nms_count"]) for row in ordered_summaries
        ),
        "raw_oracle": sum(bool(row["raw_oracle"]) for row in ordered_labels),
        "mask_validated_oracle": sum(
            bool(row["mask_validated_oracle"]) for row in ordered_labels
        ),
        "nms_oracle": sum(bool(row["nms_oracle"]) for row in ordered_labels),
        "execution_failures": 0,
    }
    candidate_stage_artifacts = merge_stage_parquets(
        shard_roots,
        configs,
        output_root,
        tmp_root,
        manifest_root=published_root,
        low_peak_context=low_peak_context,
    )
    merged_selected_identity_sha256 = canonical_json_hash(
        [
            {
                "sample_index": int(row["sample_index"]),
                "sample_id": str(row["sample_id"]),
                "question_index": int(row["question_index"]),
                "scene_id": str(row["scene_id"]),
            }
            for row in prediction_rows
        ]
    )
    merged_protocol_identity = {
        **dict(configs[0]["protocol_family_identity"]),
        "protocol_family_identity_sha256": configs[0][
            "protocol_family_identity_sha256"
        ],
        "selected_identity_sha256": merged_selected_identity_sha256,
    }
    merged_config = {
        **{
            key: value
            for key, value in configs[0].items()
            if key not in {"counts", "fresh_samples", "elapsed_seconds", "selection"}
        },
        "status": "COMPLETED",
        "counts": counts,
        "candidate_stage_artifacts": candidate_stage_artifacts,
        "protocol_identity": merged_protocol_identity,
        "protocol_identity_sha256": canonical_json_hash(
            merged_protocol_identity
        ),
        "source_shard_protocol_identity_sha256": [
            config["protocol_identity_sha256"] for config in configs
        ],
        "fresh_samples": sum(int(config["fresh_samples"]) for config in configs),
        "elapsed_seconds_sum": sum(
            float(config["elapsed_seconds"]) for config in configs
        ),
        "selection": {
            "merged_scene_grouped_shards": len(shard_roots),
            "shard_roots": [str(root) for root in shard_roots],
        },
        "prediction_manifest": str(prediction_manifest),
        "scorer_compatible": not args.compact_only,
        "merge_storage": {
            "mode": (
                "low_peak_stagewise_compact_tables"
                if low_peak_context is not None
                else "compact_stage_tables_only"
                if args.compact_only
                else "scorer_compatible_hardlinks"
            ),
            "candidate_files_hardlinked": not args.compact_only,
            "label_files_hardlinked": not args.compact_only,
            "verbose_per_sample_tree_retained": not args.compact_only,
            "scene_caches_not_duplicated_into_merged_root": True,
            "source_stage_parquets_released_after_verification": (
                low_peak_context is not None
            ),
            "transaction_receipt": (
                str(low_peak_context["receipt_path"])
                if low_peak_context is not None
                else None
            ),
            "max_run_bytes": (
                int(args.max_run_bytes)
                if low_peak_context is not None
                else None
            ),
        },
    }
    atomic_json(output_root / "run_config.json", merged_config, tmp_root)
    atomic_text(
        output_root / "run_command.txt",
        " ".join([sys.executable, *sys.argv]) + "\n",
        tmp_root,
    )
    published_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(output_root, published_root)
    low_peak.fsync_directory(published_root.parent)
    if low_peak_context is not None:
        low_peak.fault_point("candidate:after_publish")
        receipt = low_peak_context["receipt"]
        receipt.update(
            {
                "status": "COMPLETED",
                "output_run_config_sha256": sha256_file(
                    published_root / "run_config.json"
                ),
                "output_root": str(published_root),
            }
        )
        low_peak.durable_atomic_json(
            low_peak_context["receipt_path"],
            receipt,
            tmp_root=tmp_root,
        )
    print(json.dumps(merged_config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
