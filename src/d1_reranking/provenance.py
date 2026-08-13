"""Fail-closed D1 snapshot reconciliation without opening Test labels."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from unified_reranking.artifacts import load_verified_json, verified_artifact_path
from unified_reranking.hashing import (
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)

from .contracts import EXPECTED_EVALUATOR_SHA256, EXPECTED_UNIFIED_FINAL_LOCK_SHA256


EXPECTED_SOURCE_HASHES = {
    "snapshot_a_run_manifest": "f5f04cbfa948cc28aa97c201cad7dba6fe2514859beb38febc0b274a4bfe27e3",
    "snapshot_a_frozen_protocol": "ff4798c1049f836bb624202b48b6895bc0605e65867133037ba662826e48261b",
    "snapshot_a_final_output_manifest": "938d1e0298f90016ef9977e5e6ba5c8f615d7b7712093c8cd126141143fea911",
    "snapshot_a_completed": "ff7a031761db305ed291cfdc5d28d8c2f3c56ae98d14fb90b174fb301e8b6b22",
    "snapshot_a_test_nms": "00a2cd9ce1d3d5007b0a25ff4c448497772e9b9944c7d46129213d63b34eb8cd",
    "compact_test_scores": "5cef2e0fdd2bdc965ed05037171c005403489b1399873188f37583216ffb2a66",
    "compact_test_candidate_manifest": "4a44b48229042e476bfb070d819321a8e019d41874c3632816ca1b0deecf0a5e",
    "compact_test_score_manifest": "85f8066e684d3a3d763d83058d24a7b9c23525a7e9ce831297c000e4522f185b",
    "development_feature_registry": "83b2b855624b3674ef8ed4880a798220fdf345b6d1919f0415126ed5d7687ff5",
    "unified_final_lock": EXPECTED_UNIFIED_FINAL_LOCK_SHA256,
    "canonical_evaluator": EXPECTED_EVALUATOR_SHA256,
    "development_train_labels": "7c6a4d1a420ea28d77d5fc6c6f210da5d2a756dedc6f82dd8726a0f582c09a0b",
    "development_validation_labels": "00c7fc082ee797b6ba8cdc0351bf5fc001bfb1e1e551d14d9f7c0c441bd79c58",
    "opaque_test_ground_truth": "6262b2189c02b1af239537cf814734e138fbaa6c7535fb58cf1bd65222014e70",
    "opaque_test_visual_ground_truth": "406f69c1d5aa923bfe4fe68cf20fefec9a1b73b4b2bd69767542e5c914bb064a",
}
SOURCE_CLOSURE_POINTER = "D1_SOURCE_RECONCILIATION_ACTIVE.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_A_ROOT = (
    REPO_ROOT
    / "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528"
)
COMPACT_SOURCE_ROOT = (
    REPO_ROOT
    / "HiFi_reproduction/runs/modular_reranking_repeatedfilm_v1_20260729_203147"
)
UNIFIED_RUN_ROOT = REPO_ROOT / "runs/fair_unified_reranking_20260809_103012"
SNAPSHOT_B_ROOT = REPO_ROOT / "HiFi_reproduction/docs"
SNAPSHOT_C_ROOT = (
    REPO_ROOT / "HiFi_reproduction/runs/grasp_backend_comparison_20260807_090155"
)
DEVELOPMENT_LABEL_ROOT = (
    REPO_ROOT
    / "HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500/manifests"
)
TEST_VISUAL_GROUND_TRUTH = (
    REPO_ROOT
    / "runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523/01_manifest/paired_manifest.parquet"
)


def _authoritative_named_paths() -> dict[str, Path]:
    values = {
        "snapshot_a_run_manifest": SNAPSHOT_A_ROOT / "run_manifest.json",
        "snapshot_a_frozen_protocol": SNAPSHOT_A_ROOT / "frozen_protocol.yaml",
        "snapshot_a_final_output_manifest": SNAPSHOT_A_ROOT
        / "final_output_manifest.json",
        "snapshot_a_completed": SNAPSHOT_A_ROOT / "COMPLETED",
        "snapshot_a_test_nms": SNAPSHOT_A_ROOT
        / "candidates/dexnet_nms_candidates.parquet",
        "compact_test_scores": COMPACT_SOURCE_ROOT
        / "compact_inputs/test/gqcnn_scores.parquet",
        "compact_test_candidate_manifest": COMPACT_SOURCE_ROOT
        / "compact_inputs/test/candidate_tables.manifest.json",
        "compact_test_score_manifest": COMPACT_SOURCE_ROOT
        / "compact_inputs/test/gqcnn_scores.manifest.json",
        "development_feature_registry": COMPACT_SOURCE_ROOT
        / "features/per_candidate.parquet.manifest.json",
        "unified_final_lock": UNIFIED_RUN_ROOT / "FINAL_RUN_LOCK.json",
        "canonical_evaluator": UNIFIED_RUN_ROOT / "configs/canonical_evaluator.py",
        "development_train_labels": DEVELOPMENT_LABEL_ROOT / "train_labels.parquet",
        "development_validation_labels": (
            DEVELOPMENT_LABEL_ROOT / "validation_labels.parquet"
        ),
        "opaque_test_ground_truth": DEVELOPMENT_LABEL_ROOT / "test_labels.parquet",
        "opaque_test_visual_ground_truth": TEST_VISUAL_GROUND_TRUTH,
        "d1_provenance_contract": Path(__file__),
    }
    for index, name in enumerate(
        (
            "DEXNET_FULL_HIFICS_GENERATION_RESULTS.md",
            "GQCNN_FULL_HIFICS_SCORING_RESULTS.md",
            "MODULAR_RERANKING_V1_RESULTS.md",
            "MODULAR_RERANKING_V1_AUDIT.md",
        )
    ):
        values[f"snapshot_b_evidence_{index}"] = SNAPSHOT_B_ROOT / name
    for index, name in enumerate(
        (
            "artifact_manifest_phase0.json",
            "audit/ranking_contract_audit.json",
            "audit/repeatedfilm_source_manifest.json",
            "D1_order_sensitivity.csv",
        )
    ):
        values[f"snapshot_c_evidence_{index}"] = SNAPSHOT_C_ROOT / name
    for split, source_name in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
    ):
        feature_dir = COMPACT_SOURCE_ROOT / "features" / source_name
        values[f"{split}_native_features"] = feature_dir / "per_candidate.parquet"
        values[f"{split}_feature_allowlist"] = (
            feature_dir / "inference_feature_allowlist.json"
        )
        values[f"{split}_feature_manifest"] = feature_dir / "dataset_manifest.json"
    return {name: path.resolve() for name, path in values.items()}


def _expected_canonical_input_paths() -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    for split, source_name, shard_count in (
        ("train", "train", 8),
        ("validation", "val", 4),
    ):
        values[split] = {
            "candidate_sources": [
                str(
                    (
                        COMPACT_SOURCE_ROOT
                        / "tmp"
                        / source_name
                        / "candidates"
                        / f"shard_{index}"
                        / "nms_candidates.parquet"
                    ).resolve()
                )
                for index in range(shard_count)
            ],
            "score_sources": [
                str(
                    (
                        COMPACT_SOURCE_ROOT
                        / "tmp"
                        / source_name
                        / "gqcnn_score_shards"
                        / f"shard_{index}.parquet"
                    ).resolve()
                )
                for index in range(shard_count)
            ],
            "paired_manifest": str(
                (
                    UNIFIED_RUN_ROOT / "01_manifests" / f"paired_{split}.parquet"
                ).resolve()
            ),
            "development_labels": str(
                (DEVELOPMENT_LABEL_ROOT / f"{split}_labels.parquet").resolve()
            ),
        }
    values["test"] = {
        "candidate_sources": [
            str(
                (SNAPSHOT_A_ROOT / "candidates/dexnet_nms_candidates.parquet").resolve()
            )
        ],
        "score_sources": [
            str(
                (
                    COMPACT_SOURCE_ROOT / "compact_inputs/test/gqcnn_scores.parquet"
                ).resolve()
            )
        ],
        "paired_manifest": str(
            (UNIFIED_RUN_ROOT / "01_manifests/paired_test.parquet").resolve()
        ),
        "opaque_ground_truth": str(
            (DEVELOPMENT_LABEL_ROOT / "test_labels.parquet").resolve()
        ),
        "opaque_visual_ground_truth": str(TEST_VISUAL_GROUND_TRUTH.resolve()),
    }
    for split, source_name in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
    ):
        feature_dir = COMPACT_SOURCE_ROOT / "features" / source_name
        values[split].update(
            {
                "native_features": str(
                    (feature_dir / "per_candidate.parquet").resolve()
                ),
                "feature_allowlist": str(
                    (feature_dir / "inference_feature_allowlist.json").resolve()
                ),
                "feature_manifest": str(
                    (feature_dir / "dataset_manifest.json").resolve()
                ),
            }
        )
    return values


def _closure_identity(value: dict[str, Any]) -> str:
    development = value.get("development")
    test = value.get("test")
    if not isinstance(development, dict) or not isinstance(test, dict):
        raise RuntimeError("D1 source closure identity fields are absent")
    return canonical_sha256(
        {
            "verified_artifacts": value.get("verified_artifacts"),
            "development": development,
            "test_key_audit": test.get("key_audit"),
            "test_model_contract": test.get("model_contract"),
            "canonical_inputs": value.get("canonical_inputs"),
        }
    )[:20]


def _completed_p1_ledger_binding(run_dir: Path, closure_path: Path) -> dict[str, Any]:
    ledger_path = run_dir / "run_ledger.sqlite"
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise RuntimeError("D1 source closure has no authoritative ledger")
    with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, stage, substage, route, method, status, command,
                   artifact_path, artifact_sha256
            FROM stages
            WHERE stage='P1' AND status='COMPLETE'
              AND artifact_path=? AND artifact_sha256=?
            ORDER BY id DESC
            """,
            (str(closure_path.resolve()), sha256_file(closure_path)),
        ).fetchall()
    if not rows:
        raise RuntimeError("D1 source closure has no matching COMPLETE P1 ledger row")
    row = dict(rows[0])
    if row.get("route") != "D1" or row.get("method") != "provenance":
        raise RuntimeError("D1 source closure ledger semantics differ")
    return row


def publish_source_closure(run_dir: str | Path, closure_path: str | Path) -> None:
    """Publish the active pointer only after its immutable P1 row is COMPLETE."""

    root = Path(run_dir).expanduser().resolve()
    source = Path(closure_path).expanduser().resolve()
    closure = _load_content(
        source, name="D1 immutable source closure", statuses=("PASS",)
    )
    closure_id = _closure_identity(closure)
    expected = (
        root / "00_audit" / "source_reconciliations" / closure_id / "manifest.json"
    ).resolve()
    if source != expected or closure.get("closure_id") != closure_id:
        raise RuntimeError("D1 source closure publication identity differs")
    ledger_binding = _completed_p1_ledger_binding(root, source)
    pointer: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "closure_id": closure_id,
        "source_closure": _artifact_record(source),
        "ledger_binding": ledger_binding,
        "candidate_test_labels_read": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(root / "00_audit" / SOURCE_CLOSURE_POINTER, pointer)
    atomic_json(
        root / "pipeline_status.json",
        {
            "schema_version": 1,
            "status": "AUDIT_COMPLETE",
            "first_incomplete_stage": "P2_CANDIDATES",
            "formal_test_executed": False,
            "test_candidate_labels_read": False,
        },
    )


def _artifact_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _load_content(
    path: Path, *, name: str, statuses: tuple[str, ...]
) -> dict[str, Any]:
    value = load_verified_json(path, name=name, statuses=statuses)
    unsigned = dict(value)
    expected = unsigned.pop("content_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash mismatch")
    return value


def _write_immutable_text(path: Path, text: str) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"D1 immutable source artifact is not a regular file: {path}"
            )
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"D1 immutable source artifact differs: {path}")
        return
    atomic_text(path, text)


def load_source_closure(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load the current immutable P1 source closure and re-hash every input."""

    root = Path(run_dir).expanduser().resolve()
    pointer_path = root / "00_audit" / SOURCE_CLOSURE_POINTER
    pointer = _load_content(
        pointer_path, name="D1 source closure pointer", statuses=("PASS",)
    )
    closure_path = verified_artifact_path(
        pointer.get("source_closure", {}), name="D1 immutable source closure"
    )
    closure = _load_content(
        closure_path, name="D1 immutable source closure", statuses=("PASS",)
    )
    closure_id = _closure_identity(closure)
    expected_path = (
        root / "00_audit" / "source_reconciliations" / closure_id / "manifest.json"
    ).resolve()
    if closure_path != expected_path:
        raise RuntimeError("D1 source closure path/identity differs")
    if (
        closure.get("canonical_snapshot") != "A"
        or closure.get("candidate_test_labels_read") is not False
        or closure.get("selection_used_test_metrics") is not False
        or closure.get("repo_root") != str(REPO_ROOT)
        or closure.get("closure_id") != closure_id
        or pointer.get("closure_id") != closure_id
        or pointer.get("source_closure") != _artifact_record(closure_path)
    ):
        raise RuntimeError("D1 source closure semantics differ")
    if pointer.get("ledger_binding") != _completed_p1_ledger_binding(
        root, closure_path
    ):
        raise RuntimeError("D1 source closure pointer/ledger binding differs")
    verified = closure.get("verified_artifacts")
    if not isinstance(verified, list) or not verified:
        raise RuntimeError("D1 source closure verified-artifact inventory is absent")
    named: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(verified):
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise RuntimeError(f"D1 source closure evidence {index} has no name")
        name = str(record["name"])
        if name in named:
            raise RuntimeError(f"D1 source closure evidence name is duplicated: {name}")
        verified_artifact_path(record, name=f"D1 source closure evidence {name}")
        named[name] = record
    expected_paths = _authoritative_named_paths()
    if set(named) != set(expected_paths):
        raise RuntimeError("D1 source closure named evidence inventory differs")
    for name, expected_path in expected_paths.items():
        record = named[name]
        if Path(str(record.get("path", ""))).resolve() != expected_path:
            raise RuntimeError(f"D1 source closure evidence path differs: {name}")
        expected_sha = EXPECTED_SOURCE_HASHES.get(name)
        if expected_sha is not None and record.get("sha256") != expected_sha:
            raise RuntimeError(f"D1 source closure evidence hash differs: {name}")
    canonical_paths = _expected_canonical_input_paths()
    for split in ("train", "validation", "test"):
        value = closure.get("canonical_inputs", {}).get(split)  # type: ignore[union-attr]
        if not isinstance(value, dict):
            raise RuntimeError(f"D1 source closure misses {split} canonical inputs")
        expected = canonical_paths[split]
        for key in ("candidate_sources", "score_sources"):
            records = value.get(key)
            if (
                not isinstance(records, list)
                or [
                    str(Path(str(record.get("path", ""))).resolve())
                    for record in records
                    if isinstance(record, dict)
                ]
                != expected[key]
            ):
                raise RuntimeError(f"D1 {split} source closure {key} paths differ")
        for key in (
            "paired_manifest",
            "native_features",
            "feature_allowlist",
            "feature_manifest",
        ):
            record = value.get(key)
            if (
                not isinstance(record, dict)
                or str(Path(str(record.get("path", ""))).resolve()) != expected[key]
            ):
                raise RuntimeError(f"D1 {split} source closure {key} path differs")
        if split in {"train", "validation"}:
            record = value.get("development_labels")
            if (
                not isinstance(record, dict)
                or str(Path(str(record.get("path", ""))).resolve())
                != expected["development_labels"]
            ):
                raise RuntimeError(
                    f"D1 {split} source closure development_labels path differs"
                )
        else:
            for key in ("opaque_ground_truth", "opaque_visual_ground_truth"):
                record = value.get(key)
                if (
                    not isinstance(record, dict)
                    or str(Path(str(record.get("path", ""))).resolve()) != expected[key]
                ):
                    raise RuntimeError(f"D1 Test source closure {key} path differs")
        records = [
            *value.get("candidate_sources", []),
            *value.get("score_sources", []),
            value.get("paired_manifest", {}),
            value.get("native_features", {}),
            value.get("feature_allowlist", {}),
            value.get("feature_manifest", {}),
        ]
        if split in {"train", "validation"}:
            records.append(value.get("development_labels", {}))
        else:
            records.extend(
                (
                    value.get("opaque_ground_truth", {}),
                    value.get("opaque_visual_ground_truth", {}),
                )
            )
        for index, record in enumerate(records):
            verified_artifact_path(
                record, name=f"D1 {split} source closure input {index}"
            )
    return closure_path, closure


def _verify(path: Path, expected: str, name: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"D1 source drift for {name}: observed={observed} expected={expected}"
        )
    return {
        "name": name,
        "path": str(path.resolve()),
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def _verify_existing(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"D1 provenance evidence is missing: {name}: {path}")
    return {
        "name": name,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _key_audit(
    candidate_path: Path,
    score_path: Path,
    paired_path: Path,
) -> dict[str, Any]:
    candidates = pd.read_parquet(candidate_path, columns=["sample_id", "candidate_id"])
    scores = pd.read_parquet(score_path, columns=["sample_id", "candidate_id"])
    paired = pd.read_parquet(paired_path, columns=["sample_id"])
    for name, frame in (("candidates", candidates), ("scores", scores)):
        if (
            frame.isna().any().any()
            or frame.duplicated(["sample_id", "candidate_id"]).any()
        ):
            raise RuntimeError(f"D1 {name} candidate keys are invalid")
    candidate_keys = set(map(tuple, candidates.astype(str).to_numpy()))
    score_keys = set(map(tuple, scores.astype(str).to_numpy()))
    if candidate_keys != score_keys:
        raise RuntimeError("D1 candidate/score exact keys differ")
    paired_ids = set(paired["sample_id"].astype(str))
    candidate_ids = set(candidates["sample_id"].astype(str))
    foreign = candidate_ids.difference(paired_ids)
    if foreign:
        raise RuntimeError(
            f"D1 candidates contain foreign paired samples: {len(foreign)}"
        )
    return {
        "candidate_keys": len(candidate_keys),
        "candidate_key_sha256": canonical_sha256(sorted(candidate_keys)),
        "paired_samples": len(paired_ids),
        "candidate_samples": len(candidate_ids),
        "no_output_samples": len(paired_ids.difference(candidate_ids)),
        "duplicate_candidate_keys": 0,
        "duplicate_score_keys": 0,
        "foreign_samples": 0,
    }


def _shard_inventory(
    source_root: Path, split: str, paired_path: Path
) -> dict[str, Any]:
    source_name = "train" if split == "train" else "val"
    candidate_paths = sorted(
        (source_root / "tmp" / source_name / "candidates").glob(
            "shard_*/nms_candidates.parquet"
        )
    )
    score_paths = sorted(
        (source_root / "tmp" / source_name / "gqcnn_score_shards").glob(
            "shard_*.parquet"
        )
    )
    manifest_paths = sorted(
        (source_root / "tmp" / source_name / "gqcnn_score_shards").glob(
            "shard_*.manifest.json"
        )
    )
    expected_shards = 8 if split == "train" else 4
    if not (
        len(candidate_paths)
        == len(score_paths)
        == len(manifest_paths)
        == expected_shards
    ):
        raise RuntimeError(f"D1 {split} shard inventory is incomplete")
    candidate_rows = sum(_parquet_rows(path) for path in candidate_paths)
    score_rows = sum(_parquet_rows(path) for path in score_paths)
    if candidate_rows != score_rows:
        raise RuntimeError(f"D1 {split} candidate/score row totals differ")
    records = []
    samples = 0
    empty_samples = 0
    protocol_families = set()
    model_contracts = set()
    candidate_sample_ids: set[str] = set()
    paired_ids = set(
        pd.read_parquet(paired_path, columns=["sample_id"])["sample_id"].astype(str)
    )
    for index, (candidate, score, manifest_path) in enumerate(
        zip(candidate_paths, score_paths, manifest_paths, strict=True)
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETED" or manifest.get("gt_free") is not True:
            raise RuntimeError(
                f"D1 {split} score shard is not completed and GT-free: {index}"
            )
        if manifest.get("gqcnn_scores_parquet_sha256") != sha256_file(score):
            raise RuntimeError(f"D1 {split} score shard hash mismatch: {index}")
        if int(manifest.get("rows", -1)) != _parquet_rows(score):
            raise RuntimeError(f"D1 {split} score shard row mismatch: {index}")
        samples += int(manifest.get("samples", 0))
        empty_samples += int(manifest.get("empty_samples", 0))
        protocol_families.add(
            str(manifest.get("candidate_protocol_family_identity_sha256"))
        )
        model = manifest.get("model", {})
        model_contracts.add(
            (
                str(model.get("model_commit")),
                str(model.get("model_config_hash")),
                str(model.get("model_file_manifest_hash")),
            )
        )
        key_audit = _key_audit(candidate, score, paired_path)
        candidate_sample_ids.update(
            pd.read_parquet(candidate, columns=["sample_id"])["sample_id"].astype(str)
        )
        run_config_path = candidate.parent / "run_config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        nms_record = run_config.get("candidate_stage_artifacts", {}).get("nms", {})
        candidate_sha = sha256_file(candidate)
        if nms_record.get("sha256") != candidate_sha or int(
            nms_record.get("rows", -1)
        ) != _parquet_rows(candidate):
            raise RuntimeError(
                f"D1 {split} candidate shard provenance mismatch: {index}"
            )
        records.append(
            {
                "shard": index,
                "candidate_path": str(candidate.resolve()),
                "score_path": str(score.resolve()),
                "candidate_sha256": candidate_sha,
                "score_sha256": sha256_file(score),
                "candidate_run_config_path": str(run_config_path.resolve()),
                "candidate_run_config_sha256": sha256_file(run_config_path),
                "legacy_candidate_run_manifest_sha256": manifest.get(
                    "candidate_run_manifest_sha256"
                ),
                "legacy_candidate_run_manifest_note": (
                    "legacy pre-finalization identity; current run_config and NMS bytes "
                    "are independently frozen"
                ),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "rows": _parquet_rows(score),
                "key_audit": key_audit,
            }
        )
    if len(protocol_families) != 1 or len(model_contracts) != 1:
        raise RuntimeError(
            f"D1 {split} shards do not share one protocol/model contract"
        )
    expected_samples = 26295 if split == "train" else 3778
    expected_rows = 655433 if split == "train" else 94845
    if samples != expected_samples or candidate_rows != expected_rows:
        raise RuntimeError(
            f"D1 {split} totals drift: samples={samples} rows={candidate_rows}"
        )
    if not candidate_sample_ids.issubset(paired_ids):
        raise RuntimeError(f"D1 {split} candidates contain foreign paired samples")
    observed_empty = len(paired_ids.difference(candidate_sample_ids))
    if observed_empty != empty_samples:
        raise RuntimeError(
            f"D1 {split} no-output count differs: {observed_empty} != {empty_samples}"
        )
    return {
        "split": split,
        "shards": expected_shards,
        "samples": samples,
        "candidate_rows": candidate_rows,
        "empty_samples": empty_samples,
        "paired_manifest": {
            "path": str(paired_path.resolve()),
            "sha256": sha256_file(paired_path),
        },
        "exact_candidate_score_keys": True,
        "paired_sample_membership_verified": True,
        "candidate_test_labels_read": False,
        "candidate_protocol_family_identity_sha256": next(iter(protocol_families)),
        "model_contract": list(next(iter(model_contracts))),
        "records": records,
    }


def reconcile_sources(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    snapshot_a: str | Path,
    compact_source: str | Path,
    unified_run: str | Path,
    snapshot_b: str | Path,
    snapshot_c: str | Path,
) -> dict[str, Any]:
    """Select Snapshot A by provenance and explicitly reject mixing A/B/C."""

    root = Path(repo_root).resolve()
    run = Path(run_dir).resolve()
    snapshot = Path(snapshot_a).resolve()
    compact = Path(compact_source).resolve()
    unified = Path(unified_run).resolve()
    if (
        root != REPO_ROOT
        or snapshot != SNAPSHOT_A_ROOT
        or compact != COMPACT_SOURCE_ROOT
        or unified != UNIFIED_RUN_ROOT
        or Path(snapshot_b).resolve() != SNAPSHOT_B_ROOT
        or Path(snapshot_c).resolve() != SNAPSHOT_C_ROOT
    ):
        raise RuntimeError(
            "D1 reconciliation source roots differ from the canonical registry"
        )
    verified = [
        _verify(
            snapshot / "run_manifest.json",
            EXPECTED_SOURCE_HASHES["snapshot_a_run_manifest"],
            "snapshot_a_run_manifest",
        ),
        _verify(
            snapshot / "frozen_protocol.yaml",
            EXPECTED_SOURCE_HASHES["snapshot_a_frozen_protocol"],
            "snapshot_a_frozen_protocol",
        ),
        _verify(
            snapshot / "final_output_manifest.json",
            EXPECTED_SOURCE_HASHES["snapshot_a_final_output_manifest"],
            "snapshot_a_final_output_manifest",
        ),
        _verify(
            snapshot / "COMPLETED",
            EXPECTED_SOURCE_HASHES["snapshot_a_completed"],
            "snapshot_a_completed",
        ),
        _verify(
            snapshot / "candidates/dexnet_nms_candidates.parquet",
            EXPECTED_SOURCE_HASHES["snapshot_a_test_nms"],
            "snapshot_a_test_nms",
        ),
        _verify(
            compact / "compact_inputs/test/gqcnn_scores.parquet",
            EXPECTED_SOURCE_HASHES["compact_test_scores"],
            "compact_test_scores",
        ),
        _verify(
            compact / "compact_inputs/test/candidate_tables.manifest.json",
            EXPECTED_SOURCE_HASHES["compact_test_candidate_manifest"],
            "compact_test_candidate_manifest",
        ),
        _verify(
            compact / "compact_inputs/test/gqcnn_scores.manifest.json",
            EXPECTED_SOURCE_HASHES["compact_test_score_manifest"],
            "compact_test_score_manifest",
        ),
        _verify(
            compact / "features/per_candidate.parquet.manifest.json",
            EXPECTED_SOURCE_HASHES["development_feature_registry"],
            "development_feature_registry",
        ),
        _verify(
            unified / "FINAL_RUN_LOCK.json",
            EXPECTED_SOURCE_HASHES["unified_final_lock"],
            "unified_final_lock",
        ),
        _verify(
            unified / "configs/canonical_evaluator.py",
            EXPECTED_SOURCE_HASHES["canonical_evaluator"],
            "canonical_evaluator",
        ),
        _verify(
            DEVELOPMENT_LABEL_ROOT / "train_labels.parquet",
            EXPECTED_SOURCE_HASHES["development_train_labels"],
            "development_train_labels",
        ),
        _verify(
            DEVELOPMENT_LABEL_ROOT / "validation_labels.parquet",
            EXPECTED_SOURCE_HASHES["development_validation_labels"],
            "development_validation_labels",
        ),
        _verify(
            DEVELOPMENT_LABEL_ROOT / "test_labels.parquet",
            EXPECTED_SOURCE_HASHES["opaque_test_ground_truth"],
            "opaque_test_ground_truth",
        ),
        _verify(
            TEST_VISUAL_GROUND_TRUTH,
            EXPECTED_SOURCE_HASHES["opaque_test_visual_ground_truth"],
            "opaque_test_visual_ground_truth",
        ),
        _verify_existing(Path(__file__), "d1_provenance_contract"),
    ]
    test_nms = snapshot / "candidates/dexnet_nms_candidates.parquet"
    test_scores = compact / "compact_inputs/test/gqcnn_scores.parquet"
    if _parquet_rows(test_nms) != 187077 or _parquet_rows(test_scores) != 187077:
        raise RuntimeError("Snapshot A Test candidate/score row count drift")
    train = _shard_inventory(
        compact, "train", unified / "01_manifests/paired_train.parquet"
    )
    validation = _shard_inventory(
        compact,
        "validation",
        unified / "01_manifests/paired_validation.parquet",
    )
    test_key_audit = _key_audit(
        test_nms,
        test_scores,
        unified / "01_manifests/paired_test.parquet",
    )
    test_score_manifest = json.loads(
        (compact / "compact_inputs/test/gqcnn_scores.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    test_model = test_score_manifest.get("model", {})
    test_model_contract = [
        str(test_model.get("model_commit")),
        str(test_model.get("model_config_hash")),
        str(test_model.get("model_file_manifest_hash")),
    ]
    if (
        train["model_contract"] != validation["model_contract"]
        or train["model_contract"] != test_model_contract
    ):
        raise RuntimeError("D1 Train/Validation/Test GQ-CNN model contracts differ")
    snapshot_b_root = Path(snapshot_b).resolve()
    b_documents = [
        snapshot_b_root / "DEXNET_FULL_HIFICS_GENERATION_RESULTS.md",
        snapshot_b_root / "GQCNN_FULL_HIFICS_SCORING_RESULTS.md",
        snapshot_b_root / "MODULAR_RERANKING_V1_RESULTS.md",
        snapshot_b_root / "MODULAR_RERANKING_V1_AUDIT.md",
    ]
    snapshot_c_root = Path(snapshot_c).resolve()
    c_evidence = [
        snapshot_c_root / "artifact_manifest_phase0.json",
        snapshot_c_root / "audit/ranking_contract_audit.json",
        snapshot_c_root / "audit/repeatedfilm_source_manifest.json",
        snapshot_c_root / "D1_order_sensitivity.csv",
    ]
    snapshot_b_records = [
        _verify_existing(path, f"snapshot_b_evidence_{index}")
        for index, path in enumerate(b_documents)
    ]
    snapshot_c_records = [
        _verify_existing(path, f"snapshot_c_evidence_{index}")
        for index, path in enumerate(c_evidence)
    ]
    verified.extend(snapshot_b_records)
    verified.extend(snapshot_c_records)
    feature_inputs: dict[str, dict[str, Any]] = {}
    for split, source_name in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
    ):
        feature_dir = compact / "features" / source_name
        feature_inputs[split] = {
            "native_features": _artifact_record(feature_dir / "per_candidate.parquet"),
            "feature_allowlist": _artifact_record(
                feature_dir / "inference_feature_allowlist.json"
            ),
            "feature_manifest": _artifact_record(feature_dir / "dataset_manifest.json"),
        }
        verified.extend(
            _verify_existing(path, f"{split}_{name}")
            for name, path in (
                ("native_features", feature_dir / "per_candidate.parquet"),
                (
                    "feature_allowlist",
                    feature_dir / "inference_feature_allowlist.json",
                ),
                ("feature_manifest", feature_dir / "dataset_manifest.json"),
            )
        )
    canonical_inputs = {
        split: {
            "candidate_sources": [
                {
                    "path": record["candidate_path"],
                    "sha256": record["candidate_sha256"],
                }
                for record in inventory["records"]
            ],
            "score_sources": [
                {
                    "path": record["score_path"],
                    "sha256": record["score_sha256"],
                }
                for record in inventory["records"]
            ],
            "paired_manifest": inventory["paired_manifest"],
        }
        for split, inventory in (("train", train), ("validation", validation))
    }
    canonical_inputs["test"] = {
        "candidate_sources": [_artifact_record(test_nms)],
        "score_sources": [_artifact_record(test_scores)],
        "paired_manifest": _artifact_record(
            unified / "01_manifests/paired_test.parquet"
        ),
        "opaque_ground_truth": _artifact_record(
            DEVELOPMENT_LABEL_ROOT / "test_labels.parquet"
        ),
        "opaque_visual_ground_truth": _artifact_record(TEST_VISUAL_GROUND_TRUTH),
    }
    for split in canonical_inputs:
        canonical_inputs[split].update(feature_inputs[split])
    for split in ("train", "validation"):
        canonical_inputs[split]["development_labels"] = _artifact_record(
            DEVELOPMENT_LABEL_ROOT / f"{split}_labels.parquet"
        )
    closure_id = _closure_identity(
        {
            "verified_artifacts": verified,
            "development": {"train": train, "validation": validation},
            "test": {
                "key_audit": test_key_audit,
                "model_contract": test_model_contract,
            },
            "canonical_inputs": canonical_inputs,
        }
    )
    closure_dir = run / "00_audit" / "source_reconciliations" / closure_id
    closure_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.DataFrame(
        [
            {
                "snapshot": "A",
                "run_path": str(snapshot),
                "physical_candidate_pool": True,
                "candidate_rows": 187077,
                "paired_samples": 7675,
                "no_output_samples": 108,
                "train_validation_pools": True,
                "current_evaluator_labels": False,
                "canonical_use": "candidate membership/geometry/native q and rank only",
                "decision": "SELECTED",
                "metric_status": "VERIFIED_FROM_PHYSICAL_ARTIFACTS",
                "reason": "only complete GT-free pool with exact paired universe, score and development provenance",
            },
            {
                "snapshot": "B",
                "run_path": "UNAVAILABLE_PHYSICAL_POOL",
                "evidence_root": str(snapshot_b_root),
                "physical_candidate_pool": False,
                "candidate_rows": 206538,
                "paired_samples": 7675,
                "no_output_samples": 55,
                "train_validation_pools": False,
                "current_evaluator_labels": False,
                "canonical_use": "provenance appendix only",
                "decision": "REJECTED",
                "metric_status": "DOCUMENTED_NOT_PHYSICALLY_VERIFIABLE",
                "reason": "documented pool is physically unavailable and upstream checkpoint/config differ",
            },
            {
                "snapshot": "C",
                "run_path": str(snapshot_c_root),
                "evidence_root": str(snapshot_c_root),
                "physical_candidate_pool": True,
                "candidate_rows": 187077,
                "paired_samples": 7675,
                "no_output_samples": 108,
                "train_validation_pools": False,
                "current_evaluator_labels": False,
                "canonical_use": "evaluator/geometry sensitivity appendix only",
                "decision": "REJECTED",
                "metric_status": "VERIFIED_MIGRATION_VIEW_NOT_CANDIDATE_SOURCE",
                "reason": "migration view over A; changed width/evaluator/tie semantics and is not a new pool",
            },
        ]
    )
    registry_path = closure_dir / "D1_RUN_REGISTRY.csv"
    _write_immutable_text(registry_path, registry.to_csv(index=False))
    hashes_path = closure_dir / "D1_ARTIFACT_HASHES.sha256"
    _write_immutable_text(
        hashes_path,
        "".join(f"{record['sha256']}  {record['path']}\n" for record in verified),
    )
    baseline = pd.DataFrame(
        [
            {
                "snapshot": name,
                "status": "NOT_RUN_PRELOCK",
                "reason": "Test outcomes are not used to select the canonical snapshot",
                "candidate_test_labels_read": False,
            }
            for name in ("A", "B", "C")
        ]
    )
    _write_immutable_text(
        closure_dir / "D1_BASELINE_RECOMPUTATION.csv",
        baseline.to_csv(index=False),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "closure_id": closure_id,
        "canonical_snapshot": "A",
        "selection_used_test_metrics": False,
        "candidate_test_labels_read": False,
        "repo_root": str(root),
        "verified_artifacts": verified,
        "development": {"train": train, "validation": validation},
        "test": {
            "samples": 7675,
            "candidate_rows": 187077,
            "no_output_samples": 108,
            "key_audit": test_key_audit,
            "model_contract": test_model_contract,
        },
        "snapshot_b_evidence": snapshot_b_records,
        "snapshot_c_evidence": snapshot_c_records,
        "cross_split_model_contract_equal": True,
        "canonical_inputs": canonical_inputs,
        "paired_inference_contract_limitation": {
            "train_validation": "5cc746f20db6025df3705fd93ab143dadfd848ed6bd527cb63cdc6fbb3f8ccd9",
            "test": "2d8428f784fdbc1d9d505f9f609c9109a7508fbdc356839fdb7437bca59a5639",
            "interpretation": (
                "paired inference-contract bytes differ across official split builders; "
                "checkpoint/config and candidate protocol are frozen and equal"
            ),
        },
        "geometry_contract": {
            "center": "source center_u_px/center_v_px",
            "angle": "degrees(source angle_rad), no adapter sign flip",
            "width": "source configured width_px",
            "height_px": 20.0,
            "contact_span": "sensitivity only",
        },
        "native_order": "full precision gqcnn_q_value descending; exact ties candidate_id ascending",
        "prohibited_mixing": [
            "Snapshot B candidates",
            "Snapshot C labels/order/geometry",
        ],
    }
    payload["content_sha256"] = canonical_sha256(payload)
    closure_path = closure_dir / "manifest.json"
    if closure_path.exists():
        existing = _load_content(
            closure_path, name="D1 immutable source closure", statuses=("PASS",)
        )
        if existing != payload:
            raise RuntimeError("D1 immutable source closure ID collision")
    else:
        atomic_json(closure_path, payload)
    report = f"""# D1 provenance reconciliation

## Decision

Snapshot **A** is selected for candidate membership, configured geometry,
full-precision GQ-CNN q and original rank. This decision is based on provenance,
not on historical Test accuracy. Snapshot B is physically incomplete and has a
different upstream contract. Snapshot C is an evaluator-migration view of A,
not an independent candidate pool.

## Verified inventory

- Test: 7,675 samples, 187,077 NMS candidates, 108 no-output samples.
- Train: {train["samples"]} samples, {train["candidate_rows"]} NMS candidates,
  {train["empty_samples"]} no-output samples, {train["shards"]} complete shards.
- Validation: {validation["samples"]} samples,
  {validation["candidate_rows"]} NMS candidates,
  {validation["empty_samples"]} no-output samples,
  {validation["shards"]} complete shards.
- Test candidate labels opened: no.

## Migration rule

Old outcome columns are not reused. Train/Validation labels will be regenerated
with the byte-frozen fair evaluator. Test ground truth remains unavailable until
the extension formal lock and exactly-once execution claim exist.
"""
    _write_immutable_text(closure_dir / "D1_PROVENANCE_RECONCILIATION.md", report)
    return payload
