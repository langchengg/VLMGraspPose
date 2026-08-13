"""Fail-closed P13 readiness assembly for the retrospective D1 extension.

This module deliberately has no formal-Test evaluator import.  It may inspect
label-free Test schemas and rows, but development candidate labels are the only
candidate-label tables it is permitted to open.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from unified_reranking.artifacts import (
    load_verified_json,
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import (
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.datasets import FoldPreprocessor
from unified_reranking.metrics import evaluate_order_only
from unified_reranking.training import FORMAL_SEEDS

from .candidates import (
    pool_hash_rows,
    verify_canonical_candidate_frame,
)
from .contracts import POOL_LIMITS, assert_label_free_parquet_schema
from .ablation_schema_adapter import validate_adapted_ablation_replay
from .execution import load_content_manifest
from .gate_validation import validate_gate_selection_semantics
from .four_route_test_inputs import validate_four_route_test_inputs
from .four_route_validation import validate_p12_prelock
from .feature_replay_adapter import validate_adapted_feature_execution_replay
from .k_formal_replay_adapter import validate_adapted_k_formal_replay
from .k_replay_adapter import validate_adapted_k_sensitivity_replay
from .primary_source_adapter import (
    load_primary_plan_with_source_adapter,
    verify_primary_cell_records_with_source_adapter,
    verify_records_with_primary_source_adapter,
)
from .provenance import load_source_closure
from .postformal_evidence import validate_postformal_evidence_prelock
from .lightweight_audits import load_lightweight_audits
from .run import transition_pipeline_status
from .selection import (
    ensemble_seed_scores,
    select_validation_winner,
    trial_configuration,
    trial_id,
)
from .splits import validate_fold_assignments


SPLITS = ("train", "validation", "test")
POOLS = tuple(POOL_LIMITS)
TRACKS = ("T1_native_available", "T2_matched_common", "T3_route_rich")
PRIMARY_POOL = "top5"
PRIMARY_TRACK = "T2_matched_common"
MANDATORY_TABLES = {
    "r0_r7_full": "07_validation/tables/r0_r7_full.csv",
    "selected_primary_ungated": "07_validation/tables/selected_primary_ungated.csv",
    "gate_operating_point": "07_validation/tables/gate_operating_point.csv",
    "evidence_track_ablation": "07_validation/tables/evidence_track_ablation.csv",
    "k_comparison": "11_k_sensitivity/k_comparison.csv",
    "feature_ablation": "12_feature_ablation/feature_ablation.csv",
    "four_route_validation": "13_four_route_extension/validation_router_union.csv",
}

REQUIRED_LABEL_FREE_TEST_STAGES = frozenset(
    {
        "d1_calibration_top5_test",
        "d1_calibration_top10_test",
        "d1_calibration_allnms_test",
        "d1_raw_feature_extraction",
        "d1_route_rich_test_features",
        "d1_selected_ranker_test_application",
        "d1_gate_test_application",
        "d1_k_scenario_test_ranker",
        "d1_k_scenario_test_gate",
        "d1_formal_input_normalization",
        "d1_p12_four_route_test_inputs",
        "d1_p12_four_route_test_application",
        "d1_read_only_reference_systems",
        "d1_r5_candidate_contributions",
        "d1_postformal_label_free_sources",
    }
)


def _regular_file(path: Path, *, name: str) -> Path:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{name} is missing or not a regular file: {source}")
    return source


def _record(path: Path) -> dict[str, Any]:
    source = _regular_file(path, name="D1 prelock source")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _same_record(observed: object, expected: Mapping[str, Any], *, name: str) -> None:
    if not isinstance(observed, Mapping):
        raise RuntimeError(f"{name} record is absent")
    for key in ("path", "sha256"):
        if observed.get(key) != expected.get(key):
            raise RuntimeError(f"{name} does not bind the current artifact")
    if (
        "bytes" in observed
        and "bytes" in expected
        and observed.get("bytes") != expected.get("bytes")
    ):
        raise RuntimeError(f"{name} byte count differs")


def _load_content(
    path: Path, *, name: str, statuses: tuple[str, ...]
) -> dict[str, Any]:
    return load_content_manifest(path, name=name, statuses=statuses)


def _verify_manifest_records(value: Mapping[str, Any], *, name: str) -> None:
    verify_artifact_records_recursive(
        value,
        name=name,
        require_at_least_one=True,
    )


def _require_label_free(value: Mapping[str, Any], *, name: str) -> None:
    if value.get("candidate_test_labels_read") is not False:
        raise RuntimeError(f"{name} does not declare candidate_test_labels_read=false")
    configuration = value.get("configuration")
    if (
        isinstance(configuration, Mapping)
        and configuration.get("candidate_test_labels_read", False) is not False
    ):
        raise RuntimeError(f"{name} configuration violates Test isolation")


def _candidate_manifest_path(root: Path, split: str) -> Path:
    if split == "test":
        return root / "02_candidates" / "test_manifest.json"
    return root / "02_candidates" / split / "manifest.json"


def _paired_path(root: Path, split: str) -> Path:
    filename = (
        "d1_paired_manifest.parquet"
        if split == "test"
        else f"d1_paired_{split}.parquet"
    )
    return root / "01_manifests" / filename


def _read_denominators(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    pairing_path = _regular_file(
        root / "01_manifests" / "d1_pairing_manifest.json",
        name="D1 pairing manifest",
    )
    pairing = load_verified_json(
        pairing_path, name="D1 pairing manifest", statuses=("PASS",)
    )
    if (
        pairing.get("same_denominator_and_assets_as_unified_run") is not True
        or pairing.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 pairing manifest semantics differ")
    frames: dict[str, pd.DataFrame] = {}
    records: dict[str, Any] = {}
    for split in SPLITS:
        path = _regular_file(_paired_path(root, split), name=f"D1 {split} denominator")
        if split == "test":
            assert_label_free_parquet_schema(path, name="Test paired denominator")
        frame = pd.read_parquet(path)
        required = {"sample_id", "scene_id"}
        if not required.issubset(frame.columns):
            raise RuntimeError(
                f"D1 {split} denominator misses {sorted(required - set(frame.columns))}"
            )
        ids = frame["sample_id"].astype(str)
        if ids.eq("").any() or ids.duplicated().any():
            raise RuntimeError(
                f"D1 {split} denominator sample IDs are not unique/non-empty"
            )
        frames[split] = frame
        records[split] = _record(path)
        _same_record(
            pairing.get("artifacts", {}).get(split),  # type: ignore[union-attr]
            records[split],
            name=f"D1 pairing {split}",
        )
    if any(
        set(frames[left]["sample_id"].astype(str)).intersection(
            frames[right]["sample_id"].astype(str)
        )
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise RuntimeError("D1 official split sample IDs overlap")
    return frames, {"pairing_manifest": _record(pairing_path), "paired": records}


def _assert_frame_keys(
    frame: pd.DataFrame, expected: pd.DataFrame, *, name: str
) -> None:
    columns = ["sample_id", "candidate_id"]
    if not set(columns).issubset(frame.columns):
        raise RuntimeError(f"{name} misses candidate key columns")
    if frame.duplicated(columns).any():
        raise RuntimeError(f"{name} contains duplicate candidate keys")
    observed_keys = set(map(tuple, frame[columns].astype(str).to_numpy()))
    expected_keys = set(map(tuple, expected[columns].astype(str).to_numpy()))
    if observed_keys != expected_keys or len(frame) != len(expected):
        raise RuntimeError(f"{name} candidate membership differs")


def _validate_candidate_split(
    root: Path,
    *,
    split: str,
    paired: pd.DataFrame,
    source_closure_record: Mapping[str, Any],
    source_paired_record: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    manifest_path = _candidate_manifest_path(root, split)
    manifest = _load_content(
        manifest_path, name=f"D1 {split} candidate manifest", statuses=("COMPLETE",)
    )
    _require_label_free(manifest, name=f"D1 {split} candidate manifest")
    verify_artifact_records_recursive(
        {
            "configuration": manifest.get("configuration"),
            "execution_contract": manifest.get("execution_contract"),
            "artifacts": manifest.get("artifacts"),
        },
        name=f"D1 {split} candidate closure",
        require_at_least_one=True,
    )
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping) or any(
        configuration.get(key) != value
        for key, value in {
            "route": "D1",
            "split": split,
            "pool_limits": POOL_LIMITS,
        }.items()
    ):
        raise RuntimeError(f"D1 {split} candidate configuration differs")
    source_contract = configuration.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise RuntimeError(f"D1 {split} source contract is absent")
    _same_record(
        source_contract.get("source_closure"),
        source_closure_record,
        name=f"D1 {split} source closure",
    )
    if (
        source_contract.get("snapshot") != "A"
        or source_contract.get("source_paths_from_closure_only") is not True
        or source_contract.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 {split} candidate source closure semantics differ")
    _same_record(
        configuration.get("paired_manifest"),
        source_paired_record,
        name=f"D1 {split} paired source",
    )
    if (
        source_paired_record.get("sha256")
        != _record(_paired_path(root, split))["sha256"]
    ):
        raise RuntimeError(f"D1 {split} local/source paired denominator bytes differ")

    pools: dict[str, pd.DataFrame] = {}
    artifacts = manifest.get("artifacts")
    summaries = manifest.get("summaries")
    if not isinstance(artifacts, Mapping) or not isinstance(summaries, Mapping):
        raise RuntimeError(f"D1 {split} candidate artifacts/summaries are absent")
    denominator_ids = paired["sample_id"].astype(str).tolist()
    for pool in POOLS:
        path = verified_artifact_path(
            artifacts.get(pool, {}), name=f"D1 {split}/{pool} candidates"
        )
        if split == "test":
            assert_label_free_parquet_schema(path, name=f"Test {pool} candidates")
        frame = pd.read_parquet(path)
        verify_canonical_candidate_frame(frame, split=split)
        foreign = set(frame["sample_id"].astype(str)).difference(denominator_ids)
        if foreign:
            raise RuntimeError(f"D1 {split}/{pool} contains foreign samples")
        limit = POOL_LIMITS[pool]
        if limit is not None and (pd.to_numeric(frame["native_rank"]) > limit).any():
            raise RuntimeError(f"D1 {split}/{pool} exceeds its native-rank limit")
        counts = (
            frame.groupby("sample_id").size().reindex(denominator_ids, fill_value=0)
        )
        expected_summary = {
            "rows": len(frame),
            "denominator_samples": len(denominator_ids),
            "samples_with_candidates": int((counts > 0).sum()),
            "no_output_samples": int((counts == 0).sum()),
            "maximum_candidates": int(counts.max()) if len(counts) else 0,
        }
        if summaries.get(pool) != expected_summary:
            raise RuntimeError(f"D1 {split}/{pool} denominator summary differs")
        pools[pool] = frame

    ordered = (
        pools["allnms"]
        .sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    for pool, limit in (("top5", 5), ("top10", 10)):
        expected = ordered.loc[ordered["native_rank"] <= limit].reset_index(drop=True)
        observed = (
            pools[pool]
            .sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        if not observed.equals(expected):
            raise RuntimeError(
                f"D1 {split}/{pool} is not the exact frozen AllNMS prefix"
            )

    hash_path = verified_artifact_path(
        artifacts.get("candidate_hashes", {}), name=f"D1 {split} candidate hashes"
    )
    if split == "test":
        assert_label_free_parquet_schema(
            hash_path, name="Test candidate contract hashes"
        )
    observed_hashes = (
        pd.read_parquet(hash_path)
        .sort_values(["pool", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected_hashes = (
        pool_hash_rows(pools, paired, split=split)
        .sort_values(["pool", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(
            observed_hashes.loc[:, expected_hashes.columns],
            expected_hashes,
            check_dtype=False,
            check_like=False,
        )
    except (AssertionError, KeyError) as error:
        raise RuntimeError(
            f"D1 {split} membership/geometry/native-score hashes differ"
        ) from error
    return (
        pools,
        manifest,
        {
            "manifest": _record(manifest_path),
            "candidate_hashes": _record(hash_path),
            "pools": {
                pool: _record(verified_artifact_path(artifacts[pool], name=pool))
                for pool in POOLS
            },
        },
    )


def _validate_candidate_registry(
    root: Path, manifests: Mapping[str, Mapping[str, Any]], records: Mapping[str, Any]
) -> dict[str, Any]:
    path = root / "02_candidates" / "d1_candidate_manifest.json"
    registry = _load_content(path, name="D1 candidate registry", statuses=("COMPLETE",))
    _require_label_free(registry, name="D1 candidate registry")
    if registry.get("required_splits") != list(SPLITS):
        raise RuntimeError("D1 candidate registry split inventory differs")
    observed = registry.get("split_manifests")
    if not isinstance(observed, Mapping):
        raise RuntimeError("D1 candidate registry has no split manifests")
    for split in SPLITS:
        expected = {
            "path": records[split]["manifest"]["path"],
            "sha256": records[split]["manifest"]["sha256"],
            "content_sha256": manifests[split]["content_sha256"],
            "summaries": manifests[split]["summaries"],
        }
        if observed.get(split) != expected:
            raise RuntimeError(f"D1 candidate registry {split} binding differs")
    return _record(path)


def _feature_artifact(manifest: Mapping[str, Any], track: str, *, name: str) -> Path:
    if track == "T1_native_available":
        return verified_artifact_path(manifest.get("artifact", {}), name=name)
    return verified_artifact_path(
        manifest.get("artifacts", {}).get("candidate_features", {}),  # type: ignore[union-attr]
        name=name,
    )


def _validate_calibrations(
    root: Path, candidate_records: Mapping[str, Any]
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    fold_record = _record(root / "04_splits" / "fold_assignments.parquet")
    for pool in POOLS:
        path = root / "05_calibration" / pool / "calibration_manifest.json"
        value = _load_content(
            path, name=f"D1 {pool} calibration", statuses=("COMPLETE",)
        )
        _require_label_free(value, name=f"D1 {pool} calibration")
        _verify_manifest_records(value, name=f"D1 {pool} calibration")
        configuration = value.get("configuration")
        if not isinstance(configuration, Mapping) or any(
            configuration.get(key) != expected
            for key, expected in {"route": "D1", "pool": pool}.items()
        ):
            raise RuntimeError(f"D1 {pool} calibration configuration differs")
        _same_record(
            value.get("sources", {}).get("fold_assignments"),
            fold_record,
            name=f"D1 {pool} calibration folds",
        )  # type: ignore[union-attr]
        for split in ("train", "validation"):
            source = value.get("sources", {}).get(split)  # type: ignore[union-attr]
            if not isinstance(source, Mapping):
                raise RuntimeError(f"D1 {pool} calibration misses {split} source")
            _same_record(
                source.get("candidate_manifest"),
                candidate_records[split]["manifest"],
                name=f"D1 {pool} calibration {split} candidates manifest",
            )
            _same_record(
                source.get("candidates"),
                candidate_records[split]["pools"][pool],
                name=f"D1 {pool} calibration {split} candidates",
            )
        if (
            not value.get("selected_method")
            or value.get("validation_denominator", 0) <= 0
        ):
            raise RuntimeError(f"D1 {pool} calibration selection is incomplete")
        application_path = (
            root / "05_calibration" / pool / "test_application_manifest.json"
        )
        application = _load_content(
            application_path,
            name=f"D1 {pool} Test calibration application",
            statuses=("COMPLETE",),
        )
        _require_label_free(application, name=f"D1 {pool} Test calibration application")
        _verify_manifest_records(
            application, name=f"D1 {pool} Test calibration application"
        )
        _same_record(
            application.get("sources", {}).get("calibration_manifest"),
            _record(path),
            name=f"D1 {pool} Test calibration source",
        )  # type: ignore[union-attr]
        _same_record(
            application.get("sources", {}).get("candidate_manifest"),
            candidate_records["test"]["manifest"],
            name=f"D1 {pool} Test calibration candidates manifest",
        )  # type: ignore[union-attr]
        _same_record(
            application.get("sources", {}).get("candidates"),
            candidate_records["test"]["pools"][pool],
            name=f"D1 {pool} Test calibration candidates",
        )  # type: ignore[union-attr]
        predictions = verified_artifact_path(
            application.get("artifacts", {}).get("test_predictions", {}),  # type: ignore[union-attr]
            name=f"D1 {pool} Test calibration predictions",
        )
        assert_label_free_parquet_schema(
            predictions, name=f"Test {pool} calibration predictions"
        )
        records[pool] = {
            "calibration": _record(path),
            "test_application": _record(application_path),
            "selected_method": str(value["selected_method"]),
        }
    return records


def _validate_features(
    root: Path,
    *,
    candidate_frames: Mapping[str, Mapping[str, pd.DataFrame]],
    candidate_records: Mapping[str, Any],
    closure_record: Mapping[str, Any],
    calibration_records: Mapping[str, Any],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for split in SPLITS:
        records[split] = {}
        for pool in POOLS:
            records[split][pool] = {}
            current: dict[str, tuple[Path, dict[str, Any]]] = {}
            for track in TRACKS:
                path = root / "03_features" / split / pool / track / "manifest.json"
                value = _load_content(
                    path, name=f"D1 {split}/{pool}/{track}", statuses=("COMPLETE",)
                )
                _require_label_free(value, name=f"D1 {split}/{pool}/{track}")
                verify_artifact_records_recursive(
                    value,
                    name=f"D1 {split}/{pool}/{track}",
                    require_at_least_one=True,
                )
                artifact = _feature_artifact(
                    value, track, name=f"D1 {split}/{pool}/{track} features"
                )
                if split == "test":
                    assert_label_free_parquet_schema(
                        artifact, name=f"Test {pool}/{track} features"
                    )
                feature_keys = pd.read_parquet(
                    artifact, columns=["sample_id", "candidate_id"]
                )
                _assert_frame_keys(
                    feature_keys,
                    candidate_frames[split][pool],
                    name=f"D1 {split}/{pool}/{track}",
                )
                candidate_manifest_record = candidate_records[split]["manifest"]
                candidate_pool_record = candidate_records[split]["pools"][pool]
                if track == "T1_native_available":
                    configuration = value.get("configuration")
                    if not isinstance(configuration, Mapping):
                        raise RuntimeError(
                            f"D1 {split}/{pool} T1 configuration is absent"
                        )
                    _same_record(
                        configuration.get("candidate_manifest"),
                        candidate_manifest_record,
                        name=f"D1 {split}/{pool} T1 candidate manifest",
                    )
                    _same_record(
                        configuration.get("candidates"),
                        candidate_pool_record,
                        name=f"D1 {split}/{pool} T1 candidates",
                    )
                    _same_record(
                        configuration.get("source_closure"),
                        closure_record,
                        name=f"D1 {split}/{pool} T1 source closure",
                    )
                elif track == "T2_matched_common":
                    sources = value.get("sources")
                    if not isinstance(sources, Mapping):
                        raise RuntimeError(f"D1 {split}/{pool} T2 sources are absent")
                    _same_record(
                        sources.get("candidate_manifest"),
                        candidate_manifest_record,
                        name=f"D1 {split}/{pool} T2 candidate manifest",
                    )
                    _same_record(
                        sources.get("candidates"),
                        candidate_pool_record,
                        name=f"D1 {split}/{pool} T2 candidates",
                    )
                    _same_record(
                        sources.get("calibration_manifest"),
                        calibration_records[pool]["calibration"],
                        name=f"D1 {split}/{pool} T2 calibration",
                    )
                else:
                    sources = value.get("sources")
                    if not isinstance(sources, Mapping):
                        raise RuntimeError(f"D1 {split}/{pool} T3 sources are absent")
                    _same_record(
                        sources.get("T1_manifest"),
                        _record(current["T1_native_available"][0]),
                        name=f"D1 {split}/{pool} T3 T1",
                    )
                    _same_record(
                        sources.get("T2_manifest"),
                        _record(current["T2_matched_common"][0]),
                        name=f"D1 {split}/{pool} T3 T2",
                    )
                    _same_record(
                        sources.get("candidate_manifest"),
                        candidate_manifest_record,
                        name=f"D1 {split}/{pool} T3 candidate manifest",
                    )
                    _same_record(
                        sources.get("candidates"),
                        candidate_pool_record,
                        name=f"D1 {split}/{pool} T3 candidates",
                    )
                current[track] = (path, value)
                records[split][pool][track] = {
                    "manifest": _record(path),
                    "features": _record(artifact),
                }
    return records


def _validate_splits(root: Path, paired: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    path = root / "04_splits" / "split_leakage_audit.json"
    value = _load_content(
        path,
        name="D1 split leakage audit",
        statuses=("PASS_WITH_SEQUENCE_OVERLAP_LIMITATION",),
    )
    _require_label_free(value, name="D1 split leakage audit")
    folds_path = verified_artifact_path(
        value.get("fold_assignments", {}), name="D1 fold assignments"
    )
    summary = validate_fold_assignments(pd.read_parquet(folds_path), paired["train"])
    if value.get("fold_summary") != summary:
        raise RuntimeError("D1 fold summary differs from current assignments")
    for split in SPLITS:
        expected = {
            "rows": len(paired[split]),
            "sample_identity_sha256": canonical_sha256(
                sorted(paired[split]["sample_id"].astype(str))
            ),
        }
        if value.get("paired_manifests", {}).get(split) != expected:  # type: ignore[union-attr]
            raise RuntimeError(f"D1 split audit {split} denominator differs")
    return {"audit": _record(path), "fold_assignments": _record(folds_path)}


def _metric(value: Mapping[str, Any], name: str, *, context: str) -> float:
    try:
        result = float(value[name])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{context} misses numeric {name}") from error
    if not 0.0 <= result <= 1.0:
        raise RuntimeError(f"{context} {name} is outside [0,1]")
    return result


def _light_record(path: Path) -> dict[str, str]:
    record = _record(path)
    return {"path": str(record["path"]), "sha256": str(record["sha256"])}


def _assert_frame_exact(
    observed_path: Path, expected: pd.DataFrame, *, name: str
) -> None:
    observed = pd.read_parquet(observed_path)
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as error:
        raise RuntimeError(f"{name} differs from semantic replay") from error


def _primary_labels_and_denominator(
    root: Path, split: str
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    manifest_path = root / "03_features" / split / "top5" / "labels" / "manifest.json"
    manifest = _load_content(
        manifest_path, name=f"D1 {split} labels", statuses=("COMPLETE",)
    )
    label_path = verified_artifact_path(
        manifest.get("artifact", {}), name=f"D1 {split} labels"
    )
    denominator_path = _paired_path(root, split)
    labels = pd.read_parquet(
        label_path, columns=["sample_id", "candidate_id", "candidate_success"]
    )
    denominator = (
        pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
        .astype(str)
        .tolist()
    )
    return (
        labels,
        denominator,
        {
            "label_manifest": _light_record(manifest_path),
            "labels": _light_record(label_path),
            "denominator": _light_record(denominator_path),
        },
    )


def _evaluate_primary_ensemble(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    denominator: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    evaluation = predictions.merge(
        labels,
        on=["sample_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(evaluation) != len(predictions) or len(evaluation) != len(labels):
        raise RuntimeError("D1 primary ensemble/label membership differs")
    return evaluate_order_only(
        denominator, evaluation, score_column="ensemble_score", max_k=5
    )


def _replay_primary_selection(
    root: Path,
    *,
    planned: Mapping[str, Mapping[str, Any]],
    cell_paths: Mapping[str, Path],
    cell_predictions: Mapping[str, pd.DataFrame],
    selection: Mapping[str, Any],
    plan_path: Path,
    execution_path: Path,
) -> None:
    train_labels, train_denominator, train_sources = _primary_labels_and_denominator(
        root, "train"
    )
    validation_labels, validation_denominator, validation_sources = (
        _primary_labels_and_denominator(root, "validation")
    )
    grouped: dict[str, list[str]] = {}
    for job_id, configuration in planned.items():
        grouped.setdefault(trial_id(configuration), []).append(job_id)
    if len(grouped) != 20 or any(len(job_ids) != 18 for job_ids in grouped.values()):
        raise RuntimeError("D1 primary replay trial/job inventory differs")

    table_rows: list[dict[str, object]] = []
    trial_payloads: dict[str, dict[str, Any]] = {}
    for identifier, job_ids in sorted(grouped.items()):
        representative = trial_configuration(planned[job_ids[0]])
        seed_validation: dict[int, pd.DataFrame] = {}
        seed_oof_parts: dict[int, list[pd.DataFrame]] = {
            seed: [] for seed in FORMAL_SEEDS
        }
        cell_records: list[dict[str, str]] = []
        for job_id in job_ids:
            configuration = planned[job_id]
            if trial_configuration(configuration) != representative:
                raise RuntimeError(
                    f"D1 primary trial {identifier} mixes configurations"
                )
            seed = int(configuration["seed"])
            frame = cell_predictions[job_id]
            if configuration["mode"] == "validation":
                if seed in seed_validation:
                    raise RuntimeError(
                        f"D1 primary trial {identifier} duplicates Validation seed"
                    )
                seed_validation[seed] = frame
            else:
                seed_oof_parts[seed].append(frame)
            cell_records.append(_light_record(cell_paths[job_id]))
        if set(seed_validation) != set(FORMAL_SEEDS) or any(
            len(parts) != 5 for parts in seed_oof_parts.values()
        ):
            raise RuntimeError(
                f"D1 primary trial {identifier} seed/fold inventory differs"
            )
        validation = ensemble_seed_scores(seed_validation)
        oof = ensemble_seed_scores(
            {
                seed: pd.concat(parts, ignore_index=True)
                for seed, parts in seed_oof_parts.items()
            }
        )
        validation_metrics, validation_decisions = _evaluate_primary_ensemble(
            validation, validation_labels, validation_denominator
        )
        oof_metrics, oof_decisions = _evaluate_primary_ensemble(
            oof, train_labels, train_denominator
        )
        trial_manifest_path = (
            root
            / "07_validation"
            / "primary_selection"
            / "trials"
            / identifier
            / "manifest.json"
        )
        trial = _load_content(
            trial_manifest_path,
            name=f"D1 primary trial {identifier}",
            statuses=("COMPLETE",),
        )
        _verify_manifest_records(trial, name=f"D1 primary trial {identifier}")
        if (
            trial.get("trial_id") != identifier
            or trial.get("configuration") != representative
            or trial.get("validation_metrics") != validation_metrics
            or trial.get("oof_metrics") != oof_metrics
            or trial.get("sources", {}).get("cells")  # type: ignore[union-attr]
            != sorted(cell_records, key=lambda row: row["path"])
            or trial.get("sources", {}).get("train_labels") != train_sources  # type: ignore[union-attr]
            or trial.get("sources", {}).get("validation_labels")  # type: ignore[union-attr]
            != validation_sources
        ):
            raise RuntimeError(f"D1 primary trial {identifier} semantic replay differs")
        artifacts = trial.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(f"D1 primary trial {identifier} artifacts are absent")
        _assert_frame_exact(
            verified_artifact_path(
                artifacts.get("validation_predictions", {}),
                name=f"D1 primary trial {identifier} Validation predictions",
            ),
            validation,
            name=f"D1 primary trial {identifier} Validation predictions",
        )
        _assert_frame_exact(
            verified_artifact_path(
                artifacts.get("validation_decisions", {}),
                name=f"D1 primary trial {identifier} Validation decisions",
            ),
            validation_decisions,
            name=f"D1 primary trial {identifier} Validation decisions",
        )
        _assert_frame_exact(
            verified_artifact_path(
                artifacts.get("oof_predictions", {}),
                name=f"D1 primary trial {identifier} OOF predictions",
            ),
            oof,
            name=f"D1 primary trial {identifier} OOF predictions",
        )
        _assert_frame_exact(
            verified_artifact_path(
                artifacts.get("oof_decisions", {}),
                name=f"D1 primary trial {identifier} OOF decisions",
            ),
            oof_decisions,
            name=f"D1 primary trial {identifier} OOF decisions",
        )
        trial_payloads[identifier] = trial
        table_rows.append(
            {
                "trial_id": identifier,
                "method": representative["method"],
                "configuration_sha256": canonical_sha256(representative),
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
                **{f"oof_{key}": value for key, value in oof_metrics.items()},
                "manifest_path": str(trial_manifest_path),
                "manifest_sha256": sha256_file(trial_manifest_path),
            }
        )

    method_winners = {
        method: dict(
            select_validation_winner(
                [row for row in table_rows if row["method"] == method],
                final_tie_column="trial_id",
            )
        )
        for method in ("R2", "R3", "R4", "R5", "R6")
    }
    primary_row = dict(
        select_validation_winner(
            [method_winners[method] for method in ("R3", "R5", "R6")],
            final_tie_column="method",
        )
    )
    winner = trial_payloads[str(primary_row["trial_id"])]
    expected_sources = {
        "plan": _light_record(plan_path),
        "execution": _light_record(execution_path),
        "train_labels": train_sources,
        "validation_labels": validation_sources,
        "selector": _light_record(
            Path(__file__).resolve().parents[2]
            / "tools/d1_reranking/select_primary_ranker.py"
        ),
    }
    trial_table_path = root / "07_validation" / "tables" / "r2_r6_trials.csv"
    selected_table_path = (
        root / "07_validation" / "tables" / "selected_primary_ungated.csv"
    )
    if trial_table_path.read_text(encoding="utf-8") != pd.DataFrame(table_rows).to_csv(
        index=False
    ):
        raise RuntimeError("D1 primary trial table differs from semantic replay")
    if selected_table_path.read_text(encoding="utf-8") != pd.DataFrame(
        [method_winners[method] for method in method_winners]
    ).to_csv(index=False):
        raise RuntimeError("D1 selected-method table differs from semantic replay")
    if (
        selection.get("source_signature_sha256") != canonical_sha256(expected_sources)
        or selection.get("method_winners") != method_winners
        or selection.get("selected_method") != primary_row["method"]
        or selection.get("selected_trial_id") != primary_row["trial_id"]
        or selection.get("selected_configuration") != winner["configuration"]
        or selection.get("validation_metrics") != winner["validation_metrics"]
        or selection.get("oof_metrics") != winner["oof_metrics"]
        or selection.get("sources") != expected_sources
    ):
        raise RuntimeError("D1 selected primary winner differs from semantic replay")
    artifacts = selection.get("artifacts")
    if not isinstance(artifacts, Mapping) or any(
        artifacts.get(name) != winner["artifacts"][winner_name]
        for name, winner_name in (
            ("selected_validation_predictions", "validation_predictions"),
            ("selected_validation_decisions", "validation_decisions"),
            ("selected_oof_predictions", "oof_predictions"),
            ("selected_oof_decisions", "oof_decisions"),
        )
    ):
        raise RuntimeError("D1 selected primary artifacts differ from winning trial")
    if artifacts.get("trial_table") != _light_record(trial_table_path) or artifacts.get(
        "selected_method_table"
    ) != _light_record(selected_table_path):
        raise RuntimeError("D1 selected primary table records differ")
    if artifacts.get("selected_trial_manifest") != _light_record(
        root
        / "07_validation"
        / "primary_selection"
        / "trials"
        / str(primary_row["trial_id"])
        / "manifest.json"
    ):
        raise RuntimeError("D1 selected primary manifest differs from winning trial")


def _development_oracles(
    root: Path,
    *,
    paired: Mapping[str, pd.DataFrame],
    candidates: Mapping[str, Mapping[str, pd.DataFrame]],
    candidate_records: Mapping[str, Any],
    closure: Mapping[str, Any],
    closure_record: Mapping[str, Any],
    evaluator_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    values: dict[str, Any] = {"train": {}, "validation": {}}
    records: dict[str, Any] = {"train": {}, "validation": {}}
    for split in ("train", "validation"):
        denominator = set(paired[split]["sample_id"].astype(str))
        for pool in POOLS:
            manifest_path = (
                root / "03_features" / split / pool / "labels" / "manifest.json"
            )
            manifest = _load_content(
                manifest_path, name=f"D1 {split}/{pool} labels", statuses=("COMPLETE",)
            )
            _require_label_free(manifest, name=f"D1 {split}/{pool} development labels")
            sources = manifest.get("sources")
            if not isinstance(sources, Mapping):
                raise RuntimeError(
                    f"D1 {split}/{pool} development-label sources are absent"
                )
            closure_inputs = closure.get("canonical_inputs", {}).get(split)  # type: ignore[union-attr]
            if not isinstance(closure_inputs, Mapping):
                raise RuntimeError(f"D1 source closure misses {split} canonical inputs")
            source_labels = closure_inputs.get("development_labels")
            if not isinstance(source_labels, Mapping):
                raise RuntimeError(
                    f"D1 source closure misses {split} development labels"
                )
            _same_record(
                sources.get("source_closure"),
                closure_record,
                name=f"D1 {split}/{pool} label source closure",
            )
            _same_record(
                sources.get("candidate_manifest"),
                candidate_records[split]["manifest"],
                name=f"D1 {split}/{pool} label candidate manifest",
            )
            _same_record(
                sources.get("candidate_hashes"),
                candidate_records[split]["candidate_hashes"],
                name=f"D1 {split}/{pool} label candidate hashes",
            )
            _same_record(
                sources.get("candidates"),
                candidate_records[split]["pools"][pool],
                name=f"D1 {split}/{pool} label candidates",
            )
            _same_record(
                sources.get("sample_labels"),
                source_labels,
                name=f"D1 {split}/{pool} immutable development labels",
            )
            _same_record(
                sources.get("evaluator"),
                evaluator_record,
                name=f"D1 {split}/{pool} canonical evaluator",
            )
            verify_artifact_records_recursive(
                sources,
                name=f"D1 {split}/{pool} development-label sources",
                require_at_least_one=True,
            )
            label_path = verified_artifact_path(
                manifest.get("artifact", {}), name=f"D1 {split}/{pool} labels"
            )
            labels = pd.read_parquet(
                label_path, columns=["sample_id", "candidate_id", "candidate_success"]
            )
            _assert_frame_keys(
                labels, candidates[split][pool], name=f"D1 {split}/{pool} labels"
            )
            positives = set(
                labels.loc[
                    labels["candidate_success"].astype(bool), "sample_id"
                ].astype(str)
            )
            foreign = positives.difference(denominator)
            if foreign:
                raise RuntimeError(f"D1 {split}/{pool} labels contain foreign samples")
            numerator = len(positives)
            values[split][pool] = {
                "numerator": numerator,
                "denominator": len(denominator),
                "oracle": numerator / len(denominator) if denominator else 0.0,
            }
            records[split][pool] = {
                "manifest": _record(manifest_path),
                "labels": _record(label_path),
            }
        ordered = [values[split][pool]["oracle"] for pool in POOLS]
        if not (ordered[0] <= ordered[1] <= ordered[2]):
            raise RuntimeError(f"D1 {split} Top5/Top10/AllNMS oracle is not monotone")
    return values, records


def _validate_primary(
    root: Path,
    *,
    oracles: Mapping[str, Any],
    splits_record: Mapping[str, Any],
    calibration_records: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    (
        plan_path,
        plan,
        primary_source_adapter,
        primary_source_adapter_value,
    ) = load_primary_plan_with_source_adapter(root)
    if (
        plan.get("route") != "D1"
        or plan.get("primary_contract")
        != {"pool": PRIMARY_POOL, "track": PRIMARY_TRACK}
        or plan.get("formal_seeds") != list(FORMAL_SEEDS)
        or plan.get("outer_folds") != 5
        or plan.get("fold_local_preprocessing") is not True
        or plan.get("fold_local_calibration", {}).get("required") is not True  # type: ignore[union-attr]
        or plan.get("fold_local_calibration", {}).get(
            "global_train_oof_base_logit_for_ranker_cells"
        )
        is not False  # type: ignore[union-attr]
        or plan.get("job_count") != 360
    ):
        raise RuntimeError("D1 primary plan semantics differ")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 360:
        raise RuntimeError("D1 primary plan must contain exactly 360 jobs")
    planned = {
        str(job.get("job_id")): job.get("configuration")
        for job in jobs
        if isinstance(job, Mapping) and isinstance(job.get("configuration"), Mapping)
    }
    if len(planned) != 360:
        raise RuntimeError("D1 primary plan job IDs/configurations are incomplete")
    method_counts: Counter[str] = Counter()
    for job_id, configuration in planned.items():
        method = str(configuration.get("method", ""))
        mode = str(configuration.get("mode", ""))
        held_fold = configuration.get("held_fold")
        if (
            method not in {"R2", "R3", "R4", "R5", "R6"}
            or configuration.get("route") != "D1"
            or configuration.get("pool") != PRIMARY_POOL
            or configuration.get("track") != PRIMARY_TRACK
            or configuration.get("seed") not in FORMAL_SEEDS
            or mode not in {"oof", "validation"}
            or (mode == "oof")
            != (held_fold in range(5) if isinstance(held_fold, int) else False)
        ):
            raise RuntimeError(f"D1 primary planned job {job_id} semantics differ")
        method_counts[method] += 1
    if method_counts != Counter(
        {method: 72 for method in ("R2", "R3", "R4", "R5", "R6")}
    ):
        raise RuntimeError(
            "D1 primary plan method budgets are not equal 72-cell budgets"
        )

    execution_path = root / "07_validation" / "primary_matrix_execution.json"
    execution = _load_content(
        execution_path, name="D1 primary matrix execution", statuses=("COMPLETE",)
    )
    _same_record(
        execution.get("plan"), _record(plan_path), name="D1 primary execution plan"
    )
    outputs = execution.get("outputs")
    if (
        not isinstance(outputs, Mapping)
        or set(outputs) != set(planned)
        or execution.get("expected_jobs") != 360
        or execution.get("completed_jobs") != 360
    ):
        raise RuntimeError("D1 primary execution inventory differs")
    fold_record = splits_record["fold_assignments"]
    calibration_record = calibration_records[PRIMARY_POOL]["calibration"]
    cell_paths: dict[str, Path] = {}
    cell_predictions: dict[str, pd.DataFrame] = {}
    for job_id, record in outputs.items():
        cell_path = verified_artifact_path(record, name=f"D1 primary cell {job_id}")
        cell = _load_content(
            cell_path, name=f"D1 primary cell {job_id}", statuses=("COMPLETE",)
        )
        verify_primary_cell_records_with_source_adapter(
            cell,
            adapter=primary_source_adapter_value,
            name=f"D1 primary cell {job_id}",
        )
        configuration = cell.get("configuration")
        if not isinstance(configuration, Mapping):
            raise RuntimeError(f"D1 primary cell {job_id} configuration is absent")
        if (
            configuration.get("planned_job_id") != job_id
            or configuration.get("planned_configuration") != planned[job_id]
        ):
            raise RuntimeError(f"D1 primary cell {job_id} differs from plan")
        calibrator = configuration.get("fold_local_calibrator")
        if not isinstance(calibrator, Mapping):
            raise RuntimeError(f"D1 primary cell {job_id} lacks fold-local calibrator")
        unsigned_calibrator = dict(calibrator)
        calibrator_content = unsigned_calibrator.pop("content_sha256", None)
        planned_configuration = planned[job_id]
        held_fold = planned_configuration.get("held_fold")
        early_fold = (int(held_fold) + 1) % 5 if held_fold is not None else 0
        excluded = {early_fold} | ({int(held_fold)} if held_fold is not None else set())
        expected_fit_folds = [fold for fold in range(5) if fold not in excluded]
        recorded_fit_folds = configuration.get("fit_fold_ids")
        try:
            normalized_fit_folds = (
                [int(value) for value in recorded_fit_folds]
                if isinstance(recorded_fit_folds, list)
                else []
            )
        except (TypeError, ValueError):
            normalized_fit_folds = []
        if (
            calibrator_content != canonical_sha256(unsigned_calibrator)
            or calibrator.get("method")
            != calibration_records[PRIMARY_POOL]["selected_method"]
            or calibrator.get("fit_fold_ids") != expected_fit_folds
            or normalized_fit_folds != expected_fit_folds
            or configuration.get("early_stop_fold") != early_fold
            or int(calibrator.get("fit_sample_count", 0)) <= 0
            or int(calibrator.get("fit_candidate_count", 0)) <= 0
            or not isinstance(calibrator.get("model"), Mapping)
        ):
            raise RuntimeError(
                f"D1 primary cell {job_id} fold-local calibration differs"
            )
        preprocessor = cell.get("preprocessor")
        if not isinstance(preprocessor, Mapping):
            raise RuntimeError(
                f"D1 primary cell {job_id} lacks fold-local preprocessor"
            )
        try:
            FoldPreprocessor.from_artifact(dict(preprocessor))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"D1 primary cell {job_id} preprocessor is invalid"
            ) from error
        preprocessor_path = verified_artifact_path(
            cell.get("artifacts", {}).get("preprocessor", {}),  # type: ignore[union-attr]
            name=f"D1 primary cell {job_id} preprocessor",
        )
        try:
            persisted_preprocessor = json.loads(
                preprocessor_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"D1 primary cell {job_id} preprocessor JSON is invalid"
            ) from error
        if persisted_preprocessor != preprocessor:
            raise RuntimeError(
                f"D1 primary cell {job_id} preprocessor manifest/file differ"
            )
        _same_record(
            cell.get("sources", {}).get("folds"),
            fold_record,
            name=f"D1 primary cell {job_id} folds",
        )  # type: ignore[union-attr]
        _same_record(
            cell.get("sources", {}).get("calibration_manifest"),
            calibration_record,
            name=f"D1 primary cell {job_id} calibration",
        )  # type: ignore[union-attr]
        if (
            cell.get("execution_provenance", {}).get("candidate_test_labels_read")
            is not False
        ):  # type: ignore[union-attr]
            raise RuntimeError(f"D1 primary cell {job_id} violates Test isolation")
        prediction_path = verified_artifact_path(
            cell.get("artifacts", {}).get("predictions", {}),  # type: ignore[union-attr]
            name=f"D1 primary cell {job_id} predictions",
        )
        cell_paths[job_id] = cell_path
        cell_predictions[job_id] = pd.read_parquet(prediction_path)

    r0_path = root / "07_validation" / "r0_r1_selection.json"
    r0 = _load_content(r0_path, name="D1 R0/R1 selection", statuses=("COMPLETE",))
    _require_label_free(r0, name="D1 R0/R1 selection")
    _verify_manifest_records(r0, name="D1 R0/R1 selection")
    configuration = r0.get("configuration")
    if not isinstance(configuration, Mapping) or any(
        configuration.get(key) != expected
        for key, expected in {
            "route": "D1",
            "pool": PRIMARY_POOL,
            "track": PRIMARY_TRACK,
        }.items()
    ):
        raise RuntimeError("D1 R0/R1 primary contract differs")
    for split, metric_prefix in (("train", "train"), ("validation", "validation")):
        oracle = float(oracles[split][PRIMARY_POOL]["oracle"])
        for method in ("r0", "r1"):
            metrics = r0.get(f"{method}_{metric_prefix}_metrics")
            if not isinstance(metrics, Mapping):
                raise RuntimeError(f"D1 {method.upper()} {split} metrics are absent")
            if int(metrics.get("sample_count", -1)) != int(
                oracles[split][PRIMARY_POOL]["denominator"]
            ):
                raise RuntimeError(f"D1 {method.upper()} {split} denominator differs")
            j1 = _metric(metrics, "j_at_1", context=f"D1 {method.upper()} {split}")
            j5 = _metric(metrics, "j_at_5", context=f"D1 {method.upper()} {split}")
            recorded_oracle = _metric(
                metrics, "oracle_at_5", context=f"D1 {method.upper()} {split}"
            )
            if (
                j1 > oracle + 1e-12
                or abs(j5 - oracle) > 1e-12
                or abs(recorded_oracle - oracle) > 1e-12
            ):
                raise RuntimeError(
                    f"D1 {method.upper()} {split} J/oracle invariants differ"
                )

    selection_path = root / "07_validation" / "selected_primary_ungated.json"
    selection = _load_content(
        selection_path, name="D1 selected primary ungated", statuses=("COMPLETE",)
    )
    _require_label_free(selection, name="D1 selected primary ungated")
    _verify_manifest_records(selection, name="D1 selected primary ungated")
    if (
        selection.get("eligible_primary_methods") != ["R3", "R5", "R6"]
        or selection.get("selected_method") not in {"R3", "R5", "R6"}
        or selection.get("selected_configuration", {}).get("pool") != PRIMARY_POOL  # type: ignore[union-attr]
        or selection.get("selected_configuration", {}).get("track") != PRIMARY_TRACK  # type: ignore[union-attr]
    ):
        raise RuntimeError("D1 selected primary method semantics differ")
    _replay_primary_selection(
        root,
        planned=planned,
        cell_paths=cell_paths,
        cell_predictions=cell_predictions,
        selection=selection,
        plan_path=plan_path,
        execution_path=execution_path,
    )
    for split, key in (("train", "oof_metrics"), ("validation", "validation_metrics")):
        metrics = selection.get(key)
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"D1 selected primary {split} metrics are absent")
        oracle = float(oracles[split][PRIMARY_POOL]["oracle"])
        if (
            int(metrics.get("sample_count", -1))
            != int(oracles[split][PRIMARY_POOL]["denominator"])
            or _metric(metrics, "j_at_1", context=f"D1 selected {split}")
            > oracle + 1e-12
            or abs(_metric(metrics, "j_at_5", context=f"D1 selected {split}") - oracle)
            > 1e-12
            or abs(
                _metric(metrics, "oracle_at_5", context=f"D1 selected {split}") - oracle
            )
            > 1e-12
        ):
            raise RuntimeError(
                f"D1 selected primary {split} oracle/denominator differs"
            )

    grid_path = root / "configs" / "d1_gate_grid.json"
    grid = _load_content(grid_path, name="D1 R7 gate grid", statuses=("PLANNED",))
    _require_label_free(grid, name="D1 R7 gate grid")
    point_count = 1
    for key in (
        "lambda_harm",
        "utility_thresholds",
        "score_margin_thresholds",
        "reliability_thresholds",
        "stability_thresholds",
    ):
        values = grid.get(key)
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"D1 R7 grid misses {key}")
        point_count *= len(values)
    if point_count != 108 or grid.get("minimum_seed_votes") != 2:
        raise RuntimeError("D1 R7 gate grid differs from the frozen 108-point grid")
    verify_artifact_records_recursive(
        grid.get("sources"), name="D1 R7 gate grid sources", require_at_least_one=True
    )
    inputs_path = root / "07_validation" / "gate_inputs" / "d1" / "manifest.json"
    inputs = _load_content(inputs_path, name="D1 gate inputs", statuses=("COMPLETE",))
    _require_label_free(inputs, name="D1 gate inputs")
    _verify_manifest_records(inputs, name="D1 gate inputs")
    _same_record(
        inputs.get("sources", {}).get("primary_selection"),
        _record(selection_path),
        name="D1 gate input primary selection",
    )  # type: ignore[union-attr]
    gate_path = root / "07_validation" / "gate" / "d1" / "gate_selection.json"
    gate = _load_content(gate_path, name="D1 R7 gate selection", statuses=("COMPLETE",))
    _require_label_free(gate, name="D1 R7 gate selection")
    _verify_manifest_records(gate, name="D1 R7 gate selection")
    _same_record(
        gate.get("sources", {}).get("input_manifest"),
        _record(inputs_path),
        name="D1 gate selection inputs",
    )  # type: ignore[union-attr]
    _same_record(
        gate.get("sources", {}).get("grid"),
        _record(grid_path),
        name="D1 gate selection grid",
    )  # type: ignore[union-attr]
    validate_gate_selection_semantics(gate)
    gate_metrics = gate.get("validation_metrics")
    if not isinstance(gate_metrics, Mapping):
        raise RuntimeError("D1 gate Validation metrics are absent")
    validation_oracle = float(oracles["validation"][PRIMARY_POOL]["oracle"])
    if (
        int(gate_metrics.get("sample_count", -1))
        != int(oracles["validation"][PRIMARY_POOL]["denominator"])
        or _metric(gate_metrics, "gated_j_at_1", context="D1 gated Validation")
        > validation_oracle + 1e-12
    ):
        raise RuntimeError("D1 gate Validation denominator/oracle invariant differs")
    return {
        "plan": _record(plan_path),
        "source_adapter": primary_source_adapter,
        "execution": _record(execution_path),
        "r0_r1": _record(r0_path),
        "selection": _record(selection_path),
        "gate_grid": _record(grid_path),
        "gate_inputs": _record(inputs_path),
        "gate_selection": _record(gate_path),
    }, {
        "selection": selection,
        "gate": gate,
        "source_adapter": primary_source_adapter_value,
    }


def _validate_test_outputs(
    root: Path,
    *,
    paired: pd.DataFrame,
    top5_candidates: pd.DataFrame,
    candidate_records: Mapping[str, Any],
    feature_records: Mapping[str, Any],
    primary_records: Mapping[str, Any],
    primary_values: Mapping[str, Any],
) -> dict[str, Any]:
    ranker_path = root / "08_lock" / "label_free_test_rankers" / "d1" / "manifest.json"
    ranker = _load_content(
        ranker_path, name="D1 label-free Test ranker", statuses=("COMPLETE",)
    )
    _require_label_free(ranker, name="D1 label-free Test ranker")
    verify_records_with_primary_source_adapter(
        ranker,
        adapter=primary_values["source_adapter"],
        name="D1 label-free Test ranker",
    )
    ranker_sources = ranker.get("sources")
    if not isinstance(ranker_sources, Mapping):
        raise RuntimeError("D1 label-free Test ranker sources are absent")
    expected_ranker = {
        "selection": primary_records["selection"],
        "candidate_manifest": candidate_records["manifest"],
        "candidates": candidate_records["pools"][PRIMARY_POOL],
        "denominator": _record(_paired_path(root, "test")),
        "final_feature_manifest": feature_records[PRIMARY_POOL][PRIMARY_TRACK][
            "manifest"
        ],
    }
    for key, expected in expected_ranker.items():
        _same_record(ranker_sources.get(key), expected, name=f"D1 Test ranker {key}")
    decisions_path = verified_artifact_path(
        ranker.get("artifacts", {}).get("per_sample_decisions", {}),  # type: ignore[union-attr]
        name="D1 Test ranker decisions",
    )
    scores_path = verified_artifact_path(
        ranker.get("artifacts", {}).get("per_candidate_scores", {}),  # type: ignore[union-attr]
        name="D1 Test ranker scores",
    )
    for path, name in (
        (decisions_path, "Test ranker decisions"),
        (scores_path, "Test ranker scores"),
    ):
        assert_label_free_parquet_schema(path, name=name)
    decisions = pd.read_parquet(
        decisions_path, columns=["sample_id", "candidate_count"]
    )
    if decisions["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("D1 Test ranker decisions duplicate samples")
    expected_ids = paired["sample_id"].astype(str).tolist()
    if set(decisions["sample_id"].astype(str)) != set(expected_ids) or len(
        decisions
    ) != len(expected_ids):
        raise RuntimeError("D1 Test ranker does not preserve the paired denominator")
    expected_counts = (
        top5_candidates.groupby("sample_id").size().reindex(expected_ids, fill_value=0)
    )
    observed_counts = decisions.set_index(decisions["sample_id"].astype(str))[
        "candidate_count"
    ].reindex(expected_ids)
    if (
        not pd.to_numeric(observed_counts)
        .astype(int)
        .reset_index(drop=True)
        .equals(expected_counts.astype(int).reset_index(drop=True))
    ):
        raise RuntimeError("D1 Test ranker no-output/candidate counts differ from Top5")
    scores = pd.read_parquet(scores_path, columns=["sample_id", "candidate_id"])
    _assert_frame_keys(scores, top5_candidates, name="D1 Test ranker scores")

    gate_path = root / "08_lock" / "label_free_test_gates" / "d1" / "manifest.json"
    gate = _load_content(
        gate_path, name="D1 label-free Test gate", statuses=("COMPLETE",)
    )
    _require_label_free(gate, name="D1 label-free Test gate")
    _verify_manifest_records(gate, name="D1 label-free Test gate")
    if gate.get("prediction_source") != "test_label_free" or int(
        gate.get("sample_count", -1)
    ) != len(expected_ids):
        raise RuntimeError("D1 Test gate prediction source/denominator differs")
    gate_sources = gate.get("sources")
    if not isinstance(gate_sources, Mapping):
        raise RuntimeError("D1 label-free Test gate sources are absent")
    for key, expected in {
        "gate_selection": primary_records["gate_selection"],
        "primary_selection": primary_records["selection"],
        "ranker_application": _record(ranker_path),
        "candidate_manifest": candidate_records["manifest"],
        "candidates": candidate_records["pools"][PRIMARY_POOL],
        "denominator": _record(_paired_path(root, "test")),
        "feature_manifest": feature_records[PRIMARY_POOL][PRIMARY_TRACK]["manifest"],
    }.items():
        _same_record(gate_sources.get(key), expected, name=f"D1 Test gate {key}")
    for key in ("inputs", "decisions"):
        path = verified_artifact_path(
            gate.get("artifacts", {}).get(key, {}),  # type: ignore[union-attr]
            name=f"D1 Test gate {key}",
        )
        assert_label_free_parquet_schema(path, name=f"Test gate {key}")
        frame = pd.read_parquet(path, columns=["sample_id"])
        if set(frame["sample_id"].astype(str)) != set(expected_ids) or len(
            frame
        ) != len(expected_ids):
            raise RuntimeError(f"D1 Test gate {key} does not preserve denominator")
    return {"ranker": _record(ranker_path), "gate": _record(gate_path)}


def _validate_access_log(root: Path) -> dict[str, Any]:
    path = _regular_file(
        root / "09_formal_test" / "test_access.log", name="D1 Test access log"
    )
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"D1 Test access log line {line_number} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError(
                f"D1 Test access log line {line_number} is not an object"
            )
        events.append(value)
    if not events:
        raise RuntimeError("D1 Test access log is empty")
    forbidden = []
    for event in events:
        if (
            event.get("event") == "candidate_test_label_access_authorized"
            or event.get("candidate_labels_opened_as_table") is True
            or event.get("candidate_test_labels_read") is True
            or int(event.get("candidate_label_rows_read", 0) or 0) != 0
        ):
            forbidden.append(event)
    if forbidden:
        raise PermissionError("D1 candidate-level Test label access count is not zero")
    label_free_stages = [
        event
        for event in events
        if event.get("event") == "prelock_label_free_test_stage"
        and event.get("candidate_labels_opened_as_table") is False
    ]
    observed_stages = {str(event.get("stage", "")) for event in label_free_stages}
    if observed_stages != REQUIRED_LABEL_FREE_TEST_STAGES:
        raise RuntimeError(
            "D1 access log label-free stage inventory differs: "
            f"missing={sorted(REQUIRED_LABEL_FREE_TEST_STAGES - observed_stages)} "
            f"extra={sorted(observed_stages - REQUIRED_LABEL_FREE_TEST_STAGES)}"
        )
    event_ids = [
        str(event["event_id"])
        for event in events
        if isinstance(event.get("event_id"), str) and event.get("event_id")
    ]
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError("D1 access log event_id inventory is duplicated")
    latest_output_records: dict[tuple[str, str], tuple[int, Mapping[str, Any]]] = {}
    for index, event in enumerate(label_free_stages):
        stage = str(event.get("stage", ""))
        event_records: list[Mapping[str, Any]] = []
        output = event.get("output_manifest")
        output_sha = event.get("output_manifest_sha256")
        if isinstance(output, str) and output:
            event_records.append({"path": output, "sha256": output_sha})
        elif isinstance(output, Mapping):
            event_records.append(output)
        for field in ("output_manifests", "outputs"):
            records = event.get(field)
            if isinstance(records, Mapping):
                for record in records.values():
                    if isinstance(record, Mapping):
                        event_records.append(record)
        if not event_records:
            raise RuntimeError(
                f"D1 label-free access event {index} has no output binding"
            )
        for record in event_records:
            output_path = record.get("path")
            if not isinstance(output_path, str) or not output_path:
                raise RuntimeError(
                    f"D1 label-free access event {index} output path is absent"
                )
            # Reruns append new audit events without deleting history.  The
            # latest record for each stable stage/output path is authoritative;
            # earlier hashes remain immutable audit history rather than active
            # source bindings.
            latest_output_records[(stage, str(Path(output_path).resolve()))] = (
                index,
                record,
            )
    for (stage, _output_path), (index, record) in latest_output_records.items():
        try:
            verified_artifact_path(
                record,
                name=f"D1 label-free access {stage} output {index}",
            )
        except (FileNotFoundError, RuntimeError) as error:
            raise RuntimeError(
                f"D1 latest label-free access output differs at event {index}"
            ) from error
    return {
        "artifact": _record(path),
        "event_count": len(events),
        "label_free_stage_event_count": len(label_free_stages),
        "candidate_label_access_count": 0,
        "label_free_test_stage_inventory": sorted(observed_stages),
    }


def _validate_mandatory_tables(root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    text_by_name: dict[str, str] = {}
    for name, relative in MANDATORY_TABLES.items():
        path = _regular_file(root / relative, name=f"D1 mandatory table {name}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"D1 mandatory table {name} is empty")
        frame = pd.read_csv(path)
        if frame.empty or not len(frame.columns):
            raise RuntimeError(f"D1 mandatory table {name} has no evidence rows")
        forbidden_columns = [
            str(column)
            for column in frame.columns
            if str(column).lower().startswith("test_")
            or "candidate_test" in str(column).lower()
        ]
        if forbidden_columns:
            raise PermissionError(
                f"D1 prelock table {name} exposes Test-derived columns: {forbidden_columns}"
            )
        records[name] = _record(path)
        text_by_name[name] = frame.to_csv(index=False).lower().replace("_", "")
    if any(f"r{index}" not in text_by_name["r0_r7_full"] for index in range(8)):
        raise RuntimeError("D1 r0_r7_full table does not cover R0-R7")
    if any(
        token not in text_by_name["evidence_track_ablation"]
        for token in ("t1", "t2", "t3")
    ):
        raise RuntimeError("D1 evidence-track table does not cover T1/T2/T3")
    if any(
        token not in text_by_name["k_comparison"]
        for token in ("top5", "top10", "allnms")
    ):
        raise RuntimeError("D1 K comparison does not cover Top5/Top10/AllNMS")
    if any(
        route.lower() not in text_by_name["four_route_validation"]
        for route in ("C1", "G1", "CROG", "D1")
    ):
        raise RuntimeError(
            "D1 four-route Validation table does not cover C1/G1/CROG/D1"
        )
    return records


def _assert_preformal_state(root: Path) -> dict[str, Any]:
    forbidden = (
        root / "08_lock" / "FORMAL_TEST_LOCK.json",
        root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json",
        root / "FINAL_RUN_LOCK.json",
        root / "COMPLETE",
    )
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise PermissionError(
            f"D1 prelock readiness refuses formal/postformal state: {present}"
        )
    run_manifest_path = _regular_file(root / "manifest.json", name="D1 run manifest")
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("D1 run manifest is invalid JSON") from error
    if not isinstance(run_manifest, dict):
        raise RuntimeError("D1 run manifest is not an object")
    if int(run_manifest.get("formal_test_execution_count", -1)) != 0:
        raise PermissionError("D1 formal Test execution count is not zero")
    status_path = _regular_file(
        root / "pipeline_status.json", name="D1 pipeline status"
    )
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("D1 pipeline status is invalid JSON") from error
    if not isinstance(status, dict):
        raise RuntimeError("D1 pipeline status is not an object")
    if (
        status.get("formal_test_executed") is not False
        or status.get("test_candidate_labels_read") is not False
    ):
        raise PermissionError("D1 pipeline status is not preformal/label-free")
    return {
        "run_manifest": _record(run_manifest_path),
        "pipeline_status": _record(status_path),
    }


def _validate_reference_systems(root: Path) -> dict[str, Any]:
    names = (
        "three_route_crog_default_router_reference",
        "top15_union_reference",
    )
    records: dict[str, Any] = {}
    for name in names:
        path = root / f"08_lock/formal_inputs/{name}/manifest.json"
        value = _load_content(path, name=f"D1 {name}", statuses=("COMPLETE",))
        configuration = value.get("configuration")
        if (
            value.get("candidate_test_labels_read") is not False
            or value.get("selection_used_test_metrics") is not False
            or not isinstance(configuration, Mapping)
            or configuration.get("role") != "READ_ONLY_REFERENCE_DIAGNOSTIC"
            or configuration.get("routes") != ["CROG", "G1", "C1"]
        ):
            raise RuntimeError(f"D1 {name} provenance differs")
        verify_artifact_records_recursive(
            {"sources": value.get("sources"), "artifacts": value.get("artifacts")},
            name=f"D1 {name}",
            require_at_least_one=True,
        )
        records[name] = _record(path)
    access_path = _regular_file(
        root / "09_formal_test" / "test_access.log",
        name="D1 reference-system access log",
    )
    events = [
        json.loads(line)
        for line in access_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matching = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event") == "prelock_label_free_test_stage"
        and event.get("stage") == "d1_read_only_reference_systems"
        and event.get("output_manifests") == records
        and event.get("candidate_test_labels_read") is False
        and event.get("candidate_labels_opened_as_table") is False
        and event.get("selection_used_test_metrics") is False
    ]
    if len(matching) != 1:
        raise RuntimeError("D1 read-only reference access event is absent/duplicated")
    return records


def _validate_all(
    root: Path, *, code_paths: Iterable[Path]
) -> tuple[dict[str, Any], dict[str, Any]]:
    preformal = _assert_preformal_state(root)
    closure_path, closure = load_source_closure(root)
    if (
        closure.get("canonical_snapshot") != "A"
        or closure.get("candidate_test_labels_read") is not False
        or closure.get("selection_used_test_metrics") is not False
    ):
        raise RuntimeError(
            "D1 active source closure is not the exact label-free Snapshot A contract"
        )
    closure_record = _record(closure_path)
    evaluator_record = _record(root / "configs" / "canonical_evaluator.py")
    verified_evidence = closure.get("verified_artifacts")
    if not isinstance(verified_evidence, list):
        raise RuntimeError("D1 active source closure evidence inventory is absent")
    canonical_evaluator = next(
        (
            record
            for record in verified_evidence
            if isinstance(record, Mapping)
            and record.get("name") == "canonical_evaluator"
        ),
        None,
    )
    if (
        not isinstance(canonical_evaluator, Mapping)
        or canonical_evaluator.get("sha256") != evaluator_record["sha256"]
    ):
        raise RuntimeError(
            "D1 copied evaluator differs from the source-closure evaluator"
        )
    paired, paired_records = _read_denominators(root)
    candidate_frames: dict[str, Any] = {}
    candidate_manifests: dict[str, Any] = {}
    candidate_records: dict[str, Any] = {}
    for split in SPLITS:
        closure_inputs = closure.get("canonical_inputs", {}).get(split)  # type: ignore[union-attr]
        if not isinstance(closure_inputs, Mapping) or not isinstance(
            closure_inputs.get("paired_manifest"), Mapping
        ):
            raise RuntimeError(f"D1 source closure misses {split} paired manifest")
        frames, manifest, records = _validate_candidate_split(
            root,
            split=split,
            paired=paired[split],
            source_closure_record=closure_record,
            source_paired_record=closure_inputs["paired_manifest"],
        )
        candidate_frames[split] = frames
        candidate_manifests[split] = manifest
        candidate_records[split] = records
    registry_record = _validate_candidate_registry(
        root, candidate_manifests, candidate_records
    )
    splits_record = _validate_splits(root, paired)
    calibration_records = _validate_calibrations(root, candidate_records)
    feature_records = _validate_features(
        root,
        candidate_frames=candidate_frames,
        candidate_records=candidate_records,
        closure_record=closure_record,
        calibration_records=calibration_records,
    )
    oracles, label_records = _development_oracles(
        root,
        paired=paired,
        candidates=candidate_frames,
        candidate_records=candidate_records,
        closure=closure,
        closure_record=closure_record,
        evaluator_record=evaluator_record,
    )
    primary_records, primary_values = _validate_primary(
        root,
        oracles=oracles,
        splits_record=splits_record,
        calibration_records=calibration_records,
    )
    test_records = _validate_test_outputs(
        root,
        paired=paired["test"],
        top5_candidates=candidate_frames["test"][PRIMARY_POOL],
        candidate_records=candidate_records["test"],
        feature_records=feature_records["test"],
        primary_records=primary_records,
        primary_values=primary_values,
    )
    raw_feature_records, raw_feature_checks = validate_adapted_feature_execution_replay(
        root
    )
    k_records, k_checks = validate_adapted_k_sensitivity_replay(root)
    k_formal_records, k_formal_checks = validate_adapted_k_formal_replay(root)
    ablation_replay = validate_adapted_ablation_replay(root)
    access = _validate_access_log(root)
    tables = _validate_mandatory_tables(root)
    p12_records = validate_p12_prelock(root)
    p12_test_inputs = validate_four_route_test_inputs(root, require_applications=True)
    postformal_evidence = validate_postformal_evidence_prelock(
        root,
        source_closure=closure,
    )
    lightweight_audit_path, lightweight_audits = load_lightweight_audits(root)
    reference_systems = _validate_reference_systems(root)
    code = [_record(path) for path in code_paths]
    sources = {
        "preformal_state": preformal,
        "source_closure": closure_record,
        "canonical_evaluator": evaluator_record,
        "paired": paired_records,
        "candidate_registry": registry_record,
        "candidates": candidate_records,
        "splits": splits_record,
        "calibrations": calibration_records,
        "features": feature_records,
        "raw_feature_execution": raw_feature_records,
        "development_labels": label_records,
        "primary": primary_records,
        "label_free_test": test_records,
        "k_sensitivity": k_records,
        "k_formal_inputs": k_formal_records,
        "p10_ablation": ablation_replay["sources"],
        "p12_four_route": p12_records,
        "p12_four_route_test_inputs": p12_test_inputs,
        "postformal_evidence": postformal_evidence,
        "lightweight_audits": {
            "manifest": _record(lightweight_audit_path),
            "sources": lightweight_audits["sources"],
            "artifacts": lightweight_audits["artifacts"],
        },
        "reference_systems": reference_systems,
        "access_log": access["artifact"],
        "mandatory_tables": tables,
        "validator_code": code,
    }
    checks = {
        "canonical_snapshot": "A",
        "candidate_pools": list(POOLS),
        "feature_tracks": list(TRACKS),
        "denominators": {split: len(paired[split]) for split in SPLITS},
        "no_output_retained": {
            split: {
                pool: candidate_manifests[split]["summaries"][pool]["no_output_samples"]
                for pool in POOLS
            }
            for split in SPLITS
        },
        "candidate_hash_invariants": "recomputed membership/native-score/geometry contract hashes exact",
        "development_oracles": oracles,
        "primary_selected_method": primary_values["selection"]["selected_method"],
        "r7_gate_decision": primary_values["gate"]["decision"],
        "fold_local_preprocessing_calibration": True,
        "raw_feature_execution_semantic_replay": raw_feature_checks,
        "label_free_test_ranker_gate": True,
        "k_sensitivity_semantic_replay": k_checks,
        "k_formal_inputs_semantic_replay": k_formal_checks,
        "p10_ablation_semantic_replay": ablation_replay["checks"],
        "p12_four_route_semantic_replay": True,
        "p12_four_route_test_input_semantic_replay": True,
        "candidate_test_label_access_count": access["candidate_label_access_count"],
        "label_free_test_stage_inventory": access["label_free_test_stage_inventory"],
        "mandatory_tables": list(MANDATORY_TABLES),
        "formal_test_lock_created": False,
    }
    return sources, checks


def _declarations(checks: Mapping[str, Any]) -> tuple[str, str]:
    primary = (
        "# D1 primary method declaration\n\n"
        "- Route: `D1`.\n"
        f"- Candidate pool: `{PRIMARY_POOL}` (frozen Snapshot A GQ-CNN order).\n"
        f"- Evidence track: `{PRIMARY_TRACK}`.\n"
        f"- Ungated ranker: `{checks['primary_selected_method']}` selected only from R3/R5/R6 on Validation.\n"
        f"- Expected-gain gate: `R7`, decision `{checks['r7_gate_decision']}`.\n"
        "- Seeds: `42, 123, 2026`; fold-local preprocessing and calibration are mandatory.\n"
        "- No-output samples remain in every denominator.\n"
        "- Candidate-level Test labels opened before this declaration: `0`.\n"
        "- This declaration is prelock readiness evidence; it is not `FORMAL_TEST_LOCK`.\n"
    )
    secondary = (
        "# D1 secondary-track declaration\n\n"
        "- Candidate-pool sensitivity: `top10` and `allnms`; `top5` remains primary.\n"
        "- Feature evidence: `T1_native_available` and `T3_route_rich`; `T2_matched_common` remains primary.\n"
        "- R0-R7, feature ablation, K sensitivity, and four-route router/union analyses are Validation/development evidence only.\n"
        "- Secondary results cannot replace the declared D1 Top5/T2 primary selector after Test evaluation.\n"
        "- Candidate-level Test labels opened before this declaration: `0`.\n"
    )
    return primary, secondary


def assemble_prelock_readiness(
    run_dir: str | Path,
    *,
    resume: bool = False,
    code_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Validate and atomically publish D1 P13 readiness without creating a lock."""

    root = Path(run_dir).expanduser().resolve()
    lock_dir = root / "08_lock"
    readiness_path = lock_dir / "PRELOCK_READINESS.json"
    primary_path = lock_dir / "PRIMARY_METHOD_DECLARATION.md"
    secondary_path = lock_dir / "SECONDARY_TRACK_DECLARATION.md"
    resolved_code = tuple(Path(path).expanduser().resolve() for path in code_paths)
    sources, checks = _validate_all(root, code_paths=resolved_code)
    pipeline_record = transition_pipeline_status(
        root,
        status="PRELOCK_LABEL_FREE",
        first_incomplete_stage="P14_FORMAL_LOCK",
        formal_test_executed=False,
        test_candidate_labels_read=False,
        formal_test_execution_count=0,
    )
    sources.setdefault("preformal_state", {})["pipeline_status"] = pipeline_record
    source_signature = canonical_sha256(sources)
    primary_text, secondary_text = _declarations(checks)

    if readiness_path.exists():
        existing = _load_content(
            readiness_path, name="D1 prelock readiness", statuses=("PASS",)
        )
        if not resume:
            raise FileExistsError(
                f"D1 prelock readiness already exists; pass --resume: {readiness_path}"
            )
        if (
            existing.get("source_signature_sha256") != source_signature
            or existing.get("checks") != checks
            or primary_path.read_text(encoding="utf-8") != primary_text
            or secondary_path.read_text(encoding="utf-8") != secondary_text
        ):
            raise RuntimeError(
                "D1 prelock readiness differs from current validated sources"
            )
        if (root / "08_lock" / "FORMAL_TEST_LOCK.json").exists():
            raise PermissionError("D1 prelock resume found an unexpected formal lock")
        return existing
    stale_outputs = [
        str(path) for path in (primary_path, secondary_path) if path.exists()
    ]
    if stale_outputs:
        raise FileExistsError(
            f"D1 prelock declarations exist without readiness manifest: {stale_outputs}"
        )

    # Close the validation/write race by re-hashing every recorded source before
    # publishing any declaration.  The records include all transitive manifests
    # and every table consumed by this stage.
    verify_artifact_records_recursive(
        sources, name="D1 P13 readiness source recheck", require_at_least_one=True
    )
    _assert_preformal_state(root)
    atomic_text(primary_path, primary_text)
    atomic_text(secondary_path, secondary_text)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "P13",
        "route": "D1",
        "source_signature_sha256": source_signature,
        "candidate_test_labels_read": False,
        "formal_test_execution_count": 0,
        "checks": checks,
        "sources": sources,
        "artifacts": {
            "primary_method_declaration": _record(primary_path),
            "secondary_track_declaration": _record(secondary_path),
        },
        "formal_test_lock_created": False,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(readiness_path, result)
    if (root / "08_lock" / "FORMAL_TEST_LOCK.json").exists():
        raise PermissionError("D1 P13 validator must never create FORMAL_TEST_LOCK")
    return result
