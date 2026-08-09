#!/usr/bin/env python3
"""Generate the five evidence-bound final 4-DoF experiment reports.

This command is a pure reporting step.  It requires completed audit,
validation, formal-test, oracle, statistical, failure-analysis, independent
recomputation, and experiment-lock artifacts.  It never recomputes a metric,
selects a configuration, or changes an existing result artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.results import (  # noqa: E402
    assert_aggregate_matches_sample_rows,
)
from src.grasping.common.statistics import (  # noqa: E402
    DEFAULT_PAIR_SPECS,
    analyze_paired_methods,
    load_aligned_predictions,
)
from tools.grasp4dof.run_formal_inference import (  # noqa: E402
    locked_test_sample_ids,
    validate_r0_reference_evidence,
)


REPORT_NAMES = (
    "REPEATEDFILM_4DOF_IMPLEMENTATION.md",
    "REPEATEDFILM_4DOF_AUDIT.md",
    "REPEATEDFILM_4DOF_VALIDATION.md",
    "REPEATEDFILM_4DOF_RESULTS.md",
    "REPEATEDFILM_4DOF_FAILURE_ANALYSIS.md",
)
PREDICTED_METHODS = ("R0", "G0", "G1", "C0", "C1", "A0")
FORMAL_DISPLAY_METHODS = (*PREDICTED_METHODS, "locked_primary_4dof_backend")
VALIDATION_METHODS = ("G0", "G1", "C0", "C1", "A0")
ORACLE_METHODS = ("G0", "G1", "C0", "C1", "A0")
PRIMARY_CANDIDATES = ("G1", "C1", "A0")
GALLERY_CATEGORIES = {
    "success",
    "ranking_failure",
    "no_candidate",
    "wrong_mask",
    "angle_failure",
    "width_failure",
}
METHOD_LABELS = {
    "R0": "repeatedfilm_dexnet_gqcnn_reference",
    "G0": "repeatedfilm_grconvnet_pretrained_transfer",
    "G1": "repeatedfilm_grconvnet_ocidvlg_finetuned",
    "C0": "repeatedfilm_ggcnn2_pretrained_transfer",
    "C1": "repeatedfilm_ggcnn2_ocidvlg_finetuned",
    "A0": "repeatedfilm_mask_depth_analytic",
    "locked_primary_4dof_backend": "locked_primary_4dof_backend",
}
SCIENTIFIC_SCOPE_SENTENCE = (
    "J@1/J@5 measure consistency with annotated OCID-VLG 2D grasp rectangles."
)
OFFICIAL_SOURCE_ALLOWLIST: Mapping[str, Mapping[str, str]] = {
    "https://github.com/skumra/robotic-grasping.git": {
        "project": "https://github.com/skumra/robotic-grasping",
        "paper": "https://arxiv.org/abs/1909.04810",
        "doi": "https://doi.org/10.1109/IROS45743.2020.9340777",
    },
    "https://github.com/dougsm/ggcnn.git": {
        "project": "https://github.com/dougsm/ggcnn",
        "paper": "https://arxiv.org/abs/1804.05172",
        "doi": "https://doi.org/10.15607/RSS.2018.XIV.021",
    },
}

REQUIRED_FILES = (
    "audit/source_inventory.json",
    "audit/dataset_split_audit.json",
    "audit/reference_baseline_inventory.json",
    "audit/basic_device_smoke.json",
    "audit/formal_input_reference_preflight.json",
    "audit/protected_source_final_verification.json",
    "third_party/grconvnet_source_manifest.json",
    "third_party/ggcnn2_source_manifest.json",
    "environment.json",
    "package_lock.txt",
    "validation_results.csv",
    "selected_configs.json",
    "primary_validation_selection.json",
    "formal_test_results.csv",
    "per_method_metrics.csv",
    "oracle_results.csv",
    "common_subset_comparison.csv",
    "per_sample_predictions.parquet",
    "per_candidate_predictions.parquet",
    "runtime_metrics.json",
    "memory_metrics.json",
    "training_curves.csv",
    "results_bundle.json",
    "statistical_tests.json",
    "bootstrap_intervals.json",
    "subgroup_results.csv",
    "subgroup_sample_features.parquet",
    "subgroup_analysis_manifest.json",
    "per_sample_failure_stage.parquet",
    "gallery/selection_manifest.json",
    "gallery/index.html",
    "independent_recompute_results.json",
    "manifests/experiment_lock.json",
    ".EXPERIMENT_LOCKED",
    "frozen_4dof_backends_experiment_manifest.json",
    "commands.log",
    "storage_budget.json",
    "storage_usage_by_stage.csv",
)
BUNDLE_OUTPUT_NAMES = {
    "per_sample_predictions.parquet",
    "per_candidate_predictions.parquet",
    "formal_test_results.csv",
    "per_method_metrics.csv",
    "oracle_results.csv",
    "common_subset_comparison.csv",
    "training_curves.csv",
}
METHOD_SOURCE_RELATIVES = {
    "R0": "formal_test/R0",
    "G0": "formal_test/G0",
    "G1": "formal_test/G1",
    "C0": "formal_test/C0",
    "C1": "formal_test/C1",
    "A0": "formal_test/A0",
    "G0-O": "oracle/G0-O",
    "G1-O": "oracle/G1-O",
    "C0-O": "oracle/C0-O",
    "C1-O": "oracle/C1-O",
    "A0-O": "oracle/A0-O",
}

FORMAL_COLUMNS = (
    "method_id",
    "j_at_1",
    "j_at_5",
    "candidate_pool_oracle",
    "mrr",
    "non_empty_rate",
    "no_grasp_rate",
    "p50_latency_seconds",
    "p95_latency_seconds",
)
ORACLE_COLUMNS = (
    "method_id",
    "pred_mask_j_at_1",
    "gt_mask_j_at_1",
    "j_at_1_gap",
    "pred_mask_oracle",
    "gt_mask_oracle",
)
FAILURE_COLUMNS = (
    "method",
    "sample_id",
    "failure_stage",
    "analysis_scope",
    "configuration_selection_performed",
)
SUBGROUP_COLUMNS = (
    "method_id",
    "dimension",
    "group",
    "sample_count",
    "j_at_1",
    "j_at_5",
    "candidate_pool_oracle",
    "non_empty_rate",
)


class FinalReportError(ValueError):
    """Raised when final evidence is missing, inconsistent, or mutable."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_file(path: Path, *, label: str | None = None) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label or path.name}: {path}")
    return path


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(_require_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalReportError(f"expected JSON object: {path}")
    return value


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, label: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FinalReportError(f"{label} missing columns: {missing}")
    if frame.empty:
        raise FinalReportError(f"{label} must not be empty")


def _validate_parquet_schema(path: Path, columns: Sequence[str], *, label: str) -> int:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    missing = sorted(set(columns) - names)
    if missing:
        raise FinalReportError(f"{label} missing columns: {missing}")
    row_count = int(parquet.metadata.num_rows)
    if row_count <= 0:
        raise FinalReportError(f"{label} must not be empty")
    return row_count


def _strict_method_rows(
    frame: pd.DataFrame, methods: Sequence[str], *, label: str
) -> pd.DataFrame:
    _require_columns(frame, ("method_id",), label=label)
    result = frame.copy()
    result["method_id"] = result["method_id"].astype(str)
    selected = result.loc[result["method_id"].isin(methods)].copy()
    counts = Counter(selected["method_id"].tolist())
    if counts != Counter(methods):
        raise FinalReportError(
            f"{label} method coverage mismatch: expected={list(methods)}, observed={dict(counts)}"
        )
    order = {method: index for index, method in enumerate(methods)}
    selected["_order"] = selected["method_id"].map(order)
    return selected.sort_values("_order", kind="mergesort").drop(columns="_order")


def _finite_rate(value: Any, *, field: str, method: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise FinalReportError(f"{method}: {field} must be finite in [0,1]")
    return number


def _validate_metric_frame(frame: pd.DataFrame, *, label: str) -> None:
    _require_columns(frame, FORMAL_COLUMNS, label=label)
    for row in frame.itertuples(index=False):
        method = str(row.method_id)
        for field in (
            "j_at_1",
            "j_at_5",
            "candidate_pool_oracle",
            "mrr",
            "non_empty_rate",
            "no_grasp_rate",
        ):
            _finite_rate(getattr(row, field), field=field, method=method)
        for field in ("p50_latency_seconds", "p95_latency_seconds"):
            value = float(getattr(row, field))
            if not math.isfinite(value) or value < 0.0:
                raise FinalReportError(
                    f"{method}: {field} must be finite and non-negative"
                )
        if float(row.j_at_1) > float(row.j_at_5):
            raise FinalReportError(f"{method}: J@1 exceeds J@5")
        if float(row.j_at_5) > float(row.candidate_pool_oracle):
            raise FinalReportError(f"{method}: J@5 exceeds candidate-pool oracle")
        if not math.isclose(
            float(row.non_empty_rate) + float(row.no_grasp_rate), 1.0, abs_tol=1e-10
        ):
            raise FinalReportError(
                f"{method}: non-empty/no-grasp rates do not sum to one"
            )


def _load_selected_configs(
    selected: Mapping[str, Any], *, expected_methods: Sequence[str]
) -> dict[str, dict[str, Any]]:
    if set(selected) != set(expected_methods):
        raise FinalReportError(
            f"selected_configs.json must contain exactly {list(expected_methods)}"
        )
    result: dict[str, dict[str, Any]] = {}
    for method in expected_methods:
        record = selected[method]
        if not isinstance(record, Mapping) or not {"path", "sha256"}.issubset(record):
            raise FinalReportError(f"{method}: selected config record is incomplete")
        path = _require_file(Path(str(record["path"])).expanduser().resolve())
        observed = sha256_file(path)
        if observed != str(record["sha256"]):
            raise FinalReportError(f"{method}: selected config hash mismatch")
        config = _json_object(path)
        result[method] = {
            "path": str(path),
            "sha256": observed,
            "config": config,
        }
    return result


def _validate_lock(
    *,
    run: Path,
    lock: Mapping[str, Any],
    frozen_lock_path: Path,
    marker: Mapping[str, Any],
) -> None:
    if lock.get("lock_status") != "LOCKED" or lock.get("effective") is not True:
        raise FinalReportError("experiment lock is not effective")
    if str(lock.get("run_dir")) != str(run):
        raise FinalReportError("experiment lock run_dir mismatch")
    expected_hash = lock.get("manifest_content_sha256")
    unhashed = dict(lock)
    unhashed.pop("manifest_content_sha256", None)
    if expected_hash != canonical_json_sha256(unhashed):
        raise FinalReportError("experiment lock canonical hash mismatch")
    if (
        marker.get("lock_status") != "LOCKED"
        or marker.get("manifest_content_sha256") != expected_hash
    ):
        raise FinalReportError("experiment lock marker mismatch")
    lock_path = run / "manifests/experiment_lock.json"
    if lock_path.read_bytes() != frozen_lock_path.read_bytes():
        raise FinalReportError("frozen lock alias differs from effective lock")
    for logical_name, raw in lock.get("artifacts", {}).items():
        if not isinstance(raw, Mapping) or not {"path", "bytes", "sha256"}.issubset(
            raw
        ):
            raise FinalReportError(
                f"locked artifact record is malformed: {logical_name}"
            )
        path = (run / str(raw["path"])).resolve()
        try:
            path.relative_to(run)
        except ValueError as error:
            raise FinalReportError(
                f"locked artifact escapes run directory: {logical_name}"
            ) from error
        _require_file(path, label=f"locked artifact {logical_name}")
        if (
            path.stat().st_size != int(raw["bytes"])
            or sha256_file(path) != raw["sha256"]
        ):
            raise FinalReportError(f"locked artifact changed: {logical_name}")
    repository = Path(str(lock.get("repository", {}).get("root", ""))).resolve()
    for relative, expected in lock.get("source_hashes", {}).items():
        path = (repository / str(relative)).resolve()
        try:
            path.relative_to(repository)
        except ValueError as error:
            raise FinalReportError(
                f"locked source escapes repository: {relative}"
            ) from error
        _require_file(path, label=f"locked source {relative}")
        if sha256_file(path) != expected:
            raise FinalReportError(f"locked source changed: {relative}")


def _validate_bundle_outputs(
    bundle: Mapping[str, Any], *, run: Path, expected_count: int
) -> None:
    outputs = bundle.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != BUNDLE_OUTPUT_NAMES:
        raise FinalReportError("results bundle output coverage mismatch")
    if (
        int(bundle.get("method_count", -1)) != 11
        or int(bundle.get("expected_sample_count_per_method", -1)) != expected_count
    ):
        raise FinalReportError("results bundle method/sample count mismatch")
    for name, raw in outputs.items():
        if not isinstance(raw, Mapping) or not {"path", "sha256"}.issubset(raw):
            raise FinalReportError(f"results bundle output is malformed: {name}")
        path = _require_file(
            Path(str(raw["path"])).resolve(), label=f"bundle output {name}"
        )
        if path != (run / name).resolve():
            raise FinalReportError(f"results bundle output path mismatch: {name}")
        if sha256_file(path) != raw["sha256"]:
            raise FinalReportError(f"results bundle output hash mismatch: {name}")


def _validate_consolidated_sources(
    bundle: Mapping[str, Any],
    *,
    run: Path,
    lock: Mapping[str, Any],
    expected_count: int,
    per_sample_path: Path,
    per_candidate_path: Path,
) -> None:
    """Prove the root prediction tables are the exact formal-source concatenation."""

    sources = bundle.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(METHOD_SOURCE_RELATIVES):
        raise FinalReportError("results bundle source coverage mismatch")
    sample_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    expected_ids = locked_test_sample_ids(lock, expected_count=expected_count)
    for method_id, relative in METHOD_SOURCE_RELATIVES.items():
        record = sources[method_id]
        if not isinstance(record, Mapping):
            raise FinalReportError(f"results bundle source is malformed: {method_id}")
        directory = (run / relative).resolve()
        expected_record = {
            "directory": str(directory),
            "per_sample_sha256": sha256_file(
                directory / "per_sample_predictions.parquet"
            ),
            "per_candidate_sha256": sha256_file(
                directory / "per_candidate_predictions.parquet"
            ),
            "metrics_sha256": sha256_file(directory / "metrics.json"),
            "run_config_sha256": sha256_file(directory / "run_config.json"),
            "complete_sha256": sha256_file(directory / "COMPLETE.json"),
            "complete_filename": "COMPLETE.json",
        }
        if dict(record) != expected_record:
            raise FinalReportError(
                f"results bundle source provenance mismatch: {method_id}"
            )
        complete = _json_object(directory / "COMPLETE.json")
        run_config = _json_object(directory / "run_config.json")
        base_method = method_id.removesuffix("-O")
        oracle = method_id.endswith("-O")
        artifact_names = {
            "metrics.json",
            "runtime_metrics.json",
            "memory_metrics.json",
            "run_config.json",
            "per_sample_predictions.parquet",
            "per_candidate_predictions.parquet",
        }
        if method_id == "R0":
            artifact_names.add("independent_reference_recompute.json")
        artifacts = complete.get("artifacts")
        if (
            complete.get("schema_version") != 2
            or complete.get("status") != "COMPLETE"
            or complete.get("method_id") != base_method
            or complete.get("split") != "test"
            or complete.get("oracle") is not oracle
            or int(complete.get("sample_count", -1)) != expected_count
            or complete.get("experiment_lock_sha256") != lock["manifest_content_sha256"]
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != artifact_names
            or any(
                artifacts[name] != sha256_file(directory / name)
                for name in artifact_names
            )
        ):
            raise FinalReportError(f"formal source completion mismatch: {method_id}")
        if (
            run_config.get("method_id") != base_method
            or run_config.get("split") != "test"
            or run_config.get("oracle") is not oracle
            or int(run_config.get("sample_count", -1)) != expected_count
            or run_config.get("experiment_lock_sha256")
            != lock["manifest_content_sha256"]
            or run_config.get("samples_manifest_sha256")
            != lock["artifacts"]["test_samples"]["sha256"]
            or run_config.get("labels_manifest_sha256")
            != lock["artifacts"]["test_labels"]["sha256"]
            or run_config.get("per_sample_sha256")
            != expected_record["per_sample_sha256"]
            or run_config.get("per_candidate_sha256")
            != expected_record["per_candidate_sha256"]
        ):
            raise FinalReportError(
                f"formal source run provenance mismatch: {method_id}"
            )
        if method_id == "R0":
            if (
                complete.get("config_sha256") is not None
                or run_config.get("config_sha256") is not None
            ):
                raise FinalReportError(
                    "R0 formal source unexpectedly claims a backend config"
                )
            try:
                validate_r0_reference_evidence(
                    directory, lock=lock, expected_count=expected_count
                )
            except (KeyError, TypeError, ValueError) as error:
                raise FinalReportError("R0 retained-source lineage mismatch") from error
        else:
            locked_config = lock.get("artifacts", {}).get(f"config_{base_method}")
            if (
                not isinstance(locked_config, Mapping)
                or complete.get("config_sha256") != locked_config.get("sha256")
                or run_config.get("config_sha256") != locked_config.get("sha256")
            ):
                raise FinalReportError(
                    f"formal source selected-config mismatch: {method_id}"
                )
        samples = pd.read_parquet(directory / "per_sample_predictions.parquet")
        candidates = pd.read_parquet(directory / "per_candidate_predictions.parquet")
        if (
            len(samples) != expected_count
            or samples["sample_id"].duplicated().any()
            or tuple(samples["sample_id"].astype(str)) != expected_ids
        ):
            raise FinalReportError(
                f"formal source sample coverage mismatch: {method_id}"
            )
        if "method" in samples:
            samples = samples.rename(columns={"method": "method_name"})
        if "method" in candidates:
            candidates = candidates.rename(columns={"method": "method_name"})
        samples.insert(0, "method_id", method_id)
        samples.insert(1, "method", method_id)
        candidates.insert(0, "method_id", method_id)
        candidates.insert(1, "method", method_id)
        sample_frames.append(samples)
        candidate_frames.append(candidates)
    expected_samples = pd.concat(sample_frames, ignore_index=True)
    expected_candidates = pd.concat(candidate_frames, ignore_index=True)
    observed_samples = pd.read_parquet(per_sample_path)
    observed_candidates = pd.read_parquet(per_candidate_path)
    try:
        pd.testing.assert_frame_equal(
            observed_samples,
            expected_samples,
            check_dtype=False,
            check_exact=True,
            check_like=False,
        )
        pd.testing.assert_frame_equal(
            observed_candidates,
            expected_candidates,
            check_dtype=False,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as error:
        raise FinalReportError(
            "consolidated predictions are not the exact formal-source concatenation"
        ) from error


def _verify_protected_sources(
    source_inventory: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    source = source_inventory.get("source")
    if not isinstance(source, Mapping):
        raise FinalReportError("source inventory lacks source record")
    checkpoint = _require_file(Path(str(source.get("checkpoint_path", ""))))
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != source.get("checkpoint_sha256"):
        raise FinalReportError("repeated-FiLM source checkpoint changed after audit")
    protected = reference.get("protected_key_files_before")
    if not isinstance(protected, Mapping) or not protected:
        raise FinalReportError("reference audit lacks protected_key_files_before")
    verified: dict[str, dict[str, Any]] = {}
    for name, raw in protected.items():
        if not isinstance(raw, Mapping):
            raise FinalReportError(f"protected reference record is malformed: {name}")
        path = _require_file(Path(str(raw.get("path", ""))))
        observed = sha256_file(path)
        if observed != raw.get("sha256"):
            raise FinalReportError(f"protected reference artifact changed: {name}")
        verified[str(name)] = {"path": str(path), "sha256": observed}
    return {
        "repeatedfilm_checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "reference_files": verified,
    }


def _validate_protected_verification(
    protected_final: Mapping[str, Any],
    *,
    source_inventory: Mapping[str, Any],
    reference: Mapping[str, Any],
    input_preflight: Mapping[str, Any],
    gr: Mapping[str, Any],
    gg: Mapping[str, Any],
) -> None:
    checks = protected_final.get("checks")
    required = {
        "repeatedfilm_checkpoint",
        "repeatedfilm_config",
        "reference",
        "reference_direct_inputs",
        "vendors",
    }
    if not isinstance(checks, Mapping) or set(checks) != required:
        raise FinalReportError(
            "final protected-source verification checks are incomplete"
        )

    def verify_leaf(row: Any, *, label: str) -> None:
        if not isinstance(row, Mapping) or row.get("status") != "UNCHANGED":
            raise FinalReportError(f"protected verification leaf failed: {label}")
        path = _require_file(Path(str(row.get("path", ""))).resolve(), label=label)
        if row.get("sha256") != sha256_file(path):
            raise FinalReportError(f"protected verification leaf drift: {label}")

    source = source_inventory.get("source")
    if not isinstance(source, Mapping):
        raise FinalReportError("source inventory lacks repeated-FiLM source record")
    for check_name, path_field, hash_field, label in (
        (
            "repeatedfilm_checkpoint",
            "checkpoint_path",
            "checkpoint_sha256",
            "repeated-FiLM checkpoint",
        ),
        (
            "repeatedfilm_config",
            "config_path",
            "config_sha256",
            "repeated-FiLM config",
        ),
    ):
        checked = checks[check_name]
        verify_leaf(checked, label=label)
        if Path(str(checked["path"])).resolve() != Path(
            str(source.get(path_field, ""))
        ).resolve() or checked["sha256"] != source.get(hash_field):
            raise FinalReportError(f"protected {label} does not match source audit")
    for group_name in ("reference", "reference_direct_inputs"):
        group = checks[group_name]
        if not isinstance(group, Mapping) or not group:
            raise FinalReportError(
                f"protected verification group is empty: {group_name}"
            )
        for name, row in group.items():
            verify_leaf(row, label=f"{group_name}/{name}")
    reference_inventory = reference.get("protected_key_files_before")
    if not isinstance(reference_inventory, Mapping) or set(reference_inventory) != set(
        checks["reference"]
    ):
        raise FinalReportError("protected reference inventory coverage mismatch")
    for name, record in reference_inventory.items():
        checked = checks["reference"][name]
        if (
            Path(str(checked["path"])).resolve() != Path(str(record["path"])).resolve()
            or checked["sha256"] != record["sha256"]
        ):
            raise FinalReportError(f"protected reference inventory mismatch: {name}")
    direct_inputs = input_preflight.get("r0", {}).get("direct_inputs")
    if not isinstance(direct_inputs, Mapping) or set(direct_inputs) != set(
        checks["reference_direct_inputs"]
    ):
        raise FinalReportError("protected R0 direct-input coverage mismatch")
    for name, record in direct_inputs.items():
        checked = checks["reference_direct_inputs"][name]
        if (
            Path(str(checked["path"])).resolve() != Path(str(record["path"])).resolve()
            or checked["sha256"] != record["sha256"]
        ):
            raise FinalReportError(f"protected R0 direct-input mismatch: {name}")
    vendors = checks["vendors"]
    if not isinstance(vendors, Mapping) or set(vendors) != {"grconvnet", "ggcnn2"}:
        raise FinalReportError("protected vendor coverage mismatch")
    for name, manifest in (("grconvnet", gr), ("ggcnn2", gg)):
        checked = vendors[name]
        if (
            not isinstance(checked, Mapping)
            or checked.get("clean") is not True
            or checked.get("commit") != manifest.get("pinned_commit")
            or checked.get("repository") != manifest.get("repository")
        ):
            raise FinalReportError(f"protected vendor verification mismatch: {name}")
        files = checked.get("files")
        if not isinstance(files, list) or not files:
            raise FinalReportError(f"protected vendor file checks are empty: {name}")
        for index, row in enumerate(files):
            verify_leaf(row, label=f"vendor/{name}/{index}")
        expected_files = [
            *manifest.get("source_files", {}).values(),
            *manifest.get("checkpoints", []),
            manifest.get("license", {}),
        ]
        expected = {
            (str(Path(str(row["path"])).resolve()), str(row["sha256"]))
            for row in expected_files
        }
        observed = {
            (str(Path(str(row["path"])).resolve()), str(row["sha256"])) for row in files
        }
        if observed != expected:
            raise FinalReportError(f"protected vendor file set mismatch: {name}")


def _validate_statistics(
    statistical: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    *,
    predictions_path: Path,
) -> None:
    tests = statistical.get("tests")
    intervals = bootstrap.get("intervals")
    if not isinstance(tests, list) or not tests:
        raise FinalReportError("statistical_tests.json has no tests")
    if not isinstance(intervals, list) or not intervals:
        raise FinalReportError("bootstrap_intervals.json has no intervals")
    required_pairs = {
        "R0_vs_G1",
        "R0_vs_C1",
        "R0_vs_A0",
        "G1_vs_C1",
        "G1_vs_A0",
        "C1_vs_A0",
        "G0_vs_G1",
        "C0_vs_C1",
        "G0_vs_G0-O",
        "G1_vs_G1-O",
        "C0_vs_C0-O",
        "C1_vs_C1-O",
        "A0_vs_A0-O",
    }
    test_ids = {str(row.get("pair_id")) for row in tests if isinstance(row, Mapping)}
    interval_ids = {
        str(row.get("pair_id")) for row in intervals if isinstance(row, Mapping)
    }
    if not required_pairs.issubset(test_ids) or not required_pairs.issubset(
        interval_ids
    ):
        raise FinalReportError("statistics artifacts omit preregistered comparisons")
    if int(bootstrap.get("bootstrap_draws", 0)) < 10_000:
        raise FinalReportError("scene bootstrap used fewer than 10,000 draws")
    provenance = statistical.get("provenance")
    if not isinstance(provenance, Mapping) or provenance != bootstrap.get("provenance"):
        raise FinalReportError("statistics provenance is missing or inconsistent")
    pair_records = [asdict(pair) for pair in DEFAULT_PAIR_SPECS]
    if (
        Path(str(provenance.get("predictions_path", ""))).resolve()
        != predictions_path.resolve()
        or provenance.get("predictions_sha256") != sha256_file(predictions_path)
        or provenance.get("pair_specifications") != pair_records
        or provenance.get("pair_specifications_sha256")
        != canonical_json_sha256(pair_records)
    ):
        raise FinalReportError("statistics input or pair-spec provenance mismatch")
    aligned = load_aligned_predictions(predictions_path)
    expected_statistical, expected_bootstrap = analyze_paired_methods(
        aligned,
        DEFAULT_PAIR_SPECS,
        bootstrap_draws=int(bootstrap["bootstrap_draws"]),
        seed=int(bootstrap["seed"]),
        alpha=float(statistical["multiple_comparison"]["alpha"]),
    )
    expected_statistical["provenance"] = dict(provenance)
    expected_bootstrap["provenance"] = dict(provenance)
    if (
        dict(statistical) != expected_statistical
        or dict(bootstrap) != expected_bootstrap
    ):
        raise FinalReportError(
            "saved statistics do not exactly recompute from predictions"
        )


def _validate_consolidated_aggregates(
    *,
    per_sample_path: Path,
    common_subset: pd.DataFrame,
    oracle: pd.DataFrame,
    formal: pd.DataFrame,
    primary: str,
    expected_count: int,
) -> None:
    expected_methods = {
        "R0",
        "G0",
        "G1",
        "C0",
        "C1",
        "A0",
        "G0-O",
        "G1-O",
        "C0-O",
        "C1-O",
        "A0-O",
    }
    _require_columns(common_subset, FORMAL_COLUMNS, label="common-subset metrics")
    if (
        len(common_subset) != len(expected_methods)
        or set(common_subset["method_id"].astype(str)) != expected_methods
    ):
        raise FinalReportError("common-subset method coverage mismatch")
    sample_table = pq.read_table(per_sample_path)
    sample_frame = sample_table.to_pandas()
    _require_columns(
        sample_frame,
        (
            "method_id",
            "method",
            "sample_id",
            "scene_id",
            "j_at_1",
            "j_at_5",
            "candidate_pool_oracle",
            "first_valid_rank",
            "reciprocal_rank",
            "non_empty",
            "raw_candidate_count",
            "nms_candidate_count",
            "latency_seconds",
        ),
        label="per-sample predictions",
    )
    for method_id in sorted(expected_methods):
        group = sample_frame.loc[
            sample_frame["method_id"].astype(str) == method_id
        ].copy()
        if (
            len(group) != expected_count
            or group["sample_id"].astype(str).duplicated().any()
        ):
            raise FinalReportError(f"per-sample method coverage mismatch: {method_id}")
        declared = (
            common_subset.loc[common_subset["method_id"].astype(str) == method_id]
            .iloc[0]
            .to_dict()
        )
        group["method"] = str(declared["method"])
        try:
            assert_aggregate_matches_sample_rows(
                pa.Table.from_pandas(group, preserve_index=False).to_pylist(), declared
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FinalReportError(
                f"saved aggregate does not match per-sample outcomes: {method_id}"
            ) from error
    by_common = common_subset.set_index(common_subset["method_id"].astype(str))
    for method_id in PREDICTED_METHODS:
        formal_row = formal.loc[formal["method_id"].astype(str) == method_id].iloc[0]
        for field in FORMAL_COLUMNS[1:]:
            if not math.isclose(
                float(formal_row[field]),
                float(by_common.loc[method_id, field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise FinalReportError(
                    f"formal/common-subset metric drift: {method_id}/{field}"
                )
    primary_row = formal.loc[
        formal["method_id"].astype(str) == "locked_primary_4dof_backend"
    ].iloc[0]
    for field in FORMAL_COLUMNS[1:]:
        if not math.isclose(
            float(primary_row[field]),
            float(by_common.loc[primary, field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FinalReportError(f"locked primary metric drift: {field}")
    oracle_by_id = oracle.set_index(oracle["method_id"].astype(str))
    for method_id in ORACLE_METHODS:
        predicted = by_common.loc[method_id]
        gt_mask = by_common.loc[f"{method_id}-O"]
        expected = {
            "pred_mask_j_at_1": float(predicted["j_at_1"]),
            "gt_mask_j_at_1": float(gt_mask["j_at_1"]),
            "j_at_1_gap": float(gt_mask["j_at_1"] - predicted["j_at_1"]),
            "pred_mask_oracle": float(predicted["candidate_pool_oracle"]),
            "gt_mask_oracle": float(gt_mask["candidate_pool_oracle"]),
        }
        for field, value in expected.items():
            if not math.isclose(
                float(oracle_by_id.loc[method_id, field]),
                value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise FinalReportError(f"oracle summary drift: {method_id}/{field}")


def _validate_independent(
    value: Mapping[str, Any], *, run: Path, expected_count: int
) -> None:
    if (
        value.get("status") != "EXACT_MATCH"
        or value.get("independent_recompute") is not True
    ):
        raise FinalReportError("independent recomputation did not report EXACT_MATCH")
    if value.get("configuration_selection_read") is not False:
        raise FinalReportError("independent recomputation read configuration selection")
    methods = value.get("methods")
    if not isinstance(methods, list) or not methods:
        raise FinalReportError("independent recomputation has no method records")
    if any(
        not isinstance(row, Mapping) or row.get("status") != "EXACT_MATCH"
        for row in methods
    ):
        raise FinalReportError(
            "one or more independent method comparisons did not match"
        )
    expected_methods = {
        "R0",
        "G0",
        "G1",
        "C0",
        "C1",
        "A0",
        "G0-O",
        "G1-O",
        "C0-O",
        "C1-O",
        "A0-O",
    }
    observed = {str(row.get("method")) for row in methods if isinstance(row, Mapping)}
    if observed != expected_methods or int(value.get("method_count", -1)) != len(
        expected_methods
    ):
        raise FinalReportError("independent recomputation method coverage mismatch")
    if int(value.get("sample_count", -1)) != expected_count:
        raise FinalReportError("independent recomputation sample count mismatch")
    labels = (run / "manifests/test_labels.parquet").resolve()
    if Path(str(value.get("frozen_test_labels", ""))).resolve() != labels or value.get(
        "frozen_test_labels_sha256"
    ) != sha256_file(labels):
        raise FinalReportError("independent recomputation label provenance mismatch")
    expected_directories = {
        "R0": run / "formal_test/R0",
        "G0": run / "formal_test/G0",
        "G1": run / "formal_test/G1",
        "C0": run / "formal_test/C0",
        "C1": run / "formal_test/C1",
        "A0": run / "formal_test/A0",
        "G0-O": run / "oracle/G0-O",
        "G1-O": run / "oracle/G1-O",
        "C0-O": run / "oracle/C0-O",
        "C1-O": run / "oracle/C1-O",
        "A0-O": run / "oracle/A0-O",
    }
    for row in methods:
        method = str(row["method"])
        directory = expected_directories[method].resolve()
        if Path(str(row.get("method_dir", ""))).resolve() != directory:
            raise FinalReportError(f"independent method directory mismatch: {method}")
        for filename, path_key, hash_key in (
            (
                "per_candidate_predictions.parquet",
                "per_candidate_predictions",
                "per_candidate_predictions_sha256",
            ),
            (
                "per_sample_predictions.parquet",
                "per_sample_predictions",
                "per_sample_predictions_sha256",
            ),
            ("metrics.json", "saved_metrics", "saved_metrics_sha256"),
        ):
            path = (directory / filename).resolve()
            if Path(str(row.get(path_key, ""))).resolve() != path or row.get(
                hash_key
            ) != sha256_file(path):
                raise FinalReportError(
                    f"independent method artifact provenance mismatch: {method}/{filename}"
                )


def _load_artifacts(run: Path) -> dict[str, Any]:
    paths = {relative: _require_file(run / relative) for relative in REQUIRED_FILES}
    r0_evidence_relative = "formal_test/R0/independent_reference_recompute.json"
    paths[r0_evidence_relative] = _require_file(run / r0_evidence_relative)
    source_inventory = _json_object(paths["audit/source_inventory.json"])
    split_audit = _json_object(paths["audit/dataset_split_audit.json"])
    reference = _json_object(paths["audit/reference_baseline_inventory.json"])
    device_smoke = _json_object(paths["audit/basic_device_smoke.json"])
    input_preflight = _json_object(paths["audit/formal_input_reference_preflight.json"])
    protected_final = _json_object(
        paths["audit/protected_source_final_verification.json"]
    )
    gr = _json_object(paths["third_party/grconvnet_source_manifest.json"])
    gg = _json_object(paths["third_party/ggcnn2_source_manifest.json"])
    environment = _json_object(paths["environment.json"])
    selection = _json_object(paths["primary_validation_selection.json"])
    selected_raw = _json_object(paths["selected_configs.json"])
    bundle = _json_object(paths["results_bundle.json"])
    statistical = _json_object(paths["statistical_tests.json"])
    bootstrap = _json_object(paths["bootstrap_intervals.json"])
    subgroup_manifest = _json_object(paths["subgroup_analysis_manifest.json"])
    gallery = _json_object(paths["gallery/selection_manifest.json"])
    independent = _json_object(paths["independent_recompute_results.json"])
    r0_recompute = _json_object(paths[r0_evidence_relative])
    lock = _json_object(paths["manifests/experiment_lock.json"])
    marker = _json_object(paths[".EXPERIMENT_LOCKED"])
    runtime = _json_object(paths["runtime_metrics.json"])
    memory = _json_object(paths["memory_metrics.json"])
    storage_budget = _json_object(paths["storage_budget.json"])
    storage_usage = pd.read_csv(paths["storage_usage_by_stage.csv"])

    validation = pd.read_csv(paths["validation_results.csv"])
    formal = pd.read_csv(paths["formal_test_results.csv"])
    per_method = pd.read_csv(paths["per_method_metrics.csv"])
    oracle = pd.read_csv(paths["oracle_results.csv"])
    common_subset = pd.read_csv(paths["common_subset_comparison.csv"])
    training_curves = pd.read_csv(paths["training_curves.csv"])
    subgroup = pd.read_csv(paths["subgroup_results.csv"])
    failure = pd.read_parquet(paths["per_sample_failure_stage.parquet"])

    if source_inventory.get("status") != "PHASE_0_PASS":
        raise FinalReportError("source audit did not pass")
    source = source_inventory.get("source", {})
    if (
        not isinstance(source, Mapping)
        or source.get("visual_grounding_variant") != "hierarchical_repeated_film"
        or source.get("single_film_allowed") is not False
        or source.get("checkpoint_strict_load_success") is not True
    ):
        raise FinalReportError("repeated-FiLM audit contract is not satisfied")
    if split_audit.get("all_checks_passed") is not True:
        raise FinalReportError("dataset split audit did not pass")
    for pair, overlap in split_audit.get("pairwise_overlap", {}).items():
        if not isinstance(overlap, Mapping) or any(
            int(value) != 0 for value in overlap.values()
        ):
            raise FinalReportError(f"dataset split leakage recorded for {pair}")
    if reference.get("reference_run_reusable") is not True:
        raise FinalReportError("reference baseline is not marked reusable")
    for name, manifest in (("GR-ConvNet", gr), ("GG-CNN2", gg)):
        if manifest.get("status") != "PASS":
            raise FinalReportError(f"{name} third-party audit did not pass")
        repository = str(manifest.get("repository"))
        if repository not in OFFICIAL_SOURCE_ALLOWLIST:
            raise FinalReportError(
                f"{name} repository is outside the citation allowlist"
            )
    if device_smoke.get("status") != "PASS":
        raise FinalReportError("Mac device smoke audit did not pass")
    entries = device_smoke.get("entries")
    if not isinstance(entries, list) or not entries:
        raise FinalReportError("Mac device smoke artifact has no entries")
    if any(
        not isinstance(row, Mapping) or row.get("status") != "PASS" for row in entries
    ):
        raise FinalReportError("a Mac device smoke entry failed")
    if (
        input_preflight.get("status") != "PASS"
        or input_preflight.get("loader_contract")
        != "row_declared_sha256_verified_fail_closed"
    ):
        raise FinalReportError("formal input reference preflight did not pass")

    validation = _strict_method_rows(
        validation, VALIDATION_METHODS, label="validation results"
    )
    _validate_metric_frame(validation, label="validation results")
    _require_columns(
        validation,
        ("gt_mask_oracle_j_at_1", "predicted_mask_oracle_gap_recovery"),
        label="validation results",
    )
    for row in validation.itertuples(index=False):
        _finite_rate(
            row.gt_mask_oracle_j_at_1,
            field="gt_mask_oracle_j_at_1",
            method=str(row.method_id),
        )
        _finite_rate(
            row.predicted_mask_oracle_gap_recovery,
            field="predicted_mask_oracle_gap_recovery",
            method=str(row.method_id),
        )
    formal_display = _strict_method_rows(
        formal, FORMAL_DISPLAY_METHODS, label="formal results"
    )
    _validate_metric_frame(formal_display, label="formal predicted-mask results")
    formal_predicted = _strict_method_rows(
        formal_display, PREDICTED_METHODS, label="formal base methods"
    )
    per_method_display = _strict_method_rows(
        per_method, FORMAL_DISPLAY_METHODS, label="per-method metrics"
    )
    _validate_metric_frame(per_method_display, label="per-method metrics")
    left = formal_display.loc[:, list(FORMAL_COLUMNS)].reset_index(drop=True)
    right = per_method_display.loc[:, list(FORMAL_COLUMNS)].reset_index(drop=True)
    if not left.equals(right):
        raise FinalReportError(
            "formal_test_results.csv and per_method_metrics.csv drift"
        )
    _require_columns(oracle, ORACLE_COLUMNS, label="oracle results")
    oracle = _strict_method_rows(oracle, ORACLE_METHODS, label="oracle results")
    for row in oracle.itertuples(index=False):
        for field in ORACLE_COLUMNS[1:]:
            value = float(getattr(row, field))
            if not math.isfinite(value):
                raise FinalReportError(
                    f"{row.method_id}: non-finite oracle field {field}"
                )
    _require_columns(failure, FAILURE_COLUMNS, label="failure analysis")
    if set(failure["analysis_scope"].astype(str)) != {"formal_test_only"}:
        raise FinalReportError("failure analysis is not formal-test-only")
    if any(bool(value) for value in failure["configuration_selection_performed"]):
        raise FinalReportError("failure analysis performed configuration selection")
    if (
        gallery.get("analysis_scope") != "formal_test_only"
        or gallery.get("configuration_selection_performed") is not False
    ):
        raise FinalReportError("gallery is not a pure formal-test analysis")
    if (
        selection.get("selection_split") != "validation"
        or selection.get("test_metrics_read") is not False
    ):
        raise FinalReportError("primary selection was not validation-only")
    primary = str(selection.get("primary_method_id"))
    if primary not in PRIMARY_CANDIDATES:
        raise FinalReportError("locked primary is not G1, C1, or A0")
    expected_test_count = int(lock["protocol"]["expected_test_sample_count"])
    if (
        bundle.get("status") != "COMPLETE"
        or bundle.get("locked_primary_method_id") != primary
    ):
        raise FinalReportError("results bundle/validation primary mismatch")
    _validate_bundle_outputs(bundle, run=run, expected_count=expected_test_count)
    if lock.get("protocol", {}).get("primary_method") != primary:
        raise FinalReportError("experiment lock/validation primary mismatch")
    _validate_lock(
        run=run,
        lock=lock,
        frozen_lock_path=paths["frozen_4dof_backends_experiment_manifest.json"],
        marker=marker,
    )
    _validate_consolidated_sources(
        bundle,
        run=run,
        lock=lock,
        expected_count=expected_test_count,
        per_sample_path=paths["per_sample_predictions.parquet"],
        per_candidate_path=paths["per_candidate_predictions.parquet"],
    )
    if (
        protected_final.get("status") != "PASS"
        or protected_final.get("all_protected_sources_unchanged") is not True
        or protected_final.get("experiment_lock_sha256")
        != lock.get("manifest_content_sha256")
    ):
        raise FinalReportError("final protected-source verification did not pass")
    _validate_protected_verification(
        protected_final,
        source_inventory=source_inventory,
        reference=reference,
        input_preflight=input_preflight,
        gr=gr,
        gg=gg,
    )
    selected_configs = _load_selected_configs(
        selected_raw, expected_methods=VALIDATION_METHODS
    )
    _validate_statistics(
        statistical,
        bootstrap,
        predictions_path=paths["per_sample_predictions.parquet"],
    )
    _require_columns(subgroup, SUBGROUP_COLUMNS, label="subgroup results")
    expected_dimensions = {
        "query_type",
        "target_area_group",
        "predicted_mask_iou_group",
        "predicted_mask_confidence_group",
        "candidate_count_group",
        "depth_validity_group",
        "object_width_group",
        "scene_clutter_group",
        "no_grasp_reason_group",
        "first_valid_rank_group",
    }
    if not expected_dimensions.issubset(set(subgroup["dimension"].astype(str))):
        raise FinalReportError("subgroup results omit required diagnostic dimensions")
    if (
        subgroup_manifest.get("analysis_scope") != "formal_test_only"
        or subgroup_manifest.get("configuration_selection_performed") is not False
    ):
        raise FinalReportError("subgroup analysis is not formal-test-only")
    if subgroup_manifest.get("gt_derived_features_offline_analysis_only") is not True:
        raise FinalReportError("subgroup GT-derived feature boundary is missing")
    _validate_parquet_schema(
        paths["subgroup_sample_features.parquet"],
        ("method_id", "sample_id", "query_type", "predicted_mask_iou"),
        label="subgroup sample features",
    )
    _validate_independent(independent, run=run, expected_count=expected_test_count)
    _validate_consolidated_aggregates(
        per_sample_path=paths["per_sample_predictions.parquet"],
        common_subset=common_subset,
        oracle=oracle,
        formal=formal_display,
        primary=primary,
        expected_count=expected_test_count,
    )
    protected = _verify_protected_sources(source_inventory, reference)
    _validate_parquet_schema(
        paths["per_sample_predictions.parquet"],
        ("method", "sample_id", "j_at_1"),
        label="per-sample predictions",
    )
    _validate_parquet_schema(
        paths["per_candidate_predictions.parquet"],
        ("method", "sample_id", "candidate_id", "rank"),
        label="per-candidate predictions",
    )
    if common_subset.empty or training_curves.empty:
        raise FinalReportError("common-subset comparison or training curves are empty")
    failure_counts = failure.groupby(failure["method"].astype(str)).size().to_dict()
    if set(failure_counts) != set(PREDICTED_METHODS) or any(
        int(count) != expected_test_count for count in failure_counts.values()
    ):
        raise FinalReportError(
            "failure analysis does not cover every predicted method/sample"
        )
    if failure.duplicated(["method", "sample_id"]).any():
        raise FinalReportError("failure analysis has duplicate method/sample rows")
    formal_sample_ids = pd.read_parquet(
        paths["per_sample_predictions.parquet"], columns=["method_id", "sample_id"]
    )
    for method in PREDICTED_METHODS:
        expected_ids = set(
            formal_sample_ids.loc[
                formal_sample_ids["method_id"].astype(str) == method, "sample_id"
            ].astype(str)
        )
        observed_ids = set(
            failure.loc[failure["method"].astype(str) == method, "sample_id"].astype(
                str
            )
        )
        if observed_ids != expected_ids:
            raise FinalReportError(
                f"failure analysis sample identity mismatch: {method}"
            )
    selection_counts = gallery.get("selection", {}).get("counts", {})
    if set(selection_counts) != set(PREDICTED_METHODS):
        raise FinalReportError("gallery method coverage mismatch")
    for method, categories in selection_counts.items():
        if set(categories) != GALLERY_CATEGORIES:
            raise FinalReportError(f"gallery category coverage mismatch: {method}")
        for values in categories.values():
            requested = int(values["requested"])
            available = int(values["available"])
            actual = int(values["actual"])
            if actual != min(requested, available):
                raise FinalReportError(
                    f"gallery did not exhaust requested/available quota: {method}"
                )
    cross = gallery.get("selection", {}).get("cross_method_count", {})
    if int(cross.get("actual", -1)) != min(
        int(cross.get("requested", -1)), int(cross.get("available", -1))
    ):
        raise FinalReportError("cross-method gallery quota mismatch")
    if int(storage_budget.get("budget_bytes", 0)) <= 0:
        raise FinalReportError("storage budget is invalid")
    _require_columns(
        storage_usage,
        ("stage", "run_bytes", "budget_bytes", "within_budget"),
        label="storage usage",
    )
    final_storage = storage_usage.loc[
        storage_usage["stage"] == "formal_complete_before_reports"
    ]
    within_budget = (
        False
        if len(final_storage) != 1
        else str(final_storage.iloc[0]["within_budget"]).strip().lower() == "true"
    )
    if len(final_storage) != 1 or not within_budget:
        raise FinalReportError("final storage-budget check did not pass")
    for label, telemetry in (("runtime", runtime), ("memory", memory)):
        if not set(PREDICTED_METHODS).issubset(telemetry):
            raise FinalReportError(f"{label} telemetry omits a predicted method")
        if any(
            not isinstance(telemetry[method], Mapping) for method in PREDICTED_METHODS
        ):
            raise FinalReportError(f"{label} telemetry contains an empty method record")
    for method in PREDICTED_METHODS:
        runtime_row = runtime[method]
        for field in (
            "p50_backend_latency_seconds",
            "p95_backend_latency_seconds",
            "throughput_samples_per_second",
        ):
            value = float(runtime_row[field])
            if not math.isfinite(value) or value < 0.0:
                raise FinalReportError(f"runtime telemetry invalid: {method}/{field}")
        if (
            int(runtime_row.get("sample_count", expected_test_count))
            != expected_test_count
        ):
            raise FinalReportError(f"runtime telemetry sample count mismatch: {method}")
        memory_row = memory[method]
        unavailable = memory_row.get("measurement_available") is False
        for field in ("peak_rss_bytes", "peak_mps_allocated_bytes"):
            raw_value = memory_row.get(field)
            if raw_value is None:
                if not unavailable or not str(memory_row.get("reason", "")).strip():
                    raise FinalReportError(
                        f"memory telemetry unavailable without disclosure: {method}/{field}"
                    )
                continue
            if int(raw_value) < 0:
                raise FinalReportError(f"memory telemetry invalid: {method}/{field}")

    return {
        "paths": paths,
        "evidence_hashes": {
            relative: sha256_file(path) for relative, path in paths.items()
        },
        "source_inventory": source_inventory,
        "split_audit": split_audit,
        "reference": reference,
        "device_smoke": device_smoke,
        "input_preflight": input_preflight,
        "protected_final": protected_final,
        "gr": gr,
        "gg": gg,
        "environment": environment,
        "selection": selection,
        "selected_configs": selected_configs,
        "bundle": bundle,
        "validation": validation,
        "formal": formal_display,
        "formal_base": formal_predicted,
        "oracle": oracle,
        "statistical": statistical,
        "bootstrap": bootstrap,
        "subgroup": subgroup,
        "subgroup_manifest": subgroup_manifest,
        "failure": failure,
        "gallery": gallery,
        "independent": independent,
        "r0_recompute": r0_recompute,
        "runtime": runtime,
        "memory": memory,
        "storage_budget": storage_budget,
        "storage_usage": storage_usage,
        "protected": protected,
        "lock": lock,
    }


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    heading = "| " + " | ".join(_escape(value) for value in headers) + " |"
    divider = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(_escape(value) for value in row) + " |" for row in rows]
    return "\n".join([heading, divider, *body])


def _rate(value: Any) -> str:
    return f"{float(value):.4f}"


def _seconds(value: Any) -> str:
    return f"{float(value):.6f}"


def _metric_table(frame: pd.DataFrame) -> str:
    return _table(
        (
            "Method",
            "J@1",
            "J@5/J@Any",
            "Oracle",
            "MRR",
            "Non-empty",
            "No-grasp",
            "p50 latency (s)",
            "p95 latency (s)",
        ),
        [
            (
                f"{row.method_id} — {METHOD_LABELS[str(row.method_id)]}",
                _rate(row.j_at_1),
                _rate(row.j_at_5),
                _rate(row.candidate_pool_oracle),
                _rate(row.mrr),
                _rate(row.non_empty_rate),
                _rate(row.no_grasp_rate),
                _seconds(row.p50_latency_seconds),
                _seconds(row.p95_latency_seconds),
            )
            for row in frame.itertuples(index=False)
        ],
    )


def _oracle_table(frame: pd.DataFrame) -> str:
    return _table(
        (
            "Method",
            "Pred-mask J@1",
            "GT-mask J@1",
            "Gap (GT−pred)",
            "Pred-mask Oracle",
            "GT-mask Oracle",
        ),
        [
            (
                row.method_id,
                _rate(row.pred_mask_j_at_1),
                _rate(row.gt_mask_j_at_1),
                f"{float(row.j_at_1_gap):+.4f}",
                _rate(row.pred_mask_oracle),
                _rate(row.gt_mask_oracle),
            )
            for row in frame.itertuples(index=False)
        ],
    )


def _validation_oracle_table(frame: pd.DataFrame) -> str:
    _require_columns(
        frame,
        (
            "method_id",
            "j_at_1",
            "gt_mask_oracle_j_at_1",
            "predicted_mask_oracle_gap_recovery",
        ),
        label="validation oracle results",
    )
    return _table(
        ("Method", "Pred-mask J@1", "GT-mask J@1", "Retained fraction"),
        [
            (
                row.method_id,
                _rate(row.j_at_1),
                _rate(row.gt_mask_oracle_j_at_1),
                _rate(row.predicted_mask_oracle_gap_recovery),
            )
            for row in frame.itertuples(index=False)
        ],
    )


def _statistics_table(
    statistical: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> str:
    intervals = {str(row["pair_id"]): row for row in bootstrap["intervals"]}
    rows = []
    for test in statistical["tests"]:
        interval = intervals[str(test["pair_id"])]
        rows.append(
            (
                test["pair_id"],
                f"{float(test['delta_j_at_1_b_minus_a']):+.4f}",
                f"{float(test['p_value_exact_two_sided']):.6g}",
                f"{float(test['p_value_holm']):.6g}",
                str(bool(test["reject_holm_at_alpha"])),
                f"[{float(interval['ci_lower']):+.4f}, {float(interval['ci_upper']):+.4f}]",
            )
        )
    return _table(
        (
            "Pair",
            "ΔJ@1 (B−A)",
            "Exact p",
            "Holm p",
            "Holm reject",
            "Scene-bootstrap 95% CI",
        ),
        rows,
    )


def _subgroup_summary(frame: pd.DataFrame, primary: str) -> str:
    selected = frame.loc[frame["method_id"].astype(str) == primary].copy()
    if selected.empty:
        raise FinalReportError(f"subgroup results omit locked primary {primary}")
    rows = []
    for dimension, group in selected.groupby("dimension", sort=True):
        minimum = group.sort_values(
            ["j_at_1", "group"], ascending=[True, True], kind="mergesort"
        ).iloc[0]
        maximum = group.sort_values(
            ["j_at_1", "group"], ascending=[False, True], kind="mergesort"
        ).iloc[0]
        rows.append(
            (
                dimension,
                len(group),
                f"{minimum['group']} ({float(minimum['j_at_1']):.4f}, n={int(minimum['sample_count'])})",
                f"{maximum['group']} ({float(maximum['j_at_1']):.4f}, n={int(maximum['sample_count'])})",
            )
        )
    return _table(
        ("Dimension", "Groups", "Lowest primary J@1", "Highest primary J@1"), rows
    )


def _source_links(gr: Mapping[str, Any], gg: Mapping[str, Any]) -> str:
    rows = []
    for name, manifest in (("GR-ConvNet", gr), ("GG-CNN2", gg)):
        repository = str(manifest["repository"])
        links = OFFICIAL_SOURCE_ALLOWLIST[repository]
        rows.append(
            (
                name,
                f"[{links['project']}]({links['project']})",
                f"[{links['paper']}]({links['paper']})",
                f"[{links['doi']}]({links['doi']})",
                manifest["pinned_commit"],
            )
        )
    return _table(
        ("Backend", "Official project", "Paper", "DOI", "Locked commit"), rows
    )


def _scope_block() -> str:
    return f"""## Scientific scope and limitations

{SCIENTIFIC_SCOPE_SENTENCE}

The study reports offline annotation consistency, not physical grasp success.
No physical robot experiment was conducted, and the evidence does not establish
force closure, collision-free motion, robot reachability, or successful lifting.
The GT-mask oracle is a non-deployable diagnostic upper bound. The visible
collision proxy is not a complete collision check. All methods use a fixed-height
2D rectangle convention and single-view depth; Mac MPS operator support may still
require CPU post-processing or fallback.
"""


def _evidence_ledger(evidence: Mapping[str, str]) -> str:
    rows = [(path, digest) for path, digest in sorted(evidence.items())]
    return "## Evidence ledger\n\n" + _table(
        ("Run artifact", "SHA-256 read by reporter"), rows
    )


def _config_table(configs: Mapping[str, Mapping[str, Any]]) -> str:
    rows = []
    for method in VALIDATION_METHODS:
        record = configs[method]
        config = record["config"]
        rows.append(
            (
                method,
                config.get(
                    "conditioning_variant",
                    "analytic_mask_depth" if method == "A0" else "unavailable",
                ),
                config.get(
                    "device", "CPU geometry" if method == "A0" else "unavailable"
                ),
                config.get("input_size", "native"),
                record["sha256"],
                record["path"],
            )
        )
    return _table(
        (
            "Method",
            "Conditioning",
            "Device",
            "Input size",
            "Config SHA-256",
            "Config path",
        ),
        rows,
    )


def _failure_summary(failure: pd.DataFrame) -> tuple[str, dict[str, int], str]:
    methods = sorted(failure["method"].astype(str).unique())
    stages = sorted(failure["failure_stage"].astype(str).unique())
    pivot = pd.crosstab(
        failure["method"].astype(str), failure["failure_stage"].astype(str)
    )
    table = _table(
        ("Method", *stages),
        [
            (method, *(int(pivot.loc[method].get(stage, 0)) for stage in stages))
            for method in methods
        ],
    )
    categories = {
        "grounding": {
            "grounding_wrong_target",
            "grounding_fragmented_mask",
            "empty_mask",
        },
        "candidate_generation": {
            "invalid_depth",
            "no_candidate_generated",
            "candidate_pool_has_no_positive",
            "crop_mapping_failure",
            "visible_collision_proxy_failure",
        },
        "ranking": {"ranking_failure"},
        "angle": {"angle_failure"},
        "width": {"width_failure"},
        "evaluator": {"evaluator_ambiguity"},
    }
    observed = Counter(failure["failure_stage"].astype(str))
    totals = {
        category: int(sum(observed.get(stage, 0) for stage in member_stages))
        for category, member_stages in categories.items()
    }
    maximum = max(totals.values(), default=0)
    bottlenecks = sorted(name for name, count in totals.items() if count == maximum)
    bottleneck_text = ", ".join(bottlenecks) if bottlenecks else "unavailable"
    return table, totals, bottleneck_text


def _best_methods(
    frame: pd.DataFrame, field: str, *, minimise: bool = False
) -> tuple[list[str], float]:
    values = frame[field].astype(float)
    target = float(values.min() if minimise else values.max())
    methods = sorted(
        frame.loc[np.isclose(values, target, rtol=0.0, atol=1e-12), "method_id"].astype(
            str
        )
    )
    return methods, target


def _pair_maps(
    statistical: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    return (
        {str(row["pair_id"]): row for row in statistical["tests"]},
        {str(row["pair_id"]): row for row in bootstrap["intervals"]},
    )


def _conclusion_answers(artifacts: Mapping[str, Any]) -> str:
    formal = artifacts["formal_base"]
    oracle = artifacts["oracle"]
    primary = str(artifacts["selection"]["primary_method_id"])
    tests, intervals = _pair_maps(artifacts["statistical"], artifacts["bootstrap"])
    highest_j1, highest_j1_value = _best_methods(formal, "j_at_1")
    highest_j5, highest_j5_value = _best_methods(formal, "j_at_5")
    highest_oracle, highest_oracle_value = _best_methods(
        formal, "candidate_pool_oracle"
    )
    lowest_empty, lowest_empty_value = _best_methods(
        formal, "no_grasp_rate", minimise=True
    )
    fastest, fastest_value = _best_methods(formal, "p50_latency_seconds", minimise=True)
    by_id = formal.set_index("method_id")
    gr_delta = float(by_id.loc["G1", "j_at_1"] - by_id.loc["G0", "j_at_1"])
    gg_delta = float(by_id.loc["C1", "j_at_1"] - by_id.loc["C0", "j_at_1"])
    gg_test = tests["C0_vs_C1"]
    gg_significant = bool(gg_test["reject_holm_at_alpha"]) and gg_delta > 0.0
    learned_best = max(
        float(by_id.loc["G1", "j_at_1"]), float(by_id.loc["C1", "j_at_1"])
    )
    analytic_delta = float(by_id.loc["A0", "j_at_1"] - learned_best)
    oracle_gaps = ", ".join(
        f"{row.method_id}={float(row.j_at_1_gap):+.4f}"
        for row in oracle.itertuples(index=False)
    )
    _, failure_totals, bottleneck = _failure_summary(artifacts["failure"])
    primary_pair_id = f"R0_vs_{primary}"
    primary_test = tests[primary_pair_id]
    primary_interval = intervals[primary_pair_id]
    primary_delta = float(primary_test["delta_j_at_1_b_minus_a"])
    reliable_improvement = (
        bool(primary_test["reject_holm_at_alpha"])
        and primary_delta > 0.0
        and float(primary_interval["ci_lower"]) > 0.0
    )
    yes_no = "Yes" if gg_significant else "No"
    reliable_text = "Yes" if reliable_improvement else "No"
    questions = (
        (
            "哪种方案在 predicted-mask 条件下 J@1 最高？",
            f"{', '.join(highest_j1)}，J@1={highest_j1_value:.4f}。",
        ),
        (
            "哪种方案 J@5/Oracle 最高？",
            f"J@5: {', '.join(highest_j5)}={highest_j5_value:.4f}；candidate-pool Oracle: {', '.join(highest_oracle)}={highest_oracle_value:.4f}。",
        ),
        (
            "哪种方案空候选率最低？",
            f"{', '.join(lowest_empty)}，no-grasp rate={lowest_empty_value:.4f}。",
        ),
        (
            "哪种方案在 Mac 上最快？",
            f"按正式 artifact 的 p50 backend latency，{', '.join(fastest)} 最快，为 {fastest_value:.6f} s；这不包含 source-file I/O。",
        ),
        (
            "GR-ConvNet 预训练到 OCID-VLG 的 domain gap 多大？",
            f"以相同 OCID-VLG test evaluator 下 G1−G0 的实测差表示，ΔJ@1={gr_delta:+.4f}。这不是用 Jacquard/Cornell 论文数字替代的结果。",
        ),
        (
            "GG-CNN2 微调是否显著优于预训练迁移？",
            f"{yes_no}。C1−C0 ΔJ@1={gg_delta:+.4f}，Holm-adjusted p={float(gg_test['p_value_holm']):.6g}，reject={bool(gg_test['reject_holm_at_alpha'])}。",
        ),
        (
            "解析几何方法是否能接近学习方法？",
            f"A0 相对 G1/C1 中较高者的 ΔJ@1={analytic_delta:+.4f}。未预注册等效性界值，因此只报告差值，不宣称统计等效。",
        ),
        (
            "GT-mask oracle 与 predicted-mask 的差距多大？",
            f"各方法 GT−pred J@1 gap：{oracle_gaps}。GT-mask oracle 不可部署。",
        ),
        (
            "主要瓶颈是 grounding、candidate generation 还是 ranking？",
            f"按冻结 failure taxonomy 的全方法计数，最大类别为 {bottleneck}；计数={json.dumps(failure_totals, sort_keys=True)}。这是 post-hoc test-only 诊断。",
        ),
        (
            "相对 Dex-Net/GQ-CNN reference 是否有统计可靠提升？",
            f"{reliable_text}。锁定 primary {primary} 相对 R0 的 ΔJ@1={primary_delta:+.4f}，Holm reject={bool(primary_test['reject_holm_at_alpha'])}，95% scene-bootstrap CI=[{float(primary_interval['ci_lower']):+.4f}, {float(primary_interval['ci_upper']):+.4f}]。",
        ),
        (
            "最适合作为 dissertation 新 modular baseline 的方法是什么？",
            f"验证集预先锁定的 {primary}（{METHOD_LABELS[primary]}）。正式 test 结果没有用于改变该选择。",
        ),
        (
            "最适合真实部署进一步验证的方法是什么？",
            f"{primary} 是按验证协议锁定、适合进入后续实体系统验证的候选；当前离线结果不构成真实机器人性能证据。",
        ),
    )
    return "\n\n".join(
        f"{index}. **{question}**  {answer}"
        for index, (question, answer) in enumerate(questions, 1)
    )


def _implementation_report(artifacts: Mapping[str, Any]) -> str:
    gr, gg = artifacts["gr"], artifacts["gg"]
    source = artifacts["source_inventory"]["source"]
    lock = artifacts["lock"]
    return f"""# Repeated-FiLM 4-DoF implementation

## Evidence status

This report was generated after every required final artifact passed the
read-only reporting gate. No metric was recomputed, no formal-test row was
filtered, and no configuration was selected or modified by this command.

## Inputs and separation

- Deployment inputs: RGB, depth, language, and the retained repeated-FiLM predicted mask.
- Repeated-FiLM variant: `{source["visual_grounding_variant"]}` with {source["film_injection_count"]} FiLM injection stages.
- GT masks and annotated grasp rectangles are restricted to training labels,
  validation evaluation, formal evaluation, oracle analysis, and post-hoc failure analysis.
- Locked test sample count: {lock["protocol"]["expected_test_sample_count"]}.
- Unified evaluator: rectangle IoU > 0.25 and 180-degree-periodic angle error <= 30 degrees,
  with a fixed {lock["protocol"]["fixed_grasp_height_px"]} px rectangle height.

## Implemented formal methods

{_table(("ID", "Formal method"), [(method, METHOD_LABELS[method]) for method in PREDICTED_METHODS])}

The GR-ConvNet and GG-CNN2 adapters reuse pinned official source and verified
weights through project-local adapters. The paper implementations were not
copied into the report generator. The analytic method uses predicted-mask and
depth geometry and does not invoke either neural grasp backend.

## Locked selected configurations

{_config_table(artifacts["selected_configs"])}

## Official source references

The following narrow URL allowlist is activated only after the locked audit
repository identity matches exactly.

{_source_links(gr, gg)}

## Backend contracts retained from the locked audit

- GR-ConvNet architecture: `{gr["architecture"]}`.
- GG-CNN2 architecture: `{gg["architecture"]}`.
- GR provenance disclosures: `{json.dumps(gr.get("provenance_disclosures", []), ensure_ascii=False)}`.
- GG provenance disclosures: `{json.dumps(gg.get("provenance_disclosures", []), ensure_ascii=False)}`.

{_scope_block()}

{_evidence_ledger(artifacts["evidence_hashes"])}
"""


def _audit_report(artifacts: Mapping[str, Any]) -> str:
    source = artifacts["source_inventory"]["source"]
    splits = artifacts["split_audit"]["splits"]
    gr, gg = artifacts["gr"], artifacts["gg"]
    env = artifacts["environment"]
    split_rows = [
        (
            name,
            values["samples"],
            values["scenes"],
            values["gt_grasp_rectangles"],
            values["predicted_mask_coverage"],
            values["frozen_manifest_sha256"],
        )
        for name, values in splits.items()
    ]
    protected_rows = [
        (name, record["path"], record["sha256"], "UNCHANGED")
        for name, record in artifacts["protected"]["reference_files"].items()
    ]
    smoke_rows = [
        (
            row["model"],
            row["device"],
            row["dtype"],
            row["input_shape"],
            row["output_shapes"],
            row["status"],
        )
        for row in artifacts["device_smoke"]["entries"]
    ]
    r0_recompute = artifacts["r0_recompute"]
    legacy = r0_recompute["legacy_outcome_comparison"]
    return f"""# Repeated-FiLM 4-DoF audit

## Audit decision

**PASS.** The retained source is five-stage hierarchical repeated-FiLM;
`single_film_allowed=false`; the checkpoint loaded strictly with no missing or
unexpected trainable keys. The effective experiment lock and its frozen alias
are byte-identical and the canonical lock digest is valid.

## Repeated-FiLM lineage

- Checkpoint: `{source["checkpoint_path"]}`.
- SHA-256: `{source["checkpoint_sha256"]}`.
- Format: `{source["checkpoint_format"]}`.
- Architecture: `{source["architecture_signature"]}`.
- Strict keys: {source["checkpoint_loaded_keys"]}/{source["checkpoint_expected_keys"]}.
- Final read-only hash check: **UNCHANGED**.

## Dataset and predicted-mask coverage

{_table(("Split", "Samples", "Scenes", "GT rectangles", "Predicted-mask coverage", "Frozen manifest SHA-256"), split_rows)}

All audited intersections between train, validation, and test are zero for
sample ID, scene ID, RGB hash, depth hash, and paired RGB-D hash.

## Official third-party lineage

{_source_links(gr, gg)}

- GR license: `{gr["license"]}`; checkpoints: `{gr["checkpoints"]}`.
- GG license: `{gg["license"]}`; checkpoints: `{gg["checkpoints"]}`.

## Mac device smoke

{_table(("Model", "Device", "dtype", "Input shape", "Output shapes", "Status"), smoke_rows)}

- Host: `{env.get("chip")}`, `{env.get("machine")}`, macOS `{env.get("macos_version")}`.
- PyTorch: `{env.get("pytorch")}`; MPS built/available: `{env.get("mps_built")}/{env.get("mps_available")}`.
- CUDA available: `{env.get("cuda_available")}`.
- Environment anomaly: `{json.dumps(env.get("environment_setup_anomaly"), ensure_ascii=False)}`.

## Protected reference lineage rechecked at reporting time

{_table(("Artifact", "Path", "SHA-256", "Status"), protected_rows)}

The repeated-FiLM source checkpoint and every protected reference artifact
still match their pre-experiment audit digests.

## R0 evaluator migration

- Frozen source candidates: `{r0_recompute["source_candidates_status"]}`.
- Formal metrics: `{r0_recompute["formal_metrics_status"]}`.
- Legacy success fields used for formal metrics: `{r0_recompute["legacy_success_fields_used_for_formal_metrics"]}`.
- Legacy compatibility status: `{legacy["status"]}`.
- Samples whose legacy success tuple differs from the locked corrected evaluator: {legacy["mismatch_count"]}.
- Mismatch-ID digest: `{legacy["mismatch_sample_ids_sha256"]}`.

The retained candidate geometry, GQ-CNN scores, and rank order are reused
exactly. Legacy `top1_correct`, `top5_correct`, and `oracle_all` columns are
reported only as migration sensitivity evidence; they are not formal labels.

{_scope_block()}

{_evidence_ledger(artifacts["evidence_hashes"])}
"""


def _validation_report(artifacts: Mapping[str, Any]) -> str:
    selection = artifacts["selection"]
    return f"""# Repeated-FiLM 4-DoF validation

## Protocol

Validation used the complete frozen validation split. The selection artifact
records `selection_split=validation` and `test_metrics_read=false`. Test results
were therefore unavailable to checkpoint, preprocessing, threshold, analytic
weight, or primary-method selection.

## Complete predicted-mask validation table

{_metric_table(artifacts["validation"])}

## Complete GT-mask-oracle validation table

{_validation_oracle_table(artifacts["validation"])}

All five G0/G1/C0/C1/A0 oracle rows use the same full validation manifest and
are separately labelled diagnostics. They do not participate in the
predicted-mask primary ranking except through the preregistered gap-recovery
tie-break for G1/C1/A0.

## Selected configurations

{_config_table(artifacts["selected_configs"])}

## Locked primary

- Primary: **{selection["primary_method_id"]} — {METHOD_LABELS[str(selection["primary_method_id"])]}**.
- Tolerance: `{selection.get("rate_tolerance")}`.
- Ordered rule: `{json.dumps(selection.get("selection_rule"), ensure_ascii=False)}`.
- Complete selection trace: `{json.dumps(selection.get("trace"), ensure_ascii=False)}`.

The full table and trace are retained even when a new method underperforms a
pretrained transfer alternative. No adverse row is removed or replaced.

{_scope_block()}

{_evidence_ledger(artifacts["evidence_hashes"])}
"""


def _results_report(artifacts: Mapping[str, Any]) -> str:
    primary = str(artifacts["selection"]["primary_method_id"])
    independent = artifacts["independent"]
    r0_recompute = artifacts["r0_recompute"]
    legacy = r0_recompute["legacy_outcome_comparison"]
    return f"""# Repeated-FiLM 4-DoF formal results

## Formal protocol status

The primary method `{primary}` was frozen on validation before formal test
inference. This report reads every all-sample row and performs no test-time
selection. Empty predictions remain failures.

## R0 reference re-evaluation

R0 preserves the frozen Dex-Net/GQ-CNN candidate geometry, score, and ranking
(`{r0_recompute["source_candidates_status"]}`), while its formal success labels
are `{r0_recompute["formal_metrics_status"]}` with the locked corrected
evaluator. The legacy source flags are not used for formal metrics. Their
compatibility audit reports `{legacy["status"]}` with {legacy["mismatch_count"]}
changed sample-level `(J@1, J@5, pool-oracle)` tuples. Legacy summary:
`{json.dumps(legacy["legacy_metrics"], sort_keys=True)}`.

## Predicted-mask main result

{_metric_table(artifacts["formal"])}

## GT-mask oracle

{_oracle_table(artifacts["oracle"])}

The GT-mask oracle isolates grounding-mask sensitivity but is not a deployable
result and is not mixed into the predicted-mask ranking.

## Statistical tests

{_statistics_table(artifacts["statistical"], artifacts["bootstrap"])}

All McNemar tests are exact and paired by sample. Holm correction is applied
over the preregistered family. Confidence intervals use at least 10,000 paired
scene-cluster bootstrap draws with the saved seed. Negative deltas and
non-rejections are retained verbatim.

## Formal-test subgroup diagnostics

{_subgroup_summary(artifacts["subgroup"], primary)}

The complete machine-readable table is `subgroup_results.csv`; its paired
sample feature table is `subgroup_sample_features.parquet`. Query type, target
area, predicted-mask IoU/confidence, candidate count, predicted-target depth
validity, GT object width, scene clutter, no-grasp reason, and first-valid rank
are all retained. GT-derived fields are post-hoc offline diagnostics only and
were not available to inference or configuration selection.

## Twelve required conclusion questions

{_conclusion_answers(artifacts)}

## Independent recomputation

- Overall status: **{independent["status"]}**.
- Candidate correctness fields trusted: `{independent.get("candidate_correctness_fields_trusted")}`.
- Configuration selection read: `{independent["configuration_selection_read"]}`.
- Methods independently checked: {independent["method_count"]}.
- Sample count: {independent["sample_count"]}.

{_scope_block()}

{_evidence_ledger(artifacts["evidence_hashes"])}
"""


def _failure_report(artifacts: Mapping[str, Any]) -> str:
    failure_table, totals, bottleneck = _failure_summary(artifacts["failure"])
    gallery = artifacts["gallery"]
    counts_rows = []
    for method, categories in sorted(
        gallery.get("selection", {}).get("counts", {}).items()
    ):
        for category, values in categories.items():
            counts_rows.append(
                (method, category, values["requested"], values["actual"])
            )
    return f"""# Repeated-FiLM 4-DoF failure analysis

## Analysis boundary

This is a post-hoc `formal_test_only` analysis. It performed no configuration,
threshold, checkpoint, or method selection. Samples retain distinct indicators
for no candidate, candidate-pool failure, and ranking failure; these mechanisms
are not collapsed.

## Primary failure-stage counts

{failure_table}

## Bottleneck grouping

{_table(("Group", "Count"), sorted(totals.items()))}

The largest observed diagnostic group is **{bottleneck}**. Grounding-derived
and GT-derived stages are diagnostic explanations, not inference inputs.

## Selected qualitative gallery

{_table(("Method", "Category", "Requested", "Actual"), counts_rows)}

- Cross-method requested/actual: `{json.dumps(gallery.get("selection", {}).get("cross_method_count"))}`.
- Rendered per-method images: `{gallery.get("rendered_method_image_count")}`.
- Rendered cross-method images: `{gallery.get("rendered_cross_method_image_count")}`.
- Dense maps: `{gallery.get("dense_maps_status")}` — `{gallery.get("dense_maps_reason")}`.
- Gallery: `gallery/index.html`.

Insufficient categories retain their actual counts. Dense maps are explicitly
unavailable when no persisted array artifact exists; no heatmap is fabricated.

## Interpretation limits

Wrong-mask, width, collision-proxy, and evaluator-ambiguity labels are diagnostic
proxies whose recorded rules must be read with the per-sample table. A visible
occupancy proxy is not a complete collision check, and a 2D failure category
does not establish the cause of a physical robot outcome.

{_scope_block()}

{_evidence_ledger(artifacts["evidence_hashes"])}
"""


def render_final_reports(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate evidence and deterministically render reports without writing."""

    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    artifacts = _load_artifacts(run)
    reports = {
        REPORT_NAMES[0]: _implementation_report(artifacts),
        REPORT_NAMES[1]: _audit_report(artifacts),
        REPORT_NAMES[2]: _validation_report(artifacts),
        REPORT_NAMES[3]: _results_report(artifacts),
        REPORT_NAMES[4]: _failure_report(artifacts),
    }
    rendered = {name: text.rstrip() + "\n" for name, text in reports.items()}
    for name, text in rendered.items():
        if SCIENTIFIC_SCOPE_SENTENCE not in text:
            raise AssertionError(f"scientific scope sentence missing from {name}")
    return artifacts, rendered


def generate_final_reports(run_dir: str | Path) -> dict[str, Any]:
    """Validate final artifacts, then exclusively materialise five reports."""

    run = Path(run_dir).expanduser().resolve()
    artifacts, reports = render_final_reports(run)
    reports_dir = run / "reports"
    if reports_dir.exists() and reports_dir.is_symlink():
        raise FinalReportError(f"refusing symlink reports directory: {reports_dir}")
    reports_dir.mkdir(parents=True, exist_ok=True)
    destinations = {name: reports_dir / name for name in REPORT_NAMES}
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite final reports: {existing}")

    temporary = Path(tempfile.mkdtemp(prefix=".final_reports.", dir=reports_dir))
    try:
        for name, text in reports.items():
            (temporary / name).write_text(text, encoding="utf-8")
        for name, destination in destinations.items():
            os.replace(temporary / name, destination)
    except BaseException:
        for destination in destinations.values():
            if destination.exists():
                destination.unlink()
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "status": "COMPLETE",
        "run_dir": str(run),
        "configuration_selection_performed": False,
        "reports": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in destinations.items()
        },
        "locked_primary_method_id": artifacts["selection"]["primary_method_id"],
        "independent_recompute_status": artifacts["independent"]["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(generate_final_reports(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
