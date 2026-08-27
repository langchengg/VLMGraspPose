"""Fail-closed publication of top-level formal-run data artifacts.

The expensive stages intentionally write condition- and group-scoped commits.
This module is the final, read-only aggregation boundary for consumers that
expect a small set of files at the run root.  It does not infer missing rows,
rerun models, or turn an empty/fixture analysis into a result.

Every component is published through a manifest-last commit.  Source files are
hashed before rendering and immediately before publication; a resume succeeds
only when both the input fingerprint and every previously published output
hash still match.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .analysis_inputs import AnalysisInputAssembly, assemble_analysis_inputs
from .experiment_analysis import (
    ANALYSIS_SCHEMA,
    FORMAL_SCOPE,
    load_analysis_input_manifest,
)
from .formal_inputs import load_committed_formal_feature_table
from .io import atomic_json, canonical_sha256, sha256_file
from .paper_artifacts import validate_formal_paper_inputs
from .provenance import RUN_MANIFEST_SCHEMA_VERSION
from .stages import LABEL_BUNDLE_SCHEMA, load_target_language_jsonl


ENVIRONMENT_OUTPUT_SCHEMA = "graspnet6d_environment_output_v1"
CANDIDATE_OUTPUT_SCHEMA = "graspnet6d_candidate_aggregate_output_v1"
SELECTED_ANALYSIS_OUTPUT_SCHEMA = "graspnet6d_selected_analysis_output_v1"
RUN_OUTPUT_SCHEMA = "graspnet6d_formal_run_outputs_v1"

CONDITIONS = (
    "oracle_gt_mask",
    "hifics_zero_shot_mask",
    "hifics_adapted_mask",
)
PREDICTED_CONDITIONS = CONDITIONS[1:]
PARTITIONS = ("train", "validation", "test")

ENVIRONMENT_MANIFEST = "environment_provenance.json"
CANDIDATE_MANIFEST = "candidate_aggregate_provenance.json"
SELECTED_ANALYSIS_MANIFEST = "selected_analysis_provenance.json"
RUN_OUTPUT_MANIFEST = "run_outputs_manifest.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORMAL_RUN_FORBIDDEN = re.compile(
    r"fixture|synthetic|dummy|unit[-_]?test|smoke", re.IGNORECASE
)

_ENVIRONMENT_OUTPUTS = ("environment.txt",)
_CANDIDATE_OUTPUTS = (
    "candidate_labels.parquet",
    "candidate_features.parquet",
    "candidate_group_universe.parquet",
    "feature_schema.json",
)
_SELECTED_COPY_OUTPUTS = (
    "paired_outcomes.csv",
    "metrics.csv",
    "metrics.json",
    "bootstrap_results.csv",
    "significance_tests.json",
    "ablation_results.csv",
    "failure_taxonomy.csv",
    "failure_summary.json",
    "frozen_pool_audit.json",
)
_SELECTED_OUTPUTS = (
    "native_predictions.parquet",
    "reranked_predictions.parquet",
    *_SELECTED_COPY_OUTPUTS,
)

_LABEL_VALUE_COLUMNS = (
    "candidate_id",
    "candidate_index",
    "target_object_id",
    "associated_object_id",
    "target_match",
    "correct_target",
    "collision",
    "empty_grasp",
    "pose_valid",
    "valid_geometry",
    "friction_required",
    "friction_score",
    "relevance",
    "success_mu_0.2",
    "success_mu_0.4",
    "success_mu_0.6",
    "success_mu_0.8",
    "success_mu_1.0",
    "success_mu_1.2",
)
_LABEL_COLUMNS = (
    "record_kind",
    "grounding_condition",
    "partition",
    "scene_id",
    "group_id",
    "generation_status",
    "grounding_failure_reason",
    "label_generation_status",
    "empty_pool_reason",
    "evaluator_calls_for_group",
    "candidate_count",
    "candidate_pool_fingerprint",
    "candidate_bundle_sha256",
    "label_bundle_path",
    "label_bundle_sha256",
    "label_bundle_fingerprint",
    "official_source_hashes_sha256",
    *_LABEL_VALUE_COLUMNS,
)
_UNIVERSE_COLUMNS = (
    "grounding_condition",
    "partition",
    "scene_id",
    "group_id",
    "candidate_count",
    "generation_status",
    "grounding_failure_reason",
    "label_generation_status",
    "empty_pool_reason",
    "candidate_pool_fingerprint",
    "candidate_bundle_sha256",
    "feature_commit_sha256",
    "label_bundle_sha256",
)


class RunOutputError(RuntimeError):
    """A formal source or a previously published aggregate is invalid."""


@dataclass(frozen=True, slots=True)
class PublishedComponent:
    """One manifest-last top-level publication."""

    manifest_path: Path
    manifest_sha256: str
    input_fingerprint: str
    outputs: Mapping[str, str]
    resumed: bool


@dataclass(frozen=True, slots=True)
class FormalRunOutputs:
    """Complete top-level publication returned by the orchestration API."""

    run_dir: Path
    manifest_path: Path
    manifest_sha256: str
    selected_predicted_condition: str
    outputs: Mapping[str, str]
    environment: PublishedComponent
    candidates: PublishedComponent
    selected_analysis: PublishedComponent
    resumed: bool


def _digest(value: Any, description: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise RunOutputError(f"{description} is not a lowercase SHA-256 digest")
    return text


def _regular_file(
    path: str | os.PathLike[str], description: str, *, nonempty: bool = True
) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise RunOutputError(f"{description} must not be a symlink: {raw}")
    source = raw.resolve()
    if not source.is_file() or (nonempty and source.stat().st_size <= 0):
        qualifier = "non-empty " if nonempty else ""
        raise RunOutputError(f"missing {qualifier}regular {description}: {source}")
    return source


def _read_json(
    path: str | os.PathLike[str], description: str
) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, description)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunOutputError(f"invalid {description} {source}: {error}") from error
    if not isinstance(payload, dict):
        raise RunOutputError(f"{description} must be a JSON object: {source}")
    return source, payload


def _formal_run(
    run_dir: Path | str, *, allow_terminal_status: bool = False
) -> tuple[Path, dict[str, Any], Mapping[str, Any]]:
    raw = Path(run_dir).expanduser()
    if raw.is_symlink():
        raise RunOutputError(f"formal run directory must not be a symlink: {raw}")
    root = raw.resolve()
    if not root.is_dir():
        raise RunOutputError(f"formal run directory is absent: {root}")
    _, manifest = _read_json(root / "run_manifest.json", "run manifest")
    if (
        manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("run_id") != root.name
    ):
        raise RunOutputError("run manifest schema/run_id does not match its directory")
    if _FORMAL_RUN_FORBIDDEN.search(root.name):
        raise RunOutputError("fixture-labelled run IDs cannot publish formal outputs")
    identity = manifest.get("immutable_identity")
    if not isinstance(identity, Mapping):
        raise RunOutputError("run manifest lacks its immutable identity")
    config = identity.get("resolved_config")
    if not isinstance(config, Mapping) or config.get("formal_results") is not True:
        raise RunOutputError("run profile is not eligible for formal results")
    if not allow_terminal_status and str(manifest.get("status", "")) in {
        "BLOCKED",
        "FAILED",
    }:
        raise RunOutputError(
            f"run status {manifest.get('status')!r} cannot publish formal outputs"
        )
    return root, manifest, config


def _write_bytes_sync(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            prefix=f".{destination.stem}.",
            suffix=".parquet",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=False)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_source_hashes(source_hashes: Mapping[str, str]) -> None:
    for raw_path, expected in source_hashes.items():
        source = _regular_file(raw_path, "component source")
        if sha256_file(source) != _digest(expected, f"source hash for {source}"):
            raise RunOutputError(
                f"component source changed during publication: {source}"
            )


def _validate_component_manifest(
    manifest_path: Path,
    *,
    schema: str,
    input_fingerprint: str,
    output_names: Sequence[str],
    verify_sources: bool = True,
) -> PublishedComponent:
    source, payload = _read_json(manifest_path, "output component manifest")
    check = dict(payload)
    observed_commit = check.pop("commit_fingerprint", None)
    if observed_commit != canonical_sha256(check):
        raise RunOutputError(f"component commit fingerprint is stale: {source}")
    if (
        payload.get("schema_version") != schema
        or payload.get("status") != "COMPLETE"
        or payload.get("input_fingerprint") != input_fingerprint
    ):
        raise RunOutputError(f"component cannot resume another input: {source}")
    source_hashes = payload.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise RunOutputError(f"component lacks source hashes: {source}")
    if verify_sources:
        _assert_source_hashes(
            {
                str(path): _digest(value, f"source hash {path}")
                for path, value in source_hashes.items()
            }
        )
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(output_names):
        raise RunOutputError(f"component output universe is incomplete: {source}")
    checked: dict[str, str] = {}
    for name in output_names:
        if Path(name).name != name:
            raise RunOutputError(f"unsafe component output name: {name!r}")
        path = _regular_file(source.parent / name, f"published output {name}")
        expected = _digest(outputs[name], f"output hash {name}")
        if sha256_file(path) != expected:
            raise RunOutputError(f"published output is stale: {path}")
        checked[name] = expected
    return PublishedComponent(
        manifest_path=source,
        manifest_sha256=sha256_file(source),
        input_fingerprint=input_fingerprint,
        outputs=checked,
        resumed=True,
    )


def _commit_component(
    run_dir: Path,
    *,
    manifest_name: str,
    schema: str,
    contract: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    output_names: Sequence[str],
    render: Callable[[Path], None],
    metadata: Mapping[str, Any],
    resume: bool,
    fingerprint_sources: bool = True,
    verify_sources_on_resume: bool = True,
) -> PublishedComponent:
    if len(output_names) != len(set(output_names)):
        raise AssertionError("component output names must be unique")
    normalized_sources = dict(
        sorted(
            (str(Path(path).resolve()), value) for path, value in source_hashes.items()
        )
    )
    _assert_source_hashes(normalized_sources)
    fingerprint_payload: dict[str, Any] = {
        "schema_version": schema,
        "contract": dict(contract),
    }
    if fingerprint_sources:
        fingerprint_payload["source_sha256"] = normalized_sources
    input_fingerprint = canonical_sha256(fingerprint_payload)
    manifest_path = run_dir / manifest_name
    if manifest_path.exists():
        if not resume:
            raise RunOutputError(
                f"component already exists: {manifest_path}; use resume=True"
            )
        return _validate_component_manifest(
            manifest_path,
            schema=schema,
            input_fingerprint=input_fingerprint,
            output_names=output_names,
            verify_sources=verify_sources_on_resume,
        )
    existing = [name for name in output_names if (run_dir / name).exists()]
    if existing:
        raise RunOutputError(
            "uncommitted top-level outputs already exist; refusing overwrite: "
            + ", ".join(existing)
        )
    with tempfile.TemporaryDirectory(dir=run_dir, prefix=".run-output-") as raw_stage:
        staging = Path(raw_stage)
        render(staging)
        observed = {
            path.name
            for path in staging.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if observed != set(output_names):
            raise RunOutputError(
                "component renderer produced another output universe: "
                f"expected={sorted(output_names)}, observed={sorted(observed)}"
            )
        outputs = {
            name: sha256_file(_regular_file(staging / name, f"rendered {name}"))
            for name in output_names
        }
        _assert_source_hashes(normalized_sources)
        payload: dict[str, Any] = {
            "schema_version": schema,
            "status": "COMPLETE",
            "run_id": run_dir.name,
            "scope": FORMAL_SCOPE,
            "fixture_only": False,
            "input_fingerprint": input_fingerprint,
            "source_sha256": normalized_sources,
            "outputs": outputs,
            **dict(metadata),
        }
        payload["commit_fingerprint"] = canonical_sha256(payload)
        atomic_json(staging / manifest_name, payload)
        for name in output_names:
            os.replace(staging / name, run_dir / name)
        os.replace(staging / manifest_name, manifest_path)
        _fsync_directory(run_dir)
    result = _validate_component_manifest(
        manifest_path,
        schema=schema,
        input_fingerprint=input_fingerprint,
        output_names=output_names,
        verify_sources=verify_sources_on_resume,
    )
    return PublishedComponent(
        manifest_path=result.manifest_path,
        manifest_sha256=result.manifest_sha256,
        input_fingerprint=result.input_fingerprint,
        outputs=result.outputs,
        resumed=False,
    )


def _strict_csv_bool(value: Any, description: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise RunOutputError(f"{description} is not boolean: {value!r}")


def _stable_environment_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude audit-time and free-space observations from host identity."""

    return {
        key: value.get(key)
        for key in ("hardware", "platform", "privacy", "python", "torch")
    }


def publish_environment(
    run_dir: Path | str,
    *,
    audit_root: Path | str | None = None,
    device_root: Path | str | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
    resume: bool = False,
) -> PublishedComponent:
    """Publish ``environment.txt`` from five mutually checked audit sources.

    ``expected_source_hashes`` is optional because the component manifest is
    itself content-addressed.  When supplied by an audit stage, it must contain
    exactly the five evidence basenames and provides an additional independent
    pre-publication binding.
    """

    root, run_manifest, config = _formal_run(run_dir, allow_terminal_status=True)
    audit = (
        Path(audit_root).expanduser().resolve()
        if audit_root is not None
        else root.parent / "audit"
    )
    devices = (
        Path(device_root).expanduser().resolve()
        if device_root is not None
        else root.parent
    )
    evidence = {
        "hardware_environment.json": _regular_file(
            audit / "hardware_environment.json", "hardware environment audit"
        ),
        "system_profile.json": _regular_file(
            audit / "system_profile.json", "system profile audit"
        ),
        "pip_freeze.txt": _regular_file(audit / "pip_freeze.txt", "pip freeze"),
        "device_benchmark.csv": _regular_file(
            devices / "device_benchmark.csv", "device benchmark"
        ),
        "device_decision.md": _regular_file(
            devices / "device_decision.md", "device decision"
        ),
    }
    hashes = {name: sha256_file(path) for name, path in evidence.items()}
    if expected_source_hashes is not None:
        if set(expected_source_hashes) != set(evidence):
            raise RunOutputError(
                "expected environment source hashes must name exactly the five audit files"
            )
        for name, observed in hashes.items():
            if (
                _digest(expected_source_hashes[name], f"expected hash {name}")
                != observed
            ):
                raise RunOutputError(f"audit evidence hash mismatch: {name}")
    _, hardware = _read_json(evidence["hardware_environment.json"], "hardware audit")
    _, system = _read_json(evidence["system_profile.json"], "system profile")
    if (
        hardware != system
        or hashes["hardware_environment.json"] != hashes["system_profile.json"]
    ):
        raise RunOutputError(
            "hardware_environment.json and system_profile.json disagree"
        )
    run_environment = run_manifest.get("hardware_environment")
    if not isinstance(run_environment, Mapping):
        raise RunOutputError("run manifest lacks its captured hardware environment")
    if _stable_environment_projection(
        run_environment
    ) != _stable_environment_projection(hardware):
        raise RunOutputError(
            "current audit host/software differs from the run-captured environment"
        )
    freeze = evidence["pip_freeze.txt"].read_text(encoding="utf-8")
    if not freeze.strip() or "Traceback (most recent call last)" in freeze:
        raise RunOutputError(
            "pip freeze evidence is empty or contains a command failure"
        )
    try:
        with evidence["device_benchmark.csv"].open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            benchmark_rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise RunOutputError(f"invalid device benchmark: {error}") from error
    required = {
        "device",
        "available",
        "successful",
        "finite_rate",
        "median_latency_s",
        "p95_latency_s",
    }
    if (
        not benchmark_rows
        or any(not required.issubset(row) for row in benchmark_rows)
        or len({str(row["device"]).lower() for row in benchmark_rows})
        != len(benchmark_rows)
    ):
        raise RunOutputError("device benchmark lacks unique required device rows")
    vgn = config.get("vgn")
    if not isinstance(vgn, Mapping):
        raise RunOutputError("formal configuration lacks a VGN section")
    selected_device = str(vgn.get("device", "")).strip().lower()
    rows_by_device = {str(row["device"]).lower(): row for row in benchmark_rows}
    if selected_device not in rows_by_device:
        raise RunOutputError("formal VGN device is absent from benchmark evidence")
    selected_row = rows_by_device[selected_device]
    try:
        finite_rate = float(selected_row["finite_rate"])
        latencies = (
            float(selected_row["median_latency_s"]),
            float(selected_row["p95_latency_s"]),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise RunOutputError(
            "selected device benchmark contains non-numeric values"
        ) from error
    if (
        not _strict_csv_bool(selected_row["available"], "device available")
        or not _strict_csv_bool(selected_row["successful"], "device successful")
        or not np.isfinite(finite_rate)
        or finite_rate != 1.0
        or not np.isfinite(latencies).all()
        or min(latencies) < 0
    ):
        raise RunOutputError(
            "selected formal device did not pass finite benchmark execution"
        )
    decision = evidence["device_decision.md"].read_text(encoding="utf-8")
    expected_decision = f"Formal inference device: **{selected_device.upper()}**"
    if expected_decision not in decision:
        raise RunOutputError("device decision disagrees with the formal configuration")

    # Global audits are intentionally refreshed before every `all` invocation.
    # Their timestamp/storage fields may change, but package and device evidence
    # must not silently change underneath an already committed environment.
    existing_environment_manifest = root / ENVIRONMENT_MANIFEST
    if existing_environment_manifest.exists() and resume:
        _, existing_payload = _read_json(
            existing_environment_manifest, "environment provenance"
        )
        prior_sources = existing_payload.get("source_sha256")
        if not isinstance(prior_sources, Mapping):
            raise RunOutputError("environment provenance lacks source hashes")
        prior_by_name = {
            Path(str(path)).name: _digest(value, f"prior environment hash {path}")
            for path, value in prior_sources.items()
        }
        for immutable_name in (
            "pip_freeze.txt",
            "device_benchmark.csv",
            "device_decision.md",
        ):
            if prior_by_name.get(immutable_name) != hashes[immutable_name]:
                raise RunOutputError(
                    f"immutable environment evidence changed across resume: {immutable_name}"
                )

    source_hashes = {str(path): hashes[name] for name, path in evidence.items()}
    run_identity_sha = canonical_sha256(
        {
            "run_id": root.name,
            "profile": run_manifest.get("profile"),
            "immutable_identity": run_manifest.get("immutable_identity"),
            "hardware_environment": dict(run_environment),
            "software": run_manifest.get("software"),
        }
    )
    contract = {
        "run_id": root.name,
        "run_identity_sha256": run_identity_sha,
        "formal_device": selected_device,
        "evidence_basenames": list(evidence),
    }
    rendered = (
        "GraspNet 6-DoF formal environment evidence\n"
        f"run_id: {root.name}\n"
        f"run_identity_sha256: {run_identity_sha}\n"
        f"formal_vgn_device: {selected_device}\n\n"
        "Source SHA-256\n"
        + "".join(f"{name}: {hashes[name]}\n" for name in evidence)
        + "\nRun-captured hardware / system profile (primary)\n"
        + json.dumps(run_environment, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n\nCurrent refreshed audit (timestamp/storage may differ)\n"
        + json.dumps(hardware, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n\nPython package freeze\n"
        + freeze.rstrip()
        + "\n\nDevice benchmark\n"
        + evidence["device_benchmark.csv"].read_text(encoding="utf-8").rstrip()
        + "\n\nDevice decision\n"
        + decision.rstrip()
        + "\n"
    )

    def render(staging: Path) -> None:
        _write_bytes_sync(staging / "environment.txt", rendered.encode("utf-8"))

    return _commit_component(
        root,
        manifest_name=ENVIRONMENT_MANIFEST,
        schema=ENVIRONMENT_OUTPUT_SCHEMA,
        contract=contract,
        source_hashes=source_hashes,
        output_names=_ENVIRONMENT_OUTPUTS,
        render=render,
        metadata={
            "formal_device": selected_device,
            "run_identity_sha256": run_identity_sha,
            "audit_root": str(audit),
            "device_root": str(devices),
        },
        resume=resume,
        fingerprint_sources=False,
        verify_sources_on_resume=False,
    )


def _analysis_binding(
    run_dir: Path, condition: str, assembly: AnalysisInputAssembly
) -> Path:
    manifest_path, manifest = _read_json(
        run_dir / "analysis" / condition / "analysis_manifest.json",
        f"{condition} analysis manifest",
    )
    if (
        manifest.get("schema_version") != ANALYSIS_SCHEMA
        or manifest.get("status") != "COMPLETE"
        or manifest.get("analysis_scope") != FORMAL_SCOPE
        or manifest.get("fixture_only") is not False
        or manifest.get("run_id") != run_dir.name
        or manifest.get("formal_report_eligible") is not True
    ):
        raise RunOutputError(f"{condition} analysis is not complete formal evidence")
    declared = Path(str(manifest.get("input_manifest_path", ""))).expanduser()
    if not declared.is_absolute():
        declared = manifest_path.parent / declared
    declared = declared.resolve()
    expected_input_sha = _digest(
        manifest.get("input_manifest_sha256"),
        f"{condition} analysis input manifest hash",
    )
    if (
        declared != assembly.manifest_path.resolve()
        or expected_input_sha != assembly.manifest_sha256
        or sha256_file(declared) != expected_input_sha
    ):
        raise RunOutputError(
            f"{condition} aggregate assembly is not the input used by its analysis"
        )
    return manifest_path


def _loaded_assembly(
    run_dir: Path, condition: str, assembly: AnalysisInputAssembly
) -> tuple[dict[str, Any], Path, Path]:
    try:
        loaded = load_analysis_input_manifest(assembly.manifest_path)
    except Exception as error:
        raise RunOutputError(
            f"invalid {condition} formal analysis input: {type(error).__name__}: {error}"
        ) from error
    if (
        loaded.status != "COMPLETE"
        or loaded.scope != FORMAL_SCOPE
        or loaded.fixture_only
        or loaded.run_id != run_dir.name
    ):
        raise RunOutputError(f"{condition} analysis input is fixture/incomplete")
    provenance_path, provenance = _read_json(
        assembly.output_dir / "assembly_provenance.json",
        f"{condition} assembly provenance",
    )
    expected = _digest(
        loaded.provenance.get("assembly_provenance_sha256"),
        f"{condition} assembly provenance hash",
    )
    if sha256_file(provenance_path) != expected:
        raise RunOutputError(f"{condition} assembly provenance is stale")
    if (
        provenance.get("status") != "COMPLETE"
        or provenance.get("scope") != FORMAL_SCOPE
        or provenance.get("fixture_only") is not False
        or provenance.get("condition") != condition
        or provenance.get("run_id") != run_dir.name
    ):
        raise RunOutputError(f"{condition} assembly provenance is not formal")
    return provenance, provenance_path, loaded.feature_schema.path


def _label_rows(
    *,
    condition: str,
    commit: Mapping[str, Any],
    payload: Mapping[str, Any],
    label_path: Path,
) -> list[dict[str, Any]]:
    check = dict(payload)
    observed_fingerprint = check.pop("bundle_fingerprint", None)
    if observed_fingerprint != canonical_sha256(check):
        raise RunOutputError(
            f"official label bundle fingerprint is stale: {label_path}"
        )
    if (
        payload.get("schema_version") != LABEL_BUNDLE_SCHEMA
        or payload.get("group_id") != commit.get("group_id")
        or payload.get("grounding_condition") != condition
        or payload.get("candidate_pool_fingerprint")
        != commit.get("candidate_pool_fingerprint")
    ):
        raise RunOutputError(
            f"official label bundle disagrees with assembly: {label_path}"
        )
    candidate_count = int(commit.get("candidate_count", -1))
    candidate_ids = payload.get("candidate_ids")
    labels = payload.get("labels")
    if (
        candidate_count < 0
        or payload.get("candidate_count") != candidate_count
        or not isinstance(candidate_ids, list)
        or not isinstance(labels, list)
        or len(candidate_ids) != candidate_count
        or len(labels) != candidate_count
    ):
        raise RunOutputError(
            f"official label candidate universe is incomplete: {label_path}"
        )
    base = {
        "grounding_condition": condition,
        "partition": str(commit.get("partition", "")),
        "scene_id": str(commit.get("scene_id", "")),
        "group_id": str(commit.get("group_id", "")),
        "generation_status": str(commit.get("generation_status", "")),
        "grounding_failure_reason": commit.get("grounding_failure_reason"),
        "label_generation_status": payload.get("label_generation_status"),
        "empty_pool_reason": payload.get("empty_pool_reason"),
        "evaluator_calls_for_group": payload.get("evaluator_calls_for_group"),
        "candidate_count": candidate_count,
        "candidate_pool_fingerprint": commit.get("candidate_pool_fingerprint"),
        "candidate_bundle_sha256": commit.get("candidate_bundle_sha256"),
        "label_bundle_path": str(label_path),
        "label_bundle_sha256": commit.get("label_bundle_sha256"),
        "label_bundle_fingerprint": observed_fingerprint,
        "official_source_hashes_sha256": canonical_sha256(
            payload.get("official_source_hashes", {})
        ),
    }
    if candidate_count == 0:
        if (
            payload.get("label_generation_status") != "skipped_empty_pool"
            or payload.get("evaluator_calls_for_group") != 0
            or payload.get("official_source_hashes") != {}
            or not str(payload.get("empty_pool_reason", ""))
        ):
            raise RunOutputError(
                f"zero-pool label provenance is incomplete: {label_path}"
            )
        return [
            {
                **base,
                "record_kind": "empty_pool_group",
                **{name: None for name in _LABEL_VALUE_COLUMNS},
                "target_object_id": payload.get("target_object_id"),
            }
        ]
    required = set(_LABEL_VALUE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for index, (candidate_id, raw) in enumerate(
        zip(candidate_ids, labels, strict=True)
    ):
        if not isinstance(raw, Mapping) or not required.issubset(raw):
            raise RunOutputError(
                f"official label row lacks canonical fields: {label_path}"
            )
        if (
            str(raw.get("candidate_id")) != str(candidate_id)
            or raw.get("candidate_index") != index
        ):
            raise RunOutputError(f"official label ordering changed: {label_path}")
        rows.append(
            {
                **base,
                "record_kind": "candidate_label",
                **{name: raw[name] for name in _LABEL_VALUE_COLUMNS},
            }
        )
    return rows


def _typed_label_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=_LABEL_COLUMNS)
    string_columns = {
        "record_kind",
        "grounding_condition",
        "partition",
        "scene_id",
        "group_id",
        "generation_status",
        "grounding_failure_reason",
        "label_generation_status",
        "empty_pool_reason",
        "candidate_pool_fingerprint",
        "candidate_bundle_sha256",
        "label_bundle_path",
        "label_bundle_sha256",
        "label_bundle_fingerprint",
        "official_source_hashes_sha256",
        "candidate_id",
    }
    integer_columns = {
        "evaluator_calls_for_group",
        "candidate_count",
        "candidate_index",
        "target_object_id",
        "associated_object_id",
        "relevance",
    }
    boolean_columns = {
        "target_match",
        "correct_target",
        "collision",
        "empty_grasp",
        "pose_valid",
        "valid_geometry",
        "success_mu_0.2",
        "success_mu_0.4",
        "success_mu_0.6",
        "success_mu_0.8",
        "success_mu_1.0",
        "success_mu_1.2",
    }
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    for column in integer_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")
    for column in boolean_columns:
        frame[column] = frame[column].astype("boolean")
    for column in ("friction_required", "friction_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    return frame


def publish_candidate_aggregates(
    run_dir: Path | str, *, resume: bool = False
) -> PublishedComponent:
    """Aggregate every formal label/feature commit across all three arms.

    Each condition is first passed through :func:`assemble_analysis_inputs`,
    then required to be the exact content-addressed input consumed by its saved
    formal analysis.  Thus this publisher cannot bless an unanalysed or stale
    alternate set of group commits.
    """

    root, _, _ = _formal_run(run_dir)
    target_path = _regular_file(
        root / "manifests" / "target_groups.jsonl", "target manifest"
    )
    language_path = _regular_file(
        root / "manifests" / "language_queries.jsonl", "language manifest"
    )
    try:
        groups = load_target_language_jsonl(target_path, language_path)
    except Exception as error:
        raise RunOutputError(f"invalid target/language universe: {error}") from error
    if not groups:
        raise RunOutputError("formal target group universe is empty")
    group_order = [group.group_id for group in groups]
    target_group_set = set(group_order)

    assemblies: dict[str, AnalysisInputAssembly] = {}
    provenances: dict[str, dict[str, Any]] = {}
    schemas: dict[str, Path] = {}
    source_hashes: dict[str, str] = {
        str(target_path): sha256_file(target_path),
        str(language_path): sha256_file(language_path),
    }
    commits_by_condition: dict[str, dict[str, Mapping[str, Any]]] = {}
    for condition in CONDITIONS:
        try:
            assembly = assemble_analysis_inputs(
                target_path,
                language_path,
                root,
                root,
                condition=condition,
                run_id=root.name,
            )
        except Exception as error:
            raise RunOutputError(
                f"cannot validate {condition} aggregate inputs: {type(error).__name__}: {error}"
            ) from error
        if (
            assembly.condition != condition
            or assembly.group_count != len(groups)
            or assembly.candidate_count <= 0
        ):
            raise RunOutputError(
                f"{condition} assembly has an incomplete formal universe"
            )
        analysis_path = _analysis_binding(root, condition, assembly)
        provenance, provenance_path, schema_path = _loaded_assembly(
            root, condition, assembly
        )
        raw_commits = provenance.get("group_commits")
        if not isinstance(raw_commits, list) or len(raw_commits) != len(groups):
            raise RunOutputError(f"{condition} assembly group provenance is incomplete")
        indexed: dict[str, Mapping[str, Any]] = {}
        for raw_commit in raw_commits:
            if not isinstance(raw_commit, Mapping):
                raise RunOutputError(
                    f"{condition} assembly has a malformed group commit"
                )
            group_id = str(raw_commit.get("group_id", ""))
            if not group_id or group_id in indexed:
                raise RunOutputError(f"{condition} group provenance IDs are invalid")
            indexed[group_id] = raw_commit
        if set(indexed) != target_group_set:
            raise RunOutputError(
                f"{condition} group universe differs from the manifest"
            )
        assemblies[condition] = assembly
        provenances[condition] = provenance
        schemas[condition] = schema_path
        commits_by_condition[condition] = indexed
        for path in (
            assembly.manifest_path,
            provenance_path,
            analysis_path,
            schema_path,
        ):
            source_hashes[str(path.resolve())] = sha256_file(path)

    schema_hashes = {sha256_file(path) for path in schemas.values()}
    if len(schema_hashes) != 1:
        raise RunOutputError("grounding conditions use different feature schemas")
    schema_path = schemas[CONDITIONS[0]]
    schema_bytes = schema_path.read_bytes()

    label_rows: list[dict[str, Any]] = []
    feature_frames: list[pd.DataFrame] = []
    universe_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for group_id in group_order:
            commit = commits_by_condition[condition][group_id]
            label_path = _regular_file(
                commit.get("label_bundle_path", ""), "official label bundle"
            )
            expected_label_sha = _digest(
                commit.get("label_bundle_sha256"), "official label bundle hash"
            )
            if sha256_file(label_path) != expected_label_sha:
                raise RunOutputError(f"official label bundle is stale: {label_path}")
            _, label_payload = _read_json(label_path, "official label bundle")
            rows = _label_rows(
                condition=condition,
                commit=commit,
                payload=label_payload,
                label_path=label_path,
            )
            label_rows.extend(rows)
            source_hashes[str(label_path)] = expected_label_sha
            official_sources = label_payload.get("official_source_hashes", {})
            if not isinstance(official_sources, Mapping):
                raise RunOutputError(
                    f"official source hashes are malformed: {label_path}"
                )
            for raw_path, raw_hash in official_sources.items():
                source = _regular_file(raw_path, "official evaluator source")
                expected = _digest(raw_hash, "official evaluator source hash")
                if sha256_file(source) != expected:
                    raise RunOutputError(
                        f"official evaluator source is stale: {source}"
                    )
                source_hashes[str(source)] = expected

            candidate_path = _regular_file(
                commit.get("candidate_bundle_path", ""), "candidate bundle"
            )
            candidate_sha = _digest(
                commit.get("candidate_bundle_sha256"), "candidate bundle hash"
            )
            if sha256_file(candidate_path) != candidate_sha:
                raise RunOutputError(f"candidate bundle is stale: {candidate_path}")
            source_hashes[str(candidate_path)] = candidate_sha

            feature_sidecar = _regular_file(
                commit.get("feature_commit_path", ""), "feature commit"
            )
            feature_commit_sha = _digest(
                commit.get("feature_commit_sha256"), "feature commit hash"
            )
            if sha256_file(feature_sidecar) != feature_commit_sha:
                raise RunOutputError(f"feature commit is stale: {feature_sidecar}")
            candidate_ids = [str(value) for value in label_payload["candidate_ids"]]
            try:
                feature_frame = load_committed_formal_feature_table(
                    feature_sidecar,
                    expected_group_id=group_id,
                    expected_condition=condition,  # type: ignore[arg-type]
                    expected_candidate_ids=candidate_ids,
                )
            except Exception as error:
                raise RunOutputError(
                    f"invalid feature commit for {condition}/{group_id}: {error}"
                ) from error
            _, feature_payload = _read_json(feature_sidecar, "feature commit")
            feature_name = str(feature_payload.get("feature_file", ""))
            if Path(feature_name).name != feature_name:
                raise RunOutputError(
                    f"feature commit has an unsafe table path: {feature_sidecar}"
                )
            feature_path = _regular_file(
                feature_sidecar.parent / feature_name, "feature table"
            )
            feature_sha = _digest(
                feature_payload.get("feature_sha256"), "feature table hash"
            )
            if sha256_file(feature_path) != feature_sha or feature_sha != _digest(
                commit.get("feature_table_sha256"), "assembled feature hash"
            ):
                raise RunOutputError(f"feature table is stale: {feature_path}")
            source_hashes[str(feature_sidecar)] = feature_commit_sha
            source_hashes[str(feature_path)] = feature_sha
            feature_frames.append(feature_frame)
            universe_rows.append(
                {
                    "grounding_condition": condition,
                    "partition": commit.get("partition"),
                    "scene_id": commit.get("scene_id"),
                    "group_id": group_id,
                    "candidate_count": commit.get("candidate_count"),
                    "generation_status": commit.get("generation_status"),
                    "grounding_failure_reason": commit.get("grounding_failure_reason"),
                    "label_generation_status": label_payload.get(
                        "label_generation_status"
                    ),
                    "empty_pool_reason": label_payload.get("empty_pool_reason"),
                    "candidate_pool_fingerprint": commit.get(
                        "candidate_pool_fingerprint"
                    ),
                    "candidate_bundle_sha256": candidate_sha,
                    "feature_commit_sha256": feature_commit_sha,
                    "label_bundle_sha256": expected_label_sha,
                }
            )

    labels = _typed_label_frame(label_rows)
    features = pd.concat(feature_frames, ignore_index=True)
    universe = pd.DataFrame(universe_rows, columns=_UNIVERSE_COLUMNS)
    candidate_labels = labels.loc[labels["record_kind"].eq("candidate_label")]
    if candidate_labels.empty or features.empty or universe.empty:
        raise RunOutputError("formal candidate aggregates cannot be empty")
    label_keys = list(
        zip(
            candidate_labels["grounding_condition"].astype(str),
            candidate_labels["group_id"].astype(str),
            candidate_labels["candidate_id"].astype(str),
            strict=True,
        )
    )
    feature_keys = list(
        zip(
            features["condition"].astype(str),
            features["group_id"].astype(str),
            features["candidate_id"].astype(str),
            strict=True,
        )
    )
    if (
        len(label_keys) != len(set(label_keys))
        or len(feature_keys) != len(set(feature_keys))
        or label_keys != feature_keys
    ):
        raise RunOutputError("label and feature candidate universes/order differ")
    expected_universe_size = len(CONDITIONS) * len(groups)
    if (
        len(universe) != expected_universe_size
        or universe.duplicated(["grounding_condition", "group_id"]).any()
        or set(universe["grounding_condition"].astype(str)) != set(CONDITIONS)
    ):
        raise RunOutputError("candidate group provenance lost a condition/group")
    empty_groups = int(labels["record_kind"].eq("empty_pool_group").sum())
    declared_empty = int(pd.to_numeric(universe["candidate_count"]).eq(0).sum())
    if empty_groups != declared_empty:
        raise RunOutputError("zero-pool label provenance is not one-to-one")

    condition_counts = {
        condition: {
            "groups": int(universe["grounding_condition"].eq(condition).sum()),
            "candidates": int(
                candidate_labels["grounding_condition"].eq(condition).sum()
            ),
            "empty_groups": int(
                universe.loc[
                    universe["grounding_condition"].eq(condition), "candidate_count"
                ]
                .eq(0)
                .sum()
            ),
        }
        for condition in CONDITIONS
    }
    contract = {
        "run_id": root.name,
        "conditions": list(CONDITIONS),
        "target_group_count": len(groups),
        "feature_schema_sha256": next(iter(schema_hashes)),
        "condition_counts": condition_counts,
        "assembly_manifest_sha256": {
            condition: assemblies[condition].manifest_sha256 for condition in CONDITIONS
        },
    }

    def render(staging: Path) -> None:
        _atomic_parquet(staging / "candidate_labels.parquet", labels)
        _atomic_parquet(staging / "candidate_features.parquet", features)
        _atomic_parquet(staging / "candidate_group_universe.parquet", universe)
        _write_bytes_sync(staging / "feature_schema.json", schema_bytes)

    return _commit_component(
        root,
        manifest_name=CANDIDATE_MANIFEST,
        schema=CANDIDATE_OUTPUT_SCHEMA,
        contract=contract,
        source_hashes=source_hashes,
        output_names=_CANDIDATE_OUTPUTS,
        render=render,
        metadata={
            "conditions": list(CONDITIONS),
            "condition_counts": condition_counts,
            "candidate_label_rows": len(labels),
            "candidate_feature_rows": len(features),
            "group_universe_rows": len(universe),
            "zero_pool_group_rows": empty_groups,
            "zero_pool_encoding": "one explicit empty_pool_group label row per zero pool",
            "feature_schema_sha256": next(iter(schema_hashes)),
        },
        resume=resume,
    )


def _prediction_universe(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    required = (
        "grounding_condition",
        "group_id",
        "candidate_id",
        "geometry_sha256",
    )
    missing = sorted(set(required) - set(frame.columns))
    if frame.empty or missing:
        raise RunOutputError(f"{description} is empty or lacks columns: {missing}")
    if frame.duplicated(["group_id", "candidate_id"]).any():
        raise RunOutputError(f"{description} has duplicate frozen candidates")
    return (
        frame[list(required)]
        .astype(str)
        .sort_values(["group_id", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _selected_source_path(
    snapshot_root: Path, filename: str, outputs: Mapping[str, Any]
) -> Path:
    path = _regular_file(snapshot_root / filename, f"selected analysis {filename}")
    expected = _digest(outputs.get(filename), f"selected analysis {filename} hash")
    if sha256_file(path) != expected:
        raise RunOutputError(f"selected analysis output is stale: {path}")
    return path


def publish_selected_analysis_outputs(
    run_dir: Path | str,
    *,
    resume: bool = False,
) -> PublishedComponent:
    """Publish the validation-selected predicted arm as unambiguous root outputs.

    The oracle arm remains a counterfactual and is referenced by hash in the
    provenance manifest; its results are never copied into these primary files.
    The public paper preflight is always rerun here; callers cannot inject a
    partially validated snapshot across this publication boundary.
    """

    root, _, _ = _formal_run(run_dir)
    try:
        validated = validate_formal_paper_inputs(root)
    except Exception as error:
        raise RunOutputError(
            f"formal publication preflight failed: {type(error).__name__}: {error}"
        ) from error
    if validated.run_dir.resolve() != root:  # pragma: no cover - defensive
        raise RunOutputError("validated paper inputs belong to another run")
    selected = str(validated.selected_condition)
    if selected not in PREDICTED_CONDITIONS:
        raise RunOutputError(
            "primary output condition was not validation-selected predicted grounding"
        )
    if set(validated.analyses) != {*CONDITIONS, "combined_a8"}:
        raise RunOutputError("formal validation did not cover every required analysis")
    snapshot = validated.analyses[selected]
    outputs = snapshot.manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RunOutputError("selected analysis manifest has no output hashes")
    if (
        snapshot.manifest.get("analysis_scope") != FORMAL_SCOPE
        or snapshot.manifest.get("fixture_only") is not False
        or snapshot.manifest.get("run_id") != root.name
    ):
        raise RunOutputError("selected analysis is fixture, incomplete, or cross-run")

    source_files: dict[str, Path] = {
        name: _selected_source_path(snapshot.root, name, outputs)
        for name in (
            "native_predictions.csv",
            "reranked_predictions.csv",
            *_SELECTED_COPY_OUTPUTS,
        )
    }
    native = snapshot.native.copy()
    reranked = snapshot.reranked.copy()
    for description, frame in (("native", native), ("reranked", reranked)):
        if set(frame["grounding_condition"].astype(str)) != {selected}:
            raise RunOutputError(f"{description} predictions contain another condition")
    native_universe = _prediction_universe(native, "native predictions")
    reranked_universe = _prediction_universe(reranked, "reranked predictions")
    if not native_universe.equals(reranked_universe):
        raise RunOutputError("native/reranked frozen candidate universes differ")

    try:
        input_manifest = load_analysis_input_manifest(
            Path(str(snapshot.manifest.get("input_manifest_path", "")))
        )
    except Exception as error:
        raise RunOutputError(
            f"selected analysis input manifest is invalid: {error}"
        ) from error
    if (
        input_manifest.status != "COMPLETE"
        or input_manifest.scope != FORMAL_SCOPE
        or input_manifest.fixture_only
        or input_manifest.run_id != root.name
    ):
        raise RunOutputError("selected analysis input is not complete formal evidence")
    expected_input_sha = _digest(
        snapshot.manifest.get("input_manifest_sha256"), "selected input manifest hash"
    )
    if input_manifest.manifest_sha256 != expected_input_sha:
        raise RunOutputError("selected analysis input binding is stale")
    test_rows = pd.read_parquet(input_manifest.partitions["test"].rows.path)
    expected_universe = _prediction_universe(test_rows, "selected test input")
    if not native_universe.equals(expected_universe):
        raise RunOutputError(
            "published predictions do not preserve the selected test pool"
        )
    test_groups = pd.read_parquet(input_manifest.partitions["test"].group_universe.path)
    if test_groups.empty or test_groups["group_id"].astype(str).duplicated().any():
        raise RunOutputError("selected test group universe is empty or duplicated")
    failure_groups = snapshot.failures["group_id"].astype(str)
    if failure_groups.duplicated().any() or set(failure_groups) != set(
        test_groups["group_id"].astype(str)
    ):
        raise RunOutputError(
            "failure taxonomy does not cover the exact test group universe"
        )

    source_hashes = {str(path): sha256_file(path) for path in source_files.values()}
    analysis_manifest_path = _regular_file(
        snapshot.root / "analysis_manifest.json", "selected analysis manifest"
    )
    source_hashes[str(analysis_manifest_path)] = sha256_file(analysis_manifest_path)
    source_hashes[str(input_manifest.source_path)] = input_manifest.manifest_sha256
    for partition in PARTITIONS:
        for item in (
            input_manifest.partitions[partition].rows,
            input_manifest.partitions[partition].group_universe,
        ):
            source_hashes[str(item.path)] = item.sha256

    roles: dict[str, Any] = {}
    for condition in CONDITIONS:
        arm = validated.analyses[condition]
        manifest_path = _regular_file(
            arm.root / "analysis_manifest.json", f"{condition} analysis manifest"
        )
        manifest_sha = sha256_file(manifest_path)
        source_hashes[str(manifest_path)] = manifest_sha
        roles[condition] = {
            "role": (
                "oracle_grounding_counterfactual"
                if condition == "oracle_gt_mask"
                else (
                    "validation_selected_complete_predicted_pipeline"
                    if condition == selected
                    else "non_selected_predicted_grounding_ablation"
                )
            ),
            "analysis_manifest_path": str(manifest_path),
            "analysis_manifest_sha256": manifest_sha,
            "copied_to_top_level": condition == selected,
        }
    source_hashes[str(input_manifest.feature_schema.path)] = (
        input_manifest.feature_schema.sha256
    )
    contract = {
        "run_id": root.name,
        "selected_predicted_condition": selected,
        "selection_rule": "validation_mean_iou_only_no_test_access",
        "top_level_role": "validation_selected_complete_predicted_pipeline",
        "arms": roles,
        "selected_analysis_fingerprint": snapshot.manifest.get("analysis_fingerprint"),
        "candidate_count": len(native),
        "test_group_count": len(test_groups),
    }

    def render(staging: Path) -> None:
        _atomic_parquet(staging / "native_predictions.parquet", native)
        _atomic_parquet(staging / "reranked_predictions.parquet", reranked)
        for name in _SELECTED_COPY_OUTPUTS:
            _write_bytes_sync(staging / name, source_files[name].read_bytes())

    return _commit_component(
        root,
        manifest_name=SELECTED_ANALYSIS_MANIFEST,
        schema=SELECTED_ANALYSIS_OUTPUT_SCHEMA,
        contract=contract,
        source_hashes=source_hashes,
        output_names=_SELECTED_OUTPUTS,
        render=render,
        metadata={
            "selected_predicted_condition": selected,
            "selection_rule": "validation_mean_iou_only_no_test_access",
            "top_level_outputs_role": "validation_selected_complete_predicted_pipeline",
            "oracle_is_separate_counterfactual": True,
            "analysis_arms": roles,
            "candidate_count": len(native),
            "test_group_count": len(test_groups),
            "source_csv_sha256": {
                "native_predictions.csv": source_hashes[
                    str(source_files["native_predictions.csv"])
                ],
                "reranked_predictions.csv": source_hashes[
                    str(source_files["reranked_predictions.csv"])
                ],
            },
        },
        resume=resume,
    )


def _load_final_manifest(
    path: Path,
    *,
    input_fingerprint: str,
    expected_outputs: Mapping[str, str],
) -> dict[str, Any]:
    source, payload = _read_json(path, "formal run output manifest")
    check = dict(payload)
    observed = check.pop("commit_fingerprint", None)
    if observed != canonical_sha256(check):
        raise RunOutputError(f"formal run output commit is stale: {source}")
    if (
        payload.get("schema_version") != RUN_OUTPUT_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("input_fingerprint") != input_fingerprint
        or payload.get("outputs") != dict(expected_outputs)
    ):
        raise RunOutputError(
            "formal run output manifest cannot resume these components"
        )
    for name, expected in expected_outputs.items():
        path = _regular_file(source.parent / name, f"formal run output {name}")
        if sha256_file(path) != expected:
            raise RunOutputError(f"formal run output is stale: {path}")
    return payload


def publish_formal_run_outputs(
    run_dir: Path | str,
    *,
    audit_root: Path | str | None = None,
    device_root: Path | str | None = None,
    expected_environment_hashes: Mapping[str, str] | None = None,
    resume: bool = False,
) -> FormalRunOutputs:
    """Publish every required top-level machine-readable formal artifact.

    This is the single API intended for CLI wiring after ``ablate`` and before
    paper rendering::

        publish_formal_run_outputs(run_dir, resume=args.resume)
    """

    root, _, _ = _formal_run(run_dir)
    environment = publish_environment(
        root,
        audit_root=audit_root,
        device_root=device_root,
        expected_source_hashes=expected_environment_hashes,
        # The CLI intentionally publishes environment evidence immediately
        # after run creation, including runs that later block on download.
        # Revalidate that earlier commit even during the initial full run.
        resume=True,
    )
    candidates = publish_candidate_aggregates(root, resume=resume)
    selected_analysis = publish_selected_analysis_outputs(root, resume=resume)
    components = {
        "environment": {
            "path": environment.manifest_path.name,
            "sha256": environment.manifest_sha256,
            "input_fingerprint": environment.input_fingerprint,
        },
        "candidates": {
            "path": candidates.manifest_path.name,
            "sha256": candidates.manifest_sha256,
            "input_fingerprint": candidates.input_fingerprint,
        },
        "selected_analysis": {
            "path": selected_analysis.manifest_path.name,
            "sha256": selected_analysis.manifest_sha256,
            "input_fingerprint": selected_analysis.input_fingerprint,
        },
    }
    outputs = {
        **dict(environment.outputs),
        **dict(candidates.outputs),
        **dict(selected_analysis.outputs),
    }
    if len(outputs) != (
        len(environment.outputs)
        + len(candidates.outputs)
        + len(selected_analysis.outputs)
    ):
        raise AssertionError("formal component output names overlap")
    selected_manifest = json.loads(
        selected_analysis.manifest_path.read_text(encoding="utf-8")
    )
    selected = str(selected_manifest["selected_predicted_condition"])
    input_fingerprint = canonical_sha256(
        {
            "schema_version": RUN_OUTPUT_SCHEMA,
            "run_id": root.name,
            "components": components,
            "selected_predicted_condition": selected,
            "outputs": outputs,
        }
    )
    final_path = root / RUN_OUTPUT_MANIFEST
    if final_path.exists():
        if not resume:
            raise RunOutputError(
                f"formal run outputs already exist: {final_path}; use resume=True"
            )
        _load_final_manifest(
            final_path,
            input_fingerprint=input_fingerprint,
            expected_outputs=outputs,
        )
        resumed = True
    else:
        payload: dict[str, Any] = {
            "schema_version": RUN_OUTPUT_SCHEMA,
            "status": "COMPLETE",
            "scope": FORMAL_SCOPE,
            "fixture_only": False,
            "run_id": root.name,
            "input_fingerprint": input_fingerprint,
            "selected_predicted_condition": selected,
            "top_level_outputs_role": "validation_selected_complete_predicted_pipeline",
            "oracle_is_separate_counterfactual": True,
            "components": components,
            "outputs": outputs,
        }
        payload["commit_fingerprint"] = canonical_sha256(payload)
        atomic_json(final_path, payload)
        _load_final_manifest(
            final_path,
            input_fingerprint=input_fingerprint,
            expected_outputs=outputs,
        )
        resumed = False
    return FormalRunOutputs(
        run_dir=root,
        manifest_path=final_path,
        manifest_sha256=sha256_file(final_path),
        selected_predicted_condition=selected,
        outputs=outputs,
        environment=environment,
        candidates=candidates,
        selected_analysis=selected_analysis,
        resumed=resumed,
    )


__all__ = [
    "CANDIDATE_OUTPUT_SCHEMA",
    "ENVIRONMENT_OUTPUT_SCHEMA",
    "FormalRunOutputs",
    "PublishedComponent",
    "RUN_OUTPUT_SCHEMA",
    "RunOutputError",
    "SELECTED_ANALYSIS_OUTPUT_SCHEMA",
    "publish_candidate_aggregates",
    "publish_environment",
    "publish_formal_run_outputs",
    "publish_selected_analysis_outputs",
]
