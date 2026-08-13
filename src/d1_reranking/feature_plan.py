"""Frozen label-free plan for the nine heavy raw common-feature jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import (
    load_verified_json,
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256
import pandas as pd

from .execution import (
    FEATURE_RESOURCE_SCOPE,
    artifact_record,
    load_content_manifest,
)
from .provenance import load_source_closure
from .contracts import assert_label_free_parquet_schema


FEATURE_SPLITS = ("train", "validation", "test")
FEATURE_POOLS = ("top5", "top10", "allnms")
FEATURE_JOB_COUNT = 9
FEATURE_RUNNER_MODULE = "tools.d1_reranking.extract_common_features"
FEATURE_PLAN_REGISTRY_RELATIVE = Path("configs/feature_plans")
FEATURE_PLAN_POINTER_RELATIVE = Path("configs/d1_feature_extraction_plan_active.json")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _plan_run_root(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if source.parent.name == "feature_plans" and source.parent.parent.name == "configs":
        return source.parent.parent.parent
    if source.parent.name == "configs":
        return source.parent.parent
    raise RuntimeError(f"D1 raw-feature plan path is outside its registry: {source}")


def _candidate_manifest_path(root: Path, split: str) -> Path:
    base = root / "02_candidates" if split == "test" else root / "02_candidates" / split
    return base / ("test_manifest.json" if split == "test" else "manifest.json")


def _paired_copy_path(root: Path, split: str) -> Path:
    name = (
        "d1_paired_manifest.parquet"
        if split == "test"
        else f"d1_paired_{split}.parquet"
    )
    return root / "01_manifests" / name


def _candidate_sources(
    root: Path, *, closure_path: Path, closure: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifests: dict[str, Any] = {}
    pools: dict[str, Any] = {}
    hashes: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for split in FEATURE_SPLITS:
        path = _candidate_manifest_path(root, split)
        manifest = load_content_manifest(
            path, name=f"D1 {split} candidate manifest", statuses=("COMPLETE",)
        )
        artifacts = _mapping(
            manifest.get("artifacts"), name=f"D1 {split} candidate artifacts"
        )
        source_contract = _mapping(
            _mapping(
                manifest.get("configuration"),
                name=f"D1 {split} candidate configuration",
            ).get("source_contract"),
            name=f"D1 {split} candidate source contract",
        )
        closure_record = _mapping(
            source_contract.get("source_closure"),
            name=f"D1 {split} candidate source closure",
        )
        if (
            verified_artifact_path(
                closure_record, name=f"D1 {split} candidate source closure"
            )
            != closure_path
            or manifest.get("candidate_test_labels_read") is not False
        ):
            raise RuntimeError(f"D1 {split} candidate/closure contract differs")
        manifests[split] = artifact_record(path)
        pools[split] = {
            pool: _mapping(
                artifacts.get(pool), name=f"D1 {split}/{pool} candidate record"
            )
            for pool in FEATURE_POOLS
        }
        hashes[split] = _mapping(
            artifacts.get("candidate_hashes"),
            name=f"D1 {split} candidate hashes",
        )
        inputs = _mapping(
            _mapping(closure.get("canonical_inputs"), name="D1 canonical inputs").get(
                split
            ),
            name=f"D1 {split} canonical inputs",
        )
        paired_record = _mapping(
            inputs.get("paired_manifest"), name=f"D1 {split} paired source"
        )
        paired_source = verified_artifact_path(
            paired_record, name=f"D1 {split} paired source"
        )
        paired_copy = _paired_copy_path(root, split)
        if (
            artifact_record(paired_copy)["sha256"]
            != artifact_record(paired_source)["sha256"]
        ):
            raise RuntimeError(f"D1 {split} paired run copy differs")
        paired[split] = {
            "source": paired_record,
            "run_copy": artifact_record(paired_copy),
        }
        if split == "test":
            assert_label_free_parquet_schema(
                paired_source, name="D1 Test paired source"
            )
            assert_label_free_parquet_schema(
                paired_copy, name="D1 Test paired run copy"
            )
            for pool, record in pools[split].items():
                assert_label_free_parquet_schema(
                    verified_artifact_path(
                        record, name=f"D1 Test/{pool} candidate pool"
                    ),
                    name=f"D1 Test/{pool} candidate pool",
                )
            assert_label_free_parquet_schema(
                verified_artifact_path(hashes[split], name="D1 Test candidate hashes"),
                name="D1 Test candidate hashes",
            )
    verify_artifact_records_recursive(
        {
            "candidate_manifests": manifests,
            "candidate_pools": pools,
            "candidate_hashes": hashes,
            "paired_manifests": paired,
        },
        name="D1 raw-feature plan input closure",
        require_at_least_one=True,
    )
    return manifests, pools, hashes, paired


def build_feature_extraction_plan(
    run_dir: str | Path,
    *,
    python_path: Path,
    tool_paths: Sequence[Path],
) -> dict[str, Any]:
    """Recompute the exact nine-job plan from current immutable inputs."""

    root = Path(run_dir).expanduser().resolve()
    closure_path, closure = load_source_closure(root)
    if (
        closure.get("canonical_snapshot") != "A"
        or closure.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 raw-feature plan requires label-free Snapshot A")
    manifests, pools, hashes, paired = _candidate_sources(
        root, closure_path=closure_path, closure=closure
    )
    executable = python_path.expanduser().resolve()
    environment_path = root / "environment.txt"
    environment = {
        "python_executable": artifact_record(executable),
        "bootstrap_environment": artifact_record(environment_path),
    }
    code = [
        artifact_record(path)
        for path in sorted(
            {Path(path).expanduser().resolve() for path in tool_paths}, key=str
        )
    ]
    jobs: list[dict[str, Any]] = []
    for split in FEATURE_SPLITS:
        for pool in FEATURE_POOLS:
            configuration: dict[str, Any] = {
                "schema_version": 1,
                "route": "D1",
                "stage": "P5",
                "operation": "raw_matched_common_feature_extraction",
                "split": split,
                "pool": pool,
                "track": "matched_common_raw",
                "method": "unified_common_extractor",
                "chunk_size": 100,
                "tag": "formal",
                "limit": None,
                "candidate_test_labels_read": False,
                "test_inputs_referenced": split == "test",
            }
            job_id = canonical_sha256(configuration)[:16]
            jobs.append(
                {
                    "job_id": job_id,
                    "configuration": configuration,
                    "inputs": {
                        "source_closure": artifact_record(closure_path),
                        "candidate_manifest": manifests[split],
                        "candidate_pool": pools[split][pool],
                        "candidate_hashes": hashes[split],
                        "paired_source": paired[split]["source"],
                        "paired_copy": paired[split]["run_copy"],
                    },
                    "worker_argv": [
                        "-m",
                        FEATURE_RUNNER_MODULE,
                        "--job-id",
                        job_id,
                        "--split",
                        split,
                        "--pool",
                        pool,
                        "--chunk-size",
                        "100",
                        "--resume",
                    ],
                    "output_manifest": (
                        f"03_features/{split}/{pool}/matched_common_raw/manifest.json"
                    ),
                }
            )
    sources = {
        "source_closure": artifact_record(closure_path),
        "candidate_manifests": manifests,
        "candidate_pools": pools,
        "candidate_hashes": hashes,
        "paired_manifests": paired,
        "environment": environment,
        "code": code,
    }
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "PLANNED",
        "route": "D1",
        "analysis": "raw_matched_common_feature_extraction",
        "scope": FEATURE_RESOURCE_SCOPE,
        "job_count": FEATURE_JOB_COUNT,
        "job_ids_sha256": canonical_sha256([job["job_id"] for job in jobs]),
        "runner_module": FEATURE_RUNNER_MODULE,
        "max_parallel": 1,
        "device": "cpu",
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
        "source_signature_sha256": canonical_sha256(sources),
        "sources": sources,
        "jobs": jobs,
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def load_feature_extraction_plan(path: str | Path) -> dict[str, Any]:
    """Load and semantically rebuild the active plan and source universe."""

    source = Path(path).expanduser().resolve()
    plan = load_content_manifest(
        source, name="D1 raw-feature extraction plan", statuses=("PLANNED",)
    )
    sources = _mapping(plan.get("sources"), name="D1 raw-feature plan sources")
    environment = _mapping(
        sources.get("environment"), name="D1 raw-feature plan environment"
    )
    executable = verified_artifact_path(
        _mapping(environment.get("python_executable"), name="D1 feature Python record"),
        name="D1 feature Python executable",
    )
    code = sources.get("code")
    if not isinstance(code, Sequence) or isinstance(code, (str, bytes)):
        raise RuntimeError("D1 raw-feature plan code inventory is invalid")
    tool_paths = tuple(
        verified_artifact_path(
            _mapping(record, name="D1 raw-feature code record"),
            name="D1 raw-feature code",
        )
        for record in code
    )
    expected = build_feature_extraction_plan(
        _plan_run_root(source), python_path=executable, tool_paths=tool_paths
    )
    if plan != expected:
        raise RuntimeError("D1 raw-feature plan/source universe differs")
    return plan


def validate_feature_extraction_result(
    value: Mapping[str, Any],
    *,
    plan_path: str | Path,
    job: Mapping[str, Any],
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Replay one COMPLETE raw-feature result against its frozen job."""

    result = _mapping(value, name="D1 raw-feature result")
    unsigned = dict(result)
    observed_content = unsigned.pop("content_sha256", None)
    if observed_content != canonical_sha256(unsigned):
        raise RuntimeError("D1 raw-feature result content hash mismatch")
    configuration = _mapping(
        job.get("configuration"), name="D1 raw-feature planned configuration"
    )
    job_id = str(job.get("job_id", ""))
    root = _plan_run_root(plan_path)
    expected_manifest = (root / str(job.get("output_manifest", ""))).resolve()
    observed_manifest = Path(manifest_path).expanduser().resolve()
    if (
        result.get("status") != "COMPLETE"
        or result.get("job_id") != job_id
        or result.get("configuration") != configuration
        or result.get("configuration_sha256") != canonical_sha256(configuration)
        or result.get("candidate_test_labels_read") is not False
        or result.get("test_inputs_referenced")
        is not (configuration.get("split") == "test")
        or observed_manifest != expected_manifest
    ):
        raise RuntimeError("D1 raw-feature result/job contract differs")
    sources = _mapping(result.get("sources"), name="D1 raw-feature result sources")
    planned_inputs = _mapping(job.get("inputs"), name="D1 raw-feature planned inputs")
    if (
        sources.get("plan") != artifact_record(plan_path)
        or sources.get("planned_inputs") != planned_inputs
        or result.get("source_signature_sha256") != canonical_sha256(sources)
    ):
        raise RuntimeError("D1 raw-feature result source binding differs")
    verify_artifact_records_recursive(
        sources, name="D1 raw-feature result sources", require_at_least_one=True
    )
    artifacts = _mapping(
        result.get("artifacts"), name="D1 raw-feature result artifacts"
    )
    if set(artifacts) != {
        "candidate_features",
        "candidate_relations",
        "sample_context",
        "feature_schema",
        "missingness_report",
    }:
        raise RuntimeError("D1 raw-feature result artifact inventory differs")
    verified = verify_artifact_records_recursive(
        artifacts, name="D1 raw-feature result artifacts", require_at_least_one=True
    )
    if any(
        Path(record["path"]).parent != observed_manifest.parent for record in verified
    ):
        raise RuntimeError("D1 raw-feature result artifact path escapes its job")
    extractor = _mapping(
        sources.get("extractor"), name="D1 raw-feature extractor sources"
    )
    common_manifest_path = verified_artifact_path(
        _mapping(
            extractor.get("unified_common_manifest"),
            name="D1 unified common manifest record",
        ),
        name="D1 unified common manifest",
    )
    common = load_verified_json(
        common_manifest_path,
        name="D1 unified common manifest",
        statuses=("COMPLETE",),
    )
    common_artifacts = _mapping(
        common.get("artifacts"), name="D1 unified common artifacts"
    )
    for kind in ("candidate_features", "candidate_relations", "sample_context"):
        if _mapping(artifacts[kind], name=f"D1 raw-feature {kind}").get(
            "sha256"
        ) != _mapping(common_artifacts.get(kind), name=f"D1 common {kind}").get(
            "sha256"
        ):
            raise RuntimeError(f"D1 raw-feature {kind} differs from unified extractor")
    feature_path = verified_artifact_path(
        _mapping(artifacts.get("candidate_features"), name="D1 raw features"),
        name="D1 raw features",
    )
    candidate_path = verified_artifact_path(
        _mapping(
            planned_inputs.get("candidate_pool"), name="D1 raw-feature candidates"
        ),
        name="D1 raw-feature candidates",
    )
    if configuration.get("split") == "test":
        assert_label_free_parquet_schema(feature_path, name="D1 Test raw features")
        assert_label_free_parquet_schema(candidate_path, name="D1 Test candidates")
    keys = ["sample_id", "candidate_id"]
    features = pd.read_parquet(feature_path, columns=keys)
    candidates = pd.read_parquet(candidate_path, columns=keys)
    feature_keys = set(map(tuple, features.astype(str).to_numpy()))
    candidate_keys = set(map(tuple, candidates.astype(str).to_numpy()))
    if (
        features.duplicated(keys).any()
        or candidates.duplicated(keys).any()
        or feature_keys != candidate_keys
        or len(features) != len(candidates)
    ):
        raise RuntimeError("D1 raw-feature result candidate membership differs")
    model_columns = result.get("model_feature_columns")
    if (
        not isinstance(model_columns, list)
        or not model_columns
        or result.get("model_feature_schema_sha256") != canonical_sha256(model_columns)
        or common.get("model_feature_columns") != model_columns
    ):
        raise RuntimeError("D1 raw-feature model feature schema differs")
    schema_path = verified_artifact_path(
        _mapping(artifacts.get("feature_schema"), name="D1 raw feature schema"),
        name="D1 raw feature schema",
    )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("D1 raw feature schema is invalid") from error
    if (
        not isinstance(schema, Mapping)
        or schema.get("model_columns") != model_columns
        or schema.get("model_schema_sha256") != canonical_sha256(model_columns)
        or schema.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 raw feature schema contract differs")
    execution = _mapping(
        result.get("execution_provenance"),
        name="D1 raw-feature execution provenance",
    )
    if (
        execution.get("scope") != FEATURE_RESOURCE_SCOPE
        or execution.get("resource_lease_path")
        != str((root.parent / ".d1_heavy_resource.lock").resolve())
        or not isinstance(execution.get("execution_id"), str)
        or execution.get("execution_manifest") != sources.get("execution_manifest")
        or execution.get("execution_event") != sources.get("execution_event")
        or execution.get("execution_claim") != sources.get("execution_claim")
    ):
        raise RuntimeError("D1 raw-feature execution provenance differs")
    verify_artifact_records_recursive(
        execution,
        name="D1 raw-feature execution provenance",
        require_at_least_one=True,
    )
    return result


def write_feature_extraction_plan(
    destination: str | Path,
    *,
    run_dir: str | Path,
    python_path: Path,
    tool_paths: Sequence[Path],
    resume: bool,
) -> dict[str, Any]:
    from .run import assert_writable_prelock

    path = Path(destination).expanduser().resolve()
    root = Path(run_dir).expanduser().resolve()
    assert_writable_prelock(root)
    expected = build_feature_extraction_plan(
        root, python_path=python_path, tool_paths=tool_paths
    )
    if path.exists():
        if not resume:
            raise FileExistsError(f"D1 raw-feature plan already exists: {path}")
        existing = load_feature_extraction_plan(path)
        if existing != expected:
            raise RuntimeError("D1 raw-feature plan resume differs")
        return existing
    assert_writable_prelock(root)
    from .feature_execution import exclusive_json

    try:
        exclusive_json(path, expected)
    except FileExistsError:
        if not resume:
            raise FileExistsError(f"D1 raw-feature plan already exists: {path}")
        existing = load_feature_extraction_plan(path)
        if existing != expected:
            raise RuntimeError("D1 raw-feature plan resume differs")
        return existing
    return expected


def publish_feature_extraction_plan(
    run_dir: str | Path,
    *,
    python_path: Path,
    tool_paths: Sequence[Path],
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    """Publish one immutable plan and atomically advance its active pointer."""

    from .feature_execution import exclusive_json
    from .run import assert_writable_prelock

    root = Path(run_dir).expanduser().resolve()
    assert_writable_prelock(root)
    expected = build_feature_extraction_plan(
        root, python_path=python_path, tool_paths=tool_paths
    )
    plan_id = str(expected["content_sha256"])[:20]
    destination = root / FEATURE_PLAN_REGISTRY_RELATIVE / f"{plan_id}.json"
    if destination.exists():
        existing = load_feature_extraction_plan(destination)
        if existing != expected:
            raise RuntimeError("immutable D1 raw-feature plan differs")
        if not resume:
            raise FileExistsError(f"D1 raw-feature plan already exists: {destination}")
    else:
        exclusive_json(destination, expected)
    pointer: dict[str, Any] = {
        "schema_version": 1,
        "status": "PLANNED_POINTER",
        "active_plan": artifact_record(destination),
        "job_count": FEATURE_JOB_COUNT,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    assert_writable_prelock(root)
    atomic_json(root / FEATURE_PLAN_POINTER_RELATIVE, pointer)
    return destination, expected


def load_active_feature_extraction_plan(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Load the active immutable plan without invalidating prior executions."""

    root = Path(run_dir).expanduser().resolve()
    pointer = load_content_manifest(
        root / FEATURE_PLAN_POINTER_RELATIVE,
        name="D1 active raw-feature plan pointer",
        statuses=("PLANNED_POINTER",),
    )
    plan_path = verified_artifact_path(
        _mapping(pointer.get("active_plan"), name="D1 active feature plan record"),
        name="D1 active raw-feature plan",
    )
    if plan_path.parent != (root / FEATURE_PLAN_REGISTRY_RELATIVE).resolve():
        raise RuntimeError("D1 active raw-feature plan is outside its registry")
    plan = load_feature_extraction_plan(plan_path)
    if (
        pointer.get("active_plan") != artifact_record(plan_path)
        or pointer.get("job_count") != FEATURE_JOB_COUNT
        or pointer.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 active raw-feature plan pointer differs")
    return plan_path, plan


__all__ = [
    "FEATURE_JOB_COUNT",
    "FEATURE_PLAN_POINTER_RELATIVE",
    "FEATURE_PLAN_REGISTRY_RELATIVE",
    "FEATURE_POOLS",
    "FEATURE_RUNNER_MODULE",
    "FEATURE_SPLITS",
    "build_feature_extraction_plan",
    "load_feature_extraction_plan",
    "load_active_feature_extraction_plan",
    "publish_feature_extraction_plan",
    "validate_feature_extraction_result",
    "write_feature_extraction_plan",
]
