"""Repository-owned callbacks for immutable V3 inference and evaluation.

The lifecycle layer deliberately accepts only a flat ``NAME=PATH`` mapping.
This module keeps that boundary while supporting multi-shard feature artifacts:
the mapping names a locked JSON descriptor and locked artifact manifests, and
the manifests bind every index, schema, and shard file by SHA-256.  No label
path is accepted by :func:`run_formal_inference`; labels enter only through the
two explicitly locked evaluator artifacts used by
:func:`run_independent_dual_track_evaluation`.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from failure_analysis.reranking_v2.schema import atomic_write_jsonl

from .evaluation import evaluate_method_suite
from .depth_fallback import (
    catalog_depth_availability,
    checkpoint_ensemble_uses_depth,
    merge_depth_fallback_rankings,
    plan_depth_aware_execution,
)
from .feature_store import FeatureCatalog
from .formal import verify_scope_candidate_identity
from .fullchain_extractor import extract_fullchain_features
from .independent_evaluation import independently_recompute_suite
from .inference import prepare_gate_inference, write_v3_predictions
from .pipeline import predict_final_ensemble
from .schema import (
    artifact_identity,
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_sample_id,
)
from .uncertainty import score_perturbation_ensemble_streaming


SCHEMA_VERSION = "3.0.0"
INFERENCE_DESCRIPTOR_KIND = "v3_formal_inference_descriptor"
SOURCE_FILE_MANIFEST_KIND = "v3_formal_source_file_manifest"
SUPPORTED_V3_METHODS = frozenset(
    {
        "v3_full_head_scalar_gate",
        "v3_fcer_native",
        "v3_fcer_rgbd",
        "v3_locked_primary",
    }
)
REQUIRED_METHODS = frozenset(
    {"q_only", "v2_locked_primary", "v3_locked_primary"}
)
POLICY_FIELDS = frozenset(
    {
        "harm_cost",
        "threshold",
        "uncertainty_kappa",
        "consensus",
        "minimum_valid_fraction",
    }
)


def build_formal_source_file_manifest(
    *, candidate_artifact: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Bind every RGB/depth file that a later fresh formal extraction may read."""
    candidate = Path(candidate_artifact).expanduser().resolve()
    paths: set[Path] = set()
    sources: list[dict[str, Any]] = []
    missing_depth_count = 0
    for record in read_jsonl(candidate):
        if "image_path" not in record or not str(record["image_path"]).strip():
            raise ValueError("candidate record is missing image_path")
        image = Path(str(record["image_path"])).expanduser().resolve()
        image_identity = artifact_identity(image)
        paths.add(image)
        raw_depth = record.get("depth_path")
        if raw_depth is None or not str(raw_depth).strip():
            depth: dict[str, Any] = {
                "status":"missing", "reason":"depth_path_missing", "path":None,
            }
            missing_depth_count += 1
        else:
            depth_path = Path(str(raw_depth)).expanduser().resolve()
            if depth_path.is_file():
                depth = {"status":"present", **artifact_identity(depth_path)}
                paths.add(depth_path)
            else:
                depth = {
                    "status":"missing", "reason":"depth_file_missing",
                    "path":str(depth_path),
                }
                missing_depth_count += 1
        sources.append({
            "sample_id":str(
                record.get("stable_sample_id")
                or stable_sample_id(str(record["split"]), record["sample_id"])
            ),
            "image":{"status":"present", **image_identity},
            "depth":depth,
        })
    if not paths:
        raise ValueError("candidate artifact contains no RGB/depth sources")
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_FILE_MANIFEST_KIND,
        "status": "complete",
        "candidate_artifact": artifact_identity(candidate),
        "source_file_count": len(paths),
        "files": [artifact_identity(path) for path in sorted(paths)],
        "sources":sources,
        "declared_missing_depth_count":missing_depth_count,
        "labels_read": False,
    }
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    atomic_write_json(output, result)
    return result


def _json_object(path: str | Path, *, description: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {source}")
    return value


def _locked_index(locked_manifest: Mapping[str, Any]) -> dict[str, str]:
    identities = locked_manifest.get("locked_artifacts")
    if not isinstance(identities, list) or not identities:
        raise ValueError("locked manifest has no artifact identities")
    result: dict[str, str] = {}
    for value in identities:
        if not isinstance(value, Mapping) or not {"path", "sha256"} <= set(value):
            raise ValueError("locked manifest contains an invalid artifact identity")
        path = str(Path(str(value["path"])).expanduser().resolve())
        digest = str(value["sha256"])
        if path in result and result[path] != digest:
            raise ValueError(f"locked manifest has conflicting hashes: {path}")
        result[path] = digest
    return result


def _verified_inputs(
    input_artifacts: Mapping[str, str | Path], locked_manifest: Mapping[str, Any]
) -> dict[str, Path]:
    if not isinstance(input_artifacts, Mapping) or not input_artifacts:
        raise ValueError("input_artifacts must be a non-empty mapping")
    locked = _locked_index(locked_manifest)
    result: dict[str, Path] = {}
    for raw_name, raw_path in input_artifacts.items():
        name = str(raw_name).strip()
        if not name or name in result:
            raise ValueError("input_artifacts contains an invalid or duplicate name")
        path = Path(raw_path).expanduser().resolve()
        identity = artifact_identity(path)
        if locked.get(identity["path"]) != identity["sha256"]:
            raise PermissionError(f"formal backend input is not locked: {name}")
        result[name] = path
    return result


def _input(
    inputs: Mapping[str, Path], name: Any, *, field: str, used: set[str]
) -> Path:
    value = str(name).strip()
    if not value or value not in inputs:
        raise ValueError(f"{field} must name an input_artifact")
    used.add(value)
    return inputs[value]


def _artifact_group_paths(
    locked_manifest: Mapping[str, Any], group: str
) -> set[Path]:
    value = locked_manifest.get("artifacts", {}).get(group)
    if isinstance(value, Mapping) and {"path", "sha256"} <= set(value):
        identities = [value]
    elif isinstance(value, Mapping):
        identities = list(value.values())
    else:
        raise ValueError(f"locked manifest has no {group} artifact group")
    result: set[Path] = set()
    for index, identity in enumerate(identities):
        if not isinstance(identity, Mapping) or "path" not in identity:
            raise ValueError(f"locked {group} artifact {index} has an invalid identity")
        result.add(Path(str(identity["path"])).expanduser().resolve())
    return result


def _require_group_membership(
    paths: Sequence[Path], *, locked_manifest: Mapping[str, Any], group: str
) -> None:
    allowed = _artifact_group_paths(locked_manifest, group)
    unexpected = set(paths) - allowed
    if unexpected:
        raise PermissionError(
            f"formal inputs are not declared in locked {group} artifacts: "
            f"{sorted(map(str, unexpected))}"
        )


def _candidate_records(path: str | Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(path):
        sample_id = str(
            value.get("stable_sample_id")
            or stable_sample_id(str(value["split"]), value["sample_id"])
        )
        if sample_id in records:
            raise ValueError(f"duplicate frozen candidate sample: {sample_id}")
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"frozen candidate record is missing candidates: {sample_id}")
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        if len(candidate_ids) != 5 or len(set(candidate_ids)) != 5:
            raise ValueError(f"candidate pool is not the frozen five: {sample_id}")
        ordered.append(sample_id)
        records[sample_id] = value
    if not ordered:
        raise ValueError("frozen candidate artifact is empty")
    return ordered, records


def _verify_identity(value: Mapping[str, Any], *, description: str) -> Path:
    if not isinstance(value, Mapping) or not {"path", "sha256"} <= set(value):
        raise ValueError(f"{description} has an invalid artifact identity")
    path = Path(str(value["path"])).expanduser().resolve()
    if sha256_file(path) != str(value["sha256"]):
        raise ValueError(f"{description} changed: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"{description} size changed: {path}")
    return path


def _verified_feature_root(
    manifest_path: Path, *, artifact_type: str
) -> Path:
    """Verify every file that FeatureCatalog may read from a directory."""
    manifest = _json_object(manifest_path, description="feature artifact manifest")
    if manifest.get("status") != "complete" or manifest.get("artifact_type") != artifact_type:
        raise ValueError(f"unexpected feature artifact manifest: {manifest_path}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"feature artifact manifest has no bound outputs: {manifest_path}")
    root = manifest_path.parent.resolve()
    verified: set[Path] = set()
    for index, identity in enumerate(outputs):
        path = _verify_identity(identity, description=f"feature output {index}")
        if not path.is_relative_to(root):
            raise PermissionError(f"feature manifest references a file outside its root: {path}")
        if path in verified:
            raise ValueError(f"feature manifest repeats an output: {path}")
        verified.add(path)
    index_path = root / "index.jsonl"
    schema_path = root / "feature_schema.json"
    if index_path not in verified or schema_path not in verified:
        raise ValueError("feature manifest does not bind index.jsonl and feature_schema.json")
    referenced_shards: set[Path] = set()
    for record in read_jsonl(index_path):
        shard = (root / "shards" / str(record["shard"])).resolve()
        if not shard.is_relative_to(root / "shards"):
            raise PermissionError("feature index contains an unsafe shard path")
        referenced_shards.add(shard)
    if not referenced_shards or not referenced_shards <= verified:
        raise ValueError("feature manifest does not bind every index-referenced shard")
    return root


def _verify_source_files(manifest_path: Path, candidate_path: Path) -> None:
    manifest = _json_object(manifest_path, description="source-file manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != SOURCE_FILE_MANIFEST_KIND
        or manifest.get("status") != "complete"
    ):
        raise ValueError("fresh extraction requires a V3 source-file manifest")
    bound_candidate = manifest.get("candidate_artifact")
    if not isinstance(bound_candidate, Mapping):
        raise ValueError("source-file manifest does not bind its candidate artifact")
    observed_candidate = artifact_identity(candidate_path)
    if (
        str(Path(str(bound_candidate.get("path", ""))).expanduser().resolve())
        != observed_candidate["path"]
        or str(bound_candidate.get("sha256", "")) != observed_candidate["sha256"]
    ):
        raise ValueError("source-file manifest is bound to a different candidate artifact")
    identities = manifest.get("files")
    if not isinstance(identities, list) or not identities:
        raise ValueError("source-file manifest has no files")
    if int(manifest.get("source_file_count", -1)) != len(identities):
        raise ValueError("source-file manifest count differs from its file identities")
    expected: set[Path] = set()
    expected_sources: list[dict[str, Any]] = []
    expected_missing_depth_count = 0
    for record in read_jsonl(candidate_path):
        if "image_path" not in record or not str(record["image_path"]).strip():
            raise ValueError("candidate record is missing image_path")
        image = Path(str(record["image_path"])).expanduser().resolve()
        image_identity = artifact_identity(image)
        expected.add(image)
        raw_depth = record.get("depth_path")
        if raw_depth is None or not str(raw_depth).strip():
            depth: dict[str, Any] = {
                "status":"missing", "reason":"depth_path_missing", "path":None,
            }
            expected_missing_depth_count += 1
        else:
            depth_path = Path(str(raw_depth)).expanduser().resolve()
            if depth_path.is_file():
                depth = {"status":"present", **artifact_identity(depth_path)}
                expected.add(depth_path)
            else:
                depth = {
                    "status":"missing", "reason":"depth_file_missing",
                    "path":str(depth_path),
                }
                expected_missing_depth_count += 1
        expected_sources.append({
            "sample_id":str(
                record.get("stable_sample_id")
                or stable_sample_id(str(record["split"]), record["sample_id"])
            ),
            "image":{"status":"present", **image_identity},
            "depth":depth,
        })
    observed: set[Path] = set()
    for index, identity in enumerate(identities):
        path = _verify_identity(identity, description=f"source file {index}")
        if path in observed:
            raise ValueError(f"source-file manifest repeats a path: {path}")
        observed.add(path)
    if observed != expected:
        raise ValueError(
            "source-file manifest differs from candidate RGB/depth references: "
            f"missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    declared_sources = manifest.get("sources")
    if declared_sources is not None:
        if declared_sources != expected_sources:
            raise ValueError("source-file manifest missing-depth declarations differ")
        if int(manifest.get("declared_missing_depth_count", -1)) != expected_missing_depth_count:
            raise ValueError("source-file manifest missing-depth count differs")
    elif expected_missing_depth_count:
        raise ValueError("source-file manifest does not declare missing depth")


def _feature_catalog(
    *,
    descriptor: Mapping[str, Any],
    inputs: Mapping[str, Path],
    candidate_path: Path,
    output_root: Path,
    used: set[str],
) -> FeatureCatalog:
    source = descriptor.get("feature_source")
    if not isinstance(source, Mapping):
        raise ValueError("descriptor.feature_source must be an object")
    mode = str(source.get("mode", "")).strip()
    if mode == "reuse":
        raw_manifests = source.get("artifact_manifest_inputs")
        if not isinstance(raw_manifests, list) or not raw_manifests:
            raise ValueError("reuse mode requires artifact_manifest_inputs")
        artifact_roots = [
            _verified_feature_root(
                _input(inputs, name, field="artifact_manifest_inputs", used=used),
                artifact_type="fullchain_candidate_features",
            )
            for name in raw_manifests
        ]
        raw_overlays = source.get("head_override_manifest_inputs", [])
        if not isinstance(raw_overlays, list):
            raise ValueError("head_override_manifest_inputs must be a list")
        overlay_roots = [
            _verified_feature_root(
                _input(inputs, name, field="head_override_manifest_inputs", used=used),
                artifact_type="fullchain_head_feature_correction_overlay",
            )
            for name in raw_overlays
        ]
        return FeatureCatalog(
            artifact_roots,
            head_override_dirs=overlay_roots,
            candidate_feature_paths=(candidate_path,),
        )
    if mode != "extract":
        raise ValueError("feature_source.mode must be reuse or extract")
    metadata_path = _input(
        inputs,
        source.get("source_metadata_input"),
        field="source_metadata_input",
        used=used,
    )
    if metadata_path != (candidate_path.parent / "metadata.json").resolve():
        raise ValueError("source metadata must be the frozen candidate sibling metadata.json")
    source_manifest = _input(
        inputs,
        source.get("source_file_manifest_input"),
        field="source_file_manifest_input",
        used=used,
    )
    _verify_source_files(source_manifest, candidate_path)
    split_path = _input(
        inputs, source.get("split_manifest_input"), field="split_manifest_input", used=used
    )
    config_path = _input(
        inputs, source.get("crog_config_input"), field="crog_config_input", used=used
    )
    checkpoint_path = _input(
        inputs,
        source.get("crog_checkpoint_input"),
        field="crog_checkpoint_input",
        used=used,
    )
    options = source.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("feature_source.options must be an object")
    allowed_options = {
        "batch_size",
        "crop_size",
        "roi_size",
        "channel_bins",
        "shard_samples",
        "seed",
    }
    unknown = set(options) - allowed_options
    if unknown:
        raise ValueError(f"unknown fresh extraction options: {sorted(unknown)}")
    feature_root = output_root / "_generated_fullchain_features"
    extract_fullchain_features(
        frozen_features_path=candidate_path,
        split_manifest_path=split_path,
        output_dir=feature_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=str(descriptor.get("device", "auto")),
        batch_size=int(options.get("batch_size", 16)),
        crop_size=int(options.get("crop_size", 32)),
        roi_size=int(options.get("roi_size", 7)),
        channel_bins=int(options.get("channel_bins", 32)),
        shard_samples=int(options.get("shard_samples", 64)),
        seed=int(options.get("seed", 20260801)),
        resume=False,
    )
    # The freshly written manifest is self-verifying before FeatureCatalog sees it.
    verified_root = _verified_feature_root(
        feature_root / "artifact_manifest.json",
        artifact_type="fullchain_candidate_features",
    )
    return FeatureCatalog(
        (verified_root,), candidate_feature_paths=(candidate_path,)
    )


def _load_prior(path: Path, ordered_ids: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_ids", "prior", "valid"}
        if not required <= set(payload.files):
            raise ValueError("deployment V2 prior is missing required arrays")
        source_ids = list(map(str, payload["sample_ids"]))
        prior = np.asarray(payload["prior"], dtype=np.float32)
        valid = np.asarray(payload["valid"], dtype=bool)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("deployment V2 prior contains duplicate sample IDs")
    if prior.shape != (len(source_ids), 5, 80) or valid.shape != (len(source_ids),):
        raise ValueError("deployment V2 prior has invalid shapes")
    if set(source_ids) != set(ordered_ids) or not valid.all():
        raise ValueError("deployment V2 prior coverage differs from the formal cohort")
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
    return {sample_id: prior[lookup[sample_id]] for sample_id in ordered_ids}


def _materialize_v2_ranking(
    *,
    candidate_path: Path,
    source_path: Path,
    output_path: Path,
) -> Path:
    ordered_ids, candidates = _candidate_records(candidate_path)
    source: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(source_path):
        sample_id = str(value["sample_id"])
        if sample_id in source:
            raise ValueError(f"duplicate V2 ranking sample: {sample_id}")
        source[sample_id] = value
    missing = set(ordered_ids) - set(source)
    if missing:
        raise ValueError(f"V2 ranking is missing formal samples: {sorted(missing)[:5]}")
    rows: list[dict[str, Any]] = []
    for sample_id in ordered_ids:
        record = dict(source[sample_id])
        expected = [str(value["candidate_id"]) for value in candidates[sample_id]["candidates"]]
        order = list(map(str, record.get("candidate_order", ())))
        if len(order) != 5 or set(order) != set(expected):
            raise ValueError(f"V2 ranking changed the candidate pool: {sample_id}")
        record["sample_id"] = sample_id
        record["method"] = "v2_locked_primary"
        rows.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_path, rows)
    return output_path


def _policy(path: Path) -> dict[str, Any]:
    value = _json_object(path, description="locked gate policy")
    for wrapper in ("selected_policy", "policy"):
        if isinstance(value.get(wrapper), Mapping):
            value = dict(value[wrapper])
            break
    missing = POLICY_FIELDS - set(value)
    if missing:
        raise ValueError(f"locked gate policy is missing fields: {sorted(missing)}")
    result = {field: value[field] for field in POLICY_FIELDS}
    numeric = np.asarray(
        [
            result["harm_cost"],
            result["threshold"],
            result["uncertainty_kappa"],
            result["minimum_valid_fraction"],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(numeric).all():
        raise ValueError("locked gate policy contains non-finite values")
    if int(result["consensus"]) <= 0:
        raise ValueError("locked gate policy consensus must be positive")
    if not 0.0 <= float(result["minimum_valid_fraction"]) <= 1.0:
        raise ValueError("minimum_valid_fraction must be in [0,1]")
    return result


def _method_inputs(
    *,
    method: str,
    method_config: Mapping[str, Any],
    inputs: Mapping[str, Path],
    locked_manifest: Mapping[str, Any],
    used: set[str],
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    allowed = {
        "checkpoint_inputs", "gate_checkpoint_inputs", "policy_input",
        "missing_depth_fallback_method",
    }
    unknown = set(method_config) - allowed
    if unknown:
        raise ValueError(f"unknown {method} descriptor fields: {sorted(unknown)}")
    raw_checkpoints = method_config.get("checkpoint_inputs")
    raw_gates = method_config.get("gate_checkpoint_inputs")
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != 3:
        raise ValueError(f"{method} requires exactly three final checkpoints")
    if not isinstance(raw_gates, list) or len(raw_gates) != 3:
        raise ValueError(f"{method} requires exactly three gate checkpoints")
    checkpoints = [
        _input(inputs, name, field=f"methods.{method}.checkpoint_inputs", used=used)
        for name in raw_checkpoints
    ]
    gates = [
        _input(inputs, name, field=f"methods.{method}.gate_checkpoint_inputs", used=used)
        for name in raw_gates
    ]
    if len(set(checkpoints)) != 3 or len(set(gates)) != 3:
        raise ValueError(f"{method} checkpoint paths must be distinct")
    policy_path = _input(
        inputs,
        method_config.get("policy_input"),
        field=f"methods.{method}.policy_input",
        used=used,
    )
    _require_group_membership(
        checkpoints, locked_manifest=locked_manifest, group="checkpoints"
    )
    _require_group_membership(
        [*gates, policy_path], locked_manifest=locked_manifest, group="gate"
    )
    return checkpoints, gates, _policy(policy_path)


def run_formal_inference(
    *,
    scope: str,
    input_artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
    locked_manifest: Mapping[str, Any],
) -> list[Path]:
    """Run the locked label-free FCER stack and materialize method rankings."""
    scope_name = str(scope).strip().lower()
    if scope_name not in {"lockcheck", "test"}:
        raise ValueError("formal inference scope must be lockcheck or test")
    inputs = _verified_inputs(input_artifacts, locked_manifest)
    used = {"inference_descriptor"}
    if "inference_descriptor" not in inputs:
        raise ValueError("input_artifacts must include inference_descriptor")
    descriptor = _json_object(
        inputs["inference_descriptor"], description="formal inference descriptor"
    )
    if (
        descriptor.get("schema_version") != SCHEMA_VERSION
        or descriptor.get("kind") != INFERENCE_DESCRIPTOR_KIND
    ):
        raise ValueError("unsupported formal inference descriptor")
    scopes = descriptor.get("scopes")
    if not isinstance(scopes, list) or scope_name not in set(map(str, scopes)):
        raise PermissionError(f"descriptor does not authorize {scope_name} inference")
    methods = list(map(str, locked_manifest.get("formal_methods", ())))
    if not REQUIRED_METHODS <= set(methods):
        raise ValueError("locked formal methods are missing the required baseline/primary suite")
    unsupported = set(methods) - {"q_only", "v2_locked_primary"} - SUPPORTED_V3_METHODS
    if unsupported:
        raise ValueError(f"formal backend does not support locked methods: {sorted(unsupported)}")
    method_configs = descriptor.get("methods")
    expected_v3 = set(methods) & SUPPORTED_V3_METHODS
    if not isinstance(method_configs, Mapping) or set(method_configs) != expected_v3:
        raise ValueError("descriptor method configs differ from locked V3 methods")
    candidate_path = _input(
        inputs,
        descriptor.get("candidate_input"),
        field="candidate_input",
        used=used,
    )
    verify_scope_candidate_identity(
        locked_manifest, scope=scope_name, candidate_artifact=candidate_path
    )
    ordered_ids, _ = _candidate_records(candidate_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    catalog = _feature_catalog(
        descriptor=descriptor,
        inputs=inputs,
        candidate_path=candidate_path,
        output_root=output,
        used=used,
    )
    cohort = set(ordered_ids)
    catalog.assert_exact_ids(cohort)
    prior_path = _input(
        inputs, descriptor.get("v2_prior_input"), field="v2_prior_input", used=used
    )
    priors = _load_prior(prior_path, ordered_ids)
    v2_source = _input(
        inputs,
        descriptor.get("v2_ranking_input"),
        field="v2_ranking_input",
        used=used,
    )
    _require_group_membership(
        [prior_path, v2_source], locked_manifest=locked_manifest, group="v2"
    )
    batch_size = int(descriptor.get("inference_batch_size", 32))
    if batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    device = str(descriptor.get("device", "auto"))
    if device not in {"auto", "mps", "cpu"}:
        raise ValueError("formal inference device must be auto, mps, or cpu")
    v2_materialized = _materialize_v2_ranking(
            candidate_path=candidate_path,
            source_path=v2_source,
            output_path=output / "v2_locked_primary.jsonl",
        )
    result_paths = [v2_materialized]
    depth_status = catalog_depth_availability(catalog, cohort)
    missing_depth_ids = {
        sample_id for sample_id, value in depth_status.items()
        if value.get("available") is not True
    }
    resolved_methods: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method not in expected_v3:
            continue
        config = method_configs[method]
        if not isinstance(config, Mapping):
            raise ValueError(f"methods.{method} must be an object")
        checkpoints, gates, policy = _method_inputs(
            method=method,
            method_config=config,
            inputs=inputs,
            locked_manifest=locked_manifest,
            used=used,
        )
        fallback_method = config.get("missing_depth_fallback_method")
        if fallback_method is not None:
            fallback_method = str(fallback_method).strip()
            if not fallback_method or fallback_method not in expected_v3 or fallback_method == method:
                raise ValueError(f"{method} declares an invalid missing-depth fallback method")
        # Checkpoint architecture is part of the frozen method identity, not a
        # property that may be skipped merely because this cohort happens to
        # have complete depth coverage.
        uses_depth = checkpoint_ensemble_uses_depth(checkpoints)
        if method == "v3_fcer_native" and uses_depth:
            raise ValueError("v3_fcer_native checkpoints must not use depth")
        if method == "v3_fcer_rgbd" and not uses_depth:
            raise ValueError("v3_fcer_rgbd checkpoints must use depth")
        if fallback_method is not None and not uses_depth:
            raise ValueError(f"Native method {method} must not declare a depth fallback")
        if uses_depth and fallback_method not in {None, "v3_fcer_native"}:
            raise ValueError(
                f"RGB-D method {method} may fall back only to v3_fcer_native or V2"
            )
        resolved_methods[method] = {
            "checkpoints":checkpoints, "gates":gates, "policy":policy,
            "uses_depth":uses_depth, "fallback_method":fallback_method,
        }

    materialized_methods: dict[str, Path] = {}
    pending = [method for method in methods if method in expected_v3]
    while pending:
        progressed = False
        for method in list(pending):
            resolved = resolved_methods[method]
            fallback_method = resolved["fallback_method"]
            if fallback_method is not None and fallback_method not in materialized_methods:
                continue
            if fallback_method is not None and resolved_methods[fallback_method]["uses_depth"]:
                raise ValueError("RGB-D missing-depth fallback method must be Native")
            checkpoints = resolved["checkpoints"]
            gates = resolved["gates"]
            policy = resolved["policy"]
            plan = plan_depth_aware_execution(
                sample_ids=cohort,
                depth_status=depth_status,
                use_depth=bool(resolved["uses_depth"]),
                native_fallback_declared=fallback_method is not None,
            )
            model_ids = set(plan["model_sample_ids"])
            fallback_ids = set(plan["fallback_sample_ids"])
            method_root = output / method
            ensemble_path = method_root / "fcer_ensemble.npz"
            uncertainty_path = method_root / "uncertainty.npz"
            primary_path: Path | None = None
            if model_ids:
                predict_final_ensemble(
                    catalog=catalog,
                    sample_ids=model_ids,
                    priors=priors,
                    checkpoint_paths=checkpoints,
                    output_path=ensemble_path,
                    device=device,
                )
                score_perturbation_ensemble_streaming(
                    catalog=catalog,
                    sample_ids=model_ids,
                    priors=priors,
                    checkpoint_paths=checkpoints,
                    output_path=uncertainty_path,
                    device=device,
                    batch_size=batch_size,
                )
                bundle = prepare_gate_inference(
                    ensemble_path=ensemble_path,
                    uncertainty_path=uncertainty_path,
                    v2_predictions_path=v2_source,
                    gate_checkpoint_paths=gates,
                    device=device,
                )
                summary = write_v3_predictions(
                    bundle=bundle,
                    policy=policy,
                    output_dir=method_root / "ranking",
                    gate_checkpoint_paths=gates,
                    method=method,
                )
                primary_path = Path(summary["prediction"]["path"]).resolve()
            if fallback_ids:
                final_path = merge_depth_fallback_rankings(
                    ordered_sample_ids=ordered_ids,
                    method=method,
                    primary_ranking_path=primary_path,
                    missing_depth_ids=fallback_ids,
                    v2_ranking_path=v2_materialized,
                    native_ranking_path=(
                        None if fallback_method is None
                        else materialized_methods[fallback_method]
                    ),
                    output_path=method_root/"depth_aware_predictions.jsonl",
                    depth_status=depth_status,
                )
            elif primary_path is None:
                raise AssertionError("depth-aware inference produced no ranking")
            else:
                final_path = primary_path
            materialized_methods[method] = final_path
            result_paths.append(final_path)
            pending.remove(method)
            progressed = True
        if not progressed:
            raise ValueError("missing-depth fallback methods contain a dependency cycle")
    if used != set(inputs):
        raise ValueError(f"formal inference received unused inputs: {sorted(set(inputs)-used)}")
    return result_paths


def _verified_evaluation_labels(
    evaluation_artifacts: Mapping[str, str | Path],
) -> dict[str, Path]:
    if not isinstance(evaluation_artifacts, Mapping):
        raise TypeError("evaluation_artifacts must be a mapping")
    expected = {"corrected_labels", "legacy_labels"}
    if set(evaluation_artifacts) != expected:
        raise ValueError(
            "independent evaluation requires exactly corrected_labels and legacy_labels"
        )
    return {
        name: Path(evaluation_artifacts[name]).expanduser().resolve()
        for name in sorted(expected)
    }


def _verify_locked_evaluator_implementations(locked_manifest: Mapping[str, Any]) -> None:
    evaluators = locked_manifest.get("artifacts", {}).get("evaluators")
    expected = {"corrected_scientific", "legacy_official_compatibility"}
    if not isinstance(evaluators, Mapping) or set(evaluators) != expected:
        raise ValueError("locked manifest must name exactly the two evaluator implementations")
    for name in sorted(expected):
        _verify_identity(
            evaluators[name], description=f"locked evaluator implementation {name}"
        )


def run_independent_dual_track_evaluation(
    *,
    candidate_artifact: str | Path,
    method_rankings: Mapping[str, str | Path | None],
    evaluation_artifacts: Mapping[str, str | Path],
    evaluation_scope: str = "test",
    output_dir: str | Path,
    locked_manifest: Mapping[str, Any],
) -> list[Path]:
    """Join labels only in the independent stage and run both evaluators."""
    scope = str(evaluation_scope).strip().lower()
    if scope not in {"lockcheck", "test"}:
        raise ValueError("evaluation_scope must be lockcheck or test")
    expected_methods = list(map(str, locked_manifest.get("formal_methods", ())))
    if set(method_rankings) != set(expected_methods):
        raise ValueError("method rankings differ from the locked method suite")
    if method_rankings.get("q_only") is not None:
        raise ValueError("q_only must be represented by None")
    _verify_locked_evaluator_implementations(locked_manifest)
    labels = _verified_evaluation_labels(evaluation_artifacts)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    evaluation_root = output / "dual_track"
    methods = {
        method: None if path is None else Path(path).expanduser().resolve()
        for method, path in method_rankings.items()
    }
    evaluate_method_suite(
        features_path=Path(candidate_artifact).expanduser().resolve(),
        corrected_labels_path=labels["corrected_labels"],
        legacy_labels_path=labels["legacy_labels"],
        methods=methods,
        output_dir=evaluation_root,
        iterations=10_000,
        seed=20260801,
    )
    independent_path = output / "independent_recomputation.json"
    independently_recompute_suite(
        features_path=Path(candidate_artifact).expanduser().resolve(),
        corrected_labels_path=labels["corrected_labels"],
        legacy_labels_path=labels["legacy_labels"],
        methods=methods,
        expected_summary_path=evaluation_root / "summary.json",
        output_path=independent_path,
    )
    backend_record = output / "backend_manifest.json"
    atomic_write_json(
        backend_record,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "v3_independent_dual_track_backend",
            "status": "complete",
            "evaluation_scope": scope,
            "bootstrap_iterations": 10_000,
            "seed": 20260801,
            "candidate_artifact": artifact_identity(candidate_artifact),
            "labels": {
                name: artifact_identity(path) for name, path in labels.items()
            },
            "methods": {
                name: None if path is None else artifact_identity(path)
                for name, path in methods.items()
            },
            "primary_summary": artifact_identity(evaluation_root / "summary.json"),
            "independent_recomputation": artifact_identity(independent_path),
        },
    )
    results = [
        evaluation_root / "results.csv",
        evaluation_root / "pairwise_statistics.csv",
        evaluation_root / "calibration_curves.json",
        evaluation_root / "summary.json",
        independent_path,
        backend_record,
    ]
    if scope == "lockcheck":
        lockcheck_results = output / "results_lockcheck.csv"
        shutil.copyfile(evaluation_root / "results.csv", lockcheck_results)
        results.append(lockcheck_results)
    return results
