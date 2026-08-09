from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import code_fingerprint
from .experiment_config import ENSEMBLE_SEEDS, PERTURBATIONS
from .protocol import (
    assert_lockcheck_complete,
    claim_stage_once,
    complete_stage_once,
    verify_locked_manifest,
    write_locked_manifest,
)
from .schema import (
    artifact_identity,
    atomic_write_json,
    canonical_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    stable_sample_id,
)
from .test_access_guard import verify_manifest_sidecar


SCHEMA_VERSION = "3.0.0"
PRELIMINARY_MANIFEST_KIND = "v3_preliminary_experiment_manifest"
FINAL_MANIFEST_KIND = "v3_final_experiment_manifest"
EXPERIMENT_DESCRIPTOR_KIND = "v3_strict_experiment_descriptor"
EVALUATION_DESCRIPTOR_KIND = "v3_formal_evaluation_descriptor"
LOCKCHECK_EVALUATION_STAGE = "lockcheck_evaluate"
PARTITION_NAMES = frozenset(
    {"train", "calibration", "select", "lockcheck", "test"}
)

FORMAL_METHOD_ALLOWLIST = (
    "q_only",
    "v2_locked_primary",
    "v3_full_head_scalar_gate",
    "v3_fcer_native",
    "v3_fcer_rgbd",
    "v3_locked_primary",
)
REQUIRED_FORMAL_METHODS = frozenset(FORMAL_METHOD_ALLOWLIST)
REQUIRED_EVALUATORS = frozenset(
    {"corrected_scientific", "legacy_official_compatibility"}
)
PRIOR_TEST_EXPOSURE_DISCLOSURE = (
    "V3 was designed after aggregate exposure to previous V1/V2 test results. "
    "All V3 feature selection, architecture selection, thresholds and "
    "hyperparameters were nevertheless fixed using development, calibration "
    "and validation partitions before one immutable V3 formal-test execution."
)

InferenceCallback = Callable[..., Sequence[str | Path] | str | Path]
IndependentEvaluatorCallback = Callable[..., Sequence[str | Path] | str | Path]


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, field: str
) -> None:
    observed = set(value)
    if observed != set(expected):
        raise ValueError(
            f"{field} fields differ: missing={sorted(set(expected)-observed)} "
            f"extra={sorted(observed-set(expected))}"
        )


def _identity_shape(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an artifact identity")
    _require_exact_keys(value, {"path", "sha256", "size_bytes"}, field=field)
    path = str(Path(str(value["path"])).expanduser().resolve())
    digest = str(value["sha256"])
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field}.sha256 must be a lowercase SHA-256 digest")
    size = int(value["size_bytes"])
    if size < 0:
        raise ValueError(f"{field}.size_bytes must be non-negative")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _verified_descriptor_identity(value: Any, *, field: str) -> dict[str, Any]:
    expected = _identity_shape(value, field=field)
    observed = _identity(expected["path"], field=field)
    if observed != expected:
        raise ValueError(f"{field} changed")
    return observed


def _identity_map_shape(value: Any, *, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty identity mapping")
    result: dict[str, dict[str, Any]] = {}
    for raw_name in sorted(value):
        name = str(raw_name).strip()
        if not name or name in result:
            raise ValueError(f"{field} contains an invalid name")
        result[name] = _identity_shape(value[raw_name], field=f"{field}.{name}")
    return result


def _identity_map_equal(
    described: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
) -> None:
    if set(described) != set(observed):
        raise ValueError(f"experiment descriptor {field} names differ from CLI artifacts")
    for name in described:
        if dict(described[name]) != dict(observed[name]):
            raise ValueError(f"experiment descriptor {field}.{name} identity differs")


def validate_evaluation_descriptor(
    path: str | Path,
    *,
    formal_methods: Sequence[str] | None = None,
    evaluator_callback_identity: str | None = None,
) -> dict[str, Any]:
    """Validate label identities syntactically without opening label files."""
    source = Path(path).expanduser().resolve()
    value = _json_object(source, description="evaluation descriptor")
    expected_fields = {
        "schema_version",
        "kind",
        "status",
        "cohorts",
        "formal_methods",
        "evaluator_callback",
        "bootstrap_iterations",
        "seed",
    }
    _require_exact_keys(value, expected_fields, field="evaluation_descriptor")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != EVALUATION_DESCRIPTOR_KIND
        or value["status"] != "frozen_before_evaluation"
    ):
        raise ValueError("evaluation descriptor header is invalid")
    methods = _formal_methods(
        value["formal_methods"], primary_method="v3_locked_primary"
    )
    if formal_methods is not None and methods != list(formal_methods):
        raise ValueError("evaluation descriptor formal methods differ")
    callback = str(value["evaluator_callback"])
    if not callback.startswith("failure_analysis.reranking_v3.") or ":" not in callback:
        raise ValueError("evaluation descriptor evaluator callback is not repository-owned")
    if evaluator_callback_identity is not None and callback != evaluator_callback_identity:
        raise ValueError("evaluation descriptor evaluator callback differs")
    if int(value["bootstrap_iterations"]) != 10_000 or int(value["seed"]) != 20260801:
        raise ValueError("evaluation descriptor statistics contract differs")
    cohorts = value["cohorts"]
    if not isinstance(cohorts, Mapping):
        raise ValueError("evaluation descriptor cohorts must be an object")
    _require_exact_keys(cohorts, {"lockcheck", "test"}, field="evaluation_descriptor.cohorts")
    normalized: dict[str, dict[str, Any]] = {}
    for scope in ("lockcheck", "test"):
        cohort = cohorts[scope]
        if not isinstance(cohort, Mapping):
            raise ValueError(f"evaluation descriptor {scope} cohort must be an object")
        _require_exact_keys(
            cohort,
            {"candidate", "corrected_labels", "legacy_labels"},
            field=f"evaluation_descriptor.cohorts.{scope}",
        )
        # Candidate artifacts are label-free and can be verified now.  Label
        # identities are intentionally shape-checked only until evaluation.
        normalized[scope] = {
            "candidate": _verified_descriptor_identity(
                cohort["candidate"], field=f"evaluation_descriptor.{scope}.candidate"
            ),
            "corrected_labels": _identity_shape(
                cohort["corrected_labels"],
                field=f"evaluation_descriptor.{scope}.corrected_labels",
            ),
            "legacy_labels": _identity_shape(
                cohort["legacy_labels"],
                field=f"evaluation_descriptor.{scope}.legacy_labels",
            ),
        }
    return {**value, "formal_methods": methods, "cohorts": normalized}


def validate_experiment_descriptor(
    path: str | Path,
    *,
    formal_methods: Sequence[str],
    config: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
    normalizers: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
    split: Mapping[str, Any],
    candidate: Mapping[str, Any],
    evaluators: Mapping[str, Mapping[str, Any]],
    v2: Mapping[str, Mapping[str, Any]],
    evaluation_descriptor: Mapping[str, Any],
    inference_callback_identity: str,
    evaluator_callback_identity: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Strictly validate the complete frozen experiment contract before lock."""
    source = Path(path).expanduser().resolve()
    value = _json_object(source, description="experiment descriptor")
    expected_fields = {
        "schema_version", "kind", "status", "selected_feature_groups",
        "excluded_feature_groups", "architecture", "partition_manifests", "oof",
        "seeds", "feature_schema", "normalizers", "checkpoints", "alpha",
        "gate_policy", "perturbations", "inference_input_allowlist",
        "formal_methods", "callbacks", "git_diff_sha256",
        "prior_test_exposure_disclosure", "artifact_bindings",
        "evaluation_descriptor",
    }
    _require_exact_keys(value, expected_fields, field="experiment_descriptor")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != EXPERIMENT_DESCRIPTOR_KIND
        or value["status"] != "frozen_before_lockcheck"
    ):
        raise ValueError("experiment descriptor header is invalid")
    selected = value["selected_feature_groups"]
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(item, str) or not item.strip() for item in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError("selected_feature_groups must be unique non-empty strings")
    excluded = value["excluded_feature_groups"]
    if not isinstance(excluded, list):
        raise ValueError("excluded_feature_groups must be a list")
    excluded_names: list[str] = []
    for index, item in enumerate(excluded):
        if not isinstance(item, Mapping):
            raise ValueError("excluded feature group entry must be an object")
        _require_exact_keys(item, {"group", "reason"}, field=f"excluded_feature_groups[{index}]")
        group, reason = str(item["group"]).strip(), str(item["reason"]).strip()
        if not group or not reason:
            raise ValueError("excluded feature groups require a group and reason")
        excluded_names.append(group)
    if len(excluded_names) != len(set(excluded_names)) or set(selected) & set(excluded_names):
        raise ValueError("selected and excluded feature groups overlap or repeat")
    if not isinstance(value["architecture"], Mapping) or not value["architecture"]:
        raise ValueError("architecture must be a non-empty object")
    partitions = value["partition_manifests"]
    if not isinstance(partitions, Mapping):
        raise ValueError("partition_manifests must be an object")
    _require_exact_keys(partitions, PARTITION_NAMES, field="partition_manifests")
    dependencies = {
        f"partition_{name}": _verified_descriptor_identity(
            partitions[name], field=f"partition_manifests.{name}"
        )
        for name in sorted(PARTITION_NAMES)
    }
    oof = value["oof"]
    if not isinstance(oof, Mapping):
        raise ValueError("oof must be an object")
    _require_exact_keys(oof, {"group_key", "fold_count", "folds", "manifest"}, field="oof")
    if str(oof["group_key"]) != "sequence_id" or int(oof["fold_count"]) != 3:
        raise ValueError("OOF must use three sequence-grouped folds")
    if list(oof["folds"]) != [0, 1, 2]:
        raise ValueError("OOF fold identities must be [0,1,2]")
    dependencies["oof_manifest"] = _verified_descriptor_identity(
        oof["manifest"], field="oof.manifest"
    )
    if list(value["seeds"]) != list(ENSEMBLE_SEEDS):
        raise ValueError("experiment descriptor ensemble seeds differ")
    dependencies["feature_schema"] = _verified_descriptor_identity(
        value["feature_schema"], field="feature_schema"
    )
    described_normalizers = _identity_map_shape(value["normalizers"], field="normalizers")
    described_checkpoints = _identity_map_shape(value["checkpoints"], field="checkpoints")
    _identity_map_equal(described_normalizers, normalizers, field="normalizers")
    _identity_map_equal(described_checkpoints, checkpoints, field="checkpoints")
    alpha = float(value["alpha"])
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0,1]")
    if not isinstance(value["gate_policy"], Mapping) or not value["gate_policy"]:
        raise ValueError("gate_policy must be a non-empty object")
    if "policy" not in gate:
        raise ValueError("gate artifacts must include a policy")
    policy_source = _json_object(gate["policy"]["path"], description="locked gate policy")
    if isinstance(policy_source.get("selected_policy"), Mapping):
        policy_source = dict(policy_source["selected_policy"])
    elif isinstance(policy_source.get("policy"), Mapping):
        policy_source = dict(policy_source["policy"])
    if canonical_json(dict(value["gate_policy"])) != canonical_json(policy_source):
        raise ValueError("experiment descriptor gate policy differs from policy artifact")
    if canonical_json(value["perturbations"]) != canonical_json(list(PERTURBATIONS)):
        raise ValueError("experiment descriptor perturbations differ from implementation")
    allowlist = value["inference_input_allowlist"]
    if not isinstance(allowlist, Mapping):
        raise ValueError("inference_input_allowlist must be an object")
    _require_exact_keys(allowlist, {"lockcheck", "test"}, field="inference_input_allowlist")
    for scope in ("lockcheck", "test"):
        names = allowlist[scope]
        if (
            not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(f"{scope} inference input allowlist is invalid")
    if list(value["formal_methods"]) != list(formal_methods):
        raise ValueError("experiment descriptor formal methods differ")
    callbacks = value["callbacks"]
    if not isinstance(callbacks, Mapping):
        raise ValueError("callbacks must be an object")
    _require_exact_keys(callbacks, {"inference", "evaluator"}, field="callbacks")
    if callbacks != {
        "inference": inference_callback_identity,
        "evaluator": evaluator_callback_identity,
    }:
        raise ValueError("experiment descriptor callback identities differ")
    for callback in callbacks.values():
        if not str(callback).startswith("failure_analysis.reranking_v3.") or ":" not in str(callback):
            raise ValueError("experiment callbacks must be repository-owned")
    git_diff = str(value["git_diff_sha256"])
    if len(git_diff) != 64 or any(character not in "0123456789abcdef" for character in git_diff):
        raise ValueError("git_diff_sha256 must be a lowercase SHA-256 digest")
    if value["prior_test_exposure_disclosure"] != PRIOR_TEST_EXPOSURE_DISCLOSURE:
        raise ValueError("experiment descriptor disclosure differs")
    described_eval = _identity_shape(value["evaluation_descriptor"], field="evaluation_descriptor")
    if described_eval != dict(evaluation_descriptor):
        raise ValueError("experiment descriptor evaluation descriptor identity differs")
    bindings = value["artifact_bindings"]
    if not isinstance(bindings, Mapping):
        raise ValueError("artifact_bindings must be an object")
    _require_exact_keys(
        bindings,
        {"config", "contract", "split", "candidate", "gate", "evaluators", "v2"},
        field="artifact_bindings",
    )
    singles = {"config": config, "contract": contract, "split": split, "candidate": candidate}
    for name, observed in singles.items():
        if _identity_shape(bindings[name], field=f"artifact_bindings.{name}") != dict(observed):
            raise ValueError(f"experiment descriptor {name} artifact differs")
    for name, observed in (("gate", gate), ("evaluators", evaluators), ("v2", v2)):
        described = _identity_map_shape(bindings[name], field=f"artifact_bindings.{name}")
        _identity_map_equal(described, observed, field=name)
    return value, dependencies


def _json_object(path: str | Path, *, description: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _identity(path: str | Path, *, field: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{field} is not a regular file: {source}")
    return artifact_identity(source)


def _identity_map(
    values: Mapping[str, str | Path], *, field: str, require_nonempty: bool = True
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field} must be a mapping")
    if require_nonempty and not values:
        raise ValueError(f"{field} must not be empty")
    result: dict[str, dict[str, Any]] = {}
    for raw_name in sorted(values):
        name = str(raw_name).strip()
        if not name or name in result:
            raise ValueError(f"{field} contains an invalid or duplicate name")
        result[name] = _identity(values[raw_name], field=f"{field}.{name}")
    return result


def _deduplicate_identities(
    identities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, identity in enumerate(identities):
        observed = _identity(identity["path"], field=f"locked_artifacts[{index}]")
        if observed["sha256"] != identity.get("sha256"):
            raise ValueError(f"locked artifact identity changed: {observed['path']}")
        existing = result.get(observed["path"])
        if existing is not None and existing["sha256"] != observed["sha256"]:
            raise ValueError(f"conflicting locked artifact hashes: {observed['path']}")
        result[observed["path"]] = observed
    return [result[path] for path in sorted(result)]


def _flatten_artifact_groups(groups: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for item in groups.values():
        if isinstance(item, Mapping) and {"path", "sha256"} <= set(item):
            values.append(dict(item))
        elif isinstance(item, Mapping):
            values.extend(dict(identity) for identity in item.values())
        else:
            raise ValueError("artifact group has an invalid shape")
    return _deduplicate_identities(values)


def _formal_methods(
    methods: Sequence[str], *, primary_method: str
) -> list[str]:
    if isinstance(methods, (str, bytes)):
        raise TypeError("formal_methods must be a sequence of method names")
    result = [str(value).strip() for value in methods]
    if not result or any(not value for value in result):
        raise ValueError("formal_methods must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("formal_methods must be unique")
    unknown = set(result) - set(FORMAL_METHOD_ALLOWLIST)
    if unknown:
        raise ValueError(f"formal_methods contains methods outside the allowlist: {sorted(unknown)}")
    missing = REQUIRED_FORMAL_METHODS - set(result)
    if missing:
        raise ValueError(f"formal_methods is missing required methods: {sorted(missing)}")
    if primary_method != "v3_locked_primary" or primary_method not in result:
        raise ValueError("primary_method must be v3_locked_primary and included in formal_methods")
    return result


def _locked_artifact_index(manifest: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    identities = manifest.get("locked_artifacts")
    if not isinstance(identities, list):
        raise ValueError("locked manifest has no artifact list")
    for identity in identities:
        path = str(Path(identity["path"]).expanduser().resolve())
        sha = str(identity["sha256"])
        if path in result and result[path] != sha:
            raise ValueError(f"locked manifest has conflicting hashes: {path}")
        result[path] = sha
    return result


def _assert_locked_inputs(
    paths: Mapping[str, str | Path], manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    locked = _locked_artifact_index(manifest)
    identities = _identity_map(paths, field="input_artifacts")
    for name, identity in identities.items():
        expected = locked.get(identity["path"])
        if expected != identity["sha256"]:
            raise PermissionError(f"input artifact is not locked by the manifest: {name}")
    return identities


def _assert_label_free_inputs(paths: Mapping[str, str | Path]) -> None:
    forbidden = (
        "label",
        "ground_truth",
        "matched_gt",
        "positive_label",
        "oracle",
        "evaluation_result",
    )
    for raw_name, raw_path in paths.items():
        values = (str(raw_name).lower(), Path(raw_path).name.lower())
        marker = next((value for value in forbidden if any(value in item for item in values)), None)
        if marker is not None:
            raise PermissionError(
                f"label-free inference input {raw_name!r} contains forbidden marker {marker!r}"
            )


def _callback_paths(
    value: Sequence[str | Path] | str | Path,
    *,
    field: str,
    required_root: Path,
    forbidden_paths: set[Path] | None = None,
) -> list[Path]:
    raw = [value] if isinstance(value, (str, Path)) else list(value)
    if not raw:
        raise ValueError(f"{field} callback returned no artifacts")
    paths = [Path(item).expanduser().resolve() for item in raw]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field} callback returned duplicate artifacts")
    denied = forbidden_paths or set()
    for path in paths:
        if not path.is_relative_to(required_root):
            raise ValueError(
                f"{field} callback output is outside the declared output directory: {path}"
            )
        if path in denied:
            raise ValueError(f"{field} callback reused an input artifact as output: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"{field} callback output is missing: {path}")
    return paths


def _record_sample_id(record: Mapping[str, Any]) -> str:
    if record.get("stable_sample_id") is not None:
        return str(record["stable_sample_id"])
    if "sample_id" not in record:
        raise ValueError("record is missing sample_id")
    if record.get("split") is not None:
        try:
            return stable_sample_id(str(record["split"]), record["sample_id"])
        except (TypeError, ValueError):
            pass
    return str(record["sample_id"])


def _candidate_catalog(path: str | Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for record in read_jsonl(path):
        sample_id = _record_sample_id(record)
        if sample_id in result:
            raise ValueError(f"duplicate candidate sample: {sample_id}")
        if "candidate_ids" in record:
            candidate_ids = tuple(map(str, record["candidate_ids"]))
        else:
            candidates = record.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"candidate record has no candidate IDs: {sample_id}")
            candidate_ids = tuple(str(value["candidate_id"]) for value in candidates)
        if len(candidate_ids) != 5 or len(set(candidate_ids)) != 5:
            raise ValueError(f"candidate pool is not the frozen five: {sample_id}")
        result[sample_id] = candidate_ids
    if not result:
        raise ValueError("candidate artifact is empty")
    return result


def _verify_ranking_identity(
    path: str | Path,
    *,
    method: str,
    candidates: Mapping[str, tuple[str, ...]],
) -> None:
    observed: set[str] = set()
    for record in read_jsonl(path):
        sample_id = _record_sample_id(record)
        if sample_id in observed:
            raise ValueError(f"duplicate ranking sample for {method}: {sample_id}")
        expected = candidates.get(sample_id)
        if expected is None:
            raise ValueError(f"ranking has an unknown sample for {method}: {sample_id}")
        if str(record.get("method", "")) != method:
            raise ValueError(f"ranking method identity mismatch: {method}/{sample_id}")
        order = tuple(map(str, record.get("candidate_order", ())))
        if len(order) != len(expected) or set(order) != set(expected):
            raise ValueError(f"ranking changed candidate identity: {method}/{sample_id}")
        observed.add(sample_id)
    if observed != set(candidates):
        raise ValueError(f"ranking cohort differs from candidate cohort: {method}")


def verify_formal_ranking_outputs(
    paths: Sequence[str | Path],
    *,
    locked_manifest: Mapping[str, Any],
    scope: str | None = None,
) -> dict[str, Path]:
    """Require one candidate-preserving ranking file per non-Q formal method.

    This check is intended for inference backends before a formal test
    completion marker is committed.  In particular, a byte-for-byte V2 source
    ranking without an explicit ``method`` field is not a valid formal output;
    the backend must materialize a distinct, method-tagged ranking file.
    """
    scope_name = _formal_scope(scope, manifest=locked_manifest)
    candidate_identity = verify_scope_candidate_identity(
        locked_manifest, scope=scope_name
    )
    raw_paths = [Path(value).expanduser().resolve() for value in paths]
    if len(raw_paths) != len(set(raw_paths)):
        raise ValueError("formal methods must not share a ranking artifact")
    expected = set(map(str, locked_manifest.get("formal_methods", ()))) - {"q_only"}
    candidates = _candidate_catalog(candidate_identity["path"])
    result: dict[str, Path] = {}
    for path in raw_paths:
        methods = {
            str(record.get("method", "")).strip() for record in read_jsonl(path)
        }
        if len(methods) != 1 or "" in methods:
            raise ValueError(f"formal ranking has no unique method identity: {path}")
        method = next(iter(methods))
        if method == "q_only":
            raise ValueError("q_only must use frozen candidate order without a ranking file")
        if method in result:
            raise ValueError(f"formal method has multiple ranking artifacts: {method}")
        _verify_ranking_identity(path, method=method, candidates=candidates)
        result[method] = path
    if set(result) != expected:
        raise ValueError(
            "formal ranking outputs differ from locked methods: "
            f"missing={sorted(expected-set(result))} extra={sorted(set(result)-expected)}"
        )
    return result


def _formal_scope(
    scope: str | None, *, manifest: Mapping[str, Any]
) -> str:
    """Resolve an explicit scope, or the lifecycle kind used by the CLI wrapper."""
    if scope is None:
        scope_name = {
            PRELIMINARY_MANIFEST_KIND: "lockcheck",
            FINAL_MANIFEST_KIND: "test",
        }.get(str(manifest.get("kind", "")))
        if scope_name is None:
            raise ValueError(
                "scope is required when the manifest kind does not identify a formal stage"
            )
        return scope_name
    scope_name = str(scope).strip().lower()
    if scope_name not in {"lockcheck", "test"}:
        raise ValueError("scope must be lockcheck or test")
    return scope_name


def _verify_stage_completion(
    path: str | Path,
    *,
    stage: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    completion_path = Path(path).expanduser().resolve()
    verify_manifest_sidecar(completion_path)
    completion = _json_object(completion_path, description=f"{stage} completion")
    expected_fields = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{stage}_run_complete",
        "status": "complete",
        "stage": stage,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": manifest["content_sha256"],
    }
    for field, expected in expected_fields.items():
        if completion.get(field) != expected:
            raise ValueError(f"{stage} completion {field} mismatch")
    expected_content = str(completion.get("content_sha256", ""))
    unsigned = {key: value for key, value in completion.items() if key != "content_sha256"}
    if sha256_bytes(canonical_json(unsigned).encode()) != expected_content:
        raise ValueError(f"{stage} completion content hash mismatch")
    claim_path = Path(str(completion.get("claim_path", ""))).expanduser().resolve()
    verify_manifest_sidecar(claim_path)
    if sha256_file(claim_path) != completion.get("claim_sha256"):
        raise ValueError(f"{stage} completion claim changed")
    claim = _json_object(claim_path, description=f"{stage} claim")
    if claim.get("claim_token") != completion.get("claim_token"):
        raise ValueError(f"{stage} completion claim token mismatch")
    identities = completion.get("result_artifacts")
    if not isinstance(identities, list) or not identities:
        raise ValueError(f"{stage} completion has no result artifacts")
    for index, identity in enumerate(identities):
        observed = _identity(identity["path"], field=f"{stage}.result_artifacts[{index}]")
        if observed["sha256"] != identity.get("sha256"):
            raise ValueError(f"{stage} result artifact changed: {observed['path']}")
    return completion


def lock_preliminary(
    *,
    output_path: str | Path,
    run_id: str,
    primary_method: str,
    formal_methods: Sequence[str],
    config_path: str | Path,
    checkpoint_paths: Mapping[str, str | Path],
    normalizer_paths: Mapping[str, str | Path],
    gate_paths: Mapping[str, str | Path],
    contract_path: str | Path,
    split_path: str | Path,
    candidate_path: str | Path,
    evaluator_paths: Mapping[str, str | Path],
    v2_paths: Mapping[str, str | Path],
    experiment_descriptor_path: str | Path,
    evaluation_descriptor_path: str | Path,
    inference_callback_identity: str,
    evaluator_callback_identity: str,
    exposure_disclosure: str = PRIOR_TEST_EXPOSURE_DISCLOSURE,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Lock every selection-time input before the one-time lockcheck run."""
    identifier = str(run_id).strip()
    if not identifier:
        raise ValueError("run_id must be non-empty")
    methods = _formal_methods(formal_methods, primary_method=primary_method)
    if exposure_disclosure != PRIOR_TEST_EXPOSURE_DISCLOSURE:
        raise ValueError("the required prior-test-exposure disclosure must be exact")
    evaluators = _identity_map(evaluator_paths, field="evaluator_paths")
    if set(evaluators) != REQUIRED_EVALUATORS:
        raise ValueError("both corrected and legacy evaluators must be locked")
    config_identity = _identity(config_path, field="config_path")
    checkpoints = _identity_map(checkpoint_paths, field="checkpoint_paths")
    normalizers = _identity_map(normalizer_paths, field="normalizer_paths")
    gate = _identity_map(gate_paths, field="gate_paths")
    contract_identity = _identity(contract_path, field="contract_path")
    split_identity = _identity(split_path, field="split_path")
    candidate_identity = _identity(candidate_path, field="candidate_path")
    v2 = _identity_map(v2_paths, field="v2_paths")
    evaluation_descriptor_identity = _identity(
        evaluation_descriptor_path, field="evaluation_descriptor_path"
    )
    evaluation_descriptor = validate_evaluation_descriptor(
        evaluation_descriptor_path,
        formal_methods=methods,
        evaluator_callback_identity=evaluator_callback_identity,
    )
    if evaluation_descriptor["cohorts"]["lockcheck"]["candidate"] != candidate_identity:
        raise ValueError("lockcheck evaluation candidate differs from preliminary candidate")
    descriptor, descriptor_dependencies = validate_experiment_descriptor(
        experiment_descriptor_path,
        formal_methods=methods,
        config=config_identity,
        checkpoints=checkpoints,
        normalizers=normalizers,
        gate=gate,
        contract=contract_identity,
        split=split_identity,
        candidate=candidate_identity,
        evaluators=evaluators,
        v2=v2,
        evaluation_descriptor=evaluation_descriptor_identity,
        inference_callback_identity=inference_callback_identity,
        evaluator_callback_identity=evaluator_callback_identity,
    )
    experiment_descriptor_identity = _identity(
        experiment_descriptor_path, field="experiment_descriptor_path"
    )
    artifacts: dict[str, Any] = {
        "config": config_identity,
        "checkpoints": checkpoints,
        "normalizers": normalizers,
        "gate": gate,
        "contract": contract_identity,
        "split": split_identity,
        "candidate": candidate_identity,
        "evaluators": evaluators,
        "v2": v2,
        "experiment_descriptor": experiment_descriptor_identity,
        "evaluation_descriptor": evaluation_descriptor_identity,
        "evaluation_cohort_candidates": {
            scope: dict(evaluation_descriptor["cohorts"][scope]["candidate"])
            for scope in ("lockcheck", "test")
        },
        "descriptor_dependencies": descriptor_dependencies,
    }
    payload = {
        "run_id": identifier,
        "primary_method": primary_method,
        "primary_anchor": "v2_locked_primary",
        "formal_methods": methods,
        "prior_test_exposure_disclosure": exposure_disclosure,
        "inference_callback": inference_callback_identity,
        "evaluator_callback": evaluator_callback_identity,
        "inference_input_allowlist": descriptor["inference_input_allowlist"],
        "artifacts": artifacts,
        "locked_artifacts": _flatten_artifact_groups(artifacts),
        "code_fingerprint": code_fingerprint(),
        "metadata": dict(metadata or {}),
    }
    return write_locked_manifest(output_path, payload, kind=PRELIMINARY_MANIFEST_KIND)


def run_locked_inference_once(
    *,
    scope: str,
    manifest_path: str | Path,
    stage_dir: str | Path,
    input_artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
    inference_callback: InferenceCallback,
    callback_identity: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Claim and complete exactly one label-free lockcheck or formal-test inference."""
    scope_name = str(scope).strip().lower()
    expected_kind = {
        "lockcheck": PRELIMINARY_MANIFEST_KIND,
        "test": FINAL_MANIFEST_KIND,
    }.get(scope_name)
    if expected_kind is None:
        raise ValueError("scope must be lockcheck or test")
    manifest_source = Path(manifest_path).expanduser().resolve()
    locked = verify_locked_manifest(manifest_source, expected_kind=expected_kind)
    if str(callback_identity) != locked.get("inference_callback"):
        raise PermissionError("inference callback differs from the locked experiment descriptor")
    claim_path = claim_stage_once(
        stage_dir,
        stage=scope_name,
        manifest_path=manifest_source,
        resume=resume,
    )
    _assert_label_free_inputs(input_artifacts)
    allowlist = locked.get("inference_input_allowlist", {}).get(scope_name)
    if not isinstance(allowlist, list) or set(input_artifacts) != set(map(str, allowlist)):
        raise PermissionError(
            f"{scope_name} inference inputs differ from the locked allowlist"
        )
    inputs = _assert_locked_inputs(input_artifacts, locked)
    expected_candidate = verify_scope_candidate_identity(
        locked, scope=scope_name
    )
    candidate_inputs = [
        name for name, identity in inputs.items() if identity == expected_candidate
    ]
    if len(candidate_inputs) != 1:
        raise PermissionError(
            f"{scope_name} inference must receive exactly one candidate artifact "
            "matching the frozen evaluation descriptor"
        )
    input_paths = {name: Path(identity["path"]) for name, identity in inputs.items()}
    output_root = Path(output_dir).expanduser().resolve()
    callback_value = inference_callback(
        scope=scope_name,
        input_artifacts=input_paths,
        output_dir=output_root,
        locked_manifest=locked,
    )
    result_paths = _callback_paths(
        callback_value,
        field=f"{scope_name} inference",
        required_root=output_root,
        forbidden_paths={Path(value["path"]) for value in inputs.values()} | {manifest_source},
    )
    # Re-hash inputs after inference to close the verification/use interval.
    for name, before in inputs.items():
        after = _identity(before["path"], field=f"input_artifacts.{name}")
        if after["sha256"] != before["sha256"]:
            raise ValueError(f"inference modified a locked input artifact: {name}")
    result_identities = [artifact_identity(path) for path in result_paths]
    stage_root = Path(stage_dir).expanduser().resolve()
    run_record_path = stage_root / f"{scope_name.upper()}_LABEL_FREE_INFERENCE.json"
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{scope_name}_label_free_inference",
        "status": "complete",
        "scope": scope_name,
        "labels_read": False,
        "manifest": artifact_identity(manifest_source),
        "claim": artifact_identity(claim_path),
        "input_artifacts": inputs,
        "result_artifacts": result_identities,
        "callback": getattr(inference_callback, "__qualname__", type(inference_callback).__name__),
    }
    atomic_write_json(run_record_path, run_record)
    completion_path = complete_stage_once(
        stage_root,
        stage=scope_name,
        result_artifacts=[*result_identities, artifact_identity(run_record_path)],
    )
    if scope_name == "lockcheck":
        assert_lockcheck_complete(completion_path)
    else:
        _verify_stage_completion(
            completion_path,
            stage="test",
            manifest_path=manifest_source,
            manifest=locked,
        )
    return {
        "scope": scope_name,
        "claim_path": str(claim_path.resolve()),
        "completion_path": str(completion_path.resolve()),
        "run_record": artifact_identity(run_record_path),
        "result_artifacts": result_identities,
        "labels_read": False,
    }


def _evaluation_descriptor_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = manifest.get("artifacts", {}).get("evaluation_descriptor")
    if not isinstance(identity, Mapping):
        raise ValueError("locked manifest has no evaluation descriptor")
    observed = _verified_descriptor_identity(identity, field="evaluation_descriptor")
    value = validate_evaluation_descriptor(
        observed["path"],
        formal_methods=manifest["formal_methods"],
        evaluator_callback_identity=str(manifest["evaluator_callback"]),
    )
    return value, observed


def verify_scope_candidate_identity(
    manifest: Mapping[str, Any],
    *,
    scope: str,
    candidate_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the scope-specific candidate against the descriptor and flat lock.

    The preliminary manifest's historical ``artifacts.candidate`` binding is
    the lockcheck cohort.  It is therefore not a valid source of truth for a
    formal test whose candidate export is different.  The frozen evaluation
    descriptor is authoritative for both cohorts, while ``locked_artifacts``
    independently binds the selected identity by its resolved path and hash.
    """
    scope_name = _formal_scope(scope, manifest=manifest)
    descriptor, _ = _evaluation_descriptor_from_manifest(manifest)
    expected = dict(descriptor["cohorts"][scope_name]["candidate"])
    locked = _locked_artifact_index(manifest)
    if locked.get(expected["path"]) != expected["sha256"]:
        raise PermissionError(
            f"{scope_name} candidate is not bound by the flat locked_artifacts hash"
        )
    if candidate_artifact is not None:
        observed = _identity(
            candidate_artifact, field=f"{scope_name}_candidate_artifact"
        )
        if observed != expected:
            raise PermissionError(
                f"{scope_name} candidate differs from the frozen evaluation descriptor"
            )
    return expected


def _verified_evaluation_inputs(
    paths: Mapping[str, str | Path],
    *,
    descriptor: Mapping[str, Any],
    scope: str,
) -> dict[str, dict[str, Any]]:
    expected_names = {"corrected_labels", "legacy_labels"}
    if set(paths) != expected_names:
        raise ValueError(
            "evaluation inputs must be exactly corrected_labels and legacy_labels"
        )
    identities = _identity_map(paths, field=f"{scope}_evaluation_artifacts")
    cohort = descriptor["cohorts"][scope]
    for name in sorted(expected_names):
        if identities[name] != cohort[name]:
            raise PermissionError(f"{scope} {name} differs from evaluation descriptor")
    return identities


def _ranking_identities_from_completion(
    *,
    manifest: Mapping[str, Any],
    completion: Mapping[str, Any],
    candidate_identity: Mapping[str, Any],
    method_rankings: Mapping[str, str | Path | None],
) -> dict[str, dict[str, Any] | None]:
    expected_methods = list(map(str, manifest["formal_methods"]))
    if set(method_rankings) != set(expected_methods):
        raise ValueError("method rankings do not exactly match the locked formal methods")
    if method_rankings.get("q_only") is not None:
        raise ValueError("q_only must use frozen candidate order rather than a ranking file")
    completed_outputs = {
        str(Path(identity["path"]).resolve()): str(identity["sha256"])
        for identity in completion["result_artifacts"]
    }
    candidates = _candidate_catalog(candidate_identity["path"])
    result: dict[str, dict[str, Any] | None] = {}
    seen_paths: set[str] = set()
    for method in expected_methods:
        ranking_path = method_rankings[method]
        if ranking_path is None:
            result[method] = None
            continue
        identity = _identity(ranking_path, field=f"method_rankings.{method}")
        if identity["path"] in seen_paths:
            raise ValueError("formal methods must not share a ranking artifact")
        seen_paths.add(identity["path"])
        if completed_outputs.get(identity["path"]) != identity["sha256"]:
            raise PermissionError(
                f"ranking was not produced by the completed inference run: {method}"
            )
        _verify_ranking_identity(identity["path"], method=method, candidates=candidates)
        result[method] = identity
    return result


def _verify_dual_track_summary(
    path: Path,
    *,
    methods: Sequence[str],
    candidate_count: int,
) -> dict[str, Any]:
    summary = _json_object(path, description="dual-track evaluation summary")
    if (
        summary.get("kind") != "v3_dual_track_evaluation"
        or summary.get("status") != "complete"
        or summary.get("oracle_unchanged") is not True
        or int(summary.get("sample_count", -1)) != candidate_count
        or list(map(str, summary.get("methods", ()))) != list(methods)
    ):
        raise ValueError("dual-track evaluation summary contract differs")
    tracks = summary.get("tracks")
    if not isinstance(tracks, Mapping) or set(tracks) != REQUIRED_EVALUATORS:
        raise ValueError("dual-track evaluation tracks differ")
    for track, track_results in tracks.items():
        if not isinstance(track_results, Mapping) or set(track_results) != set(methods):
            raise ValueError(f"dual-track method cohort differs: {track}")
        oracle_counts: set[int] = set()
        for method in methods:
            record = track_results[method]
            if int(record.get("sample_count", -1)) != candidate_count:
                raise ValueError(f"dual-track sample cohort differs: {track}/{method}")
            oracle_counts.add(int(record.get("oracle_correct", -1)))
        if len(oracle_counts) != 1:
            raise ValueError(f"Oracle@5 identity differs across methods: {track}")
    return summary


def evaluate_lockcheck_once(
    *,
    preliminary_manifest_path: str | Path,
    lockcheck_inference_completion_path: str | Path,
    stage_dir: str | Path,
    candidate_artifact: str | Path,
    method_rankings: Mapping[str, str | Path | None],
    evaluation_artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
    evaluator_callback: IndependentEvaluatorCallback,
    callback_identity: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Run exactly one dual-track lockcheck evaluation after label-free inference."""
    manifest_path = Path(preliminary_manifest_path).expanduser().resolve()
    locked = verify_locked_manifest(
        manifest_path, expected_kind=PRELIMINARY_MANIFEST_KIND
    )
    if str(callback_identity) != locked.get("evaluator_callback"):
        raise PermissionError("lockcheck evaluator callback differs from descriptor")
    inference_completion = _verify_stage_completion(
        lockcheck_inference_completion_path,
        stage="lockcheck",
        manifest_path=manifest_path,
        manifest=locked,
    )
    descriptor, descriptor_identity = _evaluation_descriptor_from_manifest(locked)
    candidate_identity = _identity(candidate_artifact, field="candidate_artifact")
    if candidate_identity != descriptor["cohorts"]["lockcheck"]["candidate"]:
        raise PermissionError("lockcheck candidate differs from evaluation descriptor")
    rankings = _ranking_identities_from_completion(
        manifest=locked,
        completion=inference_completion,
        candidate_identity=candidate_identity,
        method_rankings=method_rankings,
    )
    claim_path = claim_stage_once(
        stage_dir,
        stage=LOCKCHECK_EVALUATION_STAGE,
        manifest_path=manifest_path,
        resume=resume,
    )
    evaluation_inputs = _verified_evaluation_inputs(
        evaluation_artifacts, descriptor=descriptor, scope="lockcheck"
    )
    callback_rankings = {
        method: None if identity is None else Path(identity["path"])
        for method, identity in rankings.items()
    }
    output_root = Path(output_dir).expanduser().resolve()
    callback_value = evaluator_callback(
        evaluation_scope="lockcheck",
        candidate_artifact=Path(candidate_identity["path"]),
        method_rankings=callback_rankings,
        evaluation_artifacts={
            name: Path(identity["path"]) for name, identity in evaluation_inputs.items()
        },
        output_dir=output_root,
        locked_manifest=locked,
    )
    result_paths = _callback_paths(
        callback_value,
        field="lockcheck evaluation",
        required_root=output_root,
        forbidden_paths={
            manifest_path,
            Path(candidate_identity["path"]),
            *(Path(identity["path"]) for identity in evaluation_inputs.values()),
            *(Path(identity["path"]) for identity in rankings.values() if identity),
        },
    )
    by_name = {path.name: path for path in result_paths}
    if "results_lockcheck.csv" not in by_name or "summary.json" not in by_name:
        raise ValueError("lockcheck evaluator must return results_lockcheck.csv and summary.json")
    _verify_dual_track_summary(
        by_name["summary.json"],
        methods=locked["formal_methods"],
        candidate_count=len(_candidate_catalog(candidate_identity["path"])),
    )
    for name, before in evaluation_inputs.items():
        if _identity(before["path"], field=f"evaluation_artifacts.{name}") != before:
            raise ValueError(f"lockcheck evaluator modified {name}")
    result_identities = [artifact_identity(path) for path in result_paths]
    stage_root = Path(stage_dir).expanduser().resolve()
    record_path = stage_root / "LOCKCHECK_EVALUATION.json"
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v3_lockcheck_dual_track_evaluation",
        "status": "complete",
        "manifest": artifact_identity(manifest_path),
        "lockcheck_inference_completion": artifact_identity(
            lockcheck_inference_completion_path
        ),
        "evaluation_descriptor": descriptor_identity,
        "candidate_artifact": candidate_identity,
        "method_rankings": rankings,
        "evaluation_artifacts": evaluation_inputs,
        "results_lockcheck": artifact_identity(by_name["results_lockcheck.csv"]),
        "summary": artifact_identity(by_name["summary.json"]),
        "candidate_identity_verified": True,
        "method_identity_verified": True,
        "oracle_identity_verified": True,
        "result_artifacts": result_identities,
        "callback_identity": callback_identity,
    }
    atomic_write_json(record_path, record)
    completion_path = complete_stage_once(
        stage_root,
        stage=LOCKCHECK_EVALUATION_STAGE,
        result_artifacts=[*result_identities, artifact_identity(record_path)],
    )
    _verify_stage_completion(
        completion_path,
        stage=LOCKCHECK_EVALUATION_STAGE,
        manifest_path=manifest_path,
        manifest=locked,
    )
    return {
        "claim_path": str(claim_path.resolve()),
        "completion_path": str(completion_path.resolve()),
        "record": artifact_identity(record_path),
        "result_artifacts": result_identities,
        "candidate_identity_verified": True,
        "method_identity_verified": True,
        "oracle_identity_verified": True,
    }


def lock_final(
    *,
    output_path: str | Path,
    preliminary_manifest_path: str | Path,
    lockcheck_completion_path: str | Path,
    lockcheck_evaluation_completion_path: str | Path,
    exact_test_command: Sequence[str],
    test_input_paths: Mapping[str, str | Path] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Promote an immutable preliminary lock only after verified lockcheck completion."""
    preliminary_path = Path(preliminary_manifest_path).expanduser().resolve()
    preliminary = verify_locked_manifest(
        preliminary_path, expected_kind=PRELIMINARY_MANIFEST_KIND
    )
    completion_path = Path(lockcheck_completion_path).expanduser().resolve()
    completion = assert_lockcheck_complete(completion_path)
    if Path(completion["manifest_path"]).resolve() != preliminary_path:
        raise ValueError("lockcheck completion is bound to a different preliminary manifest")
    evaluation_completion_path = Path(
        lockcheck_evaluation_completion_path
    ).expanduser().resolve()
    evaluation_completion = _verify_stage_completion(
        evaluation_completion_path,
        stage=LOCKCHECK_EVALUATION_STAGE,
        manifest_path=preliminary_path,
        manifest=preliminary,
    )
    evaluation_records = []
    for identity in evaluation_completion["result_artifacts"]:
        path = Path(identity["path"])
        if path.suffix != ".json":
            continue
        try:
            value = _json_object(path, description="lockcheck evaluation result")
        except (ValueError, OSError):
            continue
        if value.get("kind") == "v3_lockcheck_dual_track_evaluation":
            evaluation_records.append((path, value))
    if len(evaluation_records) != 1:
        raise ValueError("lockcheck evaluation completion has no unique evaluation record")
    evaluation_record_path, evaluation_record = evaluation_records[0]
    if (
        evaluation_record.get("status") != "complete"
        or evaluation_record.get("candidate_identity_verified") is not True
        or evaluation_record.get("method_identity_verified") is not True
        or evaluation_record.get("oracle_identity_verified") is not True
    ):
        raise ValueError("lockcheck evaluation verification flags are incomplete")
    expected_inference = artifact_identity(completion_path)
    if evaluation_record.get("lockcheck_inference_completion") != expected_inference:
        raise ValueError("lockcheck evaluation is bound to different inference completion")
    descriptor, descriptor_identity = _evaluation_descriptor_from_manifest(preliminary)
    if evaluation_record.get("evaluation_descriptor") != descriptor_identity:
        raise ValueError("lockcheck evaluation is bound to a different evaluation descriptor")
    if evaluation_record.get("callback_identity") != preliminary["evaluator_callback"]:
        raise ValueError("lockcheck evaluation callback identity differs")
    if evaluation_record.get("candidate_artifact") != descriptor["cohorts"]["lockcheck"]["candidate"]:
        raise ValueError("lockcheck evaluation candidate identity differs")
    recorded_labels = evaluation_record.get("evaluation_artifacts")
    if not isinstance(recorded_labels, Mapping) or {
        name: _identity_shape(identity, field=f"lockcheck_evaluation.{name}")
        for name, identity in recorded_labels.items()
    } != {
        name: descriptor["cohorts"]["lockcheck"][name]
        for name in ("corrected_labels", "legacy_labels")
    }:
        raise ValueError("lockcheck evaluation label identities differ from descriptor")
    result_index = {
        str(Path(identity["path"]).resolve()): dict(identity)
        for identity in evaluation_completion["result_artifacts"]
    }
    for field in ("results_lockcheck", "summary"):
        identity = evaluation_record.get(field)
        if not isinstance(identity, Mapping):
            raise ValueError(f"lockcheck evaluation has no {field} identity")
        shaped = _identity_shape(identity, field=f"lockcheck_evaluation.{field}")
        if result_index.get(shaped["path"]) != shaped:
            raise ValueError(f"lockcheck evaluation {field} is not a completed result")
    _verify_dual_track_summary(
        Path(evaluation_record["summary"]["path"]),
        methods=preliminary["formal_methods"],
        candidate_count=len(
            _candidate_catalog(descriptor["cohorts"]["lockcheck"]["candidate"]["path"])
        ),
    )
    if isinstance(exact_test_command, (str, bytes)):
        raise TypeError("exact_test_command must be an argv sequence")
    command = [str(value) for value in exact_test_command]
    if not command or any(not value for value in command):
        raise ValueError("exact_test_command must not be empty")
    test_inputs = _identity_map(
        test_input_paths or {}, field="test_input_paths", require_nonempty=False
    )
    lockcheck_results = [dict(value) for value in completion["result_artifacts"]]
    additional = [
        artifact_identity(preliminary_path),
        artifact_identity(completion_path),
        artifact_identity(completion["claim_path"]),
        *lockcheck_results,
        artifact_identity(evaluation_completion_path),
        artifact_identity(evaluation_completion["claim_path"]),
        artifact_identity(evaluation_record_path),
        *[dict(value) for value in evaluation_completion["result_artifacts"]],
        *test_inputs.values(),
    ]
    payload = {
        "run_id": preliminary["run_id"],
        "primary_method": preliminary["primary_method"],
        "primary_anchor": preliminary["primary_anchor"],
        "formal_methods": list(preliminary["formal_methods"]),
        "prior_test_exposure_disclosure": preliminary[
            "prior_test_exposure_disclosure"
        ],
        "inference_callback": preliminary["inference_callback"],
        "evaluator_callback": preliminary["evaluator_callback"],
        "inference_input_allowlist": preliminary["inference_input_allowlist"],
        "artifacts": preliminary["artifacts"],
        "preliminary_manifest": artifact_identity(preliminary_path),
        "lockcheck_completion": artifact_identity(completion_path),
        "lockcheck_result_artifacts": lockcheck_results,
        "lockcheck_evaluation_completion": artifact_identity(
            evaluation_completion_path
        ),
        "lockcheck_evaluation_record": artifact_identity(evaluation_record_path),
        "lockcheck_evaluation_result_artifacts": [
            dict(value) for value in evaluation_completion["result_artifacts"]
        ],
        "test_input_artifacts": test_inputs,
        "exact_test_command": command,
        "locked_artifacts": _deduplicate_identities(
            [*preliminary["locked_artifacts"], *additional]
        ),
        "code_fingerprint": code_fingerprint(),
        "metadata": dict(metadata or {}),
    }
    return write_locked_manifest(output_path, payload, kind=FINAL_MANIFEST_KIND)


def verify_exact_test_command(
    manifest_path: str | Path,
    executed_command: Sequence[str],
) -> dict[str, Any]:
    """Bind the one-time formal test claim to the exact locked argv.

    The comparison is deliberately byte-for-byte at the argv-string level.
    Callers must therefore lock the same interpreter path, option ordering and
    path spelling that the formal invocation will actually use.
    """
    if isinstance(executed_command, (str, bytes)):
        raise TypeError("executed_command must be an argv sequence")
    observed = [str(value) for value in executed_command]
    if not observed or any(not value for value in observed):
        raise ValueError("executed_command must not be empty")
    manifest = verify_locked_manifest(
        Path(manifest_path).expanduser().resolve(),
        expected_kind=FINAL_MANIFEST_KIND,
    )
    expected = manifest.get("exact_test_command")
    if not isinstance(expected, list) or any(
        not isinstance(value, str) or not value for value in expected
    ):
        raise ValueError("final manifest has no valid exact_test_command")
    if observed != expected:
        raise PermissionError(
            "formal test argv differs from the exact command locked in the final manifest"
        )
    return manifest


def independent_evaluate_once(
    *,
    final_manifest_path: str | Path,
    test_completion_path: str | Path,
    stage_dir: str | Path,
    candidate_artifact: str | Path,
    method_rankings: Mapping[str, str | Path | None],
    evaluation_artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
    evaluator_callback: IndependentEvaluatorCallback,
    callback_identity: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate frozen rankings, then run one independently injected evaluator."""
    manifest_path = Path(final_manifest_path).expanduser().resolve()
    locked = verify_locked_manifest(manifest_path, expected_kind=FINAL_MANIFEST_KIND)
    if str(callback_identity) != locked.get("evaluator_callback"):
        raise PermissionError("independent evaluator callback differs from descriptor")
    test_completion = _verify_stage_completion(
        test_completion_path,
        stage="test",
        manifest_path=manifest_path,
        manifest=locked,
    )
    descriptor, descriptor_identity = _evaluation_descriptor_from_manifest(locked)
    candidate_identity = _identity(candidate_artifact, field="candidate_artifact")
    locked_index = _locked_artifact_index(locked)
    if locked_index.get(candidate_identity["path"]) != candidate_identity["sha256"]:
        raise PermissionError("candidate artifact is not locked by the final manifest")
    if candidate_identity != descriptor["cohorts"]["test"]["candidate"]:
        raise PermissionError("test candidate differs from evaluation descriptor")
    ranking_identities = _ranking_identities_from_completion(
        manifest=locked,
        completion=test_completion,
        candidate_identity=candidate_identity,
        method_rankings=method_rankings,
    )
    claim_path = claim_stage_once(
        stage_dir,
        stage="independent_evaluate",
        manifest_path=manifest_path,
        resume=resume,
    )
    evaluation_inputs = _verified_evaluation_inputs(
        evaluation_artifacts, descriptor=descriptor, scope="test"
    )
    callback_rankings = {
        method: None if identity is None else Path(identity["path"])
        for method, identity in ranking_identities.items()
    }
    output_root = Path(output_dir).expanduser().resolve()
    callback_value = evaluator_callback(
        evaluation_scope="test",
        candidate_artifact=Path(candidate_identity["path"]),
        method_rankings=callback_rankings,
        evaluation_artifacts={
            name: Path(identity["path"])
            for name, identity in evaluation_inputs.items()
        },
        output_dir=output_root,
        locked_manifest=locked,
    )
    forbidden = {Path(candidate_identity["path"]), manifest_path} | {
        Path(identity["path"])
        for identity in ranking_identities.values()
        if identity is not None
    }
    result_paths = _callback_paths(
        callback_value,
        field="independent evaluation",
        required_root=output_root,
        forbidden_paths=forbidden,
    )
    for name, before in evaluation_inputs.items():
        after = _identity(before["path"], field=f"evaluation_artifacts.{name}")
        if after["sha256"] != before["sha256"]:
            raise ValueError(f"evaluator modified an evaluation input artifact: {name}")
    # The independent evaluator is intentionally injected and may read labels,
    # so re-verify every label-free input after it returns before completing.
    locked = verify_locked_manifest(manifest_path, expected_kind=FINAL_MANIFEST_KIND)
    test_completion = _verify_stage_completion(
        test_completion_path,
        stage="test",
        manifest_path=manifest_path,
        manifest=locked,
    )
    result_identities = [artifact_identity(path) for path in result_paths]
    stage_root = Path(stage_dir).expanduser().resolve()
    record_path = stage_root / "INDEPENDENT_EVALUATION.json"
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v3_independent_evaluation_run",
        "status": "complete",
        "manifest": artifact_identity(manifest_path),
        "test_completion": artifact_identity(test_completion_path),
        "evaluation_descriptor": descriptor_identity,
        "candidate_artifact": candidate_identity,
        "method_rankings": ranking_identities,
        "evaluation_artifacts": evaluation_inputs,
        "candidate_identity_verified": True,
        "method_identity_verified": True,
        "result_artifacts": result_identities,
        "callback_identity": callback_identity,
    }
    atomic_write_json(record_path, record)
    completion_path = complete_stage_once(
        stage_root,
        stage="independent_evaluate",
        result_artifacts=[*result_identities, artifact_identity(record_path)],
    )
    _verify_stage_completion(
        completion_path,
        stage="independent_evaluate",
        manifest_path=manifest_path,
        manifest=locked,
    )
    return {
        "claim_path": str(claim_path.resolve()),
        "completion_path": str(completion_path.resolve()),
        "record": artifact_identity(record_path),
        "result_artifacts": result_identities,
        "candidate_identity_verified": True,
        "method_identity_verified": True,
    }
