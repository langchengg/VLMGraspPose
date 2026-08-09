#!/usr/bin/env python3
"""Independently evaluate frozen-pool rerankers and write report data tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.evaluation import (  # noqa: E402
    EvaluationError,
    evaluate_predictions,
    predictions_from_wide,
    vlm_results_to_predictions,
    vlm_runtime_metrics,
    write_evaluation_bundle,
)
from src.grasping.reranking_v1.experiment_lock import (  # noqa: E402
    selected_vlm_contract,
    verify_lock,
)
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    validate_artifact_identity,
    validate_matching_artifact_identity,
)
from src.grasping.reranking_v1.features import (  # noqa: E402
    FORBIDDEN_GT_COLUMNS,
    join_candidate_labels,
)
from src.grasping.reranking_v1.identity import (  # noqa: E402
    jsonl_prefix_sha256,
)
from src.grasping.reranking_v1.local_vlm import (  # noqa: E402
    SESSION_AUDIT_POLICY_VERSION,
    stable_session_contract_sha256,
)
from src.grasping.reranking_v1.vlm_safe_switch import (  # noqa: E402
    apply_vlm_safe_switch_policy,
)
from src.grasping.reranking_v1.vlm_visualization import (  # noqa: E402
    canonical_recipe_sha256,
    validate_vlm_visualization_recipe,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_repeatedfilm_identity(
    value: Mapping[str, Any], *, context: str
) -> None:
    try:
        validate_artifact_identity(value, context=context)
    except ValueError as error:
        raise EvaluationError(str(error)) from error


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("rows", value.get("predictions", value))
        if not isinstance(value, list):
            raise EvaluationError(f"JSON table must contain a list: {path}")
        return pd.DataFrame(value)
    raise EvaluationError(f"unsupported table extension: {path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise EvaluationError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(value)
    return rows


def _parse_mapping(values: Sequence[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise EvaluationError(f"{option} requires METHOD=VALUE, got {value!r}")
        key, selected = value.split("=", 1)
        if not key or not selected or key in result:
            raise EvaluationError(f"invalid or duplicate {option}: {value!r}")
        result[key] = selected
    return result


def _parse_named_path(
    value: str, *, default_protocol: str
) -> tuple[str | None, str, Path]:
    """Parse ``[METHOD[@PROTOCOL]=]PATH``."""

    if "=" not in value:
        return None, default_protocol, Path(value).expanduser().resolve()
    name, raw_path = value.split("=", 1)
    if "@" in name:
        method, protocol = name.rsplit("@", 1)
    else:
        method, protocol = name, default_protocol
    if not method or not protocol or not raw_path:
        raise EvaluationError(f"invalid named path: {value!r}")
    return method, protocol, Path(raw_path).expanduser().resolve()


def _derive_score_ranks(
    frame: pd.DataFrame,
    *,
    method: str,
    score_column: str,
    protocol: str,
) -> pd.DataFrame:
    required = {"sample_id", "candidate_id", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EvaluationError(
            f"score-only prediction for {method} missing columns: {missing}"
        )
    result = frame.loc[
        :,
        [
            "sample_id",
            "candidate_id",
            *(
                ["candidate_identity_sha256"]
                if "candidate_identity_sha256" in frame.columns
                else []
            ),
            score_column,
        ],
    ].rename(columns={score_column: "score"})
    scores = pd.to_numeric(result["score"], errors="coerce")
    if scores.isna().any() or not np.all(np.isfinite(scores.to_numpy(float))):
        raise EvaluationError(f"{method} scores must be finite")
    result["score"] = scores
    result["method"] = method
    result["protocol"] = protocol
    ordered = result.sort_values(
        ["sample_id", "score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ordered["rank"] = ordered.groupby("sample_id", sort=False).cumcount() + 1
    return ordered


def _prediction_table(
    path: Path,
    *,
    method_override: str | None,
    protocol: str,
) -> pd.DataFrame:
    frame = _read_table(path)
    method_column = (
        "method"
        if "method" in frame.columns
        else "reranker_method"
        if "reranker_method" in frame.columns
        else None
    )
    rank_column = (
        "rank"
        if "rank" in frame.columns
        else "reranker_rank"
        if "reranker_rank" in frame.columns
        else None
    )
    score_column = (
        "score"
        if "score" in frame.columns
        else "reranker_score"
        if "reranker_score" in frame.columns
        else None
    )
    if method_override is not None:
        if method_column is not None and frame[method_column].nunique() > 1:
            raise EvaluationError(
                f"{path} contains multiple methods; cannot apply method override"
            )
        frame["method"] = method_override
        method_column = "method"
    if "protocol" not in frame.columns:
        frame["protocol"] = protocol
    if method_column is None:
        raise EvaluationError(
            f"{path} has no method column; use METHOD=PATH to name it"
        )
    if rank_column is None:
        if score_column is None:
            raise EvaluationError(f"{path} has neither rank nor score")
        parts: list[pd.DataFrame] = []
        canonical_method_column = method_column
        for (file_protocol, method), group in frame.groupby(
            ["protocol", canonical_method_column], sort=True
        ):
            parts.append(
                _derive_score_ranks(
                    group,
                    method=str(method),
                    score_column=score_column,
                    protocol=str(file_protocol),
                )
            )
        return pd.concat(parts, ignore_index=True)
    result = frame.copy()
    if method_column != "method":
        result = result.rename(columns={method_column: "method"})
    if rank_column != "rank":
        result = result.rename(columns={rank_column: "rank"})
    if score_column is not None and score_column != "score":
        result = result.rename(columns={score_column: "score"})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument(
        "--sample-universe",
        type=Path,
        required=True,
        help=(
            "Complete per-sample parquet/csv containing sample_id and scene_id; "
            "valid-empty samples are required for the all denominator."
        ),
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="[METHOD[@PROTOCOL]=]PATH",
        help="Long prediction parquet/csv/json(l); repeat for multiple artifacts.",
    )
    parser.add_argument(
        "--rank-column",
        action="append",
        default=[],
        metavar="METHOD=COLUMN",
        help="Wide rank column inside --per-candidate; repeat as needed.",
    )
    parser.add_argument(
        "--score-column",
        action="append",
        default=[],
        metavar="METHOD=COLUMN",
        help="Optional wide score column matching --rank-column.",
    )
    parser.add_argument(
        "--vlm-results",
        action="append",
        default=[],
        metavar="METHOD[@PROTOCOL]=JSONL",
        help="Audited local-VLM result JSONL; repeat for multiple VLM methods.",
    )
    parser.add_argument(
        "--vlm-repeat",
        action="append",
        default=[],
        metavar="METHOD[@PROTOCOL]=JSONL",
        help="Deterministic repeat JSONL for a matching --vlm-results method.",
    )
    parser.add_argument(
        "--vlm-summary",
        action="append",
        default=[],
        metavar="METHOD[@PROTOCOL]=JSON",
        help="Runtime summary containing wall_time_seconds and optionally memory_peak_mib.",
    )
    parser.add_argument(
        "--vlm-memory-peak-mib",
        action="append",
        default=[],
        metavar="METHOD[@PROTOCOL]=FLOAT",
        help="Measured peak resident/unified memory for a VLM run.",
    )
    parser.add_argument("--default-protocol", default="full_nms")
    parser.add_argument("--vlm-default-protocol", default="gqcnn_top5")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-sample-count", type=int, required=True)
    parser.add_argument("--expected-candidate-count", type=int, required=True)
    parser.add_argument("--expected-nonempty-count", type=int, required=True)
    parser.add_argument("--expected-valid-empty-count", type=int, required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-lock", type=Path)
    parser.add_argument("--formal-inference-manifest", type=Path)
    parser.add_argument("--formal-label-join-manifest", type=Path)
    parser.add_argument("--formal-vlm-manifest", type=Path, action="append", default=[])
    parser.add_argument("--formal-vlm-apply-manifest", type=Path)
    args = parser.parse_args(argv)
    expected = (
        args.expected_sample_count,
        args.expected_candidate_count,
        args.expected_nonempty_count,
        args.expected_valid_empty_count,
        args.expected_scene_count,
    )
    if any(value < 0 for value in expected):
        parser.error("all expected counts must be non-negative")
    if (
        args.expected_nonempty_count + args.expected_valid_empty_count
        != args.expected_sample_count
    ):
        parser.error(
            "expected nonempty + valid-empty counts must equal expected samples"
        )
    if args.bootstrap_replicates < 10_000:
        parser.error("--bootstrap-replicates must be at least 10000")
    formal_values = (
        args.experiment_lock,
        args.formal_inference_manifest,
        args.formal_label_join_manifest,
        args.formal_vlm_apply_manifest,
    )
    formal_requested = any(value is not None for value in formal_values) or bool(
        args.formal_vlm_manifest
    )
    if formal_requested and (
        any(value is None for value in formal_values)
        or len(args.formal_vlm_manifest) != 1
    ):
        parser.error(
            "formal evaluation requires the lock, learned/label/apply manifests, "
            "and exactly one validation-selected formal VLM manifest"
        )
    return args


def _assert_locked_path(lock: Mapping[str, Any], path: Path) -> None:
    resolved = path.expanduser().resolve()
    matches = [
        item
        for item in lock["artifacts"].values()
        if Path(item["path"]).resolve() == resolved
    ]
    if len(matches) != 1 or matches[0]["sha256"] != _sha256(resolved):
        raise EvaluationError(f"formal evaluation path absent from lock: {resolved}")


def _required_formal_apply_paths(
    *,
    join: Mapping[str, Any],
    selected_contract: Mapping[str, Path],
    selection_path: Path,
    inference_manifest_path: Path,
    universe_path: Path,
) -> dict[str, Path]:
    """Return the exact GT-free artifacts that formal VLM apply must bind."""

    return {
        "formal_vlm_manifest": selected_contract["manifest"],
        "selection": selection_path,
        "formal_inference_manifest": inference_manifest_path,
        "per_candidate": Path(str(join["inference_per_candidate"])).resolve(),
        "sample_universe": universe_path,
    }


def _assert_formal_completion(
    *,
    lock_path: Path,
    lock: Mapping[str, Any],
    stage: str,
    manifest_path: Path,
) -> None:
    normalized = stage.upper()
    completion_name = (
        f"{lock_path.name}.FORMAL_TEST_COMPLETED.json"
        if normalized == "TEST"
        else f"{lock_path.name}.FORMAL_{normalized}_COMPLETED.json"
    )
    start_name = (
        f"{lock_path.name}.FORMAL_TEST_STARTED.json"
        if normalized == "TEST"
        else f"{lock_path.name}.FORMAL_{normalized}_STARTED.json"
    )
    completion_path = lock_path.with_name(completion_name)
    start_path = lock_path.with_name(start_name)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    start = json.loads(start_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        validate_matching_artifact_identity(
            start,
            lock,
            context=f"formal {normalized} start ledger",
        )
        validate_matching_artifact_identity(
            completion,
            lock,
            context=f"formal {normalized} completion ledger",
        )
        validate_matching_artifact_identity(
            manifest,
            lock,
            context=f"formal {normalized} manifest",
        )
    except ValueError as error:
        raise EvaluationError(str(error)) from error
    if (
        completion.get("lock_content_sha256")
        != lock["manifest_content_sha256"]
        or completion.get("stage") != normalized
        or Path(str(completion.get("start_ledger", ""))).resolve()
        != start_path
        or completion.get("start_ledger_sha256") != _sha256(start_path)
        or Path(str(completion.get("manifest_path", ""))).resolve()
        != manifest_path
        or completion.get("manifest_sha256") != _sha256(manifest_path)
    ):
        raise EvaluationError(
            f"formal stage {normalized} completion ledger is invalid"
        )


def _validate_formal_vlm_execution_evidence(
    *,
    vlm: Mapping[str, Any],
    runtime: Mapping[str, Any],
    results_path: Path,
    monitor_path: Path,
    manifest_path: Path,
    lock_path: Path,
    lock: Mapping[str, Any],
    stage: str,
) -> None:
    """Validate the immutable, local-only attempt chain before GT evaluation."""

    stage_root = manifest_path.parent.resolve()
    attempts_root = (stage_root / "attempts").resolve()
    artifacts = vlm.get("formal_attempt_ledgers")
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or runtime.get("formal_attempt_ledgers") != artifacts
    ):
        raise EvaluationError("formal VLM attempt-ledger inventory is invalid")
    locked_audit = Path(str(vlm.get("locked_local_audit", ""))).resolve()
    if (
        not locked_audit.is_file()
        or _sha256(locked_audit) != vlm.get("locked_local_audit_sha256")
    ):
        raise EvaluationError("formal VLM locked local audit changed")
    stable_hash = str(vlm.get("stable_session_contract_sha256", ""))
    if len(stable_hash) != 64:
        raise EvaluationError("formal VLM stable runtime contract is invalid")

    prior_hash: str | None = None
    prior_prefix_count = -1
    session_paths: set[Path] = set()
    for attempt_number, artifact in enumerate(artifacts, start=1):
        if not isinstance(artifact, Mapping):
            raise EvaluationError("formal VLM attempt artifact is malformed")
        ledger_path = Path(str(artifact.get("path", ""))).resolve()
        expected_path = attempts_root / f"attempt_{attempt_number:04d}.json"
        if (
            ledger_path != expected_path
            or not ledger_path.is_file()
            or _sha256(ledger_path) != artifact.get("sha256")
        ):
            raise EvaluationError("formal VLM attempt ledger changed or escaped")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        _require_repeatedfilm_identity(
            ledger, context="formal VLM attempt ledger"
        )
        session_path = Path(
            str(ledger.get("session_local_audit", ""))
        ).resolve()
        prefix_count = int(ledger.get("prefix_result_row_count", -1))
        prefix_hash = ledger.get("prefix_results_sha256")
        if (
            int(ledger.get("schema_version", -1)) != 1
            or int(ledger.get("session_audit_policy_version", -1))
            != SESSION_AUDIT_POLICY_VERSION
            or int(ledger.get("attempt_number", -1)) != attempt_number
            or ledger.get("previous_attempt_sha256") != prior_hash
            or Path(str(ledger.get("lock_path", ""))).resolve()
            != lock_path.resolve()
            or ledger.get("lock_content_sha256")
            != lock["manifest_content_sha256"]
            or ledger.get("stage") != stage
            or Path(str(ledger.get("locked_local_audit", ""))).resolve()
            != locked_audit
            or ledger.get("locked_local_audit_sha256")
            != vlm["locked_local_audit_sha256"]
            or ledger.get("stable_session_contract_sha256") != stable_hash
            or session_path.parent != attempts_root
            or session_path in session_paths
            or not session_path.is_file()
            or _sha256(session_path)
            != ledger.get("session_local_audit_sha256")
            or int(ledger.get("listener_pid", 0)) <= 0
            or not str(ledger.get("listener_process_started", ""))
            or prefix_count < 0
            or prefix_count < prior_prefix_count
            or (prefix_count == 0 and prefix_hash is not None)
            or (
                prefix_count > 0
                and (
                    not isinstance(prefix_hash, str)
                    or len(prefix_hash) != 64
                )
            )
            or ledger.get("prefix_results_sha256")
            != jsonl_prefix_sha256(results_path, prefix_count)
        ):
            raise EvaluationError("formal VLM attempt-ledger chain is invalid")
        session_audit = json.loads(session_path.read_text(encoding="utf-8"))
        if (
            session_audit.get("local_only") is not True
            or session_audit.get("remote_api_used") is not False
            or session_audit.get("ollama", {}).get(
                "remote_established_connections", []
            )
            or int(session_audit.get("session_audit_policy_version", -1))
            != SESSION_AUDIT_POLICY_VERSION
            or session_audit.get("stable_session_contract_sha256")
            != stable_hash
            or stable_session_contract_sha256(session_audit) != stable_hash
        ):
            raise EvaluationError("formal VLM session audit is not local-only")
        session_paths.add(session_path)
        prior_prefix_count = prefix_count
        prior_hash = str(artifact["sha256"])

    if (
        vlm.get("formal_attempt_chain_tip_sha256") != prior_hash
        or runtime.get("formal_attempt_chain_tip_sha256") != prior_hash
        or runtime.get("local_only_runtime_passed") is not True
        or int(runtime.get("remote_established_connection_event_count", -1))
        != 0
        or runtime.get("remote_established_connection_events") != []
        or int(runtime.get("monitor_attempt_count", -1)) != len(artifacts)
    ):
        raise EvaluationError("formal VLM aggregate runtime evidence is invalid")
    monitor_rows = _read_jsonl(monitor_path)
    if (
        int(runtime.get("monitor_sample_count", -1)) != len(monitor_rows)
        or any(
            row.get("remote_established_connections", [])
            for row in monitor_rows
        )
    ):
        raise EvaluationError("formal VLM runtime monitor is not local-only")


def _validate_formal_vlm_summary_runtime(
    *,
    vlm: Mapping[str, Any],
    summary: Mapping[str, Any],
    runtime: Mapping[str, Any],
    results_path: Path,
) -> None:
    """Independently bind formal summary/runtime fields to hashed artifacts."""

    input_path = Path(str(vlm.get("input_jsonl", ""))).resolve()
    sample_count = int(vlm.get("sample_count", -1))
    eligible_count = int(vlm.get("eligible_sample_count", -1))
    empty_count = int(vlm.get("empty_skipped_count", -1))
    input_rows = _read_jsonl(input_path) if input_path.is_file() else []
    result_rows = _read_jsonl(results_path)
    if (
        not input_path.is_file()
        or _sha256(input_path) != vlm.get("input_jsonl_sha256")
        or len(input_rows) != sample_count
        or len(result_rows) != sample_count
        or eligible_count < 0
        or empty_count < 0
        or eligible_count + empty_count != sample_count
    ):
        raise EvaluationError("formal VLM input/result universe is invalid")
    for input_row, result_row in zip(input_rows, result_rows, strict=True):
        sample_id = str(input_row.get("sample_id", ""))
        candidate_ids = list(map(str, input_row.get("candidate_ids", [])))
        if (
            not sample_id
            or str(result_row.get("sample_id", "")) != sample_id
            or result_row.get("input_record_sha256")
            != _canonical_json_sha256(input_row)
        ):
            raise EvaluationError(
                "formal VLM result/input recipe binding is invalid"
            )
        if not candidate_ids:
            if (
                result_row.get("eligible_for_vlm") is not False
                or result_row.get("request_hash") is not None
                or result_row.get("ranking") != []
            ):
                raise EvaluationError(
                    "formal valid-empty VLM result is invalid"
                )
            continue
        recipe = input_row.get("visualization_recipe")
        if (
            input_row.get("visualization_storage_mode")
            != "on_demand_recipe"
            or not isinstance(recipe, Mapping)
            or input_row.get("visualization_recipe_sha256")
            != canonical_recipe_sha256(recipe)
        ):
            raise EvaluationError(
                "formal VLM input is not a locked on-demand recipe"
            )
        validated = validate_vlm_visualization_recipe(
            recipe, verify_sources=True
        )
        image_hashes = result_row.get("temporary_image_sha256")
        if (
            validated["sample_id"] != sample_id
            or validated["candidate_ids"] != candidate_ids
            or result_row.get("eligible_for_vlm") is not True
            or result_row.get("visualization_storage_mode")
            != "on_demand_recipe"
            or result_row.get("visualization_recipe_sha256")
            != input_row["visualization_recipe_sha256"]
            or not isinstance(image_hashes, list)
            or len(image_hashes) != 2
            or any(not _is_lower_sha256(digest) for digest in image_hashes)
            or not _is_lower_sha256(
                result_row.get(
                    "temporary_visualization_manifest_sha256"
                )
            )
            or not _is_lower_sha256(result_row.get("request_hash"))
        ):
            raise EvaluationError(
                "formal VLM temporary-visual provenance is invalid"
            )
    common = {
        "input_jsonl": input_path,
        "input_jsonl_sha256": vlm["input_jsonl_sha256"],
        "results_jsonl": results_path,
        "results_jsonl_sha256": vlm["results_jsonl_sha256"],
        "input_split": "test",
        "formal_mode": True,
        "model_name": vlm["model_name"],
        "model_digest": vlm["model_digest"],
        "stable_session_contract_sha256": vlm[
            "stable_session_contract_sha256"
        ],
        "sample_count": sample_count,
        "eligible_sample_count": eligible_count,
        "empty_skipped_count": empty_count,
    }
    for name, evidence in (("summary", summary), ("runtime", runtime)):
        for key, expected in common.items():
            actual = evidence.get(key)
            if key in {"input_jsonl", "results_jsonl"}:
                actual = Path(str(actual)).resolve()
            if actual != expected:
                raise EvaluationError(
                    f"formal VLM {name} binding is invalid: {key}"
                )
        if (
            evidence.get("visualization_storage_mode")
            != "on_demand_recipe"
            or int(evidence.get("on_demand_visual_sample_count", -1))
            != eligible_count
            or int(evidence.get("ordinary_visual_pngs_retained", -1))
            != 0
        ):
            raise EvaluationError(
                f"formal VLM {name} temporary-storage accounting is invalid"
            )
    if (
        int(runtime.get("fresh_http_call_count", -1))
        + int(runtime.get("cache_hit_count", -1))
        != eligible_count
    ):
        raise EvaluationError("formal VLM runtime call accounting is invalid")


def _formal_allowed_sources(
    args: argparse.Namespace,
    *,
    candidate_path: Path,
    candidates: pd.DataFrame,
    universe_path: Path,
) -> tuple[dict[str, Any], set[Path], dict[str, dict[str, Path]]]:
    lock_path = args.experiment_lock.expanduser().resolve()
    lock = verify_lock(lock_path)
    _assert_locked_path(lock, universe_path)
    if int(lock["expected_test_sample_count"]) != args.expected_sample_count:
        raise EvaluationError("formal expected sample count disagrees with lock")
    if int(lock["seeds"].get("bootstrap", -1)) != int(args.seed):
        raise EvaluationError("formal bootstrap seed disagrees with lock")
    evaluation_definition = lock["evaluation_definition"]
    if int(evaluation_definition.get("bootstrap_replicates", -1)) != int(
        args.bootstrap_replicates
    ):
        raise EvaluationError("formal bootstrap count disagrees with lock")
    expected_counts = evaluation_definition.get("expected_counts", {})
    caller_counts = {
        "samples": args.expected_sample_count,
        "candidates": args.expected_candidate_count,
        "nonempty": args.expected_nonempty_count,
        "valid_empty": args.expected_valid_empty_count,
        "scenes": args.expected_scene_count,
    }
    if expected_counts != caller_counts:
        raise EvaluationError("formal expected counts disagree with experiment lock")

    join_manifest_path = args.formal_label_join_manifest.expanduser().resolve()
    join = json.loads(join_manifest_path.read_text(encoding="utf-8"))
    _require_repeatedfilm_identity(
        join, context="formal label-join manifest"
    )
    if join.get("lock_content_sha256") != lock["manifest_content_sha256"]:
        raise EvaluationError("formal label join belongs to another lock")
    if (
        Path(str(join["output"])).resolve() != candidate_path
        or _sha256(candidate_path) != join["output_sha256"]
    ):
        raise EvaluationError("formal evaluation candidate-label join changed")
    _assert_locked_path(lock, Path(str(join["inference_per_candidate"])))
    _assert_locked_path(lock, Path(str(join["labels_parquet"])))
    if (
        join.get("candidate_identity_invariant") is not True
        or join.get("formal_predictions_unchanged") is not True
        or join.get("candidate_count") != args.expected_candidate_count
        or Path(str(join.get("formal_inference_manifest", ""))).resolve()
        != args.formal_inference_manifest.expanduser().resolve()
        or join.get("formal_inference_manifest_sha256")
        != _sha256(args.formal_inference_manifest.expanduser().resolve())
    ):
        raise EvaluationError("formal label-join manifest is incomplete or inconsistent")
    inference_candidates = _read_table(Path(str(join["inference_per_candidate"])))
    locked_labels = _read_table(Path(str(join["labels_parquet"])))
    locked_labels = locked_labels.loc[
        locked_labels["sample_id"].astype(str).isin(
            set(inference_candidates["sample_id"].astype(str))
        )
    ]
    rebuilt = pd.DataFrame(
        join_candidate_labels(
            inference_candidates.to_dict("records"),
            locked_labels.to_dict("records"),
        )
    )
    try:
        pd.testing.assert_frame_equal(
            candidates.reset_index(drop=True),
            rebuilt.reset_index(drop=True),
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as error:
        raise EvaluationError(
            "formal evaluation labels are not the exact locked post-inference join"
        ) from error

    formal_ledger_path = lock_path.with_name(
        f"{lock_path.name}.FORMAL_TEST_STARTED.json"
    )
    formal_ledger = json.loads(formal_ledger_path.read_text(encoding="utf-8"))
    inference_manifest_path = args.formal_inference_manifest.expanduser().resolve()
    if inference_manifest_path != (
        Path(str(formal_ledger["output_root"])) / "inference_manifest.json"
    ).resolve():
        raise EvaluationError("formal inference manifest is outside guarded output")
    inference = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    _require_repeatedfilm_identity(
        inference, context="formal reranker inference manifest"
    )
    if inference.get("lock_content_sha256") != lock["manifest_content_sha256"]:
        raise EvaluationError("formal inference belongs to another lock")
    _assert_formal_completion(
        lock_path=lock_path,
        lock=lock,
        stage="TEST",
        manifest_path=inference_manifest_path,
    )
    learned_predictions = Path(str(inference["predictions"])).resolve()
    if _sha256(learned_predictions) != inference["predictions_sha256"]:
        raise EvaluationError("formal learned predictions changed")
    learned_decisions = Path(str(inference["safe_switch_decisions"])).resolve()
    if _sha256(learned_decisions) != inference["safe_switch_decisions_sha256"]:
        raise EvaluationError("formal learned safe-switch decisions changed")

    selected_vlm = selected_vlm_contract(lock)
    vlm_manifest_path = args.formal_vlm_manifest[0].expanduser().resolve()
    vlm = json.loads(vlm_manifest_path.read_text(encoding="utf-8"))
    _require_repeatedfilm_identity(
        vlm, context="formal VLM execution manifest"
    )
    variant = selected_vlm["source_variant"]
    method = selected_vlm["source_method"]
    protocol = selected_vlm["protocol"]
    stage_name = "VLM_VISUAL" if variant == "visual" else "VLM_METADATA"
    if (
        vlm.get("variant") != variant
        or vlm.get("source_method") != method
        or vlm.get("model_digest") != selected_vlm["model_digest"]
        or Path(
            str(vlm.get("validation_selection_path", ""))
        ).resolve()
        != Path(selected_vlm["validation_selection_path"]).resolve()
        or vlm.get("validation_selection_sha256")
        != selected_vlm["validation_selection_sha256"]
    ):
        raise EvaluationError(
            "formal VLM manifest is not the validation-selected variant "
            "recorded in the immutable lock"
        )
    aggregate_path = Path(
        str(vlm.get("aggregate_visual_manifest", ""))
    ).resolve()
    if (
        not aggregate_path.is_file()
        or _sha256(aggregate_path)
        != vlm.get("aggregate_visual_manifest_sha256")
    ):
        raise EvaluationError(
            "formal VLM on-demand recipe aggregate changed"
        )
    _assert_locked_path(lock, aggregate_path)
    vlm_stage_ledger = lock_path.with_name(
        f"{lock_path.name}.FORMAL_{stage_name}_STARTED.json"
    )
    vlm_stage = json.loads(vlm_stage_ledger.read_text(encoding="utf-8"))
    if vlm_manifest_path != (
        Path(str(vlm_stage["output_root"])) / "formal_vlm_manifest.json"
    ).resolve():
        raise EvaluationError("formal VLM manifest is outside guarded output")
    if (
        vlm.get("lock_content_sha256") != lock["manifest_content_sha256"]
        or vlm.get("completed") is not True
        or vlm.get("stage") != stage_name
    ):
        raise EvaluationError("formal VLM stage is not complete for this lock")
    _assert_formal_completion(
        lock_path=lock_path,
        lock=lock,
        stage=stage_name,
        manifest_path=vlm_manifest_path,
    )
    result_path = Path(str(vlm["results_jsonl"])).resolve()
    summary_path = Path(str(vlm["summary"])).resolve()
    runtime_path = Path(str(vlm["runtime_metrics"])).resolve()
    monitor_path = Path(str(vlm["runtime_monitor"])).resolve()
    if (
        _sha256(result_path) != vlm["results_jsonl_sha256"]
        or _sha256(summary_path) != vlm["summary_sha256"]
        or _sha256(runtime_path) != vlm["runtime_metrics_sha256"]
        or _sha256(monitor_path) != vlm["runtime_monitor_sha256"]
    ):
        raise EvaluationError("formal VLM result/runtime artifact changed")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require_repeatedfilm_identity(summary, context="formal VLM summary")
    _require_repeatedfilm_identity(runtime, context="formal VLM runtime")
    _validate_formal_vlm_summary_runtime(
        vlm=vlm,
        summary=summary,
        runtime=runtime,
        results_path=result_path,
    )
    _validate_formal_vlm_execution_evidence(
        vlm=vlm,
        runtime=runtime,
        results_path=result_path,
        monitor_path=monitor_path,
        manifest_path=vlm_manifest_path,
        lock_path=lock_path,
        lock=lock,
        stage=stage_name,
    )
    selected_key = f"{protocol}/{method}"
    vlm_contracts: dict[str, dict[str, Path]] = {
        selected_key: {
            "manifest": vlm_manifest_path,
            "results": result_path,
            "summary": summary_path,
            "runtime": runtime_path,
        }
    }

    apply_stage_ledger = lock_path.with_name(
        f"{lock_path.name}.FORMAL_VLM_APPLY_STARTED.json"
    )
    apply_stage = json.loads(apply_stage_ledger.read_text(encoding="utf-8"))
    apply_manifest_path = args.formal_vlm_apply_manifest.expanduser().resolve()
    if apply_manifest_path != (
        Path(str(apply_stage["output_root"])) / "manifest.json"
    ).resolve():
        raise EvaluationError("formal VLM apply manifest is outside guarded output")
    applied = json.loads(apply_manifest_path.read_text(encoding="utf-8"))
    _require_repeatedfilm_identity(
        applied, context="formal VLM apply manifest"
    )
    if applied.get("lock_content_sha256") != lock["manifest_content_sha256"]:
        raise EvaluationError("formal VLM apply belongs to another lock")
    _assert_formal_completion(
        lock_path=lock_path,
        lock=lock,
        stage="VLM_APPLY",
        manifest_path=apply_manifest_path,
    )
    vlm_safe_predictions = Path(str(applied["predictions"])).resolve()
    if _sha256(vlm_safe_predictions) != applied["predictions_sha256"]:
        raise EvaluationError("formal VLM safe-switch predictions changed")
    selection_item = lock["artifacts"]["vlm_safe_switch_selection"]
    selection_path = Path(selection_item["path"]).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _require_repeatedfilm_identity(
        selection, context="formal VLM safe-switch selection"
    )
    selected_method = selected_vlm["source_method"]
    selected_contract = vlm_contracts.get(selected_key)
    if selected_contract is None:
        raise EvaluationError("formal VLM apply selection names an unknown comparator")
    required_apply_paths = _required_formal_apply_paths(
        join=join,
        selected_contract=selected_contract,
        selection_path=selection_path,
        inference_manifest_path=inference_manifest_path,
        universe_path=universe_path,
    )
    required_apply_paths["vlm_results"] = selected_contract["results"]
    for field, expected_path in required_apply_paths.items():
        actual_path = Path(str(applied.get(field, ""))).resolve()
        expected_hash = (
            selection_item["sha256"]
            if field == "selection"
            else _sha256(expected_path)
        )
        if (
            actual_path != expected_path
            or applied.get(f"{field}_sha256") != expected_hash
        ):
            raise EvaluationError(f"formal VLM apply {field} binding is invalid")
    if (
        applied.get("source_method") != selected_method
        or selection.get("source_method") != selected_method
        or selection.get("source_variant") != selected_vlm["source_variant"]
        or applied.get("source_variant") != selected_vlm["source_variant"]
        or Path(
            str(applied.get("validation_selection_path", ""))
        ).resolve()
        != Path(selected_vlm["validation_selection_path"]).resolve()
        or applied.get("validation_selection_sha256")
        != selected_vlm["validation_selection_sha256"]
        or applied.get("candidate_pool_modified") is not False
        or applied.get("candidate_identity_invariant") is not True
        or applied.get("GT_columns_used") is not False
    ):
        raise EvaluationError("formal VLM apply invariants are incomplete")
    leaked = sorted(set(inference_candidates.columns) & set(FORBIDDEN_GT_COLUMNS))
    if leaked:
        raise EvaluationError(
            f"GT columns reached formal VLM safe-switch inference: {leaked}"
        )
    decision_path = Path(str(applied.get("decisions", ""))).resolve()
    if (
        not decision_path.is_file()
        or applied.get("decisions_sha256") != _sha256(decision_path)
    ):
        raise EvaluationError("formal VLM safe-switch decisions changed")
    expected_predictions, expected_decisions = apply_vlm_safe_switch_policy(
        inference_candidates,
        _read_jsonl(selected_contract["results"]),
        selection,
    )
    actual_predictions = _read_table(vlm_safe_predictions)
    actual_decisions = _read_table(decision_path)
    try:
        pd.testing.assert_frame_equal(
            actual_predictions.reset_index(drop=True),
            expected_predictions.reset_index(drop=True),
            check_dtype=True,
            check_like=False,
        )
        pd.testing.assert_frame_equal(
            actual_decisions.reset_index(drop=True),
            expected_decisions.reset_index(drop=True),
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as error:
        raise EvaluationError(
            "formal VLM safe-switch outputs fail independent policy replay"
        ) from error
    expected_counts = {
        "sample_count": len(_read_table(universe_path)),
        "nonempty_sample_count": len(expected_decisions),
        "valid_empty_sample_count": len(_read_table(universe_path))
        - len(expected_decisions),
        "prediction_rows": len(expected_predictions),
        "switch_count": int(
            expected_decisions["switch_applied"].astype(bool).sum()
        ),
    }
    if any(applied.get(key) != value for key, value in expected_counts.items()):
        raise EvaluationError("formal VLM safe-switch manifest counts are invalid")
    return lock, {learned_predictions, vlm_safe_predictions}, vlm_contracts


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = args.per_candidate.expanduser().resolve()
    candidates = _read_table(candidate_path)
    universe_path = args.sample_universe.expanduser().resolve()
    universe = _read_table(universe_path)
    formal_lock: dict[str, Any] | None = None
    formal_allowed_predictions: set[Path] = set()
    formal_vlm_contracts: dict[str, dict[str, Path]] = {}
    if args.experiment_lock is not None:
        (
            formal_lock,
            formal_allowed_predictions,
            formal_vlm_contracts,
        ) = (
            _formal_allowed_sources(
                args,
                candidate_path=candidate_path,
                candidates=candidates,
                universe_path=universe_path,
            )
        )
        if (
            args.rank_column
            or args.score_column
            or args.vlm_repeat
            or args.vlm_memory_peak_mib
        ):
            raise EvaluationError("wide ad-hoc columns are forbidden in formal evaluation")
    actual_sample_count = int(universe["sample_id"].nunique())
    actual_nonempty_count = int(candidates["sample_id"].nunique())
    actual_valid_empty_count = actual_sample_count - actual_nonempty_count
    actual_scene_count = int(universe["scene_id"].nunique())
    if actual_sample_count != args.expected_sample_count:
        raise EvaluationError(
            f"sample count mismatch: {actual_sample_count} != "
            f"{args.expected_sample_count}"
        )
    if len(candidates) != args.expected_candidate_count:
        raise EvaluationError(
            f"candidate count mismatch: {len(candidates)} != "
            f"{args.expected_candidate_count}"
        )
    if actual_nonempty_count != args.expected_nonempty_count:
        raise EvaluationError(
            f"nonempty count mismatch: {actual_nonempty_count} != "
            f"{args.expected_nonempty_count}"
        )
    if actual_valid_empty_count != args.expected_valid_empty_count:
        raise EvaluationError(
            f"valid-empty count mismatch: {actual_valid_empty_count} != "
            f"{args.expected_valid_empty_count}"
        )
    if actual_scene_count != args.expected_scene_count:
        raise EvaluationError(
            f"scene count mismatch: {actual_scene_count} != "
            f"{args.expected_scene_count}"
        )

    prediction_parts: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = [
        {
            "role": "independent_candidate_labels_and_features",
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
        }
    ]
    sources.append(
        {
            "role": "sample_universe",
            "path": str(universe_path),
            "sha256": _sha256(universe_path),
        }
    )
    observed_formal_prediction_paths: set[Path] = set()
    for specification in args.prediction:
        method, protocol, path = _parse_named_path(
            specification, default_protocol=args.default_protocol
        )
        if formal_lock is not None:
            if path not in formal_allowed_predictions:
                raise EvaluationError(f"unlocked formal prediction source: {path}")
            if method is not None:
                raise EvaluationError(
                    "formal prediction artifacts may not be relabelled with "
                    "METHOD=PATH"
                )
            if path in observed_formal_prediction_paths:
                raise EvaluationError(
                    f"duplicate formal prediction source: {path}"
                )
            observed_formal_prediction_paths.add(path)
        prediction_parts.append(
            _prediction_table(
                path, method_override=method, protocol=protocol
            )
        )
        sources.append(
            {
                "role": "prediction",
                "method_override": method,
                "protocol_default": protocol,
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    if (
        formal_lock is not None
        and observed_formal_prediction_paths != formal_allowed_predictions
    ):
        raise EvaluationError(
            "formal evaluation must consume the guarded learned and selected "
            "VLM safe-switch prediction artifacts exactly once"
        )

    rank_columns = _parse_mapping(args.rank_column, "--rank-column")
    score_columns = _parse_mapping(args.score_column, "--score-column")
    unknown_scores = sorted(set(score_columns) - set(rank_columns))
    if unknown_scores:
        raise EvaluationError(
            f"score columns have no matching rank method: {unknown_scores}"
        )
    if rank_columns:
        prediction_parts.append(
            predictions_from_wide(
                candidates,
                rank_columns=rank_columns,
                score_columns=score_columns,
                protocol=args.default_protocol,
            )
        )

    repeat_paths: dict[str, Path] = {}
    for specification in args.vlm_repeat:
        method, protocol, path = _parse_named_path(
            specification, default_protocol=args.vlm_default_protocol
        )
        if method is None:
            raise EvaluationError("--vlm-repeat requires METHOD=JSONL")
        repeat_paths[f"{protocol}/{method}"] = path
    summary_paths: dict[str, Path] = {}
    for specification in args.vlm_summary:
        method, protocol, path = _parse_named_path(
            specification, default_protocol=args.vlm_default_protocol
        )
        if method is None:
            raise EvaluationError("--vlm-summary requires METHOD=JSON")
        summary_paths[f"{protocol}/{method}"] = path
    memory_values: dict[str, float] = {}
    for specification in args.vlm_memory_peak_mib:
        if "=" not in specification:
            raise EvaluationError(
                "--vlm-memory-peak-mib requires METHOD[@PROTOCOL]=FLOAT"
            )
        raw_name, raw_value = specification.split("=", 1)
        if "@" in raw_name:
            method, protocol = raw_name.rsplit("@", 1)
        else:
            method, protocol = raw_name, args.vlm_default_protocol
        key = f"{protocol}/{method}"
        if not method or not protocol or key in memory_values:
            raise EvaluationError(f"invalid VLM memory specification: {specification}")
        try:
            memory_values[key] = float(raw_value)
        except ValueError as error:
            raise EvaluationError(
                f"invalid VLM memory value: {raw_value!r}"
            ) from error

    runtime: dict[str, Mapping[str, Any]] = {}
    observed_vlm_keys: set[str] = set()
    for specification in args.vlm_results:
        method, protocol, path = _parse_named_path(
            specification, default_protocol=args.vlm_default_protocol
        )
        if method is None:
            raise EvaluationError("--vlm-results requires METHOD=JSONL")
        key = f"{protocol}/{method}"
        if formal_lock is not None:
            contract = formal_vlm_contracts.get(key)
            if contract is None or path != contract["results"]:
                raise EvaluationError(
                    f"formal VLM method/result binding is invalid: {key}={path}"
                )
        if key in observed_vlm_keys:
            raise EvaluationError(f"duplicate VLM method/protocol: {key}")
        observed_vlm_keys.add(key)
        rows = _read_jsonl(path)
        prediction_parts.append(
            vlm_results_to_predictions(rows, method=method, protocol=protocol)
        )
        repeat_rows = (
            _read_jsonl(repeat_paths[key]) if key in repeat_paths else None
        )
        summary: dict[str, Any] = {}
        if key in summary_paths:
            if (
                formal_lock is not None
                and summary_paths[key] != formal_vlm_contracts[key]["summary"]
            ):
                raise EvaluationError(
                    f"formal VLM method/summary binding is invalid: {key}"
                )
            summary_value = json.loads(
                summary_paths[key].read_text(encoding="utf-8")
            )
            if not isinstance(summary_value, dict):
                raise EvaluationError(f"VLM summary must be an object: {summary_paths[key]}")
            summary = summary_value
        runtime[key] = vlm_runtime_metrics(
            rows,
            repeat_rows=repeat_rows,
            wall_time_seconds=summary.get("wall_time_seconds"),
            memory_peak_mib=memory_values.get(
                key, summary.get("memory_peak_mib")
            ),
        )
        sources.append(
            {
                "role": "vlm_prediction_and_runtime",
                "method": method,
                "protocol": protocol,
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
        if key in repeat_paths:
            sources.append(
                {
                    "role": "vlm_deterministic_repeat",
                    "method": method,
                    "protocol": protocol,
                    "path": str(repeat_paths[key]),
                    "sha256": _sha256(repeat_paths[key]),
                }
            )
        if key in summary_paths:
            sources.append(
                {
                    "role": "vlm_runtime_summary",
                    "method": method,
                    "protocol": protocol,
                    "path": str(summary_paths[key]),
                    "sha256": _sha256(summary_paths[key]),
                }
            )
    unpaired_auxiliary = sorted(
        (set(repeat_paths) | set(summary_paths) | set(memory_values))
        - observed_vlm_keys
    )
    if unpaired_auxiliary:
        raise EvaluationError(
            f"VLM repeat/summary/memory has no matching result: {unpaired_auxiliary}"
        )
    if formal_lock is not None:
        expected_vlm_keys = set(formal_vlm_contracts)
        if len(expected_vlm_keys) != 1:
            raise EvaluationError(
                "immutable lock must expose exactly one selected formal VLM"
            )
        if observed_vlm_keys != expected_vlm_keys:
            raise EvaluationError(
                "formal evaluation requires exactly the validation-selected "
                "raw VLM comparator"
            )
        if set(summary_paths) != expected_vlm_keys:
            raise EvaluationError(
                "formal VLM summary must be the selected guarded runtime summary"
            )

    predictions = (
        pd.concat(prediction_parts, ignore_index=True, sort=False)
        if prediction_parts
        else None
    )
    result = evaluate_predictions(
        candidates,
        predictions,
        sample_universe=universe,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        runtime_metrics=runtime,
    )
    if formal_lock is not None:
        expected_pairs = {
            (str(item["protocol"]), str(item["method"]))
            for item in formal_lock["evaluation_definition"].get(
                "formal_method_protocols", []
            )
        }
        observed_pairs = {
            (str(row.protocol), str(row.method))
            for row in result.per_method_metrics.itertuples(index=False)
        }
        if not expected_pairs or observed_pairs != expected_pairs:
            raise EvaluationError(
                "formal evaluated method/protocol set disagrees with experiment lock"
            )
    output_root = args.output_root.expanduser().resolve()
    bundle = write_evaluation_bundle(
        output_root,
        result,
        provenance={
            "sources": sources,
            "command": " ".join(
                map(shlex.quote, [sys.executable, *sys.argv])
            ),
            "test_labels_used_only_after_predictions": True,
            "candidate_identity_invariant_enforced": True,
            "expected_counts": {
                "all": args.expected_sample_count,
                "candidates": args.expected_candidate_count,
                "nonempty": args.expected_nonempty_count,
                "valid_empty": args.expected_valid_empty_count,
                "scenes": args.expected_scene_count,
            },
        },
    )
    (output_root / "run_command.txt").write_text(
        " ".join(map(shlex.quote, [sys.executable, *sys.argv])) + "\n",
        encoding="utf-8",
    )
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    bundle = run(parse_args(argv))
    print(json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
