from __future__ import annotations

import hashlib
import json
import os
import fcntl
import re
import threading
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from failure_analysis.gemini_crog_evidence_v1.environment import (
    load_private_env,
    validate_gemini_environment,
)

from .api import MODEL_IDS, PairwiseInteractionsRunner
from .dataset import (
    build_pair_manifests,
    cohort_rows,
    load_feature_index,
    load_label_index,
    smoke_sample,
    stratified_cohort_sample,
)
from .features import build_pair_evidence, ordered_candidates
from .ledger import PairwiseLedger
from .renderer import PerturbationVariant, render_pairwise_board
from .security import assert_frozen_pair


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "failure_analysis/reranking_outputs/v2_20260727T174412+0100"
TRAIN_FEATURES = V2_ROOT / "base_train/features.jsonl"
VALIDATION_FEATURES = V2_ROOT / "base_val/features.jsonl"
TRAIN_CORRECTED = V2_ROOT / "labels_train/corrected/labels.jsonl"
TRAIN_LEGACY = V2_ROOT / "labels_train/legacy_official/labels.jsonl"
VALIDATION_CORRECTED = V2_ROOT / "labels_val/corrected/labels.jsonl"
VALIDATION_LEGACY = V2_ROOT / "labels_val/legacy_official/labels.jsonl"
DEFAULT_COHORTS = REPO_ROOT / "runs/vlm_safe_rerank_20260803T112138Z/offline_replay/per_sample_cohorts.parquet"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary)
    temporary.replace(path)


def assert_audited_feature_source(run_dir: str | Path, feature_file: str | Path) -> str:
    """Hard-stop API work if a Stage-0 frozen feature file changed.

    The returned audit partition is security-relevant: callers must not infer a
    source's role from a user-controlled phase name.
    """

    root = Path(run_dir)
    audit_path = root / "audit_inventory.json"
    audit_identity_path = root / "AUDIT_INVENTORY_IDENTITY.json"
    if not audit_path.is_file():
        raise RuntimeError("Stage-0 audit inventory is required before API inference")
    if not audit_identity_path.is_file():
        raise RuntimeError("Stage-0 audit inventory has no frozen identity sidecar")
    audit_identity = json.loads(audit_identity_path.read_text(encoding="utf-8"))
    if hashlib.sha256(audit_path.read_bytes()).hexdigest() != audit_identity.get("sha256"):
        raise RuntimeError("Stage-0 audit inventory identity changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    resolved = Path(feature_file).resolve()
    matching = [
        (str(partition), payload)
        for partition, payload in audit.get("frozen_sources", {}).items()
        if isinstance(payload, Mapping)
        and Path(str(payload.get("path", ""))).resolve() == resolved
        and payload.get("file_sha256")
    ]
    if len(matching) != 1:
        raise RuntimeError("feature source is not uniquely frozen by Stage-0 audit")
    partition, payload = matching[0]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != payload["file_sha256"]:
        raise RuntimeError("frozen feature/candidate/q source identity changed")
    return partition


def _generic_runner_forbids_phase(phase: str) -> bool:
    """Treat every spelling/alias containing ``formal`` as formal execution."""

    return "formal" in re.sub(r"[^a-z0-9]+", "", str(phase).lower())


def acquire_global_api_lock(run_dir: str | Path):
    """Acquire the run-wide provider lock so phase-local budgets cannot race."""

    root = Path(run_dir)
    handle = (root / "API_RUNNER.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("another provider runner already owns this run") from exc
    active = getattr(_PROVIDER_LOCK_STATE, "handles", None)
    if active is not None:
        active.append(handle)
    return handle


_PROVIDER_LOCK_STATE = threading.local()


def release_provider_locks(function):
    """Ensure provider file locks close even when a long-lived caller catches errors."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = getattr(_PROVIDER_LOCK_STATE, "handles", None)
        handles: list[Any] = []
        _PROVIDER_LOCK_STATE.handles = handles
        try:
            return function(*args, **kwargs)
        finally:
            for handle in reversed(handles):
                try:
                    handle.close()
                except OSError:
                    pass
            _PROVIDER_LOCK_STATE.handles = previous

    return wrapped


def assert_phase_inference_manifest_frozen(phase_dir: str | Path) -> None:
    phase_path = Path(phase_dir)
    manifest_path = phase_path / "inference_manifest.json"
    identity_path = phase_path / "INFERENCE_MANIFEST_IDENTITY.json"
    if not identity_path.is_file():
        raise RuntimeError("phase inference manifest has no frozen identity sidecar")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if identity.get("sha256") != actual:
        raise RuntimeError("phase inference manifest identity changed")


def prepare_cohort(
    *,
    run_dir: str | Path,
    phase: str,
    per_cohort: Mapping[str, int] | None = None,
    all_challengers: bool,
    seed: int = 20260803,
    partition: str = "train",
) -> dict[str, Any]:
    """Evaluator process: write label-free inference and separate label manifests."""

    root = Path(run_dir); phase_dir = root / phase
    inference_path = phase_dir / "inference_manifest.json"
    evaluation_path = phase_dir / "evaluation_manifest.parquet"
    sample_path = phase_dir / "sample_manifest.json"
    if any(path.exists() for path in (inference_path, evaluation_path, sample_path)):
        raise FileExistsError(f"{phase} manifests are immutable once created")
    rows = cohort_rows(DEFAULT_COHORTS)
    samples = smoke_sample(rows, seed=seed) if phase == "smoke" else stratified_cohort_sample(
        rows, per_cohort=per_cohort or {"protected_correct": 30, "recoverable_error": 30, "unrecoverable_error": 30},
        partition=partition, seed=seed
    )
    sample_ids = {str(row["sample_id"]) for row in samples}
    features = load_feature_index(TRAIN_FEATURES, sample_ids)
    corrected = load_label_index(TRAIN_CORRECTED, sample_ids)
    legacy = load_label_index(TRAIN_LEGACY, sample_ids)
    inference, evaluation = build_pair_manifests(
        samples, features, corrected_labels=corrected, legacy_labels=legacy,
        all_challengers=all_challengers, maximum_challengers=1 if phase == "smoke" else 2,
    )
    phase_dir.mkdir(parents=True, exist_ok=True)
    _exclusive_json(sample_path, {
        "schema_version": "1.0.0", "phase": phase, "seed": seed,
        "samples": [{key: row[key] for key in ("sample_id", "corrected_cohort", "query_type", "frame_id", "scene_id", "group_id")} for row in samples],
    })
    _exclusive_json(inference_path, {
        "schema_version": "1.0.0", "phase": phase, "created_at_utc": _utc(),
        "feature_file": str(TRAIN_FEATURES.resolve()), "rows": inference,
    })
    _exclusive_json(phase_dir / "INFERENCE_MANIFEST_IDENTITY.json", {
        "sha256": hashlib.sha256(inference_path.read_bytes()).hexdigest(),
        "binding": "exact inference_manifest.json bytes",
    })
    _parquet(evaluation_path, evaluation)
    summary = {
        "phase": phase, "partition": partition, "samples": len(samples), "pairs": len(inference),
        "all_challengers": all_challengers,
        "cohorts": {name: sum(row["corrected_cohort"] == name for row in samples) for name in ("protected_correct", "recoverable_error", "unrecoverable_error")},
        "inference_manifest": str(inference_path.resolve()),
        "evaluation_manifest": str(evaluation_path.resolve()),
    }
    _atomic_json(phase_dir / "PREPARE_SUMMARY.json", summary)
    return summary


@release_provider_locks
def run_pairwise_phase(
    *,
    run_dir: str | Path,
    phase: str,
    env_file: str | Path,
    models: Sequence[str] = MODEL_IDS,
    protocol: str = "P4",
    variants: Sequence[PerturbationVariant] = (PerturbationVariant.ORIGINAL,),
    max_pairs: int | None = None,
    max_transport_retries: int = 3,
    circuit_breaker_terminal_failures: int = 5,
    replay_only: bool = False,
) -> dict[str, Any]:
    """Inference-only process. It never opens evaluation manifests or label paths."""

    root = Path(run_dir); phase_dir = root / phase
    if _generic_runner_forbids_phase(phase):
        # The generic development runner deliberately cannot consume formal
        # manifests.  Formal execution must enter through the signed, dual-gate
        # command so direct Python calls cannot bypass the experiment lock.
        from .locking import FormalRunDenied

        raise FormalRunDenied("generic pairwise runner cannot execute formal data")
    phase_dir.mkdir(parents=True, exist_ok=True)
    global_lock_handle = acquire_global_api_lock(root)
    lock_handle = (phase_dir / "RUNNER.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"another runner already owns phase {phase}") from exc
    _PROVIDER_LOCK_STATE.handles.append(lock_handle)
    _atomic_json(phase_dir / "RUNNER_STATE.json", {
        "pid": os.getpid(), "phase": phase, "started_at_utc": _utc(),
        "lock_file": str((phase_dir / "RUNNER.lock").resolve()),
    })
    assert_phase_inference_manifest_frozen(phase_dir)
    manifest = json.loads((phase_dir / "inference_manifest.json").read_text(encoding="utf-8"))
    source_partition = assert_audited_feature_source(root, manifest["feature_file"])
    if _generic_runner_forbids_phase(source_partition):
        from .locking import FormalRunDenied

        raise FormalRunDenied(
            "generic pairwise runner cannot consume the audited formal feature source"
        )
    planned_models = manifest.get("needed_models")
    if planned_models is not None and set(map(str, models)) != set(map(str, planned_models)):
        raise RuntimeError("requested models differ from the frozen phase model plan")
    rows = list(manifest["rows"])
    if not models:
        if rows:
            raise RuntimeError("a zero-model phase cannot contain planned provider requests")
        # Calibration can legitimately make every API method ineligible.  That
        # produces a frozen zero-request P5 plan and must complete without an
        # API key, budget environment, SDK client, or provider request.
        with PairwiseLedger(root / "pairwise_cache.sqlite") as ledger:
            ledger_summary = ledger.summary()
        _parquet(phase_dir / "pairwise_responses.parquet", [])
        summary = {
            "phase": phase,
            "protocol": protocol,
            "pair_rows": 0,
            "model_pair_variants": 0,
            "successful": 0,
            "fallback": 0,
            "schema_valid_rate": 1.0,
            "cache_hits": 0,
            "ledger": ledger_summary,
        }
        _atomic_json(phase_dir / "API_SUMMARY.json", summary)
        _atomic_json(phase_dir / "API_PROGRESS.json", {
            "phase": phase,
            "completed_model_pair_variants": 0,
            "planned_model_pair_variants": 0,
            "ledger": ledger_summary,
            "updated_at_utc": _utc(),
        })
        _atomic_json(phase_dir / "RUNNER_STATE.json", {
            "pid": os.getpid(),
            "phase": phase,
            "status": "completed",
            "completed_at_utc": _utc(),
            "lock_file": str((phase_dir / "RUNNER.lock").resolve()),
        })
        return summary
    load_private_env(env_file)
    environment = validate_gemini_environment()
    if max_pairs is not None:
        limit = int(max_pairs)
        if limit <= 0:
            raise ValueError("max_samples must be positive")
        chosen_samples: list[str] = []
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id not in chosen_samples:
                chosen_samples.append(sample_id)
            if len(chosen_samples) == limit:
                break
        chosen = set(chosen_samples)
        rows = [row for row in rows if str(row["sample_id"]) in chosen]
    sample_ids = {str(row["sample_id"]) for row in rows}
    features = load_feature_index(manifest["feature_file"], sample_ids)
    prompt_path = REPO_ROOT / (
        "prompts/pairwise_safe_p3_v1.txt" if protocol == "P3"
        else "prompts/pairwise_safe_v1.txt"
    )
    schema_path = REPO_ROOT / "prompts/pairwise_safe_v1.schema.json"
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_hash = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    ledger_path = root / "pairwise_cache.sqlite"
    records: list[dict[str, Any]] = []
    with PairwiseLedger(ledger_path) as ledger:
        for raw_path in (root / "raw_api").glob("**/*.json"):
            try:
                raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
                ledger.backfill_response_metadata(
                    str(raw_record["request_hash"]),
                    requested_model=raw_record.get("model_id"),
                    response_model=raw_record.get("response_model"),
                    api_request_id=raw_record.get("api_request_id"),
                )
            except (KeyError, OSError, ValueError, TypeError):
                continue
        ledger.normalize_conservative_attempt_costs(
            er2_reserve_usd=float(environment["gemini_er2_cost_cap_per_request_usd"]),
            flash_reserve_usd=0.10,
        )
        runner = PairwiseInteractionsRunner(
            ledger=ledger,
            max_spend_usd=float(environment["gemini_max_spend_usd"]),
            er2_request_reserve_usd=float(environment["gemini_er2_cost_cap_per_request_usd"]),
            max_transport_retries=max_transport_retries,
            replay_only=replay_only,
        )
        rendered_identity_cache: dict[
            tuple[str, str, str], tuple[dict[str, Any], dict[str, Any]]
        ] = {}
        evidence_cache: dict[
            tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
        ] = {}
        # Model-major queues prevent a rate-limited model from starving later
        # requests to the other model. Global concurrency remains one.
        for model in models:
            consecutive_terminal_failures = 0
            observed_response_models: set[str] = set()
            for row in rows:
                feature = features[str(row["sample_id"])]
                pair_key_base = (
                    str(row["sample_id"]), str(row["challenger_candidate_id"])
                )
                cached_evidence = evidence_cache.get(pair_key_base)
                evidence = (
                    cached_evidence[0]
                    if cached_evidence is not None
                    else build_pair_evidence(
                        feature, str(row["challenger_candidate_id"])
                    )
                )
                candidates = {str(item["candidate_id"]): item for item in ordered_candidates(feature)}
                baseline_expected = candidates[str(row["baseline_candidate_id"])]
                challenger_expected = candidates[str(row["challenger_candidate_id"])]
                assert_frozen_pair(
                    {"evidence": evidence},
                    baseline_id=str(row["baseline_candidate_id"]),
                    challenger_id=str(row["challenger_candidate_id"]),
                    baseline_expected=baseline_expected,
                    challenger_expected=challenger_expected,
                )
                if protocol == "P3":
                    minimal_evidence = {
                        "feature_schema_version": "pairwise_visual_p3_v1",
                        "sample_id": evidence["sample_id"],
                        "language_instruction": evidence["language_instruction"],
                        "baseline": {"candidate_id": evidence["baseline"]["candidate_id"]},
                        "challenger": {"candidate_id": evidence["challenger"]["candidate_id"]},
                    }
                    minimal_evidence["evidence_hash"] = hashlib.sha256(
                        json.dumps(
                            minimal_evidence, sort_keys=True, separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest()
                    request_evidence = minimal_evidence
                else:
                    request_evidence = evidence
                evidence_cache[pair_key_base] = (evidence, request_evidence)
                for variant in variants:
                    render_key = (*pair_key_base, variant.value)
                    cached_render = rendered_identity_cache.get(render_key)
                    board: bytes | None = None
                    if (
                        consecutive_terminal_failures >= circuit_breaker_terminal_failures
                        and cached_render is not None
                    ):
                        request_evidence, renderer_metadata = cached_render
                    else:
                        board, renderer_metadata = render_pairwise_board(
                            feature, str(row["challenger_candidate_id"]), variant=variant,
                            include_numeric=protocol != "P3",
                        )
                        rendered_identity_cache[render_key] = (
                            request_evidence, renderer_metadata
                        )
                    call = dict(
                        sample_id=str(row["sample_id"]), model_id=model, protocol=protocol,
                        baseline_candidate_id=str(row["baseline_candidate_id"]),
                        challenger_candidate_id=str(row["challenger_candidate_id"]), evidence=request_evidence,
                        prompt_hash=prompt_hash, schema_hash=schema_hash,
                        renderer_hash=str(renderer_metadata["renderer_contract_hash"]), perturbation_variant=variant.value,
                        board_sha256=str(renderer_metadata["image_sha256"]),
                    )
                    if consecutive_terminal_failures >= circuit_breaker_terminal_failures:
                        result = runner.record_circuit_breaker_fallback(**call)
                    else:
                        if board is None:
                            board, renderer_metadata = render_pairwise_board(
                                feature, str(row["challenger_candidate_id"]), variant=variant,
                                include_numeric=protocol != "P3",
                            )
                        result = runner.run(
                            **{key: value for key, value in call.items() if key != "board_sha256"},
                            board_png=board, system_prompt=prompt,
                            baseline_expected=None if protocol == "P3" else baseline_expected,
                            challenger_expected=None if protocol == "P3" else challenger_expected,
                        )
                        if result.status == "SUCCEEDED":
                            consecutive_terminal_failures = 0
                        elif not result.cache_hit:
                            consecutive_terminal_failures += 1
                    if result.response_model:
                        observed_response_models.add(str(result.response_model))
                        if len(observed_response_models) > 1 or str(result.response_model) != model:
                            drift = {
                                "phase": phase, "requested_model": model,
                                "observed_response_models": sorted(observed_response_models),
                                "sample_id": str(row["sample_id"]), "request_hash": result.request_hash,
                            }
                            _atomic_json(phase_dir / "MODEL_DRIFT.json", drift)
                            raise RuntimeError("provider model/version metadata drift detected")
                    parsed = result.parsed.model_dump(mode="json") if result.parsed else None
                    record = {
                        "sample_id": str(row["sample_id"]), "baseline_candidate_id": str(row["baseline_candidate_id"]),
                        "challenger_candidate_id": str(row["challenger_candidate_id"]), "model_id": model,
                        "protocol": protocol, "variant": variant.value, "request_hash": result.request_hash,
                        "status": result.status, "cache_hit": result.cache_hit, "api_attempts": result.api_attempts,
                        "fallback_reason": result.fallback_reason, "latency_seconds": result.latency_seconds,
                        "estimated_cost_usd": result.estimated_cost_usd, "parsed": parsed,
                        "response_model": result.response_model, "api_request_id": result.api_request_id,
                        "board_sha256": str(renderer_metadata["image_sha256"]),
                    }
                    records.append(record)
                    raw_path = root / "raw_api" / ("er2" if "robotics" in model else "flash") / f"{result.request_hash}.json"
                    if not raw_path.exists() and result.api_attempts > 0:
                        _exclusive_json(raw_path, {**record, "raw_response": result.raw_response, "usage": result.usage})
                    _atomic_json(phase_dir / "API_PROGRESS.json", {
                        "phase": phase, "completed_model_pair_variants": len(records),
                        "planned_model_pair_variants": len(rows) * len(models) * len(variants),
                        "ledger": ledger.summary(), "updated_at_utc": _utc(),
                    })
        ledger_summary = ledger.summary()
    _parquet(phase_dir / "pairwise_responses.parquet", records)
    summary = {
        "phase": phase, "protocol": protocol, "pair_rows": len(rows), "model_pair_variants": len(records),
        "successful": sum(row["status"] == "SUCCEEDED" for row in records),
        "fallback": sum(row["status"] != "SUCCEEDED" for row in records),
        "schema_valid_rate": sum(row["parsed"] is not None for row in records) / len(records) if records else 0.0,
        "cache_hits": sum(bool(row["cache_hit"]) for row in records), "ledger": ledger_summary,
    }
    _atomic_json(phase_dir / "API_SUMMARY.json", summary)
    _atomic_json(phase_dir / "RUNNER_STATE.json", {
        "pid": os.getpid(), "phase": phase, "status": "completed",
        "completed_at_utc": _utc(),
        "lock_file": str((phase_dir / "RUNNER.lock").resolve()),
    })
    lock_handle.close()
    global_lock_handle.close()
    return summary


def prepare_confirmation_phase(
    *, run_dir: str | Path, source_phase: str, destination_phase: str
) -> dict[str, Any]:
    """Create a label-free manifest only for pairs with a potential original switch."""

    root = Path(run_dir); source = root / source_phase; destination = root / destination_phase
    if destination.exists():
        raise FileExistsError("confirmation phase manifest is immutable")
    inference = json.loads((source / "inference_manifest.json").read_text(encoding="utf-8"))
    responses = pq.read_table(source / "pairwise_responses.parquet").to_pylist()
    potential = {
        (str(row["sample_id"]), str(row["challenger_candidate_id"]))
        for row in responses
        if row.get("parsed") and row["parsed"].get("decision") == "PREFER_CHALLENGER"
    }
    rows = [
        row for row in inference["rows"]
        if (str(row["sample_id"]), str(row["challenger_candidate_id"])) in potential
    ]
    destination.mkdir(parents=True, exist_ok=False)
    destination_manifest = destination / "inference_manifest.json"
    destination_payload = {
        "schema_version": "1.0.0", "phase": destination_phase,
        "source_phase": source_phase, "feature_file": inference["feature_file"], "rows": rows,
    }
    if "needed_models" in inference:
        destination_payload["needed_models"] = inference["needed_models"]
    _exclusive_json(destination_manifest, destination_payload)
    _exclusive_json(destination / "INFERENCE_MANIFEST_IDENTITY.json", {
        "sha256": hashlib.sha256(destination_manifest.read_bytes()).hexdigest(),
        "binding": "exact inference_manifest.json bytes",
    })
    summary = {"source_phase": source_phase, "destination_phase": destination_phase, "potential_switch_pairs": len(rows)}
    _atomic_json(destination / "PREPARE_SUMMARY.json", summary)
    return summary


def prepare_perturbation_phase(
    *, run_dir: str | Path, source_phase: str, destination_phase: str
) -> dict[str, Any]:
    """Freeze a label-free copy of a diagnostic manifest for shortcut probes."""

    root = Path(run_dir)
    source = root / source_phase
    destination = root / destination_phase
    if destination.exists():
        raise FileExistsError("perturbation phase manifest is immutable")
    assert_phase_inference_manifest_frozen(source)
    source_manifest_path = source / "inference_manifest.json"
    inference = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=False)
    destination_manifest = destination / "inference_manifest.json"
    _exclusive_json(
        destination_manifest,
        {
            "schema_version": "1.0.0",
            "phase": destination_phase,
            "source_phase": source_phase,
            "source_manifest_sha256": hashlib.sha256(
                source_manifest_path.read_bytes()
            ).hexdigest(),
            "feature_file": inference["feature_file"],
            "rows": inference["rows"],
        },
    )
    _exclusive_json(
        destination / "INFERENCE_MANIFEST_IDENTITY.json",
        {
            "sha256": hashlib.sha256(destination_manifest.read_bytes()).hexdigest(),
            "binding": "exact inference_manifest.json bytes",
        },
    )
    summary = {
        "source_phase": source_phase,
        "destination_phase": destination_phase,
        "pairs": len(inference["rows"]),
        "labels_opened": False,
        "planned_variants": [
            variant.value
            for variant in PerturbationVariant
            if variant is not PerturbationVariant.ORIGINAL
        ],
    }
    _atomic_json(destination / "PREPARE_SUMMARY.json", summary)
    return summary
