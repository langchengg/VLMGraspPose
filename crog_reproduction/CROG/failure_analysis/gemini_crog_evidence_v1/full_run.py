"""Resumable Phase C--H runner for the frozen CROG Gemini experiment.

This module intentionally keeps orchestration separate from the Phase-B CLI.
The durable units are evidence shards and one JSON decision per logical request;
both may be replayed after a process crash without issuing a duplicate paid
request.  The production backend uses :class:`GoogleInteractionsRunner`; tests
inject a request executor and never contact Gemini.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sqlite3
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import MODEL_IDS
from .ablation import ablation_renderer_hash, metadata_for_ablation, render_ablation_board
from .api import (
    DEFAULT_IMAGE_RESOLUTION,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_THINKING_LEVEL,
    BudgetGuard,
    GeminiCache,
    GoogleInteractionsRunner,
    request_hash as compute_request_hash,
    sha256_json,
)
from .audit import (
    EXPECTED_TEST_FEATURES_SHA256,
    PREEXISTING_V2_TREE_SHA256,
    audit_frozen_test_baseline,
    v2_tree_digest,
)
from .calibration import (
    LockedSafeThreshold,
    apply_locked_threshold,
    calibration_grid_payload,
    sweep_safe_thresholds,
)
from .environment import load_private_env, validate_gemini_environment
from .evaluation import (
    evaluate_saved_predictions,
    load_evaluation_index,
    paired_statistical_tests,
    select_validation_primary,
    stable_candidate_id,
)
from .finalize import finalize_experiment
from .gallery import build_evaluation_gallery
from .independent_recompute_gemini_results import recompute as independent_recompute
from .planner import (
    ABLATION_PROTOCOLS,
    MAX_PLANNED_REQUEST_SLOTS,
    P1_PROTOCOL,
    canonical_json,
    sha256_file,
    write_immutable_json,
)
from .protocol import (
    build_lock_payload,
    candidate_identity_stream_sha256,
    claim_formal_test_once,
    file_identity,
    git_diff_hash,
    lock_experiment,
    verify_experiment_lock,
    verify_lock_payload,
)
from .renderer import RENDERER_VERSION
from .run_state import FullRunState, RunAlreadyActiveError, atomic_write_json, utc_now
from .schema import response_json_schema
from .security import (
    assert_candidate_identity,
    assert_display_mapping,
    assert_no_gt_leak,
    assert_no_gt_values_in_provider_text,
    wrap_untrusted_referring_expression,
)
from .stability import compute_stability


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "failure_analysis/reranking_outputs/v2_20260727T174412+0100"
TEST_ROOT = REPO_ROOT / "failure_analysis/reranking_outputs/full_test_17749_v1"
DEFAULT_SYSTEM_PROMPT = REPO_ROOT / "prompts/crog_gemini_evidence_system_v1.txt"
DEFAULT_CROG_CONFIG = REPO_ROOT / "config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml"
DEFAULT_CHECKPOINT = REPO_ROOT / "exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth"
SDK_CONTRACT = "google-genai==2.16.0"
ENDPOINT_CONTRACT = "https://generativelanguage.googleapis.com/v1beta/interactions"
PHASES = ("pilot", "stability", "ablation", "calibration", "validation", "lock", "formal_test")
REQUEST_PHASES = tuple(phase for phase in PHASES if phase != "lock")
CANONICAL_COUNTS = {
    "pilot": 100,
    "stability": 20,
    "ablation": 500,
    "calibration": 9_790,
    "validation": 8_669,
    "formal_test": 17_749,
}
PROTOCOL_ALIASES = {
    "a0_visual_only": "A0",
    "a1_visual_plus_q": "A1",
    "a3_full_evidence_without_q": "A3",
}
TERMINAL_REQUEST_STATES = {
    "SUCCEEDED",
    "ABSTAIN",
    "TECHNICAL_FALLBACK",
    "PERMANENT_FAILED",
}


class FullRunError(RuntimeError):
    """Base error for a durable, inspectable full-run failure."""


class HardStopError(FullRunError):
    """A user-registered hard-stop condition; no new request may be sent."""

    def __init__(self, reason: str, *, phase_status: str = "blocked") -> None:
        super().__init__(reason)
        self.reason = reason
        self.phase_status = phase_status


class ResumeRequiredError(FullRunError):
    """A retryable interruption whose existing cache/decision files are safe."""


@dataclass(frozen=True)
class DatasetContract:
    features: Path
    legacy_labels: Path
    corrected_labels: Path
    exporter_split: str
    raw_predictions: Path | None = None


@dataclass
class FullRunConfig:
    run_root: Path
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT
    crog_config_path: Path = DEFAULT_CROG_CONFIG
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    v2_root: Path = V2_ROOT
    split_manifest_path: Path = V2_ROOT / "split_manifest.json"
    datasets: dict[str, DatasetContract] = field(default_factory=dict)
    private_env_path: Path | None = None
    evidence_shard_size: int = 64
    device: str = "auto"
    frozen_batch_size: int = 16
    mapping_seed: int = 47
    request_upper_bound_usd: float = 0.10
    bootstrap_draws: int = 10_000
    bootstrap_seed: int = 47
    minimum_response_coverage: float = 0.98
    max_cumulative_retryable_attempts: int | None = None
    delete_committed_boards: bool = False
    verify_environment: bool = True
    verify_repository_contract: bool = True
    enforce_canonical_counts: bool = True

    def __post_init__(self) -> None:
        self.run_root = Path(self.run_root).resolve()
        self.system_prompt_path = Path(self.system_prompt_path).resolve()
        self.crog_config_path = Path(self.crog_config_path).resolve()
        self.checkpoint_path = Path(self.checkpoint_path).resolve()
        self.v2_root = Path(self.v2_root).resolve()
        self.split_manifest_path = Path(self.split_manifest_path).resolve()
        if self.private_env_path is not None:
            self.private_env_path = Path(self.private_env_path).resolve()
        if self.evidence_shard_size < 1:
            raise ValueError("evidence_shard_size must be positive")
        if self.frozen_batch_size < 1:
            raise ValueError("frozen_batch_size must be positive")
        if self.bootstrap_draws < 1:
            raise ValueError("bootstrap_draws must be positive")
        if (
            self.max_cumulative_retryable_attempts is not None
            and self.max_cumulative_retryable_attempts < 1
        ):
            raise ValueError(
                "max cumulative retryable attempts must be a positive integer"
            )
        if not self.datasets:
            self.datasets = default_dataset_contracts()
        self.datasets = {
            name: DatasetContract(
                Path(value.features).resolve(),
                Path(value.legacy_labels).resolve(),
                Path(value.corrected_labels).resolve(),
                value.exporter_split,
                None if value.raw_predictions is None else Path(value.raw_predictions).resolve(),
            )
            for name, value in self.datasets.items()
        }
        if set(self.datasets) != {"train", "calibration", "validation", "test"}:
            raise ValueError("datasets must contain train, calibration, validation, and test")

    @property
    def plan_path(self) -> Path:
        return self.run_root / "full_run_plan.json"

    @property
    def cache_path(self) -> Path:
        return self.run_root / "gemini_cache.sqlite"


def default_dataset_contracts() -> dict[str, DatasetContract]:
    train = DatasetContract(
        V2_ROOT / "base_train/features.jsonl",
        V2_ROOT / "labels_train/legacy_official/labels.jsonl",
        V2_ROOT / "labels_train/corrected/labels.jsonl",
        "train",
        V2_ROOT / "base_train/predictions.jsonl",
    )
    return {
        "train": train,
        "calibration": train,
        "validation": DatasetContract(
            V2_ROOT / "base_val/features.jsonl",
            V2_ROOT / "labels_val/legacy_official/labels.jsonl",
            V2_ROOT / "labels_val/corrected/labels.jsonl",
            "val",
            V2_ROOT / "base_val/predictions.jsonl",
        ),
        "test": DatasetContract(
            TEST_ROOT / "features.jsonl",
            V2_ROOT / "formal_test_primary_v2/labels/legacy_official/labels.jsonl",
            V2_ROOT / "formal_test_primary_v2/labels/corrected/labels.jsonl",
            "test",
            TEST_ROOT / "predictions.jsonl",
        ),
    }


@dataclass
class FullRunDependencies:
    """Injection seam used by tests and offline audits.

    ``request_executor`` receives the exact keyword arguments otherwise passed
    to ``GoogleInteractionsRunner.run``.  A production run must leave it unset.
    ``materialize_ablation`` is only a test seam; the default calls the frozen
    A0/A1/A3 renderer and metadata builders.
    """

    export_evidence: Callable[..., dict[str, Any]] = field(default_factory=lambda: _default_export_evidence)
    request_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    materialize_ablation: Callable[..., dict[str, Any]] | None = None


def _default_export_evidence(**kwargs: Any) -> dict[str, Any]:
    # CROG replay imports the dataset/model stack (including optional native
    # packages).  Keep that dependency lazy so plan/lock/resume audits and the
    # fake-runner tests remain executable without loading the inference stack.
    from .exporter import export_frozen_crog_evidence

    return export_frozen_crog_evidence(**kwargs)


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _payload_digest(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        safe_rows = []
        for row in rows:
            safe_rows.append(
                {
                    key: (
                        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in dict(row).items()
                }
            )
        pq.write_table(pa.Table.from_pylist(safe_rows), temporary, compression="zstd")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _stable_feature_id(feature: Mapping[str, Any]) -> str:
    split = "val" if str(feature["split"]).lower() in {"val", "validation"} else str(feature["split"]).lower()
    return f"multiple:{split}:{int(feature['sample_id']):08d}"


def _q_only_candidate(candidates: Sequence[Mapping[str, Any]]) -> str:
    return str(min(candidates, key=lambda row: (-float(row["q_raw"]), str(row["candidate_id"])))["candidate_id"])


def _decision_key(phase: str, protocol: str, model: str, sample: str, replicate: int) -> str:
    return "|".join((phase, protocol, model, sample, str(int(replicate))))


def _decision_filename(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"


class FullRunOrchestrator:
    def __init__(self, config: FullRunConfig, dependencies: FullRunDependencies | None = None) -> None:
        self.config = config
        self.dependencies = dependencies or FullRunDependencies()
        self.plan: dict[str, Any] = {}
        self.manifests: dict[str, dict[str, Any]] = {}
        self._feature_indices: dict[str, dict[str, dict[str, Any]]] = {}
        self._candidate_identity_hashes: dict[str, dict[str, str]] = {}
        self._evaluation_indices: dict[str, dict[str, dict[str, Any]]] = {}
        self._raw_gt_grasps: dict[str, dict[str, list[Any]]] = {}
        self._system_prompt = ""
        self._prompt_hash = ""
        self._schema_hash = ""
        self._renderer_hash = ""
        self._evidence_schema_hash: str | None = None
        self._state: FullRunState | None = None
        self._cache: GeminiCache | None = None
        self._budget: BudgetGuard | None = None
        self._runner: GoogleInteractionsRunner | None = None
        self._provider_identity: dict[str, dict[str, str]] = {}
        self._disabled_models: dict[str, dict[str, Any]] = {}
        self._live_key_preflight_done = False
        self._requests_since_flush = 0
        self._runtime_accounted_monotonic = time.monotonic()
        self._active_wall_seconds: float | None = None
        self._progress_phase: str | None = None
        self._progress_phase_started_monotonic = self._runtime_accounted_monotonic
        self._progress_phase_baseline_completed = 0

    @property
    def state(self) -> FullRunState:
        if self._state is None:
            raise FullRunError("full-run state is unavailable")
        return self._state

    @property
    def cache(self) -> GeminiCache:
        if self._cache is None:
            raise FullRunError("Gemini cache is unavailable")
        return self._cache

    def _load_private_environment(self) -> dict[str, object]:
        if not self.config.verify_environment and self.config.private_env_path is None:
            return {"validation": "disabled_for_injected_test_backend"}
        env_path = self.config.private_env_path
        if env_path is None:
            candidate = REPO_ROOT / ".env"
            env_path = candidate if candidate.exists() else None
        if env_path is not None:
            load_private_env(env_path)
            if self.config.verify_repository_contract:
                check = subprocess.run(
                    ["git", "check-ignore", "--quiet", str(env_path)],
                    cwd=REPO_ROOT,
                    check=False,
                )
                if check.returncode != 0:
                    raise HardStopError("private_env_is_not_git_ignored", phase_status="blocked_credentials")
        if not self.config.verify_environment:
            return {"validation": "disabled_for_injected_test_backend"}
        return validate_gemini_environment()

    def _validate_plan_and_manifests(self) -> None:
        self.plan = _read_json(self.config.plan_path)
        if self.plan.get("status") != "ready" or self.plan.get("blockers"):
            raise HardStopError("full_run_plan_not_ready", phase_status="blocked_plan")
        if self.plan.get("content_sha256") != _payload_digest(self.plan):
            raise HardStopError("full_run_plan_hash_mismatch", phase_status="blocked_plan")
        if list(self.plan.get("model_ids", [])) != list(MODEL_IDS):
            raise HardStopError("full_run_plan_model_identity_changed", phase_status="blocked_identity")
        if list(self.plan.get("phase_order", [])) != list(REQUEST_PHASES):
            raise HardStopError("full_run_plan_phase_order_changed", phase_status="blocked_plan")
        totals = self.plan.get("totals", {})
        if int(totals.get("planned_request_slots", -1)) > MAX_PLANNED_REQUEST_SLOTS:
            raise HardStopError("planned_request_cap_exceeded", phase_status="blocked_plan")
        planned = {str(row["phase"]): row for row in self.plan.get("phases", [])}
        if set(planned) != set(REQUEST_PHASES):
            raise HardStopError("full_run_plan_phase_set_changed", phase_status="blocked_plan")
        for phase in REQUEST_PHASES:
            manifest = _read_json(self.config.run_root / f"{phase}_manifest.json")
            if manifest.get("content_sha256") != _payload_digest(manifest):
                raise HardStopError(f"{phase}_manifest_hash_mismatch", phase_status="blocked_plan")
            rows = manifest.get("rows")
            if not isinstance(rows, list) or int(manifest.get("sample_count", -1)) != len(rows):
                raise HardStopError(f"{phase}_manifest_count_mismatch", phase_status="blocked_plan")
            if self.config.enforce_canonical_counts and len(rows) != CANONICAL_COUNTS[phase]:
                raise HardStopError(f"{phase}_canonical_count_changed", phase_status="blocked_identity")
            if int(planned[phase].get("sample_count", -1)) != len(rows):
                raise HardStopError(f"{phase}_plan_manifest_count_mismatch", phase_status="blocked_plan")
            if str(planned[phase].get("sample_manifest_sha256")) != str(manifest["content_sha256"]):
                raise HardStopError(f"{phase}_plan_manifest_hash_mismatch", phase_status="blocked_plan")
            for row in rows:
                if "evaluation_only" not in row:
                    raise HardStopError(f"{phase}_manifest_missing_evaluation_partition", phase_status="blocked_plan")
                if not str(row.get("sample_id", "")):
                    raise HardStopError(f"{phase}_manifest_missing_sample_id", phase_status="blocked_plan")
                request_projection = {key: value for key, value in row.items() if key != "evaluation_only"}
                assert_no_gt_leak(request_projection, path=f"{phase}_manifest_request_projection")
            self.manifests[phase] = manifest

    def _secret_scan(self) -> dict[str, Any]:
        # Search generic Gemini-key shapes, never the live value.  Only a count
        # is persisted so neither a matched secret nor identifying fragments
        # can reach logs/reports.
        command = [
            "rg", "--files-with-matches", "--text", "--hidden",
            "--glob", "!.git/**", "--glob", "!.env", "--glob", "!*.png",
            "--glob", "!*.parquet", "(?:AIza[0-9A-Za-z_-]{20,}|AQ\\.[0-9A-Za-z_-]{20,})",
            str(REPO_ROOT),
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        if completed.returncode not in {0, 1}:
            raise FullRunError("secret scan utility failed")
        count = len([line for line in completed.stdout.splitlines() if line.strip()])
        if count:
            raise HardStopError("api_key_secret_scan_nonzero", phase_status="blocked_credentials")
        return {"generic_key_pattern_match_file_count": 0}

    def _validate_technical_contract(self) -> dict[str, Any]:
        required_paths = [
            self.config.system_prompt_path,
            self.config.crog_config_path,
            self.config.checkpoint_path,
            self.config.split_manifest_path,
            *(value for dataset in self.config.datasets.values() for value in (
                dataset.features, dataset.legacy_labels, dataset.corrected_labels
            )),
            *(
                dataset.raw_predictions
                for dataset in self.config.datasets.values()
                if dataset.raw_predictions is not None
            ),
        ]
        missing = sorted(str(path) for path in required_paths if not Path(path).is_file())
        if missing:
            raise HardStopError(f"required_artifact_missing:{missing[0]}", phase_status="blocked_identity")
        self._system_prompt = self.config.system_prompt_path.read_text(encoding="utf-8")
        self._prompt_hash = hashlib.sha256(self._system_prompt.encode("utf-8")).hexdigest()
        self._schema_hash = sha256_json(response_json_schema())
        self._renderer_hash = sha256_file(Path(__file__).with_name("renderer.py"))
        smoke_dir = self.config.run_root / "smoke_evidence_final"
        smoke_summary = _read_json(smoke_dir / "export_summary.json") if smoke_dir.exists() else {}
        smoke_schema = smoke_dir / "evidence_schema.json"
        if smoke_summary:
            expectations = {
                "system_prompt_sha256": self._prompt_hash,
                "checkpoint_sha256": sha256_file(self.config.checkpoint_path),
                "config_sha256": sha256_file(self.config.crog_config_path),
            }
            for name, expected in expectations.items():
                if str(smoke_summary.get(name)) != expected:
                    raise HardStopError(f"frozen_{name}_changed", phase_status="blocked_identity")
        if smoke_schema.exists():
            self._evidence_schema_hash = sha256_file(smoke_schema)
        if self.config.cache_path.exists() and self.config.cache_path.stat().st_size:
            connection = sqlite3.connect(f"file:{self.config.cache_path}?mode=ro", uri=True)
            try:
                tables = {
                    str(row[0]) for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if "responses" in tables:
                    observed = connection.execute(
                        "SELECT DISTINCT prompt_hash, schema_hash, renderer_hash, "
                        "evidence_schema_hash, protocol_id FROM responses"
                    ).fetchall()
                    for prompt_hash, schema_hash, renderer_hash, evidence_hash, protocol_id in observed:
                        expected_renderer = (
                            ablation_renderer_hash()
                            if protocol_id in set(ABLATION_PROTOCOLS) - {P1_PROTOCOL}
                            else self._renderer_hash
                        )
                        expected = (self._prompt_hash, self._schema_hash, expected_renderer, self._evidence_schema_hash)
                        if tuple(map(str, (prompt_hash, schema_hash, renderer_hash, evidence_hash))) != tuple(map(str, expected)):
                            raise HardStopError("phase_b_cache_technical_hash_changed", phase_status="blocked_identity")
            finally:
                connection.close()
        if self.config.verify_repository_contract:
            try:
                sdk_version = importlib.metadata.version("google-genai")
            except importlib.metadata.PackageNotFoundError as exc:
                raise HardStopError("google_genai_sdk_missing", phase_status="blocked_environment") from exc
            if sdk_version != "2.16.0":
                raise HardStopError("google_genai_sdk_version_changed", phase_status="blocked_identity")
            observed_v2 = v2_tree_digest(self.config.v2_root)
            if observed_v2 != PREEXISTING_V2_TREE_SHA256:
                raise HardStopError("preexisting_v2_tree_changed", phase_status="blocked_identity")
            test = self.config.datasets["test"]
            if sha256_file(test.features) != EXPECTED_TEST_FEATURES_SHA256:
                raise HardStopError("frozen_test_features_changed", phase_status="blocked_identity")
            baseline = audit_frozen_test_baseline(
                features_path=test.features,
                legacy_labels_path=test.legacy_labels,
                corrected_labels_path=test.corrected_labels,
            )
        else:
            sdk_version = "injected-test-backend"
            baseline = {"candidate_identity_stream_sha256": "not_checked_in_test"}
        return {
            "status": "passed",
            "model_ids": list(MODEL_IDS),
            "sdk": sdk_version,
            "endpoint": ENDPOINT_CONTRACT,
            "prompt_sha256": self._prompt_hash,
            "response_schema_sha256": self._schema_hash,
            "renderer_sha256": self._renderer_hash,
            "checkpoint_sha256": sha256_file(self.config.checkpoint_path),
            "crog_config_sha256": sha256_file(self.config.crog_config_path),
            "split_manifest_sha256": sha256_file(self.config.split_manifest_path),
            "v2_tree_sha256": PREEXISTING_V2_TREE_SHA256,
            "candidate_identity_stream_sha256": baseline["candidate_identity_stream_sha256"],
            "store": False,
            "background": False,
            "stream": False,
            "thinking_level": DEFAULT_THINKING_LEVEL,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "temperature_policy": "model_default",
            "api_key": "SET" if os.environ.get("GEMINI_API_KEY") else "INJECTED_TEST_BACKEND",
        }

    def preflight(self) -> dict[str, Any]:
        if (self.config.run_root / ".API_KEY_ROTATION_REQUIRED").exists():
            raise HardStopError("api_key_rotation_required", phase_status="blocked_credentials")
        if (self.config.run_root / "CACHE_QUARANTINE_GT_LEAK.json").exists():
            raise HardStopError(
                "gt_leak_cache_quarantine_active",
                phase_status="blocked_contamination",
            )
        environment = self._load_private_environment()
        self._validate_plan_and_manifests()
        if self.config.verify_environment:
            assumptions = self.plan.get("assumptions", {})
            bindings = {
                "gemini_max_spend_usd": "max_spend_usd",
                "gemini_max_concurrency": "concurrency",
                "gemini_er2_cost_cap_per_request_usd": "er2_cost_cap_per_request_usd",
            }
            for environment_name, plan_name in bindings.items():
                observed = environment.get(environment_name)
                expected = assumptions.get(plan_name)
                if observed is None or expected is None or float(observed) != float(expected):
                    raise HardStopError(
                        f"frozen_plan_environment_binding_changed:{plan_name}",
                        phase_status="blocked_plan",
                    )
            if int(environment["gemini_max_concurrency"]) != 1:
                raise HardStopError(
                    "full_runner_requires_frozen_serial_concurrency",
                    phase_status="blocked_plan",
                )
            reserve = (
                (self.plan.get("totals", {}).get("expected_cost") or {}).get(
                    "total_budget_reserve_usd"
                )
            )
            if reserve is None or float(reserve) > float(environment["gemini_max_spend_usd"]):
                raise HardStopError(
                    "full_plan_conservative_reserve_exceeds_budget",
                    phase_status="blocked_budget",
                )
        contract = self._validate_technical_contract()
        secret_scan = self._secret_scan()
        safe_environment = {
            key: value for key, value in environment.items() if key != "gemini_api_key"
        }
        result = {
            **contract,
            "environment": safe_environment,
            "secret_scan": secret_scan,
            "plan_sha256": self.plan["content_sha256"],
            "checked_at_utc": utc_now(),
        }
        atomic_write_json(self.config.run_root / "full_run_preflight.json", result)
        return result

    def _open_runtime(self) -> None:
        self._cache = GeminiCache(self.config.cache_path)
        spent, counts = self.cache.budget_state()
        if self.config.verify_environment:
            max_spend = float(os.environ["GEMINI_MAX_SPEND_USD"])
            er2_cap = float(os.environ["GEMINI_ER2_COST_CAP_PER_REQUEST_USD"])
        else:
            max_spend = float(self.plan.get("assumptions", {}).get("max_spend_usd") or 1_000_000.0)
            er2_cap = float(self.plan.get("assumptions", {}).get("er2_cost_cap_per_request_usd") or 1.0)
        self._budget = BudgetGuard(
            max_spend,
            already_estimated_usd=spent,
            request_counts=counts,
            er2_cost_cap_per_request_usd=er2_cap,
        )
        if self.dependencies.request_executor is None:
            self._runner = GoogleInteractionsRunner(
                cache=self.cache,
                budget=self._budget,
                owner_id=f"full-run:{self.state.run_id}:pid-{os.getpid()}",
                max_cumulative_retryable_attempts=(
                    self.config.max_cumulative_retryable_attempts
                ),
            )

    def _close_runtime(self) -> None:
        if self._cache is not None:
            self._cache.close()
        self._cache = None
        self._runner = None

    def _feature_index(self, partition: str) -> dict[str, dict[str, Any]]:
        if partition not in self._feature_indices:
            result = {}
            identities = {}
            for row in _read_jsonl(self.config.datasets[partition].features):
                sample_id = _stable_feature_id(row)
                if sample_id in result:
                    raise HardStopError("duplicate_frozen_feature_identity", phase_status="blocked_identity")
                identities[sample_id] = assert_candidate_identity(row["candidates"])
                result[sample_id] = row
            self._feature_indices[partition] = result
            self._candidate_identity_hashes[partition] = identities
        return self._feature_indices[partition]

    def _evaluation_index(self, partition: str) -> dict[str, dict[str, Any]]:
        if partition not in self._evaluation_indices:
            dataset = self.config.datasets[partition]
            self._evaluation_indices[partition] = load_evaluation_index(
                features_path=dataset.features,
                legacy_labels_path=dataset.legacy_labels,
                corrected_labels_path=dataset.corrected_labels,
            )
        return self._evaluation_indices[partition]

    def _raw_gt_grasp_index(self, partition: str) -> dict[str, list[Any]]:
        if partition not in self._raw_gt_grasps:
            path = self.config.datasets[partition].raw_predictions
            if path is None or not path.is_file():
                self._raw_gt_grasps[partition] = {}
            else:
                self._raw_gt_grasps[partition] = {
                    _stable_feature_id(row): list(row.get("gt_grasps") or [])
                    for row in _read_jsonl(path)
                }
        return self._raw_gt_grasps[partition]

    def _assert_request_gt_value_isolation(
        self,
        *,
        partition: str,
        sample_id: str,
        provider_text: str,
    ) -> str:
        if self.dependencies.request_executor is not None and not self.config.verify_repository_contract:
            return "provenance_only_in_injected_test"
        truth = self._evaluation_index(partition)[sample_id]
        correct_ids = {
            candidate_id
            for track in ("legacy_by_candidate_id", "corrected_by_candidate_id")
            for candidate_id, correct in truth[track].items()
            if bool(correct)
        }
        grasps = self._raw_gt_grasp_index(partition).get(sample_id, [])
        assert_no_gt_values_in_provider_text(
            provider_text,
            correct_candidate_ids=correct_ids,
            gt_grasps=grasps,
        )
        return "passed_identifier_tuple_and_provenance_audit"

    def _phase_evaluation_index(self, phase: str) -> dict[str, dict[str, Any]]:
        """Return evaluator truth with preregistered frame and scene/group clusters."""

        partition = str(self.manifests[phase]["partition"])
        base = self._evaluation_index(partition)
        result: dict[str, dict[str, Any]] = {}
        for manifest_row in self.manifests[phase]["rows"]:
            sample_id = str(manifest_row["sample_id"])
            if sample_id not in base:
                raise HardStopError("evaluation_sample_missing", phase_status="blocked_identity")
            row = dict(base[sample_id])
            row["frame_id"] = str(manifest_row.get("frame_id") or row["frame_id"])
            row["scene_id"] = str(
                manifest_row.get("group_id")
                or manifest_row.get("scene_id")
                or row["scene_id"]
            )
            result[sample_id] = row
        return result

    def _valid_evidence_source(self, directory: Path) -> bool:
        required = (
            directory / "export_summary.json",
            directory / "request_manifest.jsonl",
            directory / "candidate_evidence.parquet",
            directory / "sample_evidence.parquet",
            directory / "evidence_schema.json",
        )
        if not all(path.is_file() for path in required):
            return False
        try:
            summary = _read_json(directory / "export_summary.json")
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return summary.get("status") == "complete"

    def _evidence_sources(self) -> list[Path]:
        # The final Phase-B evidence is first by design: those ten boards and
        # maps are the byte-identical source for the 20 resolved smoke rows.
        sources = []
        smoke = self.config.run_root / "smoke_evidence_final"
        if self._valid_evidence_source(smoke):
            sources.append(smoke)
        for phase in REQUEST_PHASES:
            root = self.config.run_root / phase / "evidence"
            if not root.exists():
                continue
            for directory in sorted(root.glob("shard_*")):
                if self._valid_evidence_source(directory):
                    sources.append(directory)
        for directory in (
            self.config.run_root / "pilot" / "evidence_new_90",
            self.config.run_root / "ablation" / "evidence_new_400",
        ):
            if self._valid_evidence_source(directory):
                sources.append(directory)
        return sources

    def _evidence_catalog(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for source in self._evidence_sources():
            summary = _read_json(source / "export_summary.json")
            expected_summary = {
                "checkpoint_sha256": sha256_file(self.config.checkpoint_path),
                "config_sha256": sha256_file(self.config.crog_config_path),
                "system_prompt_sha256": self._prompt_hash,
                "no_ground_truth_forward": True,
                "forward_identity_max_difference": 0.0,
            }
            for name, expected in expected_summary.items():
                if summary.get(name) != expected:
                    raise HardStopError(
                        f"evidence_provenance_changed:{name}",
                        phase_status="blocked_identity",
                    )
            schema_hash = sha256_file(source / "evidence_schema.json")
            if self._evidence_schema_hash is None:
                self._evidence_schema_hash = schema_hash
            elif schema_hash != self._evidence_schema_hash:
                raise HardStopError("evidence_schema_hash_changed", phase_status="blocked_identity")
            sample_rows = {
                str(row["sample_id"]): row
                for row in pq.read_table(source / "sample_evidence.parquet").to_pylist()
            }
            candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in pq.read_table(source / "candidate_evidence.parquet").to_pylist():
                candidates[str(row["sample_id"])].append(row)
            for request in _read_jsonl(source / "request_manifest.jsonl"):
                sample_id = str(request["sample_id"])
                if sample_id in result:
                    previous = result[sample_id]["request"]
                    stable_fields = ("board_sha256", "metadata", "mapping")
                    if any(previous.get(name) != request.get(name) for name in stable_fields):
                        raise HardStopError("replayed_evidence_changed", phase_status="blocked_identity")
                    continue
                if sample_id not in sample_rows or len(candidates[sample_id]) != 5:
                    raise HardStopError("evidence_candidate_count_changed", phase_status="blocked_identity")
                result[sample_id] = {
                    "source": source,
                    "request": request,
                    "sample": sample_rows[sample_id],
                    "candidate_evidence": sorted(
                        candidates[sample_id], key=lambda row: int(row["original_q_rank"])
                    ),
                }
        return result

    def _assert_evidence_binding(
        self,
        *,
        feature: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> None:
        sample_id = str(evidence["request"]["sample_id"])
        candidates = list(feature["candidates"])
        by_id = {str(row["candidate_id"]): row for row in candidates}
        evidence_by_id = {
            str(row["candidate_id"]): row for row in evidence["candidate_evidence"]
        }
        if set(by_id) != set(evidence_by_id) or len(by_id) != 5:
            raise HardStopError("evidence_candidate_identity_changed", phase_status="blocked_identity")
        mapping = evidence["request"]["mapping"]
        assert_display_mapping(mapping, candidate_ids=list(by_id))
        expected_order = sorted(
            by_id,
            key=lambda candidate_id: (-float(by_id[candidate_id]["q_raw"]), candidate_id),
        )
        if str(evidence["sample"]["original_q_top1_candidate_id"]) != expected_order[0]:
            raise HardStopError("evidence_q_top1_changed", phase_status="blocked_identity")
        for rank, candidate_id in enumerate(expected_order):
            frozen = by_id[candidate_id]
            observed = evidence_by_id[candidate_id]
            expected_scalars = {
                "candidate_checksum": str(frozen["candidate_checksum"]),
                "center_x_px": float(frozen["cx"]),
                "center_y_px": float(frozen["cy"]),
                "angle_deg_periodic_180": float(frozen["angle_deg"]),
                "width_px": float(frozen["width_px"]),
                "fixed_height_px": float(frozen["height_px"]),
                "q_original_value": float(frozen["q_raw"]),
                "original_q_rank": rank,
                "display_candidate_id": str(mapping["candidate_to_display"][candidate_id]),
                "stable_candidate_id": stable_candidate_id(sample_id, candidate_id),
            }
            for name, expected in expected_scalars.items():
                value = observed.get(name)
                if isinstance(expected, float):
                    matches = value is not None and math.isclose(
                        float(value), expected, rel_tol=0.0, abs_tol=1e-6
                    )
                else:
                    matches = value == expected
                if not matches:
                    raise HardStopError(
                        f"evidence_frozen_candidate_field_changed:{name}",
                        phase_status="blocked_identity",
                    )
            if not np.allclose(
                np.asarray(observed["rectangle_corners"], dtype=np.float64),
                np.asarray(frozen["polygon"], dtype=np.float64),
                rtol=0.0,
                atol=1e-5,
            ):
                raise HardStopError(
                    "evidence_frozen_candidate_field_changed:rectangle_corners",
                    phase_status="blocked_identity",
                )
        board = Path(str(evidence["request"]["board_path"]))
        layers_path = board.with_suffix(".layers.json")
        if not layers_path.is_file():
            raise HardStopError("evidence_board_layer_audit_missing", phase_status="blocked_identity")
        layers = _read_json(layers_path)
        if (
            layers.get("evaluation_overlay_included") is not False
            or layers.get("candidate_mapping") != mapping
            or str(layers.get("sample_id")) != sample_id
            or str(layers.get("image_sha256")) != str(evidence["request"]["board_sha256"])
        ):
            raise HardStopError("evidence_board_layer_audit_changed", phase_status="blocked_identity")

    def _export_shard(
        self,
        *,
        phase: str,
        partition: str,
        shard_index: int,
        local_ids: Sequence[int],
        keep_dense_maps: bool,
    ) -> Path:
        phase_root = self.config.run_root / phase / "evidence"
        phase_root.mkdir(parents=True, exist_ok=True)
        destination = phase_root / f"shard_{shard_index:06d}"
        if self._valid_evidence_source(destination):
            return destination
        if destination.exists():
            quarantine = destination.with_name(
                destination.name + ".incomplete." + utc_now().replace(":", "").replace("-", "")
            )
            os.replace(destination, quarantine)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=phase_root))
        # exporter insists on creating its output directory itself
        temporary.rmdir()
        dataset = self.config.datasets[partition]
        try:
            summary = self.dependencies.export_evidence(
                split=dataset.exporter_split,
                selected_local_ids=list(map(int, local_ids)),
                frozen_features_path=dataset.features,
                output_dir=temporary,
                system_prompt_path=self.config.system_prompt_path,
                config_path=self.config.crog_config_path,
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device,
                frozen_batch_size=self.config.frozen_batch_size,
                mapping_seed=self.config.mapping_seed,
                keep_dense_maps=keep_dense_maps,
            )
            if summary.get("status") != "complete" or int(summary.get("sample_count", -1)) != len(local_ids):
                raise FullRunError("evidence exporter returned an incomplete shard")
            if int(summary.get("candidate_count", -1)) != 5 * len(local_ids):
                raise HardStopError("exported_candidate_count_changed", phase_status="blocked_identity")
            os.replace(temporary, destination)
            self._relocate_evidence_paths(destination, previous_root=temporary)
        except Exception:
            # Preserve the partial directory as audit evidence.  A later resume
            # uses another unique temporary path and never mistakes it for a
            # complete evidence shard.
            raise
        return destination

    @staticmethod
    def _relocate_evidence_paths(destination: Path, *, previous_root: Path) -> None:
        """Rewrite exporter-owned absolute paths after the atomic directory rename."""

        old_prefix = str(previous_root.resolve()) + os.sep
        new_prefix = str(destination.resolve()) + os.sep

        def relocate(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(old_prefix):
                return new_prefix + value[len(old_prefix) :]
            if isinstance(value, dict):
                return {key: relocate(child) for key, child in value.items()}
            if isinstance(value, list):
                return [relocate(child) for child in value]
            return value

        request_jsonl = destination / "request_manifest.jsonl"
        request_rows = [relocate(row) for row in _read_jsonl(request_jsonl)]
        _atomic_text(
            request_jsonl,
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in request_rows),
        )
        for name in ("request_manifest.parquet", "sample_evidence.parquet"):
            path = destination / name
            if path.exists():
                rows = [relocate(row) for row in pq.read_table(path).to_pylist()]
                _atomic_parquet(path, rows)

    def _ensure_evidence(self, phase: str) -> dict[str, dict[str, Any]]:
        manifest = self.manifests[phase]
        partition = str(manifest["partition"])
        requested = [str(row["sample_id"]) for row in manifest["rows"]]
        features = self._feature_index(partition)
        missing_features = [sample for sample in requested if sample not in features]
        if missing_features:
            raise HardStopError("manifest_candidate_identity_missing", phase_status="blocked_identity")
        catalog = self._evidence_catalog()
        for shard_index, start in enumerate(range(0, len(requested), self.config.evidence_shard_size)):
            shard_samples = requested[start : start + self.config.evidence_shard_size]
            missing = [sample for sample in shard_samples if sample not in catalog]
            if not missing:
                continue
            # A3 is the only post-smoke protocol needing dense maps.  Pilot
            # maps are retained, so ablation reuses all previously exported
            # development evidence and replays only genuinely missing rows.
            self._export_shard(
                phase=phase,
                partition=partition,
                shard_index=shard_index,
                local_ids=[int(features[sample]["sample_id"]) for sample in missing],
                keep_dense_maps=phase in {"pilot", "stability", "ablation"},
            )
            catalog = self._evidence_catalog()
        missing = [sample for sample in requested if sample not in catalog]
        if missing:
            raise FullRunError(f"evidence catalog remains incomplete for {len(missing)} samples")
        selected = {sample: catalog[sample] for sample in requested}
        for sample_id, item in selected.items():
            self._assert_evidence_binding(feature=features[sample_id], evidence=item)
        index_payload = {
            "schema_version": "1.0",
            "phase": phase,
            "manifest_sha256": manifest["content_sha256"],
            "sample_count": len(selected),
            "rows": [
                {
                    "sample_id": sample,
                    "source": str(selected[sample]["source"].resolve()),
                    "board_sha256": selected[sample]["request"]["board_sha256"],
                }
                for sample in requested
            ],
        }
        atomic_write_json(self.config.run_root / phase / "evidence_index.json", index_payload)
        return selected

    def _materialize_ablation_request(
        self,
        *,
        phase: str,
        protocol: str,
        feature: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        if protocol == P1_PROTOCOL:
            return dict(evidence["request"])
        if self.dependencies.materialize_ablation is not None:
            return self.dependencies.materialize_ablation(
                phase=phase, protocol=protocol, feature=feature, evidence=evidence
            )
        short = PROTOCOL_ALIASES[protocol]
        sample_id = str(evidence["request"]["sample_id"])
        output = self.config.run_root / phase / "boards" / short / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.png"
        mapping = evidence["request"]["mapping"]
        candidate_evidence = evidence["candidate_evidence"]
        image_bgr = cv2.imread(str(feature["image_path"]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(feature["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        map_path = evidence["sample"].get("maps_path")
        maps: dict[str, Any] = {}
        if short == "A3":
            if not map_path or not Path(map_path).is_file():
                raise FullRunError("A3 requires retained frozen dense maps")
            with np.load(map_path) as archive:
                maps = {
                    "mask_probability": archive["m_probability"].astype(np.float32),
                    "sin_2theta": archive["sin_2theta"].astype(np.float32),
                    "cos_2theta": archive["cos_2theta"].astype(np.float32),
                    "width_probability": archive["w_probability"].astype(np.float32),
                }
        layer_manifest = render_ablation_board(
            protocol=short,
            rgb=image_rgb,
            candidates=feature["candidates"],
            candidate_evidence=candidate_evidence,
            mapping=mapping,
            sample_id=sample_id,
            output_path=output,
            **maps,
        )
        metadata = metadata_for_ablation(
            protocol=short,
            referring_expression_block=wrap_untrusted_referring_expression(feature["language_instruction"]),
            candidates=feature["candidates"],
            candidate_evidence=candidate_evidence,
            mapping=mapping,
            original_q_top1_candidate_id=evidence["sample"]["original_q_top1_candidate_id"],
        )
        result = {
            "sample_id": sample_id,
            "frame_id": evidence["request"]["frame_id"],
            "board_path": str(output.resolve()),
            "board_sha256": layer_manifest["image_sha256"],
            "metadata": metadata,
            "mapping": mapping,
            "system_prompt_sha256": self._prompt_hash,
            "renderer_version": layer_manifest["renderer_version"],
            "store": False,
            "background": False,
            "stream": False,
        }
        assert_no_gt_leak({"metadata": metadata, "mapping": mapping, "layer_manifest": layer_manifest})
        return result

    def _request_kwargs(
        self,
        *,
        request: Mapping[str, Any],
        model_id: str,
        protocol: str,
        replicate_id: int,
    ) -> dict[str, Any]:
        # Plain P1 deliberately omits the new namespace/protocol dimensions:
        # this preserves the Phase-B v1 request hash and its resolved cache.
        normal_p1 = protocol == P1_PROTOCOL and replicate_id == 0
        renderer_hash = self._renderer_hash if protocol == P1_PROTOCOL else ablation_renderer_hash()
        return {
            "sample_id": str(request["sample_id"]),
            "frame_id": str(request["frame_id"]),
            "model_id": model_id,
            "image_path": request["board_path"],
            "system_instruction": self._system_prompt,
            "metadata_prompt": request["metadata"],
            "mapping": request["mapping"],
            "prompt_hash": self._prompt_hash,
            "schema_hash": self._schema_hash,
            "renderer_hash": renderer_hash,
            "evidence_schema_hash": str(self._evidence_schema_hash),
            "model_metadata": {
                "requested_model_id": model_id,
                "sdk": SDK_CONTRACT,
                "api_version": "v1beta",
            },
            "request_upper_bound_usd": self.config.request_upper_bound_usd,
            "image_resolution": DEFAULT_IMAGE_RESOLUTION,
            "thinking_level": DEFAULT_THINKING_LEVEL,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "protocol_id": None if normal_p1 else protocol,
            "replicate_id": None if replicate_id == 0 else replicate_id,
            "namespace": None if normal_p1 else ("stability" if replicate_id else "inference"),
            "service_tier": "standard",
        }

    @staticmethod
    def _expected_request_hash(kwargs: Mapping[str, Any]) -> str:
        image_sha256 = sha256_file(Path(str(kwargs["image_path"])))
        generation_config = {
            "thinking_level": str(kwargs["thinking_level"]),
            "max_output_tokens": int(kwargs["max_output_tokens"]),
            "image_resolution": str(kwargs["image_resolution"]),
        }
        return compute_request_hash(
            model_id=str(kwargs["model_id"]),
            model_metadata=dict(kwargs["model_metadata"]),
            prompt_hash=str(kwargs["prompt_hash"]),
            schema_hash=str(kwargs["schema_hash"]),
            renderer_hash=str(kwargs["renderer_hash"]),
            evidence_schema_hash=str(kwargs["evidence_schema_hash"]),
            image_sha256=image_sha256,
            candidate_mapping=dict(kwargs["mapping"]),
            serialized_metadata=str(kwargs["metadata_prompt"]),
            generation_config=generation_config,
            protocol_id=kwargs.get("protocol_id"),
            replicate_id=kwargs.get("replicate_id"),
            namespace=kwargs.get("namespace"),
        )

    def _verify_resumed_decision(
        self,
        *,
        phase: str,
        protocol: str,
        replicate_id: int,
        model_id: str,
        sample_id: str,
        feature: Mapping[str, Any],
        request: Mapping[str, Any],
        kwargs: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> None:
        expected_fields = {
            "logical_request_key": _decision_key(
                phase, protocol, model_id, sample_id, replicate_id
            ),
            "phase": phase,
            "protocol": protocol,
            "replicate_id": replicate_id,
            "sample_id": sample_id,
            "model_id": model_id,
            "candidate_identity_sha256": assert_candidate_identity(feature["candidates"]),
            "candidate_mapping_sha256": sha256_json(request["mapping"]),
            "board_sha256": str(request["board_sha256"]),
        }
        for name, expected in expected_fields.items():
            if decision.get(name) != expected:
                raise HardStopError(
                    f"resumed_decision_binding_changed:{name}",
                    phase_status="blocked_identity",
                )
        candidate_ids = {str(row["candidate_id"]) for row in feature["candidates"]}
        if str(decision.get("selected_candidate_id")) not in candidate_ids:
            raise HardStopError("resumed_decision_selected_outside_top5", phase_status="blocked_identity")
        if self.dependencies.request_executor is None:
            expected_hash = self._expected_request_hash(kwargs)
            if str(decision.get("request_hash")) != expected_hash:
                raise HardStopError(
                    "resumed_decision_request_hash_changed",
                    phase_status="blocked_identity",
                )
            cached = self.cache.get(expected_hash)
            state = self.cache.request_state(expected_hash)
            uncertain = bool(
                state
                and state.get("status") == "TECHNICAL_FALLBACK"
                and state.get("last_error") == "uncertain_provider_completion_after_crash"
            )
            if cached is None and not uncertain:
                raise HardStopError(
                    "resumed_decision_cache_receipt_missing",
                    phase_status="blocked_identity",
                )
        if phase == "formal_test":
            lock_path = self._formal_lock_path()
            lock = _read_json(lock_path)
            bindings = {
                "formal_run_id": self.state.run_id,
                "formal_lock_sha256": lock["lock_sha256"],
                "formal_lock_file_sha256": sha256_file(lock_path),
            }
            for name, expected in bindings.items():
                if decision.get(name) != expected:
                    raise HardStopError(
                        f"formal_decision_binding_changed:{name}",
                        phase_status="blocked_lock",
                    )

    def _execute_request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        assert_no_gt_leak(
            {
                "system_instruction": kwargs["system_instruction"],
                "prompt": kwargs["metadata_prompt"],
                "mapping": kwargs["mapping"],
            }
        )
        assert_display_mapping(kwargs["mapping"])
        if self.dependencies.request_executor is not None:
            return dict(self.dependencies.request_executor(dict(kwargs)))
        if self._runner is None:
            raise FullRunError("production request runner is unavailable")
        return self._runner.run(**kwargs)

    @staticmethod
    def _parsed_output(result: Mapping[str, Any]) -> dict[str, Any] | None:
        parsed = result.get("parsed_output")
        if isinstance(parsed, dict):
            return parsed
        raw = result.get("parsed_output_json")
        if isinstance(raw, str) and raw:
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        return None

    def _normalize_decision(
        self,
        *,
        phase: str,
        protocol: str,
        replicate_id: int,
        model_id: str,
        feature: Mapping[str, Any],
        request: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        mapping = request["mapping"]
        candidate_ids = [str(row["candidate_id"]) for row in feature["candidates"]]
        candidate_identity = assert_candidate_identity(feature["candidates"])
        assert_display_mapping(mapping, candidate_ids=candidate_ids)
        q_only = _q_only_candidate(feature["candidates"])
        parsed = self._parsed_output(result)
        valid = bool(result.get("valid", False)) and parsed is not None
        json_parse_valid = parsed is not None
        if not json_parse_valid and isinstance(result.get("raw_output"), str):
            try:
                json_parse_valid = isinstance(json.loads(str(result["raw_output"])), dict)
            except json.JSONDecodeError:
                json_parse_valid = False
        abstain = bool(result.get("abstain", False)) or bool(parsed and parsed.get("decision") == "abstain")
        if valid and not abstain:
            display = str(parsed["selected_candidate_id"])
            if display not in mapping["display_to_candidate"]:
                raise HardStopError("response_selected_outside_display_mapping", phase_status="blocked_identity")
            selected = str(mapping["display_to_candidate"][display])
        else:
            display = str(mapping["candidate_to_display"][q_only])
            selected = q_only
        if selected not in candidate_ids:
            raise HardStopError("response_selected_outside_frozen_top5", phase_status="blocked_identity")
        ranking = []
        overall = 0.0
        selected_reason_codes: list[str] = []
        global_reason_codes: list[str] = []
        if parsed:
            ranking = [str(item["candidate_id"]) for item in parsed.get("ranking", [])]
            global_reason_codes = [str(value) for value in parsed.get("global_reason_codes", [])]
            selected_assessment = next(
                (item for item in parsed.get("ranking", []) if item.get("candidate_id") == parsed.get("selected_candidate_id")),
                None,
            )
            if selected_assessment is not None:
                overall = float(selected_assessment["overall_score"])
                selected_reason_codes = [
                    str(value) for value in selected_assessment.get("reason_codes", [])
                ]
        if len(ranking) != 5 or set(ranking) != {"A", "B", "C", "D", "E"}:
            # A deterministic q-order fallback keeps stability tables complete
            # while status/valid explicitly identify the technical fallback.
            by_id = {str(row["candidate_id"]): row for row in feature["candidates"]}
            ordered = sorted(candidate_ids, key=lambda value: (-float(by_id[value]["q_raw"]), value))
            ranking = [str(mapping["candidate_to_display"][value]) for value in ordered]
        lifecycle = str(result.get("lifecycle_status", "TECHNICAL_FALLBACK"))
        formal_binding: dict[str, Any] = {}
        if phase == "formal_test":
            lock_path = self._formal_lock_path()
            lock = _read_json(lock_path)
            formal_binding = {
                "formal_run_id": self.state.run_id,
                "formal_lock_sha256": lock["lock_sha256"],
                "formal_lock_file_sha256": sha256_file(lock_path),
            }
        return {
            "logical_request_key": _decision_key(phase, protocol, model_id, str(request["sample_id"]), replicate_id),
            "phase": phase,
            "protocol": protocol,
            "replicate_id": int(replicate_id),
            "sample_id": str(request["sample_id"]),
            "frame_id": str(request["frame_id"]),
            "model_id": model_id,
            "request_hash": str(result.get("request_hash", "")),
            "candidate_identity_sha256": candidate_identity,
            "candidate_mapping_sha256": sha256_json(mapping),
            "board_sha256": str(request["board_sha256"]),
            "request_id": result.get("request_id"),
            "lifecycle_status": lifecycle,
            "status": str(result.get("status", "fallback")),
            "valid": valid,
            "json_parse_valid": json_parse_valid,
            "schema_valid": valid,
            "abstain": abstain,
            "technical_fallback": lifecycle == "TECHNICAL_FALLBACK",
            "permanent_api_failure": lifecycle == "PERMANENT_FAILED",
            "fallback_reason": result.get("fallback_reason"),
            "selected_display_id": display,
            "selected_candidate_id": selected,
            "selected_stable_candidate_id": stable_candidate_id(str(request["sample_id"]), selected),
            "q_only_candidate_id": q_only,
            "q_only_stable_candidate_id": stable_candidate_id(str(request["sample_id"]), q_only),
            "decision": "abstain" if abstain else (str(parsed.get("decision")) if parsed else "technical_fallback"),
            "ranking": ranking,
            "global_reason_codes": global_reason_codes,
            "selected_reason_codes": selected_reason_codes,
            "reason_codes": sorted(set(global_reason_codes + selected_reason_codes)),
            "confidence": float(parsed.get("confidence", 0.0)) if parsed else 0.0,
            "score_margin_top1_top2": float(parsed.get("score_margin_top1_top2", 0.0)) if parsed else 0.0,
            "selected_overall_score": overall,
            "latency_seconds": float(result.get("latency_seconds") or 0.0),
            "retry_count": int(result.get("retry_count") or 0),
            "cache_hit": bool(result.get("cache_hit", False)),
            "api_attempted": bool(
                result.get("api_attempted", not bool(result.get("cache_hit", False)))
            ),
            "usage": dict(result.get("usage") or {}),
            "estimated_charge_usd": float(result.get("estimated_charge_usd") or 0.0),
            "response_model": result.get("response_model") or (result.get("model_metadata") or {}).get("response_model"),
            "service_tier": result.get("service_tier") or (result.get("model_metadata") or {}).get("service_tier"),
            "model_metadata": dict(result.get("model_metadata") or {}),
            "completed_at_utc": utc_now(),
            **formal_binding,
        }

    def _check_hard_stop_result(self, phase: str, decision: Mapping[str, Any]) -> None:
        fallback = str(decision.get("fallback_reason") or "")
        if fallback in {"http_401", "http_403", "missing_api_key"}:
            raise HardStopError("gemini_credentials_rejected", phase_status="blocked_credentials")
        if fallback in {"budget_would_be_exceeded", "missing_budget_smoke_limit"} or fallback.startswith(
            "budget_would_be_exceeded_"
        ):
            raise HardStopError("gemini_budget_would_be_exceeded", phase_status="blocked_budget")
        if decision.get("status") == "in_flight":
            raise ResumeRequiredError("request is owned by another active lease")
        if decision.get("lifecycle_status") == "RETRYABLE_FAILED":
            raise ResumeRequiredError("retryable request remains unresolved")
        if phase != "formal_test":
            return
        if (
            str(decision.get("model_id")) == MODEL_IDS[0]
            and fallback == "http_404"
        ):
            marker = {
                "status": "blocked_provider_model_unavailable",
                "model_id": MODEL_IDS[0],
                "first_unavailable_sample_id": decision["sample_id"],
                "first_unavailable_request_hash": decision["request_hash"],
                "fallback_reason": fallback,
                "replacement_model_forbidden": True,
                "flash_may_continue": True,
                "consensus_missing_er2_fallback": "crog_q_only",
                "detected_at_utc": utc_now(),
            }
            existing = self.config.run_root / "formal_test" / "er2_unavailable.json"
            if existing.is_file() and _read_json(existing) != marker:
                prior = _read_json(existing)
                marker = prior
            else:
                atomic_write_json(existing, marker)
            self._disabled_models[MODEL_IDS[0]] = marker
            return
        identity = {
            key: str(decision[key])
            for key in ("response_model", "service_tier")
            if decision.get(key) is not None
        }
        if not bool(decision.get("api_attempted", True)):
            return
        model_id = str(decision["model_id"])
        if bool(decision.get("valid")) and "response_model" not in identity:
            evidence = {
                "model_id": model_id,
                "sample_id": decision["sample_id"],
                "request_hash": decision["request_hash"],
                "missing_required_metadata": ["response_model"],
                "detected_at_utc": utc_now(),
            }
            atomic_write_json(
                self.config.run_root / "formal_test" / "provider_metadata_missing.json",
                evidence,
            )
            raise HardStopError(
                "provider_model_metadata_missing",
                phase_status="blocked_provider_drift",
            )
        if not identity:
            return
        expected = self._provider_identity.get(model_id)
        identity_path = (
            self.config.run_root
            / "formal_test"
            / "provider_identity"
            / f"{hashlib.sha256(model_id.encode()).hexdigest()}.json"
        )
        if expected is None:
            self._provider_identity[model_id] = identity
            write_immutable_json(
                identity_path,
                {
                    "schema_version": "1.0",
                    "model_id": model_id,
                    "identity": identity,
                    "first_sample_id": decision["sample_id"],
                    "first_request_hash": decision["request_hash"],
                    "limitation": "provider_hidden_build_changes_are_not_observable",
                },
            )
            atomic_write_json(
                self.config.run_root / "formal_test" / "provider_identity.json",
                {"models": self._provider_identity, "limitation": "provider hidden build changes are not observable"},
            )
        elif identity != expected:
            ranges = _read_json(
                self.config.run_root / "formal_test" / "provider_identity_ranges.json"
            ) if (self.config.run_root / "formal_test" / "provider_identity_ranges.json").is_file() else {}
            evidence = {
                "model_id": model_id,
                "expected": expected,
                "observed": identity,
                "accepted_range": ranges.get(model_id),
                "sample_id": decision["sample_id"],
                "request_hash": decision["request_hash"],
                "detected_at_utc": utc_now(),
            }
            atomic_write_json(self.config.run_root / "formal_test" / "provider_drift.json", evidence)
            raise HardStopError("provider_model_metadata_drift", phase_status="blocked_provider_drift")
        ranges_path = self.config.run_root / "formal_test" / "provider_identity_ranges.json"
        ranges = _read_json(ranges_path) if ranges_path.is_file() else {}
        model_range = dict(ranges.get(model_id, {}))
        model_range.setdefault("first_sample_id", decision["sample_id"])
        model_range.setdefault("first_request_hash", decision["request_hash"])
        model_range.update(
            {
                "last_sample_id": decision["sample_id"],
                "last_request_hash": decision["request_hash"],
                "identity_file_sha256": sha256_file(identity_path),
                "updated_at_utc": utc_now(),
            }
        )
        ranges[model_id] = model_range
        atomic_write_json(ranges_path, ranges)

    def _disabled_model_result(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        model_id = str(kwargs["model_id"])
        marker = self._disabled_models[model_id]
        return {
            "request_hash": self._expected_request_hash(kwargs),
            "status": "provider_model_unavailable",
            "lifecycle_status": "TECHNICAL_FALLBACK",
            "valid": False,
            "abstain": False,
            "fallback_reason": "provider_model_unavailable_after_http_404",
            "cache_hit": False,
            "api_attempted": False,
            "retry_count": 0,
            "latency_seconds": 0.0,
            "usage": {},
            "estimated_charge_usd": 0.0,
            "model_metadata": {
                "requested_model_id": model_id,
                "blocked_at_sample_id": marker["first_unavailable_sample_id"],
            },
        }

    def _decision_path(self, phase: str, key: str) -> Path:
        return self.config.run_root / phase / "decisions" / _decision_filename(key)

    def _write_decision_once(self, phase: str, decision: Mapping[str, Any]) -> None:
        path = self._decision_path(phase, str(decision["logical_request_key"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(decision)
        content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            if path.read_text(encoding="utf-8") != content:
                raise HardStopError("logical_request_decision_changed", phase_status="blocked_identity")

    def _load_phase_decisions(self, phase: str) -> list[dict[str, Any]]:
        root = self.config.run_root / phase / "decisions"
        return sorted(
            (_read_json(path) for path in root.glob("*.json")) if root.exists() else [],
            key=lambda row: str(row["logical_request_key"]),
        )

    def _update_request_progress(self, phase: str, total: int) -> None:
        decisions = self._load_phase_decisions(phase)
        completed = len(decisions)
        counts = Counter(str(row["model_id"]) for row in decisions)
        cache_hits = sum(bool(row.get("cache_hit")) for row in decisions)
        decision_retries = sum(int(row.get("retry_count", 0)) for row in decisions)
        decision_api_attempts = sum(
            0
            if bool(row.get("cache_hit")) or not bool(row.get("api_attempted", True))
            else 1 + int(row.get("retry_count", 0))
            for row in decisions
        )
        attempt_statistics = self.cache.attempt_statistics()
        # Production progress is cumulative across the experiment and comes
        # from the append-only SQLite attempt ledger, so retry history survives
        # runner recreation. Injected test executors have no ledger entries and
        # retain the decision-derived compatibility path.
        retries = (
            attempt_statistics["retryable_attempts"]
            if attempt_statistics["api_attempts"]
            else decision_retries
        )
        new_api_attempts = (
            attempt_statistics["api_attempts"]
            if attempt_statistics["api_attempts"]
            else decision_api_attempts
        )
        failures = sum(bool(row.get("permanent_api_failure")) for row in decisions)
        latencies = sorted(float(row.get("latency_seconds", 0.0)) for row in decisions)
        p50 = float(np.percentile(latencies, 50)) if latencies else None
        p95 = float(np.percentile(latencies, 95)) if latencies else None
        lifecycle_counts = Counter(str(row.get("lifecycle_status", "UNKNOWN")) for row in decisions)
        successful_times = sorted(
            str(row["completed_at_utc"])
            for row in decisions
            if row.get("completed_at_utc")
            and not bool(row.get("cache_hit"))
            and str(row.get("lifecycle_status")) in {"SUCCEEDED", "ABSTAIN"}
        )
        prior_api = _read_json(self.state.paths.api) if self.state.paths.api.is_file() else {}
        last_successful = successful_times[-1] if successful_times else prior_api.get(
            "last_successful_request_timestamp"
        )
        self.state.update_phase(phase, completed=completed, total=total)
        self.state.update_progress(
            "api",
            {
                "current_phase": phase,
                "completed": completed,
                "total": total,
                "er2_completed": counts.get("gemini-robotics-er-2-preview", 0),
                "flash_completed": counts.get("gemini-3.6-flash", 0),
                "models": dict(counts),
                "lifecycle_counts": dict(lifecycle_counts),
                "cache_hits": cache_hits,
                "new_api_attempts": new_api_attempts,
                "retries": retries,
                "permanent_failures": failures,
                "last_successful_request_timestamp": last_successful,
            },
        )
        usage = Counter()
        for row in decisions:
            request_usage = row.get("usage") or {}
            usage.update(
                input=int(request_usage.get("total_input_tokens", 0) or 0),
                output=int(request_usage.get("total_output_tokens", 0) or 0),
                thought=int(request_usage.get("total_thought_tokens", 0) or 0),
            )
        if self._budget is not None:
            remaining = None
            if self._budget.max_spend_usd is not None:
                remaining = (
                    self._budget.max_spend_usd
                    - self._budget.estimated_spend_usd
                    - self._budget.outstanding_request_reserve_usd
                )
            next_upper = max(
                self.config.request_upper_bound_usd,
                float(self._budget.er2_cost_cap_per_request_usd or 0.0),
            )
            self.state.update_progress(
                "cost",
                {
                    "current_phase": phase,
                    "max_spend_usd": self._budget.max_spend_usd,
                    "actual_estimated_spend_usd": self._budget.estimated_spend_usd,
                    "outstanding_request_reserve_usd": self._budget.outstanding_request_reserve_usd,
                    "next_request_upper_bound_usd": next_upper,
                    "remaining_budget_usd": remaining,
                    "token_usage": dict(usage),
                },
            )
        now_monotonic = time.monotonic()
        if self._active_wall_seconds is None:
            existing_runtime = _read_json(self.state.paths.runtime) if self.state.paths.runtime.is_file() else {}
            self._active_wall_seconds = float(existing_runtime.get("wall_time_seconds", 0.0) or 0.0)
        self._active_wall_seconds += max(0.0, now_monotonic - self._runtime_accounted_monotonic)
        self._runtime_accounted_monotonic = now_monotonic
        if self._progress_phase != phase:
            self._progress_phase = phase
            self._progress_phase_started_monotonic = now_monotonic
            self._progress_phase_baseline_completed = completed
        phase_elapsed = max(0.0, now_monotonic - self._progress_phase_started_monotonic)
        phase_progress = max(0, completed - self._progress_phase_baseline_completed)
        requests_per_hour = phase_progress * 3600.0 / phase_elapsed if phase_elapsed > 0 else 0.0
        eta_seconds = (
            (total - completed) * 3600.0 / requests_per_hour
            if requests_per_hour > 0 and completed < total
            else (0.0 if completed >= total else None)
        )
        self.state.update_progress(
            "runtime",
            {
                "current_phase": phase,
                "wall_time_seconds": self._active_wall_seconds,
                "p50_latency_seconds": p50,
                "p95_latency_seconds": p95,
                "requests_per_hour": requests_per_hour,
                "eta_seconds": eta_seconds,
                "active_worker_pids": [os.getpid()],
            },
        )

    def _planned_request_dimensions(self, phase: str) -> list[tuple[str, int]]:
        if phase == "stability":
            return [(P1_PROTOCOL, replicate) for replicate in (1, 2, 3)]
        if phase == "ablation":
            return [(protocol, 0) for protocol in ABLATION_PROTOCOLS]
        return [(P1_PROTOCOL, 0)]

    def _run_request_phase(self, phase: str) -> list[dict[str, Any]]:
        manifest = self.manifests[phase]
        partition = str(manifest["partition"])
        features = self._feature_index(partition)
        evidence = self._ensure_evidence(phase)
        dimensions = self._planned_request_dimensions(phase)
        total = len(manifest["rows"]) * len(dimensions) * len(MODEL_IDS)
        self._update_request_progress(phase, total)
        for manifest_row in manifest["rows"]:
            sample_id = str(manifest_row["sample_id"])
            feature = features[sample_id]
            frozen_candidates = feature["candidates"]
            expected_identity = self._candidate_identity_hashes[partition][sample_id]
            # Check the frozen row again immediately before request material is
            # built.  This catches any in-process geometry/q/order mutation;
            # the file-backed identity is separately reverified at formal lock.
            if assert_candidate_identity(feature["candidates"], expected_sha256=expected_identity) != expected_identity:
                raise HardStopError("candidate_identity_changed", phase_status="blocked_identity")
            for protocol, replicate_id in dimensions:
                request = self._materialize_ablation_request(
                    phase=phase,
                    protocol=protocol,
                    feature=feature,
                    evidence=evidence[sample_id],
                )
                assert_display_mapping(
                    request["mapping"],
                    candidate_ids=[row["candidate_id"] for row in frozen_candidates],
                )
                assert_no_gt_leak({"metadata": request["metadata"], "mapping": request["mapping"]})
                board_path = Path(request["board_path"])
                if not board_path.is_file():
                    raise FullRunError(f"request board is missing for {sample_id}")
                if sha256_file(board_path) != str(request["board_sha256"]):
                    raise HardStopError("request_board_sha256_changed", phase_status="blocked_identity")
                try:
                    gt_value_audit = self._assert_request_gt_value_isolation(
                        partition=partition,
                        sample_id=sample_id,
                        provider_text=self._system_prompt + "\n" + str(request["metadata"]),
                    )
                except ValueError as exc:
                    atomic_write_json(
                        self.config.run_root / "CACHE_QUARANTINE_GT_LEAK.json",
                        {
                            "status": "quarantined",
                            "phase": phase,
                            "sample_id": sample_id,
                            "reason": "pre_send_gt_value_leak_assertion_failed",
                            "cache_responses_must_not_be_used_until_audited": True,
                            "detected_at_utc": utc_now(),
                        },
                    )
                    raise HardStopError(
                        "gt_request_leak_detected",
                        phase_status="blocked_contamination",
                    ) from exc
                for model_id in MODEL_IDS:
                    key = _decision_key(phase, protocol, model_id, sample_id, replicate_id)
                    path = self._decision_path(phase, key)
                    kwargs = self._request_kwargs(
                        request=request,
                        model_id=model_id,
                        protocol=protocol,
                        replicate_id=replicate_id,
                    )
                    if path.exists():
                        decision = _read_json(path)
                        self._verify_resumed_decision(
                            phase=phase,
                            protocol=protocol,
                            replicate_id=replicate_id,
                            model_id=model_id,
                            sample_id=sample_id,
                            feature=feature,
                            request=request,
                            kwargs=kwargs,
                            decision=decision,
                        )
                        self._check_hard_stop_result(phase, decision)
                        continue
                    result = (
                        self._disabled_model_result(kwargs)
                        if phase == "formal_test" and model_id in self._disabled_models
                        else self._execute_request(kwargs)
                    )
                    decision = self._normalize_decision(
                        phase=phase,
                        protocol=protocol,
                        replicate_id=replicate_id,
                        model_id=model_id,
                        feature=feature,
                        request=request,
                        result=result,
                    )
                    decision["gt_value_audit"] = gt_value_audit
                    if decision.get("lifecycle_status") == "RETRYABLE_FAILED":
                        # Persist the complete ledger count before the outer
                        # automatic resume loop releases this runner instance.
                        self._update_request_progress(phase, total)
                    self._check_hard_stop_result(phase, decision)
                    if not self._live_key_preflight_done and not decision["cache_hit"]:
                        # A non-auth provider response proves the live rotated
                        # key before the rest of the paid manifest is consumed.
                        atomic_write_json(
                            self.config.run_root / "live_key_preflight.json",
                            {
                                "status": "passed",
                                "sample_id": sample_id,
                                "model_id": model_id,
                                "request_hash": decision["request_hash"],
                                "lifecycle_status": decision["lifecycle_status"],
                                "checked_at_utc": utc_now(),
                            },
                        )
                        self._live_key_preflight_done = True
                    self._write_decision_once(phase, decision)
                    self._requests_since_flush += 1
                    if self._requests_since_flush >= 25:
                        if self.dependencies.request_executor is None:
                            self.cache.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
                        self._update_request_progress(phase, total)
                        self._requests_since_flush = 0
                if self.config.delete_committed_boards and phase in {"calibration", "validation"}:
                    all_terminal = all(
                        self._decision_path(
                            phase,
                            _decision_key(phase, protocol, model, sample_id, replicate_id),
                        ).exists()
                        for model in MODEL_IDS
                    )
                    if all_terminal:
                        board = Path(request["board_path"])
                        # Phase-B audit boards and formal gallery inputs are
                        # intentionally retained.  Only generated phase-local
                        # boards are eligible for cleanup.
                        try:
                            board.relative_to(self.config.run_root / phase)
                        except ValueError:
                            pass
                        else:
                            board.unlink(missing_ok=True)
                            board.with_suffix(".layers.json").unlink(missing_ok=True)
        decisions = self._load_phase_decisions(phase)
        if len(decisions) != total:
            raise ResumeRequiredError(f"{phase} has {len(decisions)}/{total} terminal decisions")
        if self.dependencies.request_executor is None:
            self.cache.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
        self._update_request_progress(phase, total)
        _atomic_parquet(self.config.run_root / phase / "per_model_decisions.parquet", decisions)
        if phase in {"pilot", "ablation", "calibration", "validation", "formal_test"}:
            audited = []
            for row in decisions[: min(100, len(decisions))]:
                sample_id = str(row["sample_id"])
                feature = features[sample_id]
                identity = assert_candidate_identity(
                    feature["candidates"],
                    expected_sha256=self._candidate_identity_hashes[partition][sample_id],
                )
                audited.append(
                    {
                        "logical_request_key": row["logical_request_key"],
                        "request_hash": row["request_hash"],
                        "sample_id": sample_id,
                        "model_id": row["model_id"],
                        "candidate_identity_sha256": identity,
                        "no_gt_leak_assertion": "passed_before_request",
                        "display_mapping_assertion": "passed_before_request",
                        "candidate_identity_assertion": "passed_before_request_and_response",
                        "gt_value_audit": row.get("gt_value_audit", "passed_before_request"),
                    }
                )
            atomic_write_json(
                self.config.run_root / phase / "request_safety_audit.json",
                {
                    "schema_version": "1.0",
                    "status": "passed",
                    "phase": phase,
                    "population_request_count": len(decisions),
                    "audited_request_count": len(audited),
                    "audit_selection": "lexicographically_first_terminal_logical_request_keys",
                    "all_requests_received_identical_pre_send_assertions": True,
                    "provider_visible_metadata_and_hashes_persisted_for_audit": True,
                    "ground_truth_not_persisted_with_request_or_cache": True,
                    "gt_value_audit_scope": "GT candidate identities and multi-value grasp tuples; scalar equality handled by source/layer provenance",
                    "rows": audited,
                    "completed_at_utc": utc_now(),
                },
            )
        return decisions

    @staticmethod
    def _usage_summary(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        summary: dict[str, Counter] = defaultdict(Counter)
        for row in decisions:
            model = str(row["model_id"])
            usage = row.get("usage") or {}
            summary[model].update(
                {
                    "requests": 1,
                    "cache_hits": int(bool(row.get("cache_hit"))),
                    "input_tokens": int(usage.get("total_input_tokens", 0) or 0),
                    "output_tokens": int(usage.get("total_output_tokens", 0) or 0),
                    "thought_tokens": int(usage.get("total_thought_tokens", 0) or 0),
                    "retries": int(row.get("retry_count", 0) or 0),
                }
            )
        return [{"model_id": model, **dict(values)} for model, values in sorted(summary.items())]

    @staticmethod
    def _agreement(decisions: Sequence[Mapping[str, Any]], *, protocol: str = P1_PROTOCOL) -> dict[str, Any]:
        grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in decisions:
            if str(row["protocol"]) == protocol and int(row.get("replicate_id", 0)) == 0:
                grouped[str(row["sample_id"])][str(row["model_id"])] = row
        comparable = agreements = 0
        disagreements = []
        for sample_id, by_model in sorted(grouped.items()):
            if set(by_model) != set(MODEL_IDS):
                continue
            comparable += 1
            selected = {str(row["selected_candidate_id"]) for row in by_model.values()}
            if len(selected) == 1:
                agreements += 1
            else:
                disagreements.append(sample_id)
        return {
            "comparable_samples": comparable,
            "selected_id_agreement_count": agreements,
            "selected_id_agreement_rate": agreements / comparable if comparable else None,
            "disagreement_sample_ids": disagreements,
        }

    @staticmethod
    def _operational_breakdown(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in decisions:
            grouped[(str(row["model_id"]), str(row["protocol"]))].append(row)
        result = []
        for (model_id, protocol), rows in sorted(grouped.items()):
            latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in rows]
            usage = Counter()
            for row in rows:
                observed = row.get("usage") or {}
                usage.update(
                    input_tokens=int(observed.get("total_input_tokens", 0) or 0),
                    output_tokens=int(observed.get("total_output_tokens", 0) or 0),
                    thought_tokens=int(observed.get("total_thought_tokens", 0) or 0),
                )
            result.append(
                {
                    "model_id": model_id,
                    "protocol": protocol,
                    "requests": len(rows),
                    "valid_response_rate": sum(bool(row["valid"]) for row in rows) / len(rows),
                    "q_copy_rate": sum(
                        str(row["selected_candidate_id"]) == str(row["q_only_candidate_id"])
                        for row in rows
                    )
                    / len(rows),
                    "switch_rate": sum(
                        str(row["selected_candidate_id"]) != str(row["q_only_candidate_id"])
                        for row in rows
                    )
                    / len(rows),
                    "p50_latency_seconds": float(np.percentile(latencies, 50)),
                    "p95_latency_seconds": float(np.percentile(latencies, 95)),
                    **dict(usage),
                    "estimated_cost_usd": sum(float(row.get("estimated_charge_usd", 0.0)) for row in rows),
                }
            )
        return result

    def _write_query_type_analysis(
        self,
        *,
        phase: str,
        outcomes: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        query_type = {
            str(row["sample_id"]): str((row.get("evaluation_only") or {}).get("query_type", "unknown"))
            for row in self.manifests[phase]["rows"]
        }
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in outcomes:
            grouped[(str(row["method"]), query_type[str(row["sample_id"])])].append(row)
        metrics = []
        transition_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        for (method, kind), rows in sorted(grouped.items()):
            metric: dict[str, Any] = {"method": method, "query_type": kind, "total": len(rows)}
            for track in ("legacy", "corrected"):
                q = np.asarray([bool(row[f"{track}_q_only_correct"]) for row in rows], dtype=bool)
                selected = np.asarray([bool(row[f"{track}_selected_correct"]) for row in rows], dtype=bool)
                recovered = int((~q & selected).sum())
                harmful = int((q & ~selected).sum())
                metric.update(
                    {
                        f"{track}_j1": float(selected.mean()),
                        f"{track}_delta_pp": 100.0 * (recovered - harmful) / len(rows),
                        f"{track}_recovered": recovered,
                        f"{track}_harmful": harmful,
                        f"{track}_net": recovered - harmful,
                    }
                )
            metrics.append(metric)
            for row in rows:
                legacy_q = bool(row["legacy_q_only_correct"])
                legacy_selected = bool(row["legacy_selected_correct"])
                transition = (
                    "recovered" if not legacy_q and legacy_selected else
                    "harmful" if legacy_q and not legacy_selected else
                    "both_correct" if legacy_q and legacy_selected else
                    "both_wrong"
                )
                transition_counts[method][kind][transition] += 1
        _atomic_csv(self.config.run_root / phase / "query_type_metrics.csv", metrics)
        analysis = {
            method: {kind: dict(counts) for kind, counts in sorted(by_kind.items())}
            for method, by_kind in sorted(transition_counts.items())
        }
        atomic_write_json(
            self.config.run_root / phase / "recovered_harmful_analysis.json",
            {"status": "complete", "track": "legacy", "by_method_and_query_type": analysis},
        )
        return {"query_type_metrics": metrics, "legacy_transition_counts": analysis}

    def _direct_predictions(
        self,
        decisions: Sequence[Mapping[str, Any]],
        *,
        protocol: str = P1_PROTOCOL,
        method_suffix: str = "crog_evidence_direct",
    ) -> list[dict[str, Any]]:
        result = []
        for row in decisions:
            if str(row["protocol"]) != protocol or int(row.get("replicate_id", 0)) != 0:
                continue
            prefix = (
                "gemini_robotics_er2" if row["model_id"] == "gemini-robotics-er-2-preview"
                else "gemini_3_6_flash"
            )
            result.append(
                {
                    "sample_id": row["sample_id"],
                    "method": f"{prefix}_{method_suffix}",
                    "selected_stable_candidate_id": row["selected_stable_candidate_id"],
                    "status": (
                        "permanent_failed" if row["permanent_api_failure"] else
                        "technical_fallback" if row["technical_fallback"] else
                        "abstain" if row["abstain"] else "valid"
                    ),
                    "valid_response": row["valid"],
                    "technical_fallback": row["technical_fallback"],
                    "permanent_api_failure": row["permanent_api_failure"],
                    "abstain": row["abstain"],
                    "selected_score_margin": row["score_margin_top1_top2"],
                }
            )
        return result

    def _write_common_phase_metrics(
        self,
        *,
        phase: str,
        decisions: Sequence[Mapping[str, Any]],
        predictions: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        outcomes, metrics = evaluate_saved_predictions(
            evaluation_index=self._phase_evaluation_index(phase), predictions=predictions
        )
        _atomic_parquet(self.config.run_root / phase / "per_sample_outcomes.parquet", outcomes)
        _atomic_csv(self.config.run_root / phase / "per_method_metrics.csv", metrics)
        _atomic_csv(self.config.run_root / phase / "token_usage.csv", self._usage_summary(decisions))
        atomic_write_json(self.config.run_root / phase / "model_agreement.json", self._agreement(decisions))
        charges = Counter()
        for row in decisions:
            charges[str(row["model_id"])] += float(row.get("estimated_charge_usd", 0.0))
        atomic_write_json(
            self.config.run_root / phase / "cost.json",
            {"estimated_charge_usd_by_model": dict(charges), "total_estimated_usd": sum(charges.values())},
        )
        return outcomes, metrics

    def _finalize_pilot(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        predictions = self._direct_predictions(decisions)
        outcomes, metrics = self._write_common_phase_metrics(
            phase="pilot", decisions=decisions, predictions=predictions
        )
        latencies = [float(row["latency_seconds"]) for row in decisions]
        validity = {
            "request_count": len(decisions),
            "valid_json_count": sum(bool(row["json_parse_valid"]) for row in decisions),
            "valid_json_rate": sum(bool(row["json_parse_valid"]) for row in decisions) / len(decisions),
            "schema_valid_count": sum(bool(row["schema_valid"]) for row in decisions),
            "schema_valid_rate": sum(bool(row["schema_valid"]) for row in decisions) / len(decisions),
            "truncated_rate": sum("truncat" in str(row.get("fallback_reason", "")) for row in decisions) / len(decisions),
            "abstain_rate": sum(bool(row["abstain"]) for row in decisions) / len(decisions),
            "technical_fallback_rate": sum(bool(row["technical_fallback"]) for row in decisions) / len(decisions),
            "p50_latency_seconds": float(np.percentile(latencies, 50)),
            "p95_latency_seconds": float(np.percentile(latencies, 95)),
            "by_model_protocol": self._operational_breakdown(decisions),
        }
        if validity["schema_valid_rate"] < 0.98:
            raise HardStopError("pilot_schema_valid_rate_below_98_percent", phase_status="blocked_technical")
        atomic_write_json(self.config.run_root / "pilot" / "runtime_metrics.json", validity)
        summary = [
            "# Phase C pilot",
            "",
            "Technical pilot only; it is not a formal performance estimate.",
            "",
            f"- Requests: {len(decisions)}",
            f"- Schema-valid rate: {validity['schema_valid_rate']:.3%}",
            f"- Technical fallback rate: {validity['technical_fallback_rate']:.3%}",
        ]
        for row in metrics:
            summary.append(
                f"- {row['method']}: Legacy J@1={row['legacy_j1']:.6f}, "
                f"R/H/Net={row['legacy_recovered']}/{row['legacy_harmful']}/{row['legacy_net']}"
            )
        _atomic_text(self.config.run_root / "pilot" / "summary.md", "\n".join(summary) + "\n")
        return {"validity": validity, "metrics": metrics, "outcome_count": len(outcomes)}

    def _finalize_stability(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        repeated = [
            {
                "model_id": row["model_id"],
                "sample_id": row["sample_id"],
                "replicate_id": row["replicate_id"],
                "selected_candidate_id": row["selected_candidate_id"],
                "ranking": row["ranking"],
                "confidence": row["confidence"],
                "score_margin_top1_top2": row["score_margin_top1_top2"],
                "decision": row["decision"],
                "valid": row["valid"],
            }
            for row in decisions
        ]
        per_sample, summary = compute_stability(repeated)
        _atomic_parquet(self.config.run_root / "stability" / "repeated_decisions.parquet", decisions)
        _atomic_parquet(self.config.run_root / "stability" / "per_sample_stability.parquet", per_sample)
        atomic_write_json(self.config.run_root / "stability" / "per_model_summary.json", summary)
        atomic_write_json(self.config.run_root / "stability_results.json", summary)
        lines = ["# Phase C2 stability", ""]
        for model, row in summary.items():
            selected_rate = row["selected_candidate_exact_agreement_rate"]
            ranking_rate = row["complete_ranking_exact_agreement_rate"]
            lines.append(
                f"- {model}: valid 3/3 coverage {row['all_replicates_valid_coverage']:.3%}; "
                f"selected exact agreement "
                f"{('unavailable' if selected_rate is None else f'{selected_rate:.3%}')}; "
                f"ranking exact agreement "
                f"{('unavailable' if ranking_rate is None else f'{ranking_rate:.3%}')}"
                "."
            )
        _atomic_text(self.config.run_root / "stability" / "report.md", "\n".join(lines) + "\n")
        return summary

    def _development_lock_payload(self) -> dict[str, Any]:
        if self._evidence_schema_hash is None:
            raise FullRunError("evidence schema is unavailable")
        return {
            "schema_version": "1.0",
            "kind": "gemini_crog_development_protocol_lock",
            "model_ids": list(MODEL_IDS),
            "primary_protocol": P1_PROTOCOL,
            "ablation_protocols": list(ABLATION_PROTOCOLS),
            "renderer_version": RENDERER_VERSION,
            "renderer_sha256": self._renderer_hash,
            "ablation_renderer_sha256": ablation_renderer_hash(),
            "prompt_sha256": self._prompt_hash,
            "response_schema_sha256": self._schema_hash,
            "evidence_schema_sha256": self._evidence_schema_hash,
            "media_resolution": DEFAULT_IMAGE_RESOLUTION,
            "thinking_level": DEFAULT_THINKING_LEVEL,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "temperature_policy": "model_default",
            "candidate_permutation": "deterministic_candidate_mapping_seed_47",
            "store": False,
            "background": False,
            "stream": False,
        }

    def _finalize_ablation(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        predictions = []
        for protocol in ABLATION_PROTOCOLS:
            predictions.extend(
                self._direct_predictions(
                    decisions,
                    protocol=protocol,
                    method_suffix=f"{protocol}_direct",
                )
            )
        _, metrics = self._write_common_phase_metrics(
            phase="ablation", decisions=decisions, predictions=predictions
        )
        operational = self._operational_breakdown(decisions)
        _atomic_csv(self.config.run_root / "ablation" / "runtime_token_cost_metrics.csv", operational)
        agreement = {
            protocol: self._agreement(decisions, protocol=protocol)
            for protocol in ABLATION_PROTOCOLS
        }
        atomic_write_json(self.config.run_root / "ablation" / "model_agreement.json", agreement)
        metric_index = {str(row["method"]): row for row in metrics}
        decision_index = {
            (str(row["model_id"]), str(row["protocol"]), str(row["sample_id"])): row
            for row in decisions
        }
        comparisons = []
        for model in MODEL_IDS:
            prefix = "gemini_robotics_er2" if model == MODEL_IDS[0] else "gemini_3_6_flash"
            a2_name = f"{prefix}_{P1_PROTOCOL}_direct"
            for reference, label in (
                ("a1_visual_plus_q", "prediction_map_contribution_a2_minus_a1"),
                ("a3_full_evidence_without_q", "q_evidence_contribution_a2_minus_a3"),
                ("a0_visual_only", "full_evidence_contribution_a2_minus_a0"),
            ):
                reference_name = f"{prefix}_{reference}_direct"
                a2 = metric_index[a2_name]
                baseline = metric_index[reference_name]
                sample_ids = [str(row["sample_id"]) for row in self.manifests["ablation"]["rows"]]
                changed = sum(
                    str(decision_index[(model, P1_PROTOCOL, sample)]["selected_candidate_id"])
                    != str(decision_index[(model, reference, sample)]["selected_candidate_id"])
                    for sample in sample_ids
                )
                comparisons.append(
                    {
                        "model_id": model,
                        "comparison": label,
                        "reference_protocol": reference,
                        "primary_protocol": P1_PROTOCOL,
                        "selected_id_change_rate": changed / len(sample_ids),
                        "legacy_j1_delta_pp": 100.0 * (
                            float(a2["legacy_j1"]) - float(baseline["legacy_j1"])
                        ),
                        "corrected_j1_delta_pp": 100.0 * (
                            float(a2["corrected_j1"]) - float(baseline["corrected_j1"])
                        ),
                        "legacy_net_delta": int(a2["legacy_net"]) - int(baseline["legacy_net"]),
                        "q_copy_rate_delta": float(a2["q_copy_rate"]) - float(baseline["q_copy_rate"]),
                    }
                )
        atomic_write_json(
            self.config.run_root / "ablation" / "comparisons.json",
            {"status": "complete", "comparisons": comparisons},
        )
        development_lock = write_immutable_json(
            self.config.run_root / "development_protocol_lock.json",
            self._development_lock_payload(),
        )
        _atomic_text(
            self.config.run_root / "ablation" / "summary.md",
            "# Phase D development ablation\n\n"
            "A2 remains the preregistered primary protocol irrespective of these development scores.\n\n"
            "Detailed J@1, R/H/Net, q-copy, selected-ID agreement, response validity, latency, "
            "token, cost, prediction-map and q-evidence comparisons are stored in the phase CSV/JSON artifacts.\n",
        )
        return {
            "metrics": metrics,
            "operational": operational,
            "model_agreement": agreement,
            "comparisons": comparisons,
            "development_protocol_lock": development_lock,
        }

    def _calibration_examples(
        self,
        decisions: Sequence[Mapping[str, Any]],
        *,
        model_id: str,
    ) -> list[dict[str, Any]]:
        evaluation = self._phase_evaluation_index("calibration")
        rows = []
        for decision in decisions:
            if decision["model_id"] != model_id or decision["protocol"] != P1_PROTOCOL:
                continue
            truth = evaluation[str(decision["sample_id"])]
            selected = str(decision["selected_candidate_id"])
            q_only = str(truth["q_only_candidate_id"])
            rows.append(
                {
                    "sample_id": decision["sample_id"],
                    "valid": decision["valid"],
                    "abstain": decision["abstain"],
                    "decision": decision["decision"],
                    "selected_candidate_id": selected,
                    "q_only_candidate_id": q_only,
                    "confidence": decision["confidence"],
                    "score_margin_top1_top2": decision["score_margin_top1_top2"],
                    "selected_overall_score": decision["selected_overall_score"],
                    "legacy_q_only_correct": truth["legacy_by_candidate_id"][q_only],
                    "legacy_selected_correct": truth["legacy_by_candidate_id"][selected],
                    "corrected_q_only_correct": truth["corrected_by_candidate_id"][q_only],
                    "corrected_selected_correct": truth["corrected_by_candidate_id"][selected],
                }
            )
        return rows

    def _consensus_examples(self, by_model: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
        left = {str(row["sample_id"]): row for row in by_model[MODEL_IDS[0]]}
        right = {str(row["sample_id"]): row for row in by_model[MODEL_IDS[1]]}
        result = []
        for sample_id in sorted(left):
            a, b = left[sample_id], right[sample_id]
            q_only = str(a["q_only_candidate_id"])
            agreed = (
                bool(a["valid"]) and bool(b["valid"])
                and not bool(a["abstain"]) and not bool(b["abstain"])
                and str(a["decision"]) == "switch" and str(b["decision"]) == "switch"
                and str(a["selected_candidate_id"]) == str(b["selected_candidate_id"])
                and str(a["selected_candidate_id"]) != q_only
            )
            selected = str(a["selected_candidate_id"]) if agreed else q_only
            result.append(
                {
                    "sample_id": sample_id,
                    "valid": bool(a["valid"]) and bool(b["valid"]),
                    "abstain": bool(a["abstain"]) or bool(b["abstain"]),
                    "decision": "switch" if agreed else "keep_original",
                    "selected_candidate_id": selected,
                    "q_only_candidate_id": q_only,
                    "confidence": min(float(a["confidence"]), float(b["confidence"])),
                    "score_margin_top1_top2": min(float(a["score_margin_top1_top2"]), float(b["score_margin_top1_top2"])),
                    "selected_overall_score": min(float(a["selected_overall_score"]), float(b["selected_overall_score"])),
                    "legacy_q_only_correct": a["legacy_q_only_correct"],
                    "legacy_selected_correct": (
                        a["legacy_selected_correct"] if agreed else a["legacy_q_only_correct"]
                    ),
                    "corrected_q_only_correct": a["corrected_q_only_correct"],
                    "corrected_selected_correct": (
                        a["corrected_selected_correct"] if agreed else a["corrected_q_only_correct"]
                    ),
                }
            )
        return result

    def _finalize_calibration(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        grid = calibration_grid_payload()
        grid_path = self.config.run_root / "calibration_grid.json"
        if grid_path.exists():
            existing_grid = _read_json(grid_path)
            if "content_sha256" in existing_grid and existing_grid["content_sha256"] != _payload_digest(existing_grid):
                raise HardStopError("calibration_grid_content_hash_changed", phase_status="blocked_identity")
            comparable = {key: value for key, value in existing_grid.items() if key != "content_sha256"}
            if comparable != grid:
                raise HardStopError("calibration_grid_changed", phase_status="blocked_identity")
        else:
            write_immutable_json(grid_path, grid)
        examples = {model: self._calibration_examples(decisions, model_id=model) for model in MODEL_IDS}
        examples["gemini_dual_consensus"] = self._consensus_examples(examples)
        selected = {}
        summaries = {}
        all_sweeps = []
        charges = Counter()
        for row in decisions:
            charges[str(row["model_id"])] += float(row.get("estimated_charge_usd", 0.0))
        for model, rows in examples.items():
            threshold, sweep, summary = sweep_safe_thresholds(
                rows,
                model_id=model,
                model_cost_usd=(sum(charges.values()) if model == "gemini_dual_consensus" else charges[model]),
            )
            selected[model] = asdict(threshold)
            summaries[model] = summary
            all_sweeps.extend(sweep)
            name = "consensus" if model == "gemini_dual_consensus" else ("er2" if model == MODEL_IDS[0] else "flash")
            _atomic_csv(self.config.run_root / "calibration" / f"{name}_threshold_sweep.csv", sweep)
        thresholds = {
            "schema_version": "1.0",
            "grid_sha256": sha256_json(grid),
            "harmful_rate_limit": 0.01,
            "thresholds": selected,
            "summaries": summaries,
        }
        atomic_write_json(self.config.run_root / "calibration" / "selected_thresholds.json", thresholds)
        _atomic_csv(self.config.run_root / "threshold_sweeps.csv", all_sweeps)
        _atomic_parquet(self.config.run_root / "calibration" / "per_sample_decisions.parquet", decisions)
        _atomic_text(
            self.config.run_root / "calibration" / "summary.md",
            "# Phase E calibration\n\nThresholds were selected only on the frozen calibration split.\n",
        )
        return thresholds

    def _load_thresholds(self) -> dict[str, LockedSafeThreshold]:
        payload = _read_json(self.config.run_root / "calibration" / "selected_thresholds.json")
        if payload.get("grid_sha256") != sha256_json(calibration_grid_payload()):
            raise HardStopError("locked_calibration_grid_changed", phase_status="blocked_identity")
        return {
            name: LockedSafeThreshold(**row)
            for name, row in payload["thresholds"].items()
        }

    def _derived_predictions(
        self,
        *,
        phase: str,
        decisions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        evaluation = self._phase_evaluation_index(phase)
        thresholds = self._load_thresholds()
        by_sample: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in decisions:
            if row["protocol"] == P1_PROTOCOL and int(row.get("replicate_id", 0)) == 0:
                by_sample[str(row["sample_id"])][str(row["model_id"])] = row
        predictions = []
        for sample_id in [str(row["sample_id"]) for row in self.manifests[phase]["rows"]]:
            truth = evaluation[sample_id]
            q_only = str(truth["q_only_candidate_id"])
            predictions.append(
                {
                    "sample_id": sample_id,
                    "method": "crog_q_only",
                    "selected_stable_candidate_id": stable_candidate_id(sample_id, q_only),
                    "status": "valid",
                    "valid_response": True,
                    "selected_score_margin": None,
                }
            )
            safe_selected = {}
            for model in MODEL_IDS:
                row = by_sample[sample_id][model]
                prefix = "gemini_robotics_er2" if model == MODEL_IDS[0] else "gemini_3_6_flash"
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "method": f"{prefix}_crog_evidence_direct",
                        "selected_stable_candidate_id": row["selected_stable_candidate_id"],
                        "status": (
                            "permanent_failed" if row["permanent_api_failure"] else
                            "technical_fallback" if row["technical_fallback"] else
                            "abstain" if row["abstain"] else "valid"
                        ),
                        "valid_response": row["valid"],
                        "technical_fallback": row["technical_fallback"],
                        "permanent_api_failure": row["permanent_api_failure"],
                        "abstain": row["abstain"],
                        "selected_score_margin": row["score_margin_top1_top2"],
                    }
                )
                selected = apply_locked_threshold(dict(row), thresholds[model])
                safe_selected[model] = selected
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "method": f"{prefix}_crog_evidence_safe",
                        "selected_stable_candidate_id": stable_candidate_id(sample_id, selected),
                        "status": (
                            "permanent_failed" if row["permanent_api_failure"] else
                            "technical_fallback" if row["technical_fallback"] else
                            "abstain" if row["abstain"] else "valid"
                        ),
                        "valid_response": row["valid"],
                        "technical_fallback": row["technical_fallback"],
                        "permanent_api_failure": row["permanent_api_failure"],
                        "abstain": row["abstain"],
                        "selected_score_margin": row["score_margin_top1_top2"],
                    }
                )
            combined = self._consensus_examples(
                {
                    MODEL_IDS[0]: [self._calibration_shape(by_sample[sample_id][MODEL_IDS[0]], truth)],
                    MODEL_IDS[1]: [self._calibration_shape(by_sample[sample_id][MODEL_IDS[1]], truth)],
                }
            )[0]
            consensus = apply_locked_threshold(combined, thresholds["gemini_dual_consensus"])
            model_rows = [by_sample[sample_id][model] for model in MODEL_IDS]
            permanent_failure = any(bool(row["permanent_api_failure"]) for row in model_rows)
            technical_fallback = (not permanent_failure) and (
                any(bool(row["technical_fallback"]) for row in model_rows)
                or not bool(combined["valid"])
            )
            abstain = (not permanent_failure) and (not technical_fallback) and any(
                bool(row["abstain"]) for row in model_rows
            )
            consensus_status = (
                "permanent_failed" if permanent_failure else
                "technical_fallback" if technical_fallback else
                "abstain" if abstain else
                "valid"
            )
            predictions.append(
                {
                    "sample_id": sample_id,
                    "method": "gemini_dual_consensus_safe",
                    "selected_stable_candidate_id": stable_candidate_id(sample_id, consensus),
                    "status": consensus_status,
                    "valid_response": bool(combined["valid"]),
                    "technical_fallback": technical_fallback,
                    "permanent_api_failure": permanent_failure,
                    "abstain": abstain,
                    "selected_score_margin": combined["score_margin_top1_top2"],
                }
            )
        return predictions

    @staticmethod
    def _calibration_shape(row: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[str, Any]:
        selected = str(row["selected_candidate_id"])
        q_only = str(truth["q_only_candidate_id"])
        return {
            **dict(row),
            "q_only_candidate_id": q_only,
            "legacy_q_only_correct": truth["legacy_by_candidate_id"][q_only],
            "legacy_selected_correct": truth["legacy_by_candidate_id"][selected],
            "corrected_q_only_correct": truth["corrected_by_candidate_id"][q_only],
            "corrected_selected_correct": truth["corrected_by_candidate_id"][selected],
        }

    def _finalize_validation(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        predictions = self._derived_predictions(phase="validation", decisions=decisions)
        outcomes, metrics = evaluate_saved_predictions(
            evaluation_index=self._phase_evaluation_index("validation"), predictions=predictions
        )
        tests, intervals = paired_statistical_tests(
            outcomes=outcomes,
            draws=self.config.bootstrap_draws,
            seed=self.config.bootstrap_seed,
        )
        charges = Counter()
        for row in decisions:
            charges[str(row["model_id"])] += float(row.get("estimated_charge_usd", 0.0))
        method_cost = {
            "gemini_robotics_er2_crog_evidence_direct": charges[MODEL_IDS[0]],
            "gemini_robotics_er2_crog_evidence_safe": charges[MODEL_IDS[0]],
            "gemini_3_6_flash_crog_evidence_direct": charges[MODEL_IDS[1]],
            "gemini_3_6_flash_crog_evidence_safe": charges[MODEL_IDS[1]],
            "gemini_dual_consensus_safe": sum(charges.values()),
        }
        selection = select_validation_primary(
            metrics=metrics,
            bootstrap=intervals,
            estimated_cost_by_method=method_cost,
            minimum_response_coverage=self.config.minimum_response_coverage,
        )
        selection.update(
            {
                "no_gt_request_leak": True,
                "candidate_identity_unchanged": True,
                "threshold_source": "calibration/selected_thresholds.json",
                "test_results_used_for_selection": False,
            }
        )
        root = self.config.run_root / "validation"
        _atomic_parquet(root / "per_model_decisions.parquet", decisions)
        _atomic_parquet(root / "per_method_outcomes.parquet", outcomes)
        _atomic_csv(root / "per_method_metrics.csv", metrics)
        atomic_write_json(root / "statistical_tests.json", tests)
        atomic_write_json(root / "bootstrap_intervals.json", intervals)
        atomic_write_json(root / "model_agreement.json", self._agreement(decisions))
        atomic_write_json(root / "primary_selection.json", selection)
        query_analysis = self._write_query_type_analysis(phase="validation", outcomes=outcomes)
        atomic_write_json(self.config.run_root / "validation_metrics.json", {"metrics": metrics})
        _atomic_text(
            root / "summary.md",
            "# Phase F validation\n\n"
            f"Locked primary selected without test access: `{selection['locked_primary']}`.\n",
        )
        return {
            "metrics": metrics,
            "statistics": tests,
            "bootstrap": intervals,
            "selection": selection,
            "query_type_analysis": query_analysis,
        }

    def _formal_lock_path(self) -> Path:
        return self.config.run_root / "frozen_gemini_crog_evidence_manifest.json"

    def _evidence_schema_source_path(self) -> Path:
        for source in self._evidence_sources():
            path = source / "evidence_schema.json"
            if path.is_file() and sha256_file(path) == self._evidence_schema_hash:
                return path
        raise FullRunError("no file-backed frozen evidence schema is available")

    def _verify_formal_lock(self) -> dict[str, Any]:
        path = self._formal_lock_path()
        if not path.is_file():
            raise HardStopError("formal_experiment_lock_missing", phase_status="blocked_lock")
        try:
            verify_experiment_lock(path, repo_root=REPO_ROOT, expected_run_id=self.state.run_id)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise HardStopError("formal_experiment_lock_verification_failed", phase_status="blocked_lock") from exc
        return _read_json(path)

    def _lock_phase(self) -> dict[str, Any]:
        if self._formal_lock_path().exists():
            locked = self._verify_formal_lock()
            return {"status": "already_locked", "lock_sha256": locked["lock_sha256"]}
        thresholds = _read_json(self.config.run_root / "calibration" / "selected_thresholds.json")
        validation_metrics = _read_json(self.config.run_root / "validation_metrics.json")
        primary_path = self.config.run_root / "validation" / "primary_selection.json"
        primary_values = _read_json(primary_path)
        primary = primary_values["locked_primary"]
        threshold_path = self.config.run_root / "calibration" / "selected_thresholds.json"
        preflight = _read_json(self.config.run_root / "full_run_preflight.json")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
        source_files.extend(
            [REPO_ROOT / "utils/grasp_eval.py", REPO_ROOT / "utils/grasp_metrics.py"]
        )
        validation_path = self.config.run_root / "validation_metrics.json"
        cohort_manifests = {
            phase: file_identity(self.config.run_root / f"{phase}_manifest.json")
            for phase in REQUEST_PHASES
        }
        validation_artifact_names = (
            "per_method_metrics.csv",
            "statistical_tests.json",
            "bootstrap_intervals.json",
            "primary_selection.json",
            "model_agreement.json",
        )
        validation_artifacts = {
            name: file_identity(self.config.run_root / "validation" / name)
            for name in validation_artifact_names
        }
        test_dataset = self.config.datasets["test"]
        if test_dataset.raw_predictions is None:
            raise HardStopError("formal_raw_gt_source_missing", phase_status="blocked_lock")
        observed_candidate_identity = candidate_identity_stream_sha256(
            test_dataset.features
        )
        if observed_candidate_identity != preflight["candidate_identity_stream_sha256"]:
            raise HardStopError("candidate_identity_changed_before_lock", phase_status="blocked_identity")
        payload = build_lock_payload(
            experiment_id=self.config.run_root.name,
            run_id=self.state.run_id,
            locked_at_utc=utc_now(),
            source_code=[file_identity(path) for path in source_files],
            full_run_plan=file_identity(self.config.run_root / "full_run_plan.json"),
            cohort_manifests=cohort_manifests,
            git_commit=commit,
            git_diff_sha256=git_diff_hash(REPO_ROOT),
            checkpoint=file_identity(self.config.checkpoint_path),
            config=file_identity(self.config.crog_config_path),
            baseline_candidates={
                "features": file_identity(self.config.datasets["test"].features),
                "candidate_identity_stream_sha256": observed_candidate_identity,
            },
            split_manifest=file_identity(self.config.split_manifest_path),
            development_protocol_lock=file_identity(self.config.run_root / "development_protocol_lock.json"),
            renderer={**file_identity(Path(__file__).with_name("renderer.py")), "version": RENDERER_VERSION},
            prompt=file_identity(self.config.system_prompt_path),
            response_schema={"sha256": self._schema_hash, "source": file_identity(Path(__file__).with_name("schema.py"))},
            evidence_schema=file_identity(self._evidence_schema_source_path()),
            candidate_permutation={
                **file_identity(Path(__file__).with_name("renderer.py")),
                "algorithm": f"deterministic_candidate_mapping(seed={self.config.mapping_seed})",
            },
            model_ids=list(MODEL_IDS),
            sdk=SDK_CONTRACT,
            endpoint=ENDPOINT_CONTRACT,
            store=False,
            background=False,
            stream=False,
            tools_enabled=False,
            previous_interaction=None,
            thinking_level=DEFAULT_THINKING_LEVEL,
            temperature_policy="model_default",
            image_resolution=DEFAULT_IMAGE_RESOLUTION,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            safe_thresholds={"values": thresholds, "source": file_identity(threshold_path)},
            calibration_grid=file_identity(self.config.run_root / "calibration_grid.json"),
            harmful_cap=0.01,
            primary_selection_rule="validation-primary-v1",
            primary_method=primary,
            primary_selection={"values": primary_values, "source": file_identity(primary_path)},
            secondary_methods=[
                "gemini_robotics_er2_crog_evidence_direct",
                "gemini_robotics_er2_crog_evidence_safe",
                "gemini_3_6_flash_crog_evidence_direct",
                "gemini_3_6_flash_crog_evidence_safe",
                "gemini_dual_consensus_safe",
            ],
            request_hash_algorithm="sha256 canonical JSON v1; stability/protocol dimensions when non-P1",
            cache_schema="gemini_cache.sqlite append-only responses/request_states/request_attempts/aliases",
            budget={
                "max_spend_usd": None if self._budget is None else self._budget.max_spend_usd,
                "er2_cost_cap_per_request_usd": None if self._budget is None else self._budget.er2_cost_cap_per_request_usd,
            },
            retry_policy={"max_retries": 5, "retryable": [429, 500, 502, 503, 504]},
            concurrency=int(os.environ.get("GEMINI_MAX_CONCURRENCY", "1")),
            transport="standard_interactions",
            validation_metrics={"values": validation_metrics, "source": file_identity(validation_path)},
            validation_artifacts=validation_artifacts,
            ground_truth_inputs={
                "legacy_labels": file_identity(test_dataset.legacy_labels),
                "corrected_labels": file_identity(test_dataset.corrected_labels),
                "raw_predictions_with_gt": file_identity(test_dataset.raw_predictions),
            },
            formal_test_expected_sample_count=len(self.manifests["formal_test"]["rows"]),
            formal_test_expected_request_count=len(self.manifests["formal_test"]["rows"]) * len(MODEL_IDS),
            evaluator={
                "main_saved_label_join": {
                    "definition": "frozen_label_candidate_join_v1",
                    "source": file_identity(Path(__file__).with_name("evaluation.py")),
                },
                "legacy": {
                    "definition": "legacy_official_impl_v1",
                    "source": file_identity(REPO_ROOT / "utils/grasp_metrics.py"),
                },
                "corrected": {
                    "definition": "corrected_geometric_v2",
                    "source": file_identity(REPO_ROOT / "utils/grasp_metrics.py"),
                },
            },
        )
        verification = verify_lock_payload(
            payload,
            repo_root=REPO_ROOT,
            expected_run_id=self.state.run_id,
        )
        dry = lock_experiment(self._formal_lock_path(), payload, dry_run=True)
        atomic_write_json(
            self.config.run_root / "formal_lock_dry_run.json",
            {**dry, "verification": verification, "proposed_payload": payload},
        )
        result = lock_experiment(self._formal_lock_path(), payload, dry_run=False)
        self._verify_formal_lock()
        atomic_write_json(self.config.run_root / "lock_status.json", result)
        return result

    @staticmethod
    def _assert_independent_match(
        *,
        independent: Mapping[str, Any],
        outcomes: Sequence[Mapping[str, Any]],
        metrics: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        main_by_key = {(str(row["sample_id"]), str(row["method"])): row for row in outcomes}
        independent_by_key = {
            (str(row["sample_id"]), str(row["method"])): row for row in independent["per_sample"]
        }
        if set(main_by_key) != set(independent_by_key):
            raise HardStopError("independent_recompute_prediction_set_mismatch", phase_status="blocked_recompute")
        for key, row in main_by_key.items():
            other = independent_by_key[key]
            for field_name in (
                "selected_stable_candidate_id",
                "legacy_q_only_correct",
                "legacy_selected_correct",
                "corrected_q_only_correct",
                "corrected_selected_correct",
                "legacy_oracle",
                "corrected_oracle",
            ):
                if row[field_name] != other[field_name]:
                    raise HardStopError("independent_recompute_per_sample_mismatch", phase_status="blocked_recompute")
        main_metrics = {str(row["method"]): row for row in metrics}
        for method, aggregate in independent["aggregate"].items():
            main = main_metrics[method]
            if int(main["total"]) != int(aggregate["total"]):
                raise HardStopError("independent_recompute_aggregate_mismatch", phase_status="blocked_recompute")
            for track in ("legacy", "corrected"):
                for metric_name in ("recovered", "harmful", "net"):
                    if int(main[f"{track}_{metric_name}"]) != int(aggregate[f"{track}_{metric_name}"]):
                        raise HardStopError("independent_recompute_aggregate_mismatch", phase_status="blocked_recompute")
                if not math.isclose(float(main[f"{track}_j1"]), float(aggregate[f"{track}_j1"]), abs_tol=1e-12):
                    raise HardStopError("independent_recompute_float_mismatch", phase_status="blocked_recompute")
                if int(round(float(main[f"{track}_j1"]) * int(main["total"]))) != int(
                    aggregate[f"{track}_success_count"]
                ):
                    raise HardStopError("independent_recompute_success_count_mismatch", phase_status="blocked_recompute")
                if not math.isclose(
                    float(main[f"{track}_oracle_at_5"]),
                    float(aggregate[f"{track}_oracle_at_5"]),
                    abs_tol=1e-12,
                ):
                    raise HardStopError("independent_recompute_oracle_mismatch", phase_status="blocked_recompute")
        return {
            "selected_candidate_id_match_rate": 1.0,
            "legacy_correctness_match_rate": 1.0,
            "corrected_correctness_match_rate": 1.0,
            "aggregate_counts_match": True,
            "strict_float_tolerance": 1e-12,
        }

    def _finalize_formal(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        lock = self._verify_formal_lock()
        claim_path = self.config.run_root / "formal_test" / "formal_run_claim.json"
        claim_formal_test_once(
            claim_path, self._formal_lock_path(), run_id=self.state.run_id, repo_root=REPO_ROOT
        )
        predictions = self._derived_predictions(phase="formal_test", decisions=decisions)
        primary = str(lock["primary_method"])
        by_key = {(str(row["sample_id"]), str(row["method"])): row for row in predictions}
        for sample_id in [str(row["sample_id"]) for row in self.manifests["formal_test"]["rows"]]:
            source = dict(by_key[(sample_id, primary)])
            source["method"] = "locked_gemini_primary"
            predictions.append(source)
        outcomes, metrics = evaluate_saved_predictions(
            evaluation_index=self._phase_evaluation_index("formal_test"), predictions=predictions
        )
        tests, intervals = paired_statistical_tests(
            outcomes=outcomes,
            draws=self.config.bootstrap_draws,
            seed=self.config.bootstrap_seed,
        )
        root = self.config.run_root
        phase_root = root / "formal_test"
        _atomic_parquet(phase_root / "per_model_decisions.parquet", decisions)
        _atomic_parquet(phase_root / "per_sample_predictions.parquet", predictions)
        _atomic_parquet(phase_root / "per_method_outcomes.parquet", outcomes)
        _atomic_csv(phase_root / "per_method_metrics.csv", metrics)
        atomic_write_json(phase_root / "statistical_tests.json", tests)
        atomic_write_json(phase_root / "bootstrap_intervals.json", intervals)
        atomic_write_json(phase_root / "model_agreement.json", self._agreement(decisions))
        query_analysis = self._write_query_type_analysis(phase="formal_test", outcomes=outcomes)
        dataset = self.config.datasets["test"]
        if dataset.raw_predictions is None:
            raise HardStopError(
                "independent_raw_gt_source_missing",
                phase_status="blocked_recompute",
            )
        independent = independent_recompute(
            predictions_path=phase_root / "per_sample_predictions.parquet",
            features_path=dataset.features,
            raw_predictions_path=dataset.raw_predictions,
        )
        comparison = self._assert_independent_match(
            independent=independent, outcomes=outcomes, metrics=metrics
        )
        independent_result = {**independent, "comparison_to_main": comparison}
        atomic_write_json(root / "independent_recompute_results.json", independent_result)
        # Publish the formal artifacts only after independent recomputation has
        # succeeded.  Atomic replacement is appropriate for the pre-formal
        # placeholder reports that explicitly say results were not yet run.
        _atomic_parquet(root / "per_sample_predictions.parquet", predictions)
        _atomic_parquet(root / "per_model_decisions.parquet", decisions)
        _atomic_csv(root / "per_method_metrics.csv", metrics)
        atomic_write_json(root / "statistical_tests.json", tests)
        atomic_write_json(root / "bootstrap_intervals.json", intervals)
        atomic_write_json(root / "model_agreement.json", self._agreement(decisions))
        atomic_write_json(
            root / "results_bundle.json",
            {
                "status": "complete",
                "experiment_id": root.name,
                "formal_lock_sha256": lock["lock_sha256"],
                "locked_primary": primary,
                "metrics": metrics,
                "independent_recompute": comparison,
                "completed_at_utc": utc_now(),
            },
        )
        return {
            "metrics": metrics,
            "statistics": tests,
            "bootstrap": intervals,
            "independent": comparison,
            "query_type_analysis": query_analysis,
        }

    def _cleanup_formal_boards_after_gallery(self) -> dict[str, Any]:
        """Remove ordinary formal input boards only after the gallery is durable.

        Candidate evidence, mappings, request manifests, hashes, cache responses,
        and gallery composites remain available for audit.  Paths outside the
        formal-test tree (for example reused Phase-B boards) are never removed.
        """

        audit_path = self.config.run_root / "formal_test" / "board_cleanup.json"
        if audit_path.is_file():
            return _read_json(audit_path)
        if not self.config.delete_committed_boards:
            result = {
                "status": "retained_by_configuration",
                "deleted_board_count": 0,
                "deleted_layer_audit_count": 0,
                "released_bytes": 0,
                "completed_at_utc": utc_now(),
            }
            atomic_write_json(audit_path, result)
            return result

        gallery_manifest = self.config.run_root / "formal_test" / "gallery" / "gallery.json"
        if not gallery_manifest.is_file():
            raise HardStopError("formal_gallery_required_before_board_cleanup", phase_status="blocked_publication")
        gallery = _read_json(gallery_manifest)
        if gallery.get("safety", {}).get("input_boards_byte_identical") is not True:
            raise HardStopError("formal_gallery_safety_audit_failed", phase_status="blocked_publication")

        index = _read_json(self.config.run_root / "formal_test" / "evidence_index.json")
        requested = {str(row["sample_id"]) for row in self.manifests["formal_test"]["rows"]}
        sources = sorted({Path(row["source"]).resolve() for row in index["rows"]})
        formal_root = (self.config.run_root / "formal_test").resolve()
        deleted_boards = 0
        deleted_layers = 0
        released_bytes = 0
        retained_external = 0
        for source in sources:
            for request in _read_jsonl(source / "request_manifest.jsonl"):
                if str(request.get("sample_id")) not in requested:
                    continue
                board = Path(str(request["board_path"])).resolve()
                try:
                    board.relative_to(formal_root)
                except ValueError:
                    retained_external += 1
                    continue
                if board.is_file():
                    if sha256_file(board) != str(request["board_sha256"]):
                        raise HardStopError(
                            "formal_board_hash_changed_before_cleanup",
                            phase_status="blocked_identity",
                        )
                    released_bytes += board.stat().st_size
                    board.unlink()
                    deleted_boards += 1
                layers = board.with_suffix(".layers.json")
                if layers.is_file():
                    released_bytes += layers.stat().st_size
                    layers.unlink()
                    deleted_layers += 1
        result = {
            "status": "complete",
            "deleted_board_count": deleted_boards,
            "deleted_layer_audit_count": deleted_layers,
            "retained_external_board_count": retained_external,
            "released_bytes": released_bytes,
            "retained_artifacts": [
                "formal_test/gallery",
                "formal_test/evidence_index.json",
                "request_manifest.parquet",
                "candidate_mapping.parquet",
                "per_candidate_evidence.parquet",
                "gemini_cache.sqlite",
            ],
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(audit_path, result)
        return result

    def _publish_formal_outputs(self) -> dict[str, Any]:
        """Build the evaluation gallery and final machine-readable bundle offline."""

        phase_root = self.config.run_root / "formal_test"
        gallery_root = phase_root / "gallery"
        gallery_manifest = gallery_root / "gallery.json"
        if gallery_manifest.is_file():
            gallery = _read_json(gallery_manifest)
            safety = gallery.get("safety", {})
            if not (
                safety.get("input_boards_byte_identical") is True
                and safety.get("gt_only_in_separate_evaluation_panel") is True
            ):
                raise HardStopError("formal_gallery_safety_audit_failed", phase_status="blocked_publication")
            missing_images = [
                str(row.get("image_path"))
                for row in gallery.get("cases", [])
                if not Path(str(row.get("image_path", ""))).is_file()
            ]
            if missing_images:
                raise HardStopError("formal_gallery_incomplete", phase_status="blocked_publication")
        else:
            gallery = build_evaluation_gallery(
                per_method_outcomes_path=phase_root / "per_method_outcomes.parquet",
                per_model_decisions_path=phase_root / "per_model_decisions.parquet",
                evidence_index_path=phase_root / "evidence_index.json",
                output_dir=gallery_root,
            )
        cleanup = self._cleanup_formal_boards_after_gallery()
        bundle = finalize_experiment(
            self.config.run_root,
            repo_root=REPO_ROOT,
            docs_dir=REPO_ROOT / "docs",
        )
        result = {
            "status": "complete",
            "gallery_case_count": int(gallery.get("case_count", 0)),
            "gallery_manifest": str(gallery_manifest),
            "board_cleanup": cleanup,
            "publication_scope": bundle.get("publication_scope"),
            "results_bundle": str(self.config.run_root / "results_bundle.json"),
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(self.config.run_root / "publication_status.json", result)
        return result

    def _phase_marker_path(self, phase: str) -> Path:
        return self.config.run_root / phase / "PHASE_COMPLETE.json"

    def _phase_artifact_identities(self, phase: str) -> list[dict[str, Any]]:
        phase_root = self.config.run_root / phase
        paths = [
            path
            for path in sorted(phase_root.iterdir())
            if path.is_file()
            and path.name != "PHASE_COMPLETE.json"
            and not path.name.startswith(".")
        ] if phase_root.is_dir() else []
        if phase == "lock":
            paths.append(self._formal_lock_path())
        if phase == "formal_test":
            paths.extend(
                [
                    self.config.run_root / "independent_recompute_results.json",
                ]
            )
        unique = {str(path.resolve()): path for path in paths if path.is_file()}
        return [file_identity(unique[key]) for key in sorted(unique)]

    def _phase_is_complete(self, phase: str) -> bool:
        path = self._phase_marker_path(phase)
        if not path.exists():
            return False
        marker = _read_json(path)
        manifest_hash = None if phase == "lock" else self.manifests[phase]["content_sha256"]
        if marker.get("phase") != phase or marker.get("plan_sha256") != self.plan["content_sha256"]:
            raise HardStopError("phase_completion_marker_changed", phase_status="blocked_identity")
        if marker.get("status") != "complete" or marker.get("run_id") != self.state.run_id:
            raise HardStopError("phase_completion_marker_run_changed", phase_status="blocked_identity")
        if marker.get("manifest_sha256") != manifest_hash:
            raise HardStopError("phase_manifest_changed_after_completion", phase_status="blocked_identity")
        identities = marker.get("artifact_identities")
        if not isinstance(identities, list) or not identities:
            raise HardStopError("phase_completion_marker_unbound", phase_status="blocked_identity")
        for identity in identities:
            artifact = Path(str(identity.get("path", "")))
            if (
                identity.get("identity_kind") != "file_sha256"
                or not artifact.is_file()
                or sha256_file(artifact) != identity.get("sha256")
                or artifact.stat().st_size != int(identity.get("size_bytes", -1))
            ):
                raise HardStopError("phase_completion_artifact_changed", phase_status="blocked_identity")
        if phase == "formal_test":
            self._verify_formal_lock()
        return True

    def _complete_phase(self, phase: str, result: Mapping[str, Any]) -> None:
        marker = {
            "schema_version": "1.0",
            "phase": phase,
            "status": "complete",
            "run_id": self.state.run_id,
            "plan_sha256": self.plan["content_sha256"],
            "manifest_sha256": None if phase == "lock" else self.manifests[phase]["content_sha256"],
            "result_sha256": sha256_json(result),
            "artifact_identities": self._phase_artifact_identities(phase),
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(self._phase_marker_path(phase), marker)
        phase_plan = next((row for row in self.plan["phases"] if row["phase"] == phase), None)
        total = 1 if phase == "lock" else int(phase_plan["planned_request_slots"])
        self.state.update_phase(phase, status="complete", completed=total, total=total)

    def run(self, *, stop_after: str | None = None) -> dict[str, Any]:
        if stop_after is not None and stop_after not in PHASES:
            raise ValueError(f"unknown stop_after phase: {stop_after}")
        experiment_id = self.config.run_root.name
        state = FullRunState(self.config.run_root, experiment_id=experiment_id)
        self._state = state
        state.acquire()
        released = False
        try:
            state.update_phase("preflight", status="running")
            preflight = self.preflight()
            state.update_phase("preflight", status="complete", completed=1, total=1)
            self._open_runtime()
            self._live_key_preflight_done = (self.config.run_root / "live_key_preflight.json").exists()
            provider_path = self.config.run_root / "formal_test" / "provider_identity.json"
            identity_directory = self.config.run_root / "formal_test" / "provider_identity"
            if identity_directory.is_dir():
                for identity_path in identity_directory.glob("*.json"):
                    payload = _read_json(identity_path)
                    if payload.get("content_sha256") != _payload_digest(payload):
                        raise HardStopError(
                            "provider_identity_file_changed",
                            phase_status="blocked_provider_drift",
                        )
                    self._provider_identity[str(payload["model_id"])] = {
                        str(key): str(value)
                        for key, value in payload["identity"].items()
                    }
            elif provider_path.exists():
                self._provider_identity = {
                    str(key): {str(k): str(v) for k, v in value.items()}
                    for key, value in _read_json(provider_path).get("models", {}).items()
                }
            er2_unavailable = self.config.run_root / "formal_test" / "er2_unavailable.json"
            if er2_unavailable.is_file():
                marker = _read_json(er2_unavailable)
                if marker.get("model_id") != MODEL_IDS[0]:
                    raise HardStopError(
                        "provider_model_unavailable_marker_changed",
                        phase_status="blocked_identity",
                    )
                self._disabled_models[MODEL_IDS[0]] = marker
            results: dict[str, Any] = {"preflight": preflight}
            for phase in PHASES:
                if self._phase_is_complete(phase):
                    state.update_phase(phase, status="complete")
                    results[phase] = {"status": "resumed_complete"}
                else:
                    state.update_phase(phase, status="running")
                    if phase == "lock":
                        value = self._lock_phase()
                    else:
                        if phase == "formal_test":
                            self._verify_formal_lock()
                            claim_path = self.config.run_root / phase / "formal_run_claim.json"
                            claim_formal_test_once(
                                claim_path,
                                self._formal_lock_path(),
                                run_id=self.state.run_id,
                                repo_root=REPO_ROOT,
                            )
                        decisions = self._run_request_phase(phase)
                        value = {
                            "pilot": self._finalize_pilot,
                            "stability": self._finalize_stability,
                            "ablation": self._finalize_ablation,
                            "calibration": self._finalize_calibration,
                            "validation": self._finalize_validation,
                            "formal_test": self._finalize_formal,
                        }[phase](decisions)
                    self._complete_phase(phase, value)
                    results[phase] = value
                if stop_after == phase:
                    break
            if self.config.verify_repository_contract and self._phase_is_complete("formal_test"):
                state.update_phase("publication", status="running")
                results["publication"] = self._publish_formal_outputs()
                state.update_phase("publication", status="complete", completed=1, total=1)
            status = "complete" if stop_after is None else f"stopped_after_{stop_after}"
            state.release(status=status)
            released = True
            return {"status": status, "run_id": state.run_id, "phases": results}
        except HardStopError as exc:
            try:
                state.update_phase(state.current_phase, status=exc.phase_status, details={"reason": exc.reason})
            finally:
                state.release(status=exc.phase_status)
                released = True
            raise
        except Exception as exc:
            try:
                state.update_phase(state.current_phase, status="failed", details={"error_type": type(exc).__name__})
            finally:
                state.release(status="failed")
                released = True
            raise
        finally:
            self._close_runtime()
            if not released:
                state.release(status="failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume CROG Gemini Evidence V1 Phase C-H")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--env-file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evidence-shard-size", type=int, default=64)
    parser.add_argument("--stop-after", choices=PHASES)
    parser.add_argument("--max-cumulative-retryable-attempts", type=int)
    parser.add_argument("--delete-committed-boards", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FullRunConfig(
        run_root=Path(args.run_root),
        private_env_path=None if args.env_file is None else Path(args.env_file),
        device=args.device,
        evidence_shard_size=args.evidence_shard_size,
        max_cumulative_retryable_attempts=(
            args.max_cumulative_retryable_attempts
        ),
        delete_committed_boards=bool(args.delete_committed_boards),
    )
    resume_count = 0
    while True:
        try:
            result = FullRunOrchestrator(config).run(stop_after=args.stop_after)
            break
        except HardStopError as exc:
            print(json.dumps({"status": exc.phase_status, "reason": exc.reason}, sort_keys=True))
            return 2
        except ResumeRequiredError as exc:
            delay = min(60.0, 2.0 ** min(resume_count, 6))
            resume_count += 1
            print(
                json.dumps(
                    {
                        "status": "automatic_resume_pending",
                        "reason": str(exc),
                        "resume_count": resume_count,
                        "backoff_seconds": delay,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay)
        except RunAlreadyActiveError as exc:
            print(json.dumps({"status": "already_running", **exc.report}, sort_keys=True))
            return 4
    print(json.dumps({"status": result["status"], "run_id": result["run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
