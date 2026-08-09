from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .renderer import reverse_display_id
from .schema import GeminiRankingResponse
from .schema import response_json_schema


LOCK_SCHEMA_VERSION = "2.1.0"
LOCK_KIND = "frozen_gemini_crog_evidence_manifest"
EXACT_MODEL_IDS = ["gemini-robotics-er-2-preview", "gemini-3.6-flash"]
LOCKED_WORKTREE_PATHS = (
    "failure_analysis/gemini_crog_evidence_v1",
    "configs/crog_gemini_evidence_v1.yaml",
    "prompts/crog_gemini_evidence_system_v1.txt",
    ":(glob)tests/test_gemini*.py",
    "utils/grasp_eval.py",
    "utils/grasp_metrics.py",
)
PHASE_G_REQUIRED_FIELDS = {
    "experiment_id",
    "run_id",
    "locked_at_utc",
    "source_code",
    "full_run_plan",
    "cohort_manifests",
    "git_commit",
    "git_diff_sha256",
    "checkpoint",
    "config",
    "baseline_candidates",
    "split_manifest",
    "development_protocol_lock",
    "renderer",
    "prompt",
    "response_schema",
    "evidence_schema",
    "candidate_permutation",
    "model_ids",
    "sdk",
    "endpoint",
    "store",
    "background",
    "stream",
    "tools_enabled",
    "previous_interaction",
    "thinking_level",
    "temperature_policy",
    "image_resolution",
    "max_output_tokens",
    "safe_thresholds",
    "calibration_grid",
    "harmful_cap",
    "primary_selection_rule",
    "primary_method",
    "primary_selection",
    "secondary_methods",
    "request_hash_algorithm",
    "cache_schema",
    "budget",
    "retry_policy",
    "concurrency",
    "transport",
    "validation_metrics",
    "validation_artifacts",
    "ground_truth_inputs",
    "formal_test_expected_sample_count",
    "formal_test_expected_request_count",
    "evaluator",
}


@dataclass(frozen=True)
class SafeThresholds:
    confidence: float
    margin: float
    overall: float

    def __post_init__(self) -> None:
        values = (self.confidence, self.margin, self.overall)
        if any(not 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("safe thresholds must be in [0,1]")


def _response(value: GeminiRankingResponse | dict[str, Any]) -> GeminiRankingResponse:
    return value if isinstance(value, GeminiRankingResponse) else GeminiRankingResponse.model_validate(value)


def direct_selection(
    *,
    response: GeminiRankingResponse | dict[str, Any] | None,
    mapping: dict[str, Any],
    q_only_candidate_id: str,
    technical_fallback: bool = False,
) -> dict[str, Any]:
    if response is None or technical_fallback:
        return {
            "selected_candidate_id": q_only_candidate_id,
            "switched": False,
            "fallback": True,
            "fallback_reason": "technical_fallback",
        }
    parsed = _response(response)
    if parsed.decision == "abstain":
        return {
            "selected_candidate_id": q_only_candidate_id,
            "switched": False,
            "fallback": False,
            "model_abstain": True,
            "fallback_reason": "model_abstain",
        }
    selected = reverse_display_id(parsed.selected_candidate_id, mapping)
    return {
        "selected_candidate_id": selected,
        "switched": selected != q_only_candidate_id,
        "fallback": False,
        "model_abstain": False,
        "fallback_reason": None,
    }


def safe_selection(
    *,
    response: GeminiRankingResponse | dict[str, Any] | None,
    mapping: dict[str, Any],
    q_only_candidate_id: str,
    thresholds: SafeThresholds,
    technical_fallback: bool = False,
) -> dict[str, Any]:
    direct = direct_selection(
        response=response,
        mapping=mapping,
        q_only_candidate_id=q_only_candidate_id,
        technical_fallback=technical_fallback,
    )
    if response is None or technical_fallback or direct.get("model_abstain"):
        return direct
    parsed = _response(response)
    selected = direct["selected_candidate_id"]
    if selected == q_only_candidate_id:
        return {**direct, "safe_accepted": True, "safe_rejection_reason": None}
    selected_assessment = next(
        item for item in parsed.ranking if item.candidate_id == parsed.selected_candidate_id
    )
    conditions = {
        "decision_switch": parsed.decision == "switch",
        "confidence": parsed.confidence >= thresholds.confidence,
        "margin": parsed.score_margin_top1_top2 >= thresholds.margin,
        "overall": selected_assessment.overall_score >= thresholds.overall,
    }
    accepted = all(conditions.values())
    return {
        **direct,
        "selected_candidate_id": selected if accepted else q_only_candidate_id,
        "switched": bool(accepted),
        "safe_accepted": bool(accepted),
        "safe_rejection_reason": None
        if accepted
        else ",".join(name for name, passed in conditions.items() if not passed),
    }


def consensus_safe_selection(
    *,
    left_response: GeminiRankingResponse | dict[str, Any] | None,
    right_response: GeminiRankingResponse | dict[str, Any] | None,
    mapping: dict[str, Any],
    q_only_candidate_id: str,
    left_thresholds: SafeThresholds,
    right_thresholds: SafeThresholds,
    left_technical_fallback: bool = False,
    right_technical_fallback: bool = False,
) -> dict[str, Any]:
    left = safe_selection(
        response=left_response,
        mapping=mapping,
        q_only_candidate_id=q_only_candidate_id,
        thresholds=left_thresholds,
        technical_fallback=left_technical_fallback,
    )
    right = safe_selection(
        response=right_response,
        mapping=mapping,
        q_only_candidate_id=q_only_candidate_id,
        thresholds=right_thresholds,
        technical_fallback=right_technical_fallback,
    )
    agreed_switch = (
        left["switched"]
        and right["switched"]
        and left["selected_candidate_id"] == right["selected_candidate_id"]
        and left["selected_candidate_id"] != q_only_candidate_id
    )
    return {
        "selected_candidate_id": left["selected_candidate_id"] if agreed_switch else q_only_candidate_id,
        "switched": bool(agreed_switch),
        "left_selected_candidate_id": left["selected_candidate_id"],
        "right_selected_candidate_id": right["selected_candidate_id"],
        "consensus": bool(agreed_switch),
    }


def evaluate_binary_selections(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    total = recovered = harmful = neutral_switch = selected_correct = q_correct = 0
    switched = 0
    for record in records:
        total += 1
        old = bool(record["q_only_correct"])
        new = bool(record["selected_correct"])
        changed = bool(record.get("switched", False))
        q_correct += int(old)
        selected_correct += int(new)
        switched += int(changed)
        recovered += int((not old) and new)
        harmful += int(old and (not new))
        neutral_switch += int(changed and old == new)
    net = recovered - harmful
    return {
        "sample_count": total,
        "q_only_success_count": q_correct,
        "selected_success_count": selected_correct,
        "q_only_j1": q_correct / total if total else None,
        "selected_j1": selected_correct / total if total else None,
        "delta_j1_percentage_points": 100.0 * net / total if total else None,
        "recovered": recovered,
        "harmful": harmful,
        "net_recovered": net,
        "neutral_switch": neutral_switch,
        "switch_coverage": switched / total if total else None,
        "outcome_changing_precision": recovered / (recovered + harmful)
        if recovered + harmful
        else None,
    }


def calibrate_safe_thresholds(
    examples: list[dict[str, Any]],
    *,
    harmful_rate_limit: float = 0.01,
    grid: tuple[float, ...] = tuple(index / 20.0 for index in range(21)),
) -> tuple[SafeThresholds, dict[str, Any]]:
    if not examples:
        raise ValueError("calibration examples are empty")
    candidates = []
    for confidence in grid:
        for margin in grid:
            for overall in grid:
                rows = []
                for item in examples:
                    accept = (
                        bool(item["valid"])
                        and not bool(item.get("abstain", False))
                        and item["decision"] == "switch"
                        and float(item["confidence"]) >= confidence
                        and float(item["score_margin_top1_top2"]) >= margin
                        and float(item["selected_overall_score"]) >= overall
                        and item["selected_candidate_id"] != item["q_only_candidate_id"]
                    )
                    rows.append(
                        {
                            "q_only_correct": bool(item["q_only_correct"]),
                            "selected_correct": bool(item["selected_correct"])
                            if accept
                            else bool(item["q_only_correct"]),
                            "switched": bool(accept),
                        }
                    )
                metrics = evaluate_binary_selections(rows)
                harmful_rate = metrics["harmful"] / metrics["sample_count"]
                if harmful_rate <= harmful_rate_limit:
                    candidates.append((metrics["net_recovered"], metrics["outcome_changing_precision"] or 0.0, -sum((confidence, margin, overall)), confidence, margin, overall, metrics))
    if not candidates:
        thresholds = SafeThresholds(1.0, 1.0, 1.0)
        return thresholds, {"status": "q_only_fallback", "reason": "no_threshold_satisfies_harm_limit"}
    best = max(candidates, key=lambda item: item[:3])
    thresholds = SafeThresholds(best[3], best[4], best[5])
    if best[0] <= 0:
        thresholds = SafeThresholds(1.0, 1.0, 1.0)
        return thresholds, {"status": "q_only_fallback", "reason": "no_positive_net_gain", "best_diagnostic": best[6]}
    return thresholds, {"status": "selected", "metrics": best[6], "harmful_rate_limit": harmful_rate_limit}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def git_diff_hash(repo_root: str | Path) -> str:
    """Hash tracked changes and relevant untracked source files.

    ``git diff`` alone silently omits an untracked implementation.  The formal
    lock therefore hashes the tracked patch plus the path and bytes of every
    untracked file in the fixed experiment source/config/prompt/test/evaluator
    scopes.  Runtime outputs are deliberately outside these scopes, so normal
    formal progress cannot invalidate the code snapshot.
    """
    import subprocess

    root = Path(repo_root).resolve()
    tracked = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *LOCKED_WORKTREE_PATHS],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *LOCKED_WORKTREE_PATHS],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked.stdout)
    for relative in sorted(line for line in untracked.stdout.splitlines() if line):
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
        if not path.is_file():
            raise ValueError(f"untracked lock input is not a regular file: {relative}")
        digest.update(b"\0untracked-path\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0untracked-content\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"artifact identity requires a regular file: {resolved}")
    return {
        "identity_kind": "file_sha256",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def candidate_identity_stream_sha256(path: str | Path) -> str:
    """Recompute the frozen candidate stream without reading any GT labels."""

    digest = hashlib.sha256()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid frozen candidate JSONL at line {line_number}") from exc
            split = str(record["split"]).lower()
            split = "val" if split in {"val", "validation"} else split
            sample_id = f"multiple:{split}:{int(record['sample_id']):08d}"
            candidates = list(record["candidates"])
            ordered = sorted(
                candidates,
                key=lambda item: (-float(item["q_raw"]), str(item["candidate_id"])),
            )
            if len(ordered) != 5:
                raise ValueError(f"frozen candidate record must contain five candidates: {sample_id}")
            for rank, candidate in enumerate(ordered):
                identity = {
                    "sample_id": sample_id,
                    "candidate_id": str(candidate["candidate_id"]),
                    "original_rank": int(rank),
                    "q_raw": float(candidate["q_raw"]),
                    "candidate_checksum": str(candidate["candidate_checksum"]),
                }
                digest.update((canonical_json(identity) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("lock_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def build_lock_payload(**fields: Any) -> dict[str, Any]:
    missing = sorted(PHASE_G_REQUIRED_FIELDS - set(fields))
    if missing:
        raise ValueError(f"lock payload is missing fields: {missing}")
    if fields["store"] is not False:
        raise ValueError("formal lock requires store=False")
    if any(fields[name] is not False for name in ("background", "stream", "tools_enabled")):
        raise ValueError("formal lock requires background/stream/tools disabled")
    if fields["previous_interaction"] is not None:
        raise ValueError("formal lock forbids previous interaction state")
    if fields["transport"] not in {"standard_interactions", "batch_generate_content"}:
        raise ValueError("unknown transport")
    if fields["model_ids"] != EXACT_MODEL_IDS:
        raise ValueError("exact model IDs/order are required")
    payload = {"schema_version": LOCK_SCHEMA_VERSION, "kind": LOCK_KIND, **fields}
    payload["lock_sha256"] = _payload_sha256(payload)
    return payload


def _iter_file_identities(value: Any, trail: str = "payload"):
    if isinstance(value, dict):
        if value.get("identity_kind") == "file_sha256":
            yield trail, value
        for key, child in value.items():
            yield from _iter_file_identities(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_file_identities(child, f"{trail}[{index}]")


def _has_file_identity(value: Any) -> bool:
    return any(True for _ in _iter_file_identities(value))


def _resolve_identity_path(
    identity: dict[str, Any], *, repo_root: str | Path | None, base_dir: str | Path | None
) -> Path:
    path = Path(str(identity.get("path", "")))
    if not path.is_absolute():
        anchor = Path(repo_root) if repo_root is not None else Path(base_dir or ".")
        path = anchor / path
    return path.resolve()


def _primary_from_selection(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    values = value.get("values", value)
    if not isinstance(values, dict):
        return None
    for key in ("locked_primary", "primary_method", "selected_primary", "method"):
        selected = values.get(key)
        if isinstance(selected, str) and selected:
            return selected
    return None


def _identity_path_from_container(
    container: dict[str, Any],
    *,
    repo_root: str | Path | None,
    base_dir: str | Path | None,
) -> Path:
    source = container.get("source")
    if not isinstance(source, dict) or source.get("identity_kind") != "file_sha256":
        raise ValueError("value container has no file-backed source identity")
    return _resolve_identity_path(source, repo_root=repo_root, base_dir=base_dir)


def _verify_json_backed_values(
    name: str,
    container: Any,
    *,
    repo_root: str | Path | None,
    base_dir: str | Path | None,
) -> None:
    if not isinstance(container, dict) or "values" not in container:
        raise ValueError(f"{name} is not a value/source container")
    path = _identity_path_from_container(container, repo_root=repo_root, base_dir=base_dir)
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} source is not valid JSON") from exc
    if observed != container["values"]:
        raise ValueError(f"{name} values disagree with the locked source artifact")


def verify_lock_payload(
    payload: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    base_dir: str | Path | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Verify a Phase-G payload and every referenced external artifact."""

    if not isinstance(payload, dict):
        raise ValueError("experiment lock must contain a JSON object")
    if payload.get("kind") != LOCK_KIND or payload.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported experiment lock kind/schema")
    missing = sorted(PHASE_G_REQUIRED_FIELDS - set(payload))
    if missing:
        raise ValueError(f"experiment lock is missing fields: {missing}")
    recorded_lock_sha = payload.get("lock_sha256")
    observed_lock_sha = _payload_sha256(payload)
    if recorded_lock_sha != observed_lock_sha:
        raise ValueError("experiment lock payload hash mismatch")
    if expected_run_id is not None and payload.get("run_id") != expected_run_id:
        raise ValueError("experiment lock run_id mismatch")
    if payload.get("model_ids") != EXACT_MODEL_IDS:
        raise ValueError("experiment lock model IDs/order changed")
    if payload.get("sdk") != "google-genai==2.16.0":
        raise ValueError("experiment lock SDK changed")
    if payload.get("store") is not False:
        raise ValueError("experiment lock requires store=False")
    if any(payload.get(name) is not False for name in ("background", "stream", "tools_enabled")):
        raise ValueError("experiment lock requires background/stream/tools disabled")
    if payload.get("previous_interaction") is not None:
        raise ValueError("experiment lock forbids previous interaction state")
    if payload.get("endpoint") != "https://generativelanguage.googleapis.com/v1beta/interactions":
        raise ValueError("experiment lock endpoint changed")
    if payload.get("thinking_level") != "medium":
        raise ValueError("experiment lock thinking level changed")
    if int(payload.get("max_output_tokens", -1)) != 4096:
        raise ValueError("experiment lock requires max_output_tokens=4096")
    if payload.get("temperature_policy") != "model_default":
        raise ValueError("experiment lock temperature policy changed")
    if payload.get("image_resolution") != "high":
        raise ValueError("experiment lock image resolution changed")
    if payload.get("transport") != "standard_interactions":
        raise ValueError("formal experiment transport must remain standard Interactions")
    if payload.get("formal_test_expected_sample_count") != 17_749:
        raise ValueError("formal-test sample count changed")
    if payload.get("formal_test_expected_request_count") != 35_498:
        raise ValueError("formal-test request count changed")
    if float(payload.get("harmful_cap", -1.0)) != 0.01:
        raise ValueError("experiment lock harmful cap changed")
    if int(payload.get("concurrency", 0)) < 1:
        raise ValueError("experiment lock concurrency is invalid")
    budget = payload.get("budget")
    if (
        not isinstance(budget, dict)
        or float(budget.get("max_spend_usd", 0.0)) <= 0.0
        or float(budget.get("er2_cost_cap_per_request_usd", 0.0)) <= 0.0
    ):
        raise ValueError("experiment lock budget contract is invalid")
    if not isinstance(payload.get("primary_method"), str) or not payload["primary_method"]:
        raise ValueError("experiment lock has no validation-selected primary")
    selected_primary = _primary_from_selection(payload.get("primary_selection"))
    if selected_primary != payload["primary_method"]:
        raise ValueError("primary selection artifact disagrees with locked primary")
    validation = payload.get("validation_metrics")
    validation_values = validation.get("values") if isinstance(validation, dict) else None
    if not validation_values:
        raise ValueError("experiment lock has no completed validation metrics")
    baseline = payload.get("baseline_candidates")
    if (
        not isinstance(baseline, dict)
        or not baseline.get("candidate_identity_stream_sha256")
        or not _has_file_identity(baseline)
    ):
        raise ValueError("frozen candidate identity is not externally verifiable")
    thresholds = payload.get("safe_thresholds")
    if not isinstance(thresholds, dict) or "values" not in thresholds or not _has_file_identity(thresholds):
        raise ValueError("safe thresholds are not externally verifiable")
    for name in (
        "source_code",
        "full_run_plan",
        "cohort_manifests",
        "checkpoint",
        "config",
        "split_manifest",
        "development_protocol_lock",
        "renderer",
        "prompt",
        "evidence_schema",
        "candidate_permutation",
        "primary_selection",
        "validation_metrics",
        "validation_artifacts",
        "ground_truth_inputs",
        "calibration_grid",
        "evaluator",
    ):
        if not _has_file_identity(payload.get(name)):
            raise ValueError(f"{name} has no external file identity")

    checks: list[dict[str, Any]] = []
    for trail, identity in _iter_file_identities(payload):
        path = _resolve_identity_path(identity, repo_root=repo_root, base_dir=base_dir)
        if not path.is_file():
            raise FileNotFoundError(f"locked artifact is missing: {trail}: {path}")
        observed = sha256_file(path)
        if observed != identity.get("sha256"):
            raise ValueError(f"locked artifact hash mismatch: {trail}")
        size = path.stat().st_size
        if "size_bytes" in identity and int(identity["size_bytes"]) != size:
            raise ValueError(f"locked artifact size mismatch: {trail}")
        checks.append({"field": trail, "path": str(path), "sha256": observed, "size_bytes": size})

    for name in ("safe_thresholds", "primary_selection", "validation_metrics"):
        _verify_json_backed_values(
            name,
            payload[name],
            repo_root=repo_root,
            base_dir=base_dir,
        )
    candidate_source = baseline.get("features")
    if not isinstance(candidate_source, dict):
        raise ValueError("frozen candidate feature identity is missing")
    candidate_path = _resolve_identity_path(
        candidate_source,
        repo_root=repo_root,
        base_dir=base_dir,
    )
    observed_candidate_identity = candidate_identity_stream_sha256(candidate_path)
    if observed_candidate_identity != baseline["candidate_identity_stream_sha256"]:
        raise ValueError("frozen candidate identity stream changed")

    current_schema_sha = hashlib.sha256(
        canonical_json(response_json_schema()).encode("utf-8")
    ).hexdigest()
    schema_identity = payload.get("response_schema", {})
    if schema_identity.get("sha256") != current_schema_sha:
        raise ValueError("response schema identity changed")
    if repo_root is not None:
        root = Path(repo_root).resolve()
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        if commit != payload.get("git_commit"):
            raise ValueError("git commit changed after experiment lock")
        if git_diff_hash(root) != payload.get("git_diff_sha256"):
            raise ValueError("git diff changed after experiment lock")
    return {
        "status": "verified",
        "lock_sha256": observed_lock_sha,
        "run_id": payload["run_id"],
        "primary_method": payload["primary_method"],
        "verified_artifact_count": len(checks),
        "artifact_checks": checks,
    }


def verify_experiment_lock(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    lock_path = Path(path).resolve()
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("experiment lock is not valid JSON") from exc
    result = verify_lock_payload(
        payload,
        repo_root=repo_root,
        base_dir=lock_path.parent,
        expected_run_id=expected_run_id,
    )
    return {
        **result,
        "path": str(lock_path),
        "file_sha256": sha256_file(lock_path),
    }


def lock_experiment(path: str | Path, payload: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    path = Path(path)
    if payload.get("lock_sha256") != _payload_sha256(payload):
        raise ValueError("refusing to write an invalid experiment lock payload")
    if dry_run:
        return {
            "status": "dry_run",
            "would_write": str(path.resolve()),
            "lock_sha256": payload["lock_sha256"],
            "payload": payload,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "status": "locked",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "lock_sha256": payload["lock_sha256"],
    }


def claim_formal_test_once(
    claim_path: str | Path,
    lock_path: str | Path,
    *,
    run_id: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    claim_path, lock_path = Path(claim_path), Path(lock_path)
    if not lock_path.exists():
        raise FileNotFoundError("formal test requires an existing experiment lock")
    verification = verify_experiment_lock(
        lock_path,
        repo_root=repo_root,
        expected_run_id=run_id,
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    formal_run_id = str(run_id or lock["run_id"])
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "formal_test_claim",
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": verification["lock_sha256"],
        "lock_file_sha256": verification["file_sha256"],
        "run_id": formal_run_id,
        "primary_method": lock["primary_method"],
    }
    payload["claim_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if claim_path.exists():
        existing = json.loads(claim_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("formal test is already claimed by another lock or run_id")
        return {"status": "resumed", **existing}
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with claim_path.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = json.loads(claim_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("formal test was concurrently claimed by another lock or run_id")
        return {"status": "resumed", **existing}
    directory = os.open(claim_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"status": "claimed", **payload}
