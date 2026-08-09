#!/usr/bin/env python3
"""Run sequential, resumable, local-only Ollama Top-K candidate re-ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.experiment_lock import (  # noqa: E402
    complete_formal_stage_once,
    consume_formal_stage_once,
    selected_vlm_contract,
    verify_lock,
)
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
    local_vlm_preregistration_payload,
    validate_artifact_identity,
    validate_config_identity,
    validate_matching_artifact_identity,
)
from src.grasping.reranking_v1.identity import (  # noqa: E402
    jsonl_prefix_sha256,
    sha256_file,
)
from src.grasping.reranking_v1.local_vlm import (  # noqa: E402
    OllamaLocalVLMBackend,
    SESSION_AUDIT_POLICY_VERSION,
    SYSTEM_PROMPT,
    VLMGenerationOptions,
    VLMRankingRequest,
    ranking_json_schema,
    stable_session_contract,
    stable_session_contract_sha256,
    validate_effective_result_record,
)
from src.grasping.reranking_v1.vlm_visualization import (  # noqa: E402
    canonical_recipe_sha256,
    render_vlm_visualization_recipe,
    validate_vlm_visualization_recipe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--tmp-root",
        type=Path,
        help=(
            "Current protected run tmp/ root used only for per-sample "
            "on-demand visual materialization"
        ),
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--ollama-version", required=True)
    parser.add_argument(
        "--local-audit",
        type=Path,
        required=True,
        help="Successful audit_local_ollama.py evidence for this exact runtime",
    )
    parser.add_argument(
        "--session-local-audit",
        type=Path,
        help=(
            "Fresh live-session audit. In formal mode this may renew listener "
            "PID/start identity but must match the locked stable contract."
        ),
    )
    parser.add_argument(
        "--session-audit-policy-version",
        type=int,
        default=SESSION_AUDIT_POLICY_VERSION,
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434/api/chat",
        help="Must remain the exact Ollama loopback chat endpoint",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--status-every", type=int, default=10)
    parser.add_argument(
        "--experiment-lock",
        type=Path,
        help="Enables immutable, once-only formal VLM mode",
    )
    parser.add_argument(
        "--formal-inference-manifest",
        type=Path,
        help="Completed formal reranker inference manifest",
    )
    parser.add_argument(
        "--formal-variant",
        choices=("visual", "visual_metadata"),
        help="Locked formal VLM comparator variant",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_local_audit(
    args: argparse.Namespace, path_value: Path | None = None
) -> tuple[dict[str, Any], Path]:
    path = (
        path_value if path_value is not None else args.local_audit
    ).expanduser().resolve()
    audit = json.loads(path.read_text(encoding="utf-8"))
    expected_version = str(audit.get("ollama", {}).get("api_version", {}).get("version"))
    checks = {
        "local_only": audit.get("local_only") is True,
        "remote_api_unused": audit.get("remote_api_used") is False,
        "model_name": audit.get("model", {}).get("exact_name") == args.model_name,
        "model_digest": (
            "sha256:" + str(audit.get("model", {}).get("manifest_sha256"))
            == args.model_digest
        ),
        "backend_version": expected_version == args.ollama_version,
        "endpoint": audit.get("ollama", {}).get("endpoint")
        == "http://127.0.0.1:11434",
        "client_endpoint": args.endpoint
        == "http://127.0.0.1:11434/api/chat",
        "cloud_disabled": audit.get("ollama", {})
        .get("server_config", {})
        .get("disable_ollama_cloud")
        is True,
        "no_remote_connections": not audit.get("ollama", {}).get(
            "remote_established_connections", []
        ),
        "session_policy": int(
            audit.get("session_audit_policy_version", -1)
        )
        == SESSION_AUDIT_POLICY_VERSION,
        "stable_contract": audit.get("stable_session_contract")
        == stable_session_contract(audit),
        "stable_contract_hash": audit.get(
            "stable_session_contract_sha256"
        )
        == stable_session_contract_sha256(audit),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"local Ollama audit mismatch: {failed}")
    config_path = Path(str(audit["ollama"]["server_config_path"])).resolve()
    executable_path = Path(
        str(audit["ollama"]["executable_path"])
    ).resolve()
    if (
        not config_path.is_file()
        or _sha256(config_path) != audit["ollama"]["server_config_sha256"]
        or json.loads(config_path.read_text(encoding="utf-8")).get(
            "disable_ollama_cloud"
        )
        is not True
    ):
        raise ValueError("live Ollama server config changed since local audit")
    if (
        not executable_path.is_file()
        or _sha256(executable_path)
        != audit["ollama"]["executable_sha256"]
    ):
        raise ValueError("live Ollama executable changed since local audit")
    listener_output = subprocess.run(
        [
            "/usr/sbin/lsof",
            "-nP",
            "-iTCP:11434",
            "-sTCP:LISTEN",
        ],
        check=False,
        text=True,
        capture_output=True,
    ).stdout
    listener_rows = [
        line
        for line in listener_output.splitlines()[1:]
        if "127.0.0.1:11434" in line and "(LISTEN)" in line
    ]
    listener_pids = {
        int(line.split()[1])
        for line in listener_rows
        if len(line.split()) >= 2 and line.split()[1].isdigit()
    }
    audited_pid = int(audit["ollama"]["listener_pid"])
    if listener_pids != {audited_pid}:
        raise ValueError(
            "live Ollama listener PID differs from the audit; rerun local audit"
        )
    live_started = subprocess.run(
        ["/bin/ps", "-o", "lstart=", "-p", str(audited_pid)],
        check=False,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if live_started != str(audit["ollama"]["listener_process_started"]).strip():
        raise ValueError(
            "live Ollama listener start identity differs from the audit"
        )
    return audit, path


def _unit_bytes(value: float, unit: str) -> int:
    scales = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    return int(value * scales.get(unit.upper(), 1))


class RuntimeMonitor:
    def __init__(
        self,
        pid: int,
        interval_seconds: float = 2.0,
        durable_path: Path | None = None,
    ) -> None:
        self.pid = int(pid)
        self.interval_seconds = float(interval_seconds)
        self.samples: list[dict[str, Any]] = []
        self.durable_path = durable_path
        if durable_path is not None:
            durable_path.parent.mkdir(parents=True, exist_ok=True)
            with durable_path.open("x", encoding="utf-8"):
                pass
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error: str | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 2.0))
        if self._thread.is_alive():
            raise RuntimeError("local-only runtime monitor did not stop")
        if self.error is not None:
            raise RuntimeError(f"local-only runtime monitor failed: {self.error}")
        if not self.samples or not any(
            int(sample.get("ollama_process_count", 0)) > 0
            for sample in self.samples
        ):
            raise RuntimeError(
                "local-only runtime monitor produced no valid Ollama process sample"
            )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._record(self._snapshot())
                self._stop.wait(self.interval_seconds)
            self._record(self._snapshot())
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
            self._stop.set()

    def _record(self, sample: dict[str, Any]) -> None:
        self.samples.append(sample)
        if self.durable_path is not None:
            with self.durable_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _snapshot(self) -> dict[str, Any]:
        process_text = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,rss=,command="],
            check=False,
            text=True,
            capture_output=True,
        ).stdout
        processes: dict[int, dict[str, Any]] = {}
        for line in process_text.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) != 4:
                continue
            try:
                pid, ppid, rss_kib = map(int, parts[:3])
            except ValueError:
                continue
            processes[pid] = {
                "pid": pid,
                "ppid": ppid,
                "rss_bytes": rss_kib * 1024,
                "command": parts[3],
            }
        included = {self.pid}
        changed = True
        while changed:
            before = len(included)
            included.update(
                pid
                for pid, item in processes.items()
                if int(item["ppid"]) in included
            )
            changed = len(included) != before
        # Ollama may re-parent a runner while loading/unloading a model. Include
        # the local runner explicitly so unified-memory use is not attributed
        # only to the tiny listener process.
        included.update(
            pid
            for pid, item in processes.items()
            if "ollama" in str(item["command"]).lower()
            and "runner" in str(item["command"]).lower()
        )
        ollama_processes = [
            processes[pid] for pid in sorted(included) if pid in processes
        ]
        swap_text = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            check=False,
            text=True,
            capture_output=True,
        ).stdout.strip()
        match = re.search(r"used = ([0-9.]+)([BKMG])", swap_text)
        swap_used = _unit_bytes(float(match.group(1)), match.group(2)) if match else None
        pid_list = ",".join(str(item["pid"]) for item in ollama_processes)
        connections = subprocess.run(
            [
                "/usr/sbin/lsof",
                "-nP",
                "-a",
                "-p",
                pid_list or str(self.pid),
                "-iTCP",
            ],
            check=False,
            text=True,
            capture_output=True,
        ).stdout
        remote = [
            line
            for line in connections.splitlines()[1:]
            if "(ESTABLISHED)" in line
            and "127.0.0.1:" not in line
            and "[::1]:" not in line
        ]
        return {
            "monotonic_seconds": time.monotonic(),
            "wall_time_ns": time.time_ns(),
            "ollama_pid": self.pid,
            "ollama_processes": ollama_processes,
            "ollama_process_count": len(ollama_processes),
            "ollama_process_tree_rss_bytes": sum(
                int(item["rss_bytes"]) for item in ollama_processes
            ),
            "swap_used_bytes": swap_used,
            "remote_established_connections": remote,
        }


def _read_input(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    sample_ids = [str(row.get("sample_id", "")) for row in records]
    if any(not value for value in sample_ids):
        raise ValueError("every input row requires a non-empty sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("input sample_id values must be unique")
    return records


def _validate_input_split(
    records: list[Mapping[str, Any]], *, formal_requested: bool
) -> str:
    """Fail closed against using formal-test rows in any pre-lock VLM run."""

    allowed = {"test"} if formal_requested else {"val", "validation"}
    observed = {str(row.get("split", "")) for row in records}
    if not records or not observed or observed - allowed:
        mode = "formal test" if formal_requested else "pre-lock validation"
        raise ValueError(
            f"{mode} VLM inputs have invalid split values: {sorted(observed)}"
        )
    return "test" if formal_requested else "validation"


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def _append_jsonl_durable(path: Path, row: Mapping[str, Any]) -> None:
    """Append one completed result and fsync it before temporary cleanup."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def _validate_vlm_tmp_root(path: Path) -> tuple[Path, Path]:
    tmp_root = path.expanduser().resolve()
    run_root = _protected_run_root(tmp_root)
    configured_tmp = (run_root / "tmp").resolve()
    if tmp_root != configured_tmp and configured_tmp not in tmp_root.parents:
        raise ValueError(
            f"VLM temporary root must be below current protected {configured_tmp}"
        )
    tmp_root.mkdir(parents=True, exist_ok=True)
    return tmp_root, run_root


def _safe_remove_on_demand_visual_dir(
    render_dir: Path, *, tmp_root: Path
) -> None:
    """Delete only a generated directory below the protected run tmp root."""

    temporary, _run_root = _validate_vlm_tmp_root(tmp_root)
    managed_path = temporary / "vlm_ondemand_visuals"
    if managed_path.is_symlink():
        raise ValueError("refusing to use a symlinked VLM temporary root")
    managed_root = managed_path.resolve()
    resolved = render_dir.expanduser().resolve()
    if (
        resolved == managed_root
        or managed_root not in resolved.parents
        or temporary not in resolved.parents
    ):
        raise ValueError(
            f"refusing to delete VLM visual directory outside {managed_root}"
        )
    if not render_dir.exists():
        return
    paths = sorted(
        render_dir.rglob("*"),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in paths:
        if path.is_symlink():
            raise ValueError(
                f"refusing to delete symlink in VLM temporary directory: {path}"
            )
        path_resolved = path.resolve()
        if temporary not in path_resolved.parents:
            raise ValueError(
                f"refusing to delete VLM temporary artifact outside {temporary}"
            )
    for path in paths:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    render_dir.rmdir()
    parent = render_dir.parent
    if parent == managed_root and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


@dataclass(frozen=True)
class _MaterializedVLMRequest:
    request: VLMRankingRequest
    storage_mode: str
    render_dir: Path | None
    recipe_sha256: str | None
    visualization_manifest_sha256: str
    image_sha256: tuple[str, str]


def _recipe_binding(
    record: Mapping[str, Any],
    aggregate_item: Mapping[str, Any] | None = None,
    *,
    verify_sources: bool,
) -> dict[str, Any]:
    if record.get("visualization_storage_mode") != "on_demand_recipe":
        raise ValueError("VLM input row is not an on-demand recipe")
    recipe = record.get("visualization_recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("VLM input row omits visualization_recipe")
    recipe_sha256 = canonical_recipe_sha256(recipe)
    if record.get("visualization_recipe_sha256") != recipe_sha256:
        raise ValueError("VLM input visualization recipe changed")
    validated = validate_vlm_visualization_recipe(
        recipe, verify_sources=verify_sources
    )
    candidate_ids = list(map(str, record.get("candidate_ids", [])))
    if (
        validated["sample_id"] != str(record.get("sample_id", ""))
        or validated["candidate_ids"] != candidate_ids
    ):
        raise ValueError("VLM recipe sample/candidate identity changed")
    forbidden_persistent = {
        "full_scene_overlay_path",
        "candidate_contact_sheet_path",
        "visualization_manifest_path",
    }
    if forbidden_persistent & set(record):
        raise ValueError("on-demand VLM row contains persistent visual paths")
    if aggregate_item is not None and (
        aggregate_item.get("storage_mode") != "on_demand_recipe"
        or aggregate_item.get("candidate_ids") != candidate_ids
        or aggregate_item.get("candidate_records_sha256")
        != validated["candidate_records_sha256"]
        or aggregate_item.get("visualization_recipe_sha256")
        != recipe_sha256
    ):
        raise ValueError("aggregate VLM recipe provenance changed")
    return validated


def _materialize_request_from_record(
    record: Mapping[str, Any], *, tmp_root: Path
) -> _MaterializedVLMRequest:
    validated = _recipe_binding(record, verify_sources=True)
    temporary, _run_root = _validate_vlm_tmp_root(tmp_root)
    managed_root = temporary / "vlm_ondemand_visuals"
    if managed_root.is_symlink():
        raise ValueError("refusing to use a symlinked VLM temporary root")
    managed_root.mkdir(parents=True, exist_ok=True)
    if temporary not in managed_root.resolve().parents:
        raise ValueError("VLM temporary visual root escaped protected tmp")
    sample_key = hashlib.sha256(
        str(record["sample_id"]).encode("utf-8")
    ).hexdigest()[:20]
    render_dir = Path(
        tempfile.mkdtemp(prefix=f"{sample_key}-", dir=managed_root)
    )
    try:
        visuals = render_vlm_visualization_recipe(
            record["visualization_recipe"], output_dir=render_dir
        )
        manifest_sha256 = sha256_file(visuals.manifest_path)
        image_sha256 = (
            sha256_file(visuals.full_scene_overlay_path),
            sha256_file(visuals.candidate_contact_sheet_path),
        )
        candidate_ids = tuple(
            str(value) for value in record["candidate_ids"]
        )
        request = VLMRankingRequest(
            sample_id=str(record["sample_id"]),
            instruction=str(record["instruction"]),
            candidate_ids=candidate_ids,
            original_top1_candidate_id=str(
                record["original_top1_candidate_id"]
            ),
            image_paths=(
                visuals.full_scene_overlay_path,
                visuals.candidate_contact_sheet_path,
            ),
            visualization_manifest_path=visuals.manifest_path,
            candidate_metadata=record.get("candidate_metadata", {}),
            include_metadata=bool(record.get("include_metadata", False)),
            input_recipe_sha256=validated["recipe_sha256"],
        )
        return _MaterializedVLMRequest(
            request=request,
            storage_mode="on_demand_recipe",
            render_dir=render_dir,
            recipe_sha256=validated["recipe_sha256"],
            visualization_manifest_sha256=manifest_sha256,
            image_sha256=image_sha256,
        )
    except BaseException:
        _safe_remove_on_demand_visual_dir(
            render_dir, tmp_root=temporary
        )
        raise


def _request_from_record(record: Mapping[str, Any]) -> VLMRankingRequest:
    candidate_ids = tuple(str(value) for value in record["candidate_ids"])
    return VLMRankingRequest(
        sample_id=str(record["sample_id"]),
        instruction=str(record["instruction"]),
        candidate_ids=candidate_ids,
        original_top1_candidate_id=str(record["original_top1_candidate_id"]),
        image_paths=(
            Path(record["full_scene_overlay_path"]).expanduser().resolve(),
            Path(record["candidate_contact_sheet_path"]).expanduser().resolve(),
        ),
        visualization_manifest_path=Path(
            record["visualization_manifest_path"]
        ).expanduser().resolve(),
        candidate_metadata=record.get("candidate_metadata", {}),
        include_metadata=bool(record.get("include_metadata", False)),
    )


def _rank_recipe_record_durably(
    *,
    record: Mapping[str, Any],
    backend: OllamaLocalVLMBackend,
    tmp_root: Path,
    results_path: Path,
    ordered_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rank one recipe row, fsync result/cache evidence, then remove visuals."""

    materialized = _materialize_request_from_record(
        record, tmp_root=tmp_root
    )
    appended = False
    try:
        result = backend.rank_candidates(materialized.request).to_dict()
        result.update(
            {
                "eligible_for_vlm": True,
                "http_call_performed": not bool(result["cache_hit"]),
                "http_call_performed_this_invocation": not bool(
                    result["cache_hit"]
                ),
                "original_http_call_performed": True,
                "recovered_from_durable_cache": bool(result["cache_hit"]),
                "skip_reason": None,
                "input_record_sha256": _canonical_sha256(record),
                "visualization_storage_mode": "on_demand_recipe",
                "visualization_recipe_sha256": (
                    materialized.recipe_sha256
                ),
                "temporary_visualization_manifest_sha256": (
                    materialized.visualization_manifest_sha256
                ),
                "temporary_image_sha256": list(
                    materialized.image_sha256
                ),
            }
        )
        ordered_results.append(result)
        appended = True
        _append_jsonl_durable(results_path, result)
        return result
    except BaseException:
        if appended and ordered_results and ordered_results[-1] is result:
            ordered_results.pop()
        raise
    finally:
        assert materialized.render_dir is not None
        _safe_remove_on_demand_visual_dir(
            materialized.render_dir, tmp_root=tmp_root
        )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _assert_locked_artifact(lock: Mapping[str, Any], path: Path) -> None:
    resolved = path.expanduser().resolve()
    matches = [
        item
        for item in lock["artifacts"].values()
        if Path(item["path"]).resolve() == resolved
    ]
    if len(matches) != 1 or matches[0]["sha256"] != sha256_file(resolved):
        raise ValueError(f"formal VLM artifact is absent from or changed since lock: {resolved}")


def _locked_vlm_input_summary(
    lock: Mapping[str, Any],
    *,
    artifact_name: str,
    input_path: Path,
    expected_metadata: bool,
) -> dict[str, Any]:
    summary_path = Path(lock["artifacts"][artifact_name]["path"]).resolve()
    _assert_locked_artifact(lock, summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        summary, context=f"formal VLM input summary ({artifact_name})"
    )
    validate_matching_artifact_identity(
        summary,
        lock,
        context=f"formal VLM input summary ({artifact_name})",
    )
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("gt_fields_included") is not False
        or bool(summary.get("include_metadata")) != expected_metadata
        or Path(str(summary.get("output_jsonl", ""))).resolve() != input_path
        or summary.get("output_jsonl_sha256") != sha256_file(input_path)
    ):
        raise ValueError("formal VLM variant input disagrees with experiment lock")
    return summary


def _completed_formal_context(
    args: argparse.Namespace,
    *,
    records: list[dict[str, Any]],
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate immutable completed-run identity without live PID/images."""

    lock_path = args.experiment_lock.expanduser().resolve()
    lock = verify_lock(lock_path)
    config_path = Path(lock["artifacts"]["config"]["path"]).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("locked formal VLM config must be a mapping")
    validate_config_identity(config)
    preregistered = local_vlm_preregistration_payload(config)
    if int(args.max_output_tokens) != int(
        preregistered["max_output_tokens"]
    ):
        raise ValueError(
            "formal VLM max output tokens disagree with preregistered config"
        )
    selected_vlm = selected_vlm_contract(lock)
    if args.formal_variant != selected_vlm["source_variant"]:
        raise ValueError(
            "completed formal VLM variant is not validation-selected"
        )
    expected_metadata = selected_vlm["source_variant"] == "visual_metadata"
    if any(
        bool(record.get("include_metadata")) != expected_metadata
        for record in records
    ):
        raise ValueError("completed formal VLM input variant changed")
    input_artifact_name = (
        "vlm_test_input_manifest_visual_metadata"
        if expected_metadata
        else "vlm_test_input_manifest_visual"
    )
    input_manifest = _locked_vlm_input_summary(
        lock,
        artifact_name=input_artifact_name,
        input_path=input_path,
        expected_metadata=expected_metadata,
    )
    if len(records) != int(lock["expected_test_sample_count"]):
        raise ValueError("completed formal VLM input universe changed")
    locked_local_audit_path = args.local_audit.expanduser().resolve()
    _assert_locked_artifact(lock, locked_local_audit_path)
    locked_local_audit = json.loads(
        locked_local_audit_path.read_text(encoding="utf-8")
    )
    stable_hash = stable_session_contract_sha256(locked_local_audit)
    if locked_local_audit.get("stable_session_contract_sha256") != stable_hash:
        raise ValueError("locked local Ollama stable contract changed")
    aggregate_path = Path(
        str(input_manifest["aggregate_visual_manifest"])
    ).resolve()
    visual_artifact_name = (
        "vlm_visual_manifest_visual_metadata"
        if expected_metadata
        else "vlm_visual_manifest_visual"
    )
    locked_visual = lock["artifacts"][visual_artifact_name]
    if (
        Path(str(locked_visual["path"])).resolve() != aggregate_path
        or locked_visual["sha256"] != sha256_file(aggregate_path)
        or input_manifest.get("aggregate_visual_manifest_sha256")
        != locked_visual["sha256"]
    ):
        raise ValueError("completed formal VLM aggregate manifest changed")
    inference_path = args.formal_inference_manifest.expanduser().resolve()
    formal_ledger_path = lock_path.with_name(
        f"{lock_path.name}.FORMAL_TEST_STARTED.json"
    )
    formal_ledger = json.loads(
        formal_ledger_path.read_text(encoding="utf-8")
    )
    validate_matching_artifact_identity(
        formal_ledger,
        lock,
        context="formal reranker start ledger",
    )
    if inference_path != (
        Path(str(formal_ledger["output_root"])) / "inference_manifest.json"
    ).resolve():
        raise ValueError("completed formal reranker manifest escaped its stage")
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        inference, context="completed formal reranker inference manifest"
    )
    validate_matching_artifact_identity(
        inference,
        lock,
        context="completed formal reranker inference manifest",
    )
    if inference.get("lock_content_sha256") != lock["manifest_content_sha256"]:
        raise ValueError("completed formal reranker manifest changed")
    formal_stage = (
        "VLM_METADATA" if expected_metadata else "VLM_VISUAL"
    )
    return {
        "lock": lock,
        "lock_path": lock_path,
        "formal_inference_manifest": inference_path,
        "formal_inference_manifest_sha256": sha256_file(inference_path),
        "input_jsonl": input_path,
        "input_jsonl_sha256": sha256_file(input_path),
        "locked_local_audit": locked_local_audit_path,
        "locked_local_audit_sha256": sha256_file(
            locked_local_audit_path
        ),
        "stable_session_contract_sha256": stable_hash,
        "aggregate_visual_manifest_sha256": locked_visual["sha256"],
        "formal_variant": str(args.formal_variant),
        "source_method": selected_vlm["source_method"],
        "validation_selection_path": selected_vlm[
            "validation_selection_path"
        ],
        "validation_selection_sha256": selected_vlm[
            "validation_selection_sha256"
        ],
        "formal_stage": formal_stage,
        "model_name": str(
            locked_local_audit["model"]["exact_name"]
        ),
        "model_digest": "sha256:"
        + str(locked_local_audit["model"]["manifest_sha256"]),
        "expected_sample_count": len(records),
        "expected_eligible_count": sum(
            bool(record.get("candidate_ids")) for record in records
        ),
        "expected_empty_count": sum(
            not bool(record.get("candidate_ids")) for record in records
        ),
    }


def _formal_context(
    args: argparse.Namespace,
    *,
    records: list[dict[str, Any]],
    input_path: Path,
    locked_local_audit_path: Path,
    session_local_audit_path: Path,
    session_local_audit: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    formal_values = (
        args.experiment_lock,
        args.formal_inference_manifest,
        args.formal_variant,
    )
    if all(value is None for value in formal_values):
        return None
    if any(value is None for value in formal_values):
        raise ValueError(
            "--experiment-lock and --formal-inference-manifest are required together"
        )
    if args.limit is not None:
        raise ValueError("--limit is forbidden in formal VLM mode")
    lock_path = args.experiment_lock.expanduser().resolve()
    lock = verify_lock(lock_path)
    config_path = Path(lock["artifacts"]["config"]["path"]).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("locked formal VLM config must be a mapping")
    validate_config_identity(config)
    preregistered = local_vlm_preregistration_payload(config)
    if int(args.max_output_tokens) != int(
        preregistered["max_output_tokens"]
    ):
        raise ValueError(
            "formal VLM max output tokens disagree with preregistered config"
        )
    selected_vlm = selected_vlm_contract(lock)
    if args.formal_variant != selected_vlm["source_variant"]:
        raise ValueError(
            "formal VLM variant is not the validation-selected variant in the "
            "immutable experiment lock"
        )
    if getattr(args, "tmp_root", None) is None:
        raise ValueError(
            "--tmp-root is required for formal on-demand VLM visuals"
        )
    tmp_root, _run_root = _validate_vlm_tmp_root(args.tmp_root)
    expected_metadata = selected_vlm["source_variant"] == "visual_metadata"
    if any(bool(record.get("include_metadata")) != expected_metadata for record in records):
        raise ValueError("formal VLM input rows disagree with --formal-variant")
    input_artifact_name = (
        "vlm_test_input_manifest_visual_metadata"
        if expected_metadata
        else "vlm_test_input_manifest_visual"
    )
    input_manifest = _locked_vlm_input_summary(
        lock,
        artifact_name=input_artifact_name,
        input_path=input_path,
        expected_metadata=expected_metadata,
    )
    if len(records) != int(lock["expected_test_sample_count"]):
        raise ValueError("formal VLM input does not cover the locked test universe")
    _assert_locked_artifact(lock, locked_local_audit_path)
    locked_local_audit = json.loads(
        locked_local_audit_path.read_text(encoding="utf-8")
    )
    locked_contract_sha256 = stable_session_contract_sha256(
        locked_local_audit
    )
    session_contract_sha256 = stable_session_contract_sha256(
        session_local_audit
    )
    if (
        locked_contract_sha256 != session_contract_sha256
        or locked_local_audit.get("stable_session_contract_sha256")
        != locked_contract_sha256
        or session_local_audit.get("stable_session_contract_sha256")
        != session_contract_sha256
    ):
        raise ValueError(
            "formal live Ollama session differs from the locked stable contract"
        )
    attempts_root = (output_dir / "attempts").resolve()
    if session_local_audit_path.parent != attempts_root:
        raise ValueError(
            "formal --session-local-audit must be a unique file below "
            "the guarded stage attempts directory"
        )
    aggregate_paths = {
        Path(str(record.get("aggregate_visual_manifest_path", ""))).resolve()
        for record in records
    }
    if len(aggregate_paths) != 1:
        raise ValueError("formal VLM inputs require one aggregate visual manifest")
    aggregate_path = next(iter(aggregate_paths))
    _assert_locked_artifact(lock, aggregate_path)
    if (
        Path(
            str(input_manifest.get("aggregate_visual_manifest", ""))
        ).resolve()
        != aggregate_path
        or input_manifest.get("aggregate_visual_manifest_sha256")
        != sha256_file(aggregate_path)
    ):
        raise ValueError("formal VLM input summary disagrees with visual aggregate")
    visual_artifact_name = (
        "vlm_visual_manifest_visual_metadata"
        if expected_metadata
        else "vlm_visual_manifest_visual"
    )
    locked_visual = lock["artifacts"].get(visual_artifact_name, {})
    if Path(str(locked_visual.get("path", ""))).resolve() != aggregate_path:
        raise ValueError("locked VLM visual manifest is not the input aggregate")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if (
        aggregate.get("schema_version") != 2
        or aggregate.get("manifest_kind")
        != "vlm_visualization_recipes"
        or aggregate.get("storage_mode") != "on_demand_recipe"
        or aggregate.get("gt_free") is not True
        or aggregate.get("input_mode") != "formal_test"
        or aggregate.get("input_split") != "test"
        or int(aggregate.get("sample_count", -1)) != len(records)
        or int(aggregate.get("nonempty_sample_count", -1))
        != sum(bool(record.get("candidate_ids")) for record in records)
        or int(aggregate.get("empty_sample_count", -1))
        != sum(not bool(record.get("candidate_ids")) for record in records)
    ):
        raise ValueError(
            "formal VLM aggregate must be a GT-free on-demand recipe manifest"
        )
    by_visual_sample = {
        str(item["sample_id"]): item for item in aggregate.get("visuals", [])
    }
    if len(by_visual_sample) != len(aggregate.get("visuals", [])):
        raise ValueError("aggregate visual manifest has duplicate sample IDs")
    empty_ids = set(map(str, aggregate.get("empty_sample_ids", [])))
    for record in records:
        sample_id = str(record["sample_id"])
        candidate_ids = list(map(str, record.get("candidate_ids", [])))
        if not candidate_ids:
            if sample_id not in empty_ids or sample_id in by_visual_sample:
                raise ValueError("aggregate valid-empty visual accounting mismatch")
            continue
        item = by_visual_sample.get(sample_id)
        if item is None or item.get("candidate_ids") != candidate_ids:
            raise ValueError("aggregate visual candidate identity mismatch")
        _recipe_binding(
            record, aggregate_item=item, verify_sources=True
        )
    if set(by_visual_sample) | empty_ids != {
        str(record["sample_id"]) for record in records
    }:
        raise ValueError("aggregate visual manifest sample universe mismatch")
    if str(lock["vlm_backend"]) != "ollama":
        raise ValueError("formal VLM backend disagrees with experiment lock")
    if str(lock["vlm_model_digest"]) != args.model_digest:
        raise ValueError("formal VLM model digest disagrees with experiment lock")
    if int(lock["seeds"].get("vlm", -1)) != int(args.seed):
        raise ValueError("formal VLM seed disagrees with experiment lock")
    ranking_parameters = lock["ranking_parameters"]
    expected_generation = {
        "vlm_top_k": 5,
        "vlm_max_output_tokens": int(
            preregistered["max_output_tokens"]
        ),
        "vlm_temperature": 0.0,
        "vlm_stream": False,
        "vlm_think": False,
    }
    if any(
        ranking_parameters.get(key) != value
        for key, value in expected_generation.items()
    ):
        raise ValueError("formal VLM generation settings disagree with experiment lock")
    prompt_path = Path(lock["artifacts"]["vlm_prompt"]["path"])
    schema_path = Path(lock["artifacts"]["vlm_json_schema"]["path"])
    if prompt_path.read_text(encoding="utf-8").rstrip("\n") != SYSTEM_PROMPT:
        raise ValueError("locked VLM prompt content differs from runtime prompt")
    if json.loads(schema_path.read_text(encoding="utf-8")) != ranking_json_schema():
        raise ValueError("locked VLM JSON schema differs from runtime schema")

    universe_path = Path(lock["artifacts"]["test_sample_universe"]["path"])
    universe = pd.read_parquet(universe_path)
    universe_ids = universe["sample_id"].astype(str).tolist()
    input_ids = [str(record["sample_id"]) for record in records]
    if len(universe_ids) != len(set(universe_ids)) or set(input_ids) != set(universe_ids):
        raise ValueError("formal VLM input IDs disagree with locked test universe")
    candidate_path = Path(lock["artifacts"]["test_per_candidate"]["path"])
    candidates = pd.read_parquet(candidate_path)
    expected_top5: dict[str, list[str]] = {}
    for sample_id, group in candidates.groupby("sample_id", sort=False):
        expected_top5[str(sample_id)] = (
            group.loc[group["original_gqcnn_rank"].astype(int) <= 5]
            .sort_values(
                ["original_gqcnn_rank", "candidate_id"], kind="mergesort"
            )["candidate_id"]
            .astype(str)
            .tolist()
        )
    for record in records:
        sample_id = str(record["sample_id"])
        if list(map(str, record.get("candidate_ids", []))) != expected_top5.get(
            sample_id, []
        ):
            raise ValueError("formal VLM Top-5 pool disagrees with locked candidates")

    inference_path = args.formal_inference_manifest.expanduser().resolve()
    formal_ledger_path = lock_path.with_name(
        f"{lock_path.name}.FORMAL_TEST_STARTED.json"
    )
    formal_ledger = json.loads(formal_ledger_path.read_text(encoding="utf-8"))
    validate_matching_artifact_identity(
        formal_ledger,
        lock,
        context="formal reranker start ledger",
    )
    expected_inference_path = (
        Path(str(formal_ledger["output_root"])) / "inference_manifest.json"
    ).resolve()
    if inference_path != expected_inference_path:
        raise ValueError("formal reranker manifest is not the guarded stage output")
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        inference, context="formal reranker inference manifest"
    )
    validate_matching_artifact_identity(
        inference,
        lock,
        context="formal reranker inference manifest",
    )
    if inference.get("lock_content_sha256") != lock["manifest_content_sha256"]:
        raise ValueError("formal reranker inference belongs to another lock")
    predictions_path = Path(str(inference["predictions"])).resolve()
    if (
        not predictions_path.is_file()
        or sha256_file(predictions_path) != inference.get("predictions_sha256")
    ):
        raise ValueError("formal reranker prediction artifact changed")
    decisions_path = Path(str(inference["safe_switch_decisions"])).resolve()
    if (
        not decisions_path.is_file()
        or sha256_file(decisions_path)
        != inference.get("safe_switch_decisions_sha256")
    ):
        raise ValueError("formal reranker safe-switch artifact changed")
    if int(inference.get("expected_test_sample_count", -1)) != len(records):
        raise ValueError("formal reranker and VLM universes disagree")

    invocation = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input-jsonl",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(args.cache_dir.expanduser().resolve()),
        "--tmp-root",
        str(tmp_root),
        "--model-name",
        args.model_name,
        "--model-digest",
        args.model_digest,
        "--ollama-version",
        args.ollama_version,
        "--local-audit",
        str(locked_local_audit_path),
        "--session-audit-policy-version",
        str(SESSION_AUDIT_POLICY_VERSION),
        "--endpoint",
        args.endpoint,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--seed",
        str(args.seed),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--status-every",
        str(args.status_every),
        "--experiment-lock",
        str(lock_path),
        "--formal-inference-manifest",
        str(inference_path),
        "--formal-variant",
        str(args.formal_variant),
    ]
    formal_stage = (
        "VLM_METADATA" if expected_metadata else "VLM_VISUAL"
    )
    stage_marker = lock_path.with_name(
        f"{lock_path.name}.FORMAL_{formal_stage}_STARTED.json"
    )
    stage_resumed = stage_marker.exists()
    cache_dir = args.cache_dir.expanduser().resolve()
    if (
        not stage_resumed
        and cache_dir.exists()
        and any(path.is_file() for path in cache_dir.rglob("*"))
    ):
        raise FileExistsError(
            "a new formal VLM run requires an empty dedicated cache directory"
        )
    consume_formal_stage_once(
        lock_path,
        stage=formal_stage,
        output_root=output_dir,
        invocation=invocation,
        allowed_preexisting_files=(session_local_audit_path,),
    )
    return {
        "lock": lock,
        "lock_path": lock_path,
        "formal_inference_manifest": inference_path,
        "formal_inference_manifest_sha256": sha256_file(inference_path),
        "input_jsonl_sha256": sha256_file(input_path),
        "input_jsonl": input_path,
        "locked_local_audit": locked_local_audit_path,
        "locked_local_audit_sha256": sha256_file(
            locked_local_audit_path
        ),
        "session_local_audit": session_local_audit_path,
        "session_local_audit_sha256": sha256_file(
            session_local_audit_path
        ),
        "stable_session_contract_sha256": locked_contract_sha256,
        "aggregate_visual_manifest": aggregate_path,
        "aggregate_visual_manifest_sha256": sha256_file(aggregate_path),
        "invocation": invocation,
        "stage_resumed": stage_resumed,
        "formal_variant": str(args.formal_variant),
        "source_method": selected_vlm["source_method"],
        "validation_selection_path": selected_vlm[
            "validation_selection_path"
        ],
        "validation_selection_sha256": selected_vlm[
            "validation_selection_sha256"
        ],
        "formal_stage": formal_stage,
        "model_name": args.model_name,
        "model_digest": args.model_digest,
        "expected_sample_count": len(records),
        "expected_eligible_count": sum(
            bool(record.get("candidate_ids")) for record in records
        ),
        "expected_empty_count": sum(
            not bool(record.get("candidate_ids")) for record in records
        ),
    }


def _load_formal_prefix(
    path: Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_input(path)
    if len(rows) > len(records):
        raise ValueError("formal VLM result contains more rows than its input")
    for index, row in enumerate(rows):
        expected = records[index]
        if str(row["sample_id"]) != str(expected["sample_id"]):
            raise ValueError("formal VLM resume rows are not an exact input prefix")
        if row.get("input_record_sha256") != _canonical_sha256(expected):
            raise ValueError("formal VLM resume row input provenance changed")
        candidate_ids = list(map(str, expected.get("candidate_ids", [])))
        if candidate_ids:
            validate_effective_result_record(
                row,
                candidate_ids=candidate_ids,
                original_top1_candidate_id=candidate_ids[0],
            )
            if (
                not isinstance(row.get("request_hash"), str)
                or re.fullmatch(r"[0-9a-f]{64}", row["request_hash"])
                is None
            ):
                raise ValueError(
                    "formal VLM resume row has an invalid request hash"
                )
        elif (
            row.get("eligible_for_vlm") is not False
            or row.get("ranking") != []
            or row.get("selected_candidate_id") is not None
            or row.get("skip_reason") != "valid_empty_no_vlm_call"
            or row.get("request_hash") is not None
        ):
            raise ValueError("formal VLM resume valid-empty row is invalid")
    return rows


def _write_formal_attempt_ledger(
    *,
    output_dir: Path,
    formal: Mapping[str, Any],
    session_local_audit_path: Path,
    session_local_audit: Mapping[str, Any],
    results_path: Path,
    prefix_row_count: int,
    cache_dir: Path,
) -> tuple[Path, int]:
    attempts_root = output_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(attempts_root.glob("attempt_[0-9][0-9][0-9][0-9].json"))
    prior_session_paths: set[str] = set()
    previous_hash: str | None = None
    for expected_number, path in enumerate(existing, start=1):
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_session_path = Path(
            str(prior.get("session_local_audit", ""))
        ).resolve()
        if (
            prior_session_path.parent != attempts_root.resolve()
            or not prior_session_path.is_file()
            or sha256_file(prior_session_path)
            != prior.get("session_local_audit_sha256")
        ):
            raise ValueError(
                "formal VLM prior session-audit evidence changed"
            )
        prior_session = json.loads(
            prior_session_path.read_text(encoding="utf-8")
        )
        prior_prefix_count = int(
            prior.get("prefix_result_row_count", -1)
        )
        if (
            int(prior.get("attempt_number", -1)) != expected_number
            or prior.get("previous_attempt_sha256") != previous_hash
            or prior.get("lock_content_sha256")
            != formal["lock"]["manifest_content_sha256"]
            or prior.get("stage") != formal["formal_stage"]
            or prior.get("stable_session_contract_sha256")
            != formal["stable_session_contract_sha256"]
            or prior_session.get("local_only") is not True
            or prior_session.get("remote_api_used") is not False
            or prior_session.get(
                "stable_session_contract_sha256"
            )
            != formal["stable_session_contract_sha256"]
            or stable_session_contract_sha256(prior_session)
            != formal["stable_session_contract_sha256"]
            or prior.get("prefix_results_sha256")
            != jsonl_prefix_sha256(results_path, prior_prefix_count)
        ):
            raise ValueError("formal VLM attempt-ledger chain is invalid")
        prior_session_paths.add(str(prior_session_path))
        previous_hash = sha256_file(path)
    if str(session_local_audit_path) in prior_session_paths:
        raise FileExistsError(
            "each formal resume requires a new immutable session-audit file"
        )
    attempt_number = len(existing) + 1
    prefix_hash = jsonl_prefix_sha256(results_path, prefix_row_count)
    payload = {
        "schema_version": 1,
        **identity_payload(),
        "session_audit_policy_version": SESSION_AUDIT_POLICY_VERSION,
        "attempt_number": attempt_number,
        "previous_attempt_sha256": previous_hash,
        "lock_path": str(formal["lock_path"]),
        "lock_content_sha256": formal["lock"]["manifest_content_sha256"],
        "stage": formal["formal_stage"],
        "locked_local_audit": str(formal["locked_local_audit"]),
        "locked_local_audit_sha256": formal[
            "locked_local_audit_sha256"
        ],
        "session_local_audit": str(session_local_audit_path),
        "session_local_audit_sha256": sha256_file(
            session_local_audit_path
        ),
        "stable_session_contract_sha256": formal[
            "stable_session_contract_sha256"
        ],
        "listener_pid": int(
            session_local_audit["ollama"]["listener_pid"]
        ),
        "listener_process_started": str(
            session_local_audit["ollama"]["listener_process_started"]
        ),
        "prefix_result_row_count": int(prefix_row_count),
        "prefix_results_sha256": prefix_hash,
        "cache_dir": str(cache_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = attempts_root / f"attempt_{attempt_number:04d}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    return path, attempt_number


def _validate_completed_formal_run(
    output_dir: Path,
    context: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    manifest_path = output_dir / "formal_vlm_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        manifest, context="completed formal VLM manifest"
    )
    validate_matching_artifact_identity(
        manifest,
        context["lock"],
        context="completed formal VLM manifest",
    )
    expected = {
        "lock_content_sha256": context["lock"]["manifest_content_sha256"],
        "formal_inference_manifest_sha256": context[
            "formal_inference_manifest_sha256"
        ],
        "input_jsonl_sha256": context["input_jsonl_sha256"],
        "locked_local_audit_sha256": context[
            "locked_local_audit_sha256"
        ],
        "stable_session_contract_sha256": context[
            "stable_session_contract_sha256"
        ],
        "aggregate_visual_manifest_sha256": context[
            "aggregate_visual_manifest_sha256"
        ],
        "variant": context["formal_variant"],
        "source_method": context["source_method"],
        "validation_selection_path": context[
            "validation_selection_path"
        ],
        "validation_selection_sha256": context[
            "validation_selection_sha256"
        ],
        "stage": context["formal_stage"],
        "completed": True,
        "lock_path": str(context["lock_path"]),
        "formal_inference_manifest": str(
            context["formal_inference_manifest"]
        ),
        "input_jsonl": str(context["input_jsonl"]),
        "model_name": context["model_name"],
        "model_digest": context["model_digest"],
        "sample_count": context["expected_sample_count"],
        "eligible_sample_count": context["expected_eligible_count"],
        "empty_skipped_count": context["expected_empty_count"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise FileExistsError(f"completed formal VLM run mismatch for {key}")
    expected_artifacts = {
        "results_jsonl": output_dir / "vlm_ranking_results.jsonl",
        "summary": output_dir / "summary.json",
        "runtime_metrics": output_dir / "vlm_runtime_metrics.json",
        "runtime_monitor": output_dir / "runtime_monitor.jsonl",
    }
    for path_key, expected_path in expected_artifacts.items():
        hash_key = f"{path_key}_sha256"
        artifact = Path(str(manifest.get(path_key, ""))).resolve()
        if artifact != expected_path.resolve():
            raise ValueError(
                f"completed formal VLM artifact escaped stage root: {path_key}"
            )
        if not artifact.is_file() or sha256_file(artifact) != manifest[hash_key]:
            raise ValueError(f"completed formal VLM artifact changed: {artifact}")
    result_rows = _load_formal_prefix(
        expected_artifacts["results_jsonl"], records
    )
    if len(result_rows) != len(records):
        raise ValueError("completed formal VLM results do not cover all inputs")
    summary = json.loads(
        expected_artifacts["summary"].read_text(encoding="utf-8")
    )
    runtime = json.loads(
        expected_artifacts["runtime_metrics"].read_text(encoding="utf-8")
    )
    validate_artifact_identity(
        summary, context="completed formal VLM summary"
    )
    validate_matching_artifact_identity(
        summary,
        context["lock"],
        context="completed formal VLM summary",
    )
    validate_artifact_identity(
        runtime, context="completed formal VLM runtime"
    )
    validate_matching_artifact_identity(
        runtime,
        context["lock"],
        context="completed formal VLM runtime",
    )
    attempt_ledgers = manifest.get("formal_attempt_ledgers")
    previous_attempt_sha256: str | None = None
    if not isinstance(attempt_ledgers, list) or not attempt_ledgers:
        raise ValueError("completed formal VLM manifest omits attempt ledgers")
    for expected_number, artifact in enumerate(attempt_ledgers, start=1):
        ledger_path = Path(str(artifact.get("path", ""))).resolve()
        if (
            not ledger_path.is_file()
            or sha256_file(ledger_path) != artifact.get("sha256")
        ):
            raise ValueError("completed formal VLM attempt ledger changed")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        validate_artifact_identity(
            ledger, context="completed formal VLM attempt ledger"
        )
        validate_matching_artifact_identity(
            ledger,
            context["lock"],
            context="completed formal VLM attempt ledger",
        )
        session_path = Path(
            str(ledger.get("session_local_audit", ""))
        ).resolve()
        if (
            session_path.parent != (output_dir / "attempts").resolve()
            or not session_path.is_file()
            or sha256_file(session_path)
            != ledger.get("session_local_audit_sha256")
        ):
            raise ValueError("completed formal VLM session audit changed")
        session_audit = json.loads(
            session_path.read_text(encoding="utf-8")
        )
        prefix_count = int(ledger.get("prefix_result_row_count", -1))
        if (
            int(ledger.get("attempt_number", -1)) != expected_number
            or ledger.get("previous_attempt_sha256")
            != previous_attempt_sha256
            or ledger.get("lock_content_sha256")
            != context["lock"]["manifest_content_sha256"]
            or ledger.get("stable_session_contract_sha256")
            != context["stable_session_contract_sha256"]
            or session_audit.get("local_only") is not True
            or session_audit.get("remote_api_used") is not False
            or session_audit.get(
                "stable_session_contract_sha256"
            )
            != context["stable_session_contract_sha256"]
            or stable_session_contract_sha256(session_audit)
            != context["stable_session_contract_sha256"]
            or ledger.get("prefix_results_sha256")
            != jsonl_prefix_sha256(
                expected_artifacts["results_jsonl"], prefix_count
            )
        ):
            raise ValueError("completed formal VLM attempt chain is invalid")
        previous_attempt_sha256 = artifact["sha256"]
    if (
        Path(str(summary.get("input_jsonl", ""))).resolve()
        != Path(str(manifest["input_jsonl"])).resolve()
        or summary.get("input_jsonl_sha256")
        != context["input_jsonl_sha256"]
        or Path(str(summary.get("results_jsonl", ""))).resolve()
        != expected_artifacts["results_jsonl"].resolve()
        or summary.get("results_jsonl_sha256")
        != manifest["results_jsonl_sha256"]
        or summary.get("input_split") != "test"
        or summary.get("formal_mode") is not True
        or summary.get("stable_session_contract_sha256")
        != context["stable_session_contract_sha256"]
        or int(summary.get("sample_count", -1))
        != context["expected_sample_count"]
        or int(summary.get("eligible_sample_count", -1))
        != context["expected_eligible_count"]
        or int(summary.get("empty_skipped_count", -1))
        != context["expected_empty_count"]
        or summary.get("model_name") != context["model_name"]
        or summary.get("model_digest") != context["model_digest"]
        or Path(str(runtime.get("input_jsonl", ""))).resolve()
        != Path(str(manifest["input_jsonl"])).resolve()
        or runtime.get("input_jsonl_sha256")
        != context["input_jsonl_sha256"]
        or Path(str(runtime.get("results_jsonl", ""))).resolve()
        != expected_artifacts["results_jsonl"].resolve()
        or runtime.get("results_jsonl_sha256")
        != manifest["results_jsonl_sha256"]
        or runtime.get("input_split") != "test"
        or runtime.get("formal_mode") is not True
        or runtime.get("model_name") != context["model_name"]
        or runtime.get("model_digest") != context["model_digest"]
        or runtime.get("stable_session_contract_sha256")
        != context["stable_session_contract_sha256"]
        or int(runtime.get("sample_count", -1))
        != context["expected_sample_count"]
        or int(runtime.get("eligible_sample_count", -1))
        != context["expected_eligible_count"]
        or int(runtime.get("empty_skipped_count", -1))
        != context["expected_empty_count"]
        or runtime.get("local_only_runtime_passed") is not True
        or runtime.get("formal_attempt_ledgers") != attempt_ledgers
        or manifest.get("formal_attempt_chain_tip_sha256")
        != previous_attempt_sha256
        or runtime.get("formal_attempt_chain_tip_sha256")
        != previous_attempt_sha256
    ):
        raise ValueError("completed formal VLM summary/runtime contract is invalid")
    return manifest


def main() -> int:
    args = parse_args()
    if args.session_audit_policy_version != SESSION_AUDIT_POLICY_VERSION:
        raise ValueError("unsupported session audit policy version")
    formal_requested = any(
        value is not None
        for value in (
            args.experiment_lock,
            args.formal_inference_manifest,
            args.formal_variant,
        )
    )
    formal_values = (
        args.experiment_lock,
        args.formal_inference_manifest,
        args.formal_variant,
    )
    if formal_requested and any(value is None for value in formal_values):
        raise ValueError(
            "--experiment-lock, --formal-inference-manifest, and "
            "--formal-variant are required together"
        )
    if formal_requested and args.limit is not None:
        raise ValueError("--limit is forbidden in formal VLM mode")
    output_dir = args.output_dir.expanduser().resolve()
    input_path = args.input_jsonl.expanduser().resolve()
    records = _read_input(input_path)
    input_split = _validate_input_split(
        records, formal_requested=formal_requested
    )
    if args.limit is not None:
        records = records[: args.limit]

    # A completed formal stage is immutable and must remain independently
    # verifiable after temporary images are removed or the local server exits.
    if formal_requested and (output_dir / "formal_vlm_manifest.json").is_file():
        completed_context = _completed_formal_context(
            args,
            records=records,
            input_path=input_path,
            output_dir=output_dir,
        )
        completed = _validate_completed_formal_run(
            output_dir, completed_context, records
        )
        if completed is None:
            raise FileNotFoundError("completed formal VLM manifest disappeared")
        complete_formal_stage_once(
            completed_context["lock_path"],
            stage=completed_context["formal_stage"],
            manifest_path=output_dir / "formal_vlm_manifest.json",
        )
        print(json.dumps(completed, sort_keys=True), flush=True)
        return 0

    nonempty_records = [
        record for record in records if record.get("candidate_ids")
    ]
    storage_modes = {
        str(record.get("visualization_storage_mode", "persistent_visuals"))
        for record in records
    }
    if len(storage_modes) > 1:
        raise ValueError("one VLM run cannot mix visualization storage modes")
    uses_on_demand = storage_modes == {"on_demand_recipe"}
    if formal_requested and nonempty_records and not uses_on_demand:
        raise ValueError(
            "formal VLM execution requires GT-free on-demand recipes"
        )
    tmp_root: Path | None = None
    if uses_on_demand:
        if args.tmp_root is None:
            raise ValueError(
                "--tmp-root is required for on-demand VLM visuals"
            )
        tmp_root, _run_root = _validate_vlm_tmp_root(args.tmp_root)

    if formal_requested and args.session_local_audit is None:
        raise ValueError(
            "an incomplete formal VLM execution requires a fresh "
            "--session-local-audit"
        )
    locked_local_audit_path = args.local_audit.expanduser().resolve()
    locked_local_audit = json.loads(
        locked_local_audit_path.read_text(encoding="utf-8")
    )
    session_local_audit_path = (
        args.session_local_audit.expanduser().resolve()
        if args.session_local_audit is not None
        else locked_local_audit_path
    )
    session_local_audit, session_local_audit_path = _validate_local_audit(
        args, session_local_audit_path
    )
    if (
        stable_session_contract_sha256(locked_local_audit)
        != stable_session_contract_sha256(session_local_audit)
    ):
        raise ValueError(
            "live Ollama session does not match the selected stable contract"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "vlm_ranking_results.jsonl"
    formal = _formal_context(
        args,
        records=records,
        input_path=input_path,
        locked_local_audit_path=locked_local_audit_path,
        session_local_audit_path=session_local_audit_path,
        session_local_audit=session_local_audit,
        output_dir=output_dir,
    )
    if formal is not None:
        completed = _validate_completed_formal_run(
            output_dir, formal, records
        )
        if completed is not None:
            complete_formal_stage_once(
                formal["lock_path"],
                stage=formal["formal_stage"],
                manifest_path=output_dir / "formal_vlm_manifest.json",
            )
            print(json.dumps(completed, sort_keys=True), flush=True)
            return 0
    backend = OllamaLocalVLMBackend(
        model_name=args.model_name,
        model_digest=args.model_digest,
        backend_version=args.ollama_version,
        cache_dir=args.cache_dir.expanduser().resolve(),
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
        options=VLMGenerationOptions(
            seed=args.seed, max_output_tokens=args.max_output_tokens
        ),
        runtime_contract_sha256=stable_session_contract_sha256(
            session_local_audit
        ),
    )
    ordered_results = (
        _load_formal_prefix(results_path, records) if formal is not None else []
    )
    if uses_on_demand and formal is None:
        _atomic_jsonl(results_path, [])
    started = time.monotonic()
    historical_monitor_samples: list[dict[str, Any]] = []
    attempt_monitor_path: Path | None = None
    attempt_ledger_path: Path | None = None
    if formal is not None:
        attempt_paths = sorted(
            (output_dir / "attempts").glob("runtime_monitor.attempt_*.jsonl")
        )
        for attempt_path in attempt_paths:
            with attempt_path.open("r", encoding="utf-8") as stream:
                historical_monitor_samples.extend(
                    json.loads(line) for line in stream if line.strip()
                )
        historical_remote = [
            event
            for sample in historical_monitor_samples
            for event in sample.get("remote_established_connections", [])
        ]
        if historical_remote:
            raise RuntimeError(
                "a prior formal VLM attempt detected a non-loopback connection; "
                "the formal stage is permanently failed closed"
            )
        attempt_ledger_path, attempt_number = _write_formal_attempt_ledger(
            output_dir=output_dir,
            formal=formal,
            session_local_audit_path=session_local_audit_path,
            session_local_audit=session_local_audit,
            results_path=results_path,
            prefix_row_count=len(ordered_results),
            cache_dir=args.cache_dir.expanduser().resolve(),
        )
        attempt_monitor_path = (
            output_dir
            / "attempts"
            / f"runtime_monitor.attempt_{attempt_number:04d}.jsonl"
        )
    monitor = RuntimeMonitor(
        int(session_local_audit["ollama"]["listener_pid"]),
        durable_path=attempt_monitor_path,
    )
    monitor.start()
    newly_processed = 0
    cache_hits = 0
    resumed_count = len(ordered_results)
    for index, record in enumerate(records[resumed_count:], start=resumed_count + 1):
        sample_id = str(record["sample_id"])
        result_was_durably_appended = False
        if not record.get("candidate_ids"):
            result = {
                "sample_id": sample_id,
                "selected_candidate_id": None,
                "ranking": [],
                "confidence": 0.0,
                "abstain": False,
                "switch_from_original_top1": False,
                "fallback": False,
                "fallback_reason": None,
                "eligible_for_vlm": False,
                "http_call_performed": False,
                "http_call_performed_this_invocation": False,
                "original_http_call_performed": False,
                "recovered_from_durable_cache": False,
                "skip_reason": "valid_empty_no_vlm_call",
                "request_hash": None,
                "cache_hit": False,
                "latency_seconds": 0.0,
                "prompt_eval_count": None,
                "eval_count": None,
                "total_duration_ns": None,
                "raw_response": None,
                "parsed_model_response": None,
                "parser_error": None,
            }
        elif uses_on_demand:
            assert tmp_root is not None
            result = _rank_recipe_record_durably(
                record=record,
                backend=backend,
                tmp_root=tmp_root,
                results_path=results_path,
                ordered_results=ordered_results,
            )
            result_was_durably_appended = True
            if result["cache_hit"]:
                cache_hits += 1
            else:
                newly_processed += 1
        else:
            result = backend.rank_candidates(_request_from_record(record)).to_dict()
            result["eligible_for_vlm"] = True
            result["http_call_performed"] = not bool(result["cache_hit"])
            result["http_call_performed_this_invocation"] = not bool(
                result["cache_hit"]
            )
            result["original_http_call_performed"] = True
            result["recovered_from_durable_cache"] = bool(result["cache_hit"])
            result["skip_reason"] = None
            if result["cache_hit"]:
                cache_hits += 1
            else:
                newly_processed += 1
        if not result_was_durably_appended:
            result["input_record_sha256"] = _canonical_sha256(record)
            ordered_results.append(result)
            if uses_on_demand:
                _append_jsonl_durable(results_path, result)
                result_was_durably_appended = True
        should_flush = (
            not uses_on_demand
            and (
                index == len(records)
                or args.status_every <= 0
                or index % args.status_every == 0
            )
        )
        if should_flush:
            _atomic_jsonl(results_path, ordered_results)
        if args.status_every > 0 and index % args.status_every == 0:
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(records),
                        "newly_processed": newly_processed,
                        "cache_hits": cache_hits,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    monitor.stop()
    monitor_samples = historical_monitor_samples + monitor.samples

    fallback_counts: dict[str, int] = {}
    for row in ordered_results:
        reason = str(row.get("fallback_reason") or "none")
        fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
    current_attempt_wall = time.monotonic() - started
    cumulative_wall = current_attempt_wall
    if formal is not None:
        cumulative_wall = 0.0
        for attempt_path in sorted(
            (output_dir / "attempts").glob("runtime_monitor.attempt_*.jsonl")
        ):
            attempt_samples = [
                json.loads(line)
                for line in attempt_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(attempt_samples) >= 2:
                cumulative_wall += (
                    int(attempt_samples[-1]["wall_time_ns"])
                    - int(attempt_samples[0]["wall_time_ns"])
                ) / 1e9
    summary = {
        "schema_version": 1,
        **identity_payload(),
        "input_jsonl": str(args.input_jsonl.expanduser().resolve()),
        "input_jsonl_sha256": sha256_file(input_path),
        "results_jsonl": str(results_path),
        "results_jsonl_sha256": sha256_file(results_path),
        "sample_count": len(ordered_results),
        "newly_processed": newly_processed,
        "cache_hits": cache_hits,
        "resumed_result_count": resumed_count,
        "model_name": args.model_name,
        "model_digest": args.model_digest,
        "ollama_version": args.ollama_version,
        "endpoint": args.endpoint,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": False,
        "think": False,
        "max_output_tokens": args.max_output_tokens,
        "fallback_counts": fallback_counts,
        "wall_time_seconds": cumulative_wall,
        "wall_time_seconds_current_attempt": current_attempt_wall,
        "locked_local_audit_path": str(locked_local_audit_path),
        "locked_local_audit_sha256": _sha256(
            locked_local_audit_path
        ),
        "session_local_audit_path": str(session_local_audit_path),
        "session_local_audit_sha256": _sha256(
            session_local_audit_path
        ),
        "stable_session_contract_sha256": (
            stable_session_contract_sha256(session_local_audit)
        ),
        "formal_mode": formal is not None,
        "formal_lock_content_sha256": (
            formal["lock"]["manifest_content_sha256"]
            if formal is not None
            else None
        ),
        "formal_variant": (
            formal["formal_variant"] if formal is not None else None
        ),
        "formal_source_method": (
            formal["source_method"] if formal is not None else None
        ),
        "formal_validation_selection_sha256": (
            formal["validation_selection_sha256"]
            if formal is not None
            else None
        ),
        "visualization_storage_mode": (
            "on_demand_recipe"
            if uses_on_demand
            else "persistent_visuals"
        ),
        "on_demand_visual_sample_count": sum(
            row.get("visualization_storage_mode")
            == "on_demand_recipe"
            for row in ordered_results
        ),
        "ordinary_visual_pngs_retained": 0 if uses_on_demand else None,
    }
    eligible = [row for row in ordered_results if row["eligible_for_vlm"]]
    summary.update(
        {
            "input_split": input_split,
            "eligible_sample_count": len(eligible),
            "empty_skipped_count": len(ordered_results) - len(eligible),
        }
    )
    fresh = [row for row in eligible if row["http_call_performed"]]
    latencies = sorted(float(row["latency_seconds"]) for row in fresh)
    valid_structured = [
        row
        for row in eligible
        if isinstance(row.get("parsed_model_response"), Mapping)
        and row.get("fallback_reason") in {None, "abstain"}
    ]
    invalid_candidate_reasons = {
        "invalid_candidate_id",
        "duplicate_candidate_id",
        "missing_candidate",
        "selected_ranking_mismatch",
    }
    remote_events = [
        event
        for sample in monitor_samples
        for event in sample["remote_established_connections"]
    ]
    formal_attempt_ledgers = (
        [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(
                (output_dir / "attempts").glob(
                    "attempt_[0-9][0-9][0-9][0-9].json"
                )
            )
        ]
        if formal is not None
        else []
    )
    runtime_metrics = {
        "schema_version": 1,
        **identity_payload(),
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256_file(input_path),
        "results_jsonl": str(results_path),
        "results_jsonl_sha256": sha256_file(results_path),
        "input_split": input_split,
        "formal_mode": formal is not None,
        "model_name": args.model_name,
        "model_digest": args.model_digest,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": False,
        "think": False,
        "max_output_tokens": args.max_output_tokens,
        "stable_session_contract_sha256": (
            stable_session_contract_sha256(session_local_audit)
        ),
        "sample_count": len(ordered_results),
        "eligible_sample_count": len(eligible),
        "empty_skipped_count": len(ordered_results) - len(eligible),
        "visualization_storage_mode": (
            "on_demand_recipe"
            if uses_on_demand
            else "persistent_visuals"
        ),
        "on_demand_visual_sample_count": sum(
            row.get("visualization_storage_mode")
            == "on_demand_recipe"
            for row in ordered_results
        ),
        "ordinary_visual_pngs_retained": (
            0 if uses_on_demand else None
        ),
        "fresh_http_call_count": len(fresh),
        "cache_hit_count": sum(bool(row["cache_hit"]) for row in eligible),
        "valid_structured_response_count": len(valid_structured),
        "valid_structured_response_rate": (
            len(valid_structured) / len(eligible) if eligible else None
        ),
        "fallback_count": sum(bool(row["fallback"]) for row in eligible),
        "fallback_rate": (
            sum(bool(row["fallback"]) for row in eligible) / len(eligible)
            if eligible
            else None
        ),
        "abstain_count": sum(bool(row["abstain"]) for row in eligible),
        "abstain_rate": (
            sum(bool(row["abstain"]) for row in eligible) / len(eligible)
            if eligible
            else None
        ),
        "timeout_count": sum(
            row.get("fallback_reason") == "timeout" for row in eligible
        ),
        "timeout_rate": (
            sum(row.get("fallback_reason") == "timeout" for row in eligible)
            / len(eligible)
            if eligible
            else None
        ),
        "invalid_candidate_count": sum(
            row.get("fallback_reason") in invalid_candidate_reasons
            for row in eligible
        ),
        "invalid_candidate_rate": (
            sum(
                row.get("fallback_reason") in invalid_candidate_reasons
                for row in eligible
            )
            / len(eligible)
            if eligible
            else None
        ),
        "fresh_latency_mean_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "fresh_latency_p50_seconds": (
            statistics.median(latencies) if latencies else None
        ),
        "fresh_latency_p95_seconds": (
            statistics.quantiles(latencies, n=100, method="inclusive")[94]
            if len(latencies) > 1
            else latencies[0]
            if latencies
            else None
        ),
        "wall_time_seconds": summary["wall_time_seconds"],
        "fresh_samples_per_hour": (
            3600.0 * len(fresh) / summary["wall_time_seconds"]
            if summary["wall_time_seconds"] > 0
            else None
        ),
        "prompt_tokens_fresh_total": sum(
            int(row.get("prompt_eval_count") or 0) for row in fresh
        ),
        "output_tokens_fresh_total": sum(
            int(row.get("eval_count") or 0) for row in fresh
        ),
        "ollama_process_tree_peak_rss_bytes": max(
            (
                int(sample["ollama_process_tree_rss_bytes"])
                for sample in monitor_samples
                if sample["ollama_process_tree_rss_bytes"] is not None
            ),
            default=None,
        ),
        "ollama_peak_process_count": max(
            (int(sample["ollama_process_count"]) for sample in monitor_samples),
            default=0,
        ),
        "memory_measurement": (
            "sum of resident bytes for the Ollama listener process tree plus "
            "any re-parented Ollama runner; macOS unified-memory RSS proxy"
        ),
        "swap_peak_bytes": max(
            (
                int(sample["swap_used_bytes"])
                for sample in monitor_samples
                if sample["swap_used_bytes"] is not None
            ),
            default=None,
        ),
        "swap_start_bytes": next(
            (
                int(sample["swap_used_bytes"])
                for sample in monitor_samples
                if sample["swap_used_bytes"] is not None
            ),
            None,
        ),
        "remote_established_connection_events": remote_events,
        "remote_established_connection_event_count": len(remote_events),
        "local_only_runtime_passed": len(remote_events) == 0,
        "monitor_sample_count": len(monitor_samples),
        "monitor_attempt_count": (
            len(
                list(
                    (output_dir / "attempts").glob(
                        "runtime_monitor.attempt_*.jsonl"
                    )
                )
            )
            if formal is not None
            else 1
        ),
        "formal_attempt_ledgers": formal_attempt_ledgers,
        "formal_attempt_chain_tip_sha256": (
            formal_attempt_ledgers[-1]["sha256"]
            if formal_attempt_ledgers
            else None
        ),
    }
    if (
        runtime_metrics["swap_peak_bytes"] is not None
        and runtime_metrics["swap_start_bytes"] is not None
    ):
        runtime_metrics["swap_peak_delta_bytes"] = (
            runtime_metrics["swap_peak_bytes"]
            - runtime_metrics["swap_start_bytes"]
        )
    summary["memory_peak_mib"] = (
        runtime_metrics["ollama_process_tree_peak_rss_bytes"] / (1024.0**2)
        if runtime_metrics["ollama_process_tree_peak_rss_bytes"] is not None
        else None
    )
    with (output_dir / "runtime_monitor.jsonl").open("w", encoding="utf-8") as stream:
        for sample in monitor_samples:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
    (output_dir / "vlm_runtime_metrics.json").write_text(
        json.dumps(runtime_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    if remote_events:
        raise RuntimeError(
            "local-only runtime audit detected non-loopback established connections"
        )
    if formal is not None:
        formal_manifest = {
            "schema_version": 1,
            **identity_payload(),
            "stage": formal["formal_stage"],
            "variant": formal["formal_variant"],
            "source_method": formal["source_method"],
            "completed": True,
            "lock_path": str(formal["lock_path"]),
            "lock_content_sha256": formal["lock"]["manifest_content_sha256"],
            "validation_selection_path": formal[
                "validation_selection_path"
            ],
            "validation_selection_sha256": formal[
                "validation_selection_sha256"
            ],
            "formal_inference_manifest": str(
                formal["formal_inference_manifest"]
            ),
            "formal_inference_manifest_sha256": formal[
                "formal_inference_manifest_sha256"
            ],
            "input_jsonl": str(input_path),
            "input_jsonl_sha256": formal["input_jsonl_sha256"],
            "locked_local_audit": str(formal["locked_local_audit"]),
            "locked_local_audit_sha256": formal[
                "locked_local_audit_sha256"
            ],
            "stable_session_contract_sha256": formal[
                "stable_session_contract_sha256"
            ],
            "formal_attempt_ledgers": formal_attempt_ledgers,
            "formal_attempt_chain_tip_sha256": (
                formal_attempt_ledgers[-1]["sha256"]
                if formal_attempt_ledgers
                else None
            ),
            "aggregate_visual_manifest": str(
                formal["aggregate_visual_manifest"]
            ),
            "aggregate_visual_manifest_sha256": formal[
                "aggregate_visual_manifest_sha256"
            ],
            "model_name": args.model_name,
            "model_digest": args.model_digest,
            "results_jsonl": str(results_path),
            "results_jsonl_sha256": sha256_file(results_path),
            "summary": str(output_dir / "summary.json"),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "runtime_metrics": str(output_dir / "vlm_runtime_metrics.json"),
            "runtime_metrics_sha256": sha256_file(
                output_dir / "vlm_runtime_metrics.json"
            ),
            "runtime_monitor": str(output_dir / "runtime_monitor.jsonl"),
            "runtime_monitor_sha256": sha256_file(
                output_dir / "runtime_monitor.jsonl"
            ),
            "sample_count": len(ordered_results),
            "eligible_sample_count": len(eligible),
            "empty_skipped_count": len(ordered_results) - len(eligible),
        }
        temporary = output_dir / f".formal_vlm_manifest.{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps(formal_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir / "formal_vlm_manifest.json")
        complete_formal_stage_once(
            formal["lock_path"],
            stage=formal["formal_stage"],
            manifest_path=output_dir / "formal_vlm_manifest.json",
        )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
