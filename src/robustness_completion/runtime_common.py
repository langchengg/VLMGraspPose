"""Shared contracts for real deployment-runtime workers and orchestration."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import psutil

from .common import atomic_frame, atomic_json, require_run_dir, sha256_file, verify_preregistration


T = TypeVar("T")
MEMORY_POLL_SECONDS = 0.005
FOUR_D_REQUIRED_STAGES = {
    "CROG": (
        "input_io",
        "preprocess",
        "crog_model_inference",
        "candidate_decode",
        "nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    ),
    "G1": (
        "input_io",
        "preprocess",
        "hifics_inference",
        "mask_postprocess",
        "grasp_backend_inference",
        "candidate_decode",
        "nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    ),
    "C1": (
        "input_io",
        "preprocess",
        "hifics_inference",
        "mask_postprocess",
        "grasp_backend_inference",
        "candidate_decode",
        "nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    ),
    "D1": (
        "input_io",
        "preprocess",
        "hifics_inference",
        "mask_postprocess",
        "grasp_backend_inference",
        "candidate_decode",
        "nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    ),
}
SIX_D_REQUIRED_STAGES = {
    "oracle": (
        "input_io",
        "gt_mask_load",
        "depth_backprojection",
        "workspace_construction",
        "tsdf_integration",
        "vgn_inference",
        "candidate_decode",
        "pose_nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    ),
    "adapted": (
        "input_io",
        "hifics_inference",
        "mask_postprocess",
        "depth_backprojection",
        "workspace_construction",
        "tsdf_integration",
        "vgn_inference",
        "candidate_decode",
        "pose_nms",
        "runtime_feature_extraction",
        "reranker",
        "gate",
        "final_selection",
    ),
}


def synchronize_device(device: str) -> None:
    name = str(device).lower()
    if name == "cpu":
        return
    import torch

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        torch.mps.synchronize()
        return
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.synchronize()
        return
    raise ValueError(f"unsupported device: {device}")


def timed_call_ns(function: Callable[[], T], *, device: str) -> tuple[int, T]:
    synchronize_device(device)
    start = time.perf_counter_ns()
    result = function()
    synchronize_device(device)
    elapsed = time.perf_counter_ns() - start
    if elapsed < 0:
        raise AssertionError("perf_counter_ns produced a negative interval")
    return int(elapsed), result


def stable_digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def stable_sequence_round_robin(
    frame: pd.DataFrame,
    *,
    sample_column: str,
    sequence_column: str,
    count: int,
) -> pd.DataFrame:
    """Outcome-blind hash order with broad deterministic sequence coverage."""

    work = frame.drop_duplicates(sample_column).copy()
    work["_sample_hash"] = work[sample_column].astype(str).map(stable_digest)
    work["_sequence_hash"] = work[sequence_column].astype(str).map(stable_digest)
    groups = [
        group.sort_values(["_sample_hash", sample_column], kind="mergesort").reset_index(drop=True)
        for _, group in work.sort_values(["_sequence_hash", sequence_column], kind="mergesort").groupby(
            sequence_column, sort=False
        )
    ]
    rows: list[pd.Series] = []
    depth = 0
    while len(rows) < min(count, len(work)):
        changed = False
        for group in groups:
            if depth < len(group):
                rows.append(group.iloc[depth])
                changed = True
                if len(rows) == min(count, len(work)):
                    break
        if not changed:
            break
        depth += 1
    return pd.DataFrame(rows).drop(columns=["_sample_hash", "_sequence_hash"]).reset_index(drop=True)


def build_four_d_subset_manifest(repo: Path, run_dir: Path) -> dict[str, Any]:
    """Read only the formal input manifest; never import or load outcomes."""

    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    source = (
        repo
        / "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_test.parquet"
    )
    frame = pd.read_parquet(source, columns=["sample_id", "scene_id"])
    frame["sequence_id"] = frame["scene_id"].astype(str).str.rsplit(",", n=1).str[0]
    profile = stable_sequence_round_robin(
        frame,
        sample_column="sample_id",
        sequence_column="sequence_id",
        count=100,
    )
    unique = frame.drop_duplicates("sample_id").copy()
    parity = (
        unique.assign(_hash=unique["sample_id"].astype(str).map(stable_digest))
        .sort_values(["_hash", "sample_id"], kind="mergesort")
        .head(20)
    )
    payload = {
        "selection_uses_outcomes": False,
        "selection_uses_runtime": False,
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source),
        "profile_4d": {
            "count": int(len(profile)),
            "sequence_count": int(profile["sequence_id"].nunique()),
            "sample_ids": profile["sample_id"].astype(str).tolist(),
            "sequence_ids": profile["sequence_id"].astype(str).tolist(),
            "selection": "stable SHA-256 within deterministic sequence round robin",
        },
        "parity_4d": {
            "count": int(len(parity)),
            "sample_ids": parity["sample_id"].astype(str).tolist(),
            "selection": "first 20 by SHA-256(sample_id)",
        },
    }
    path = run_dir / "runtime_full/profile_subset_manifest_4d.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("4D runtime subset manifest is immutable and changed")
    else:
        atomic_json(path, payload)
    return payload


def build_six_d_subset_manifest(repo: Path, run_dir: Path) -> dict[str, Any]:
    """Select 14 groups per formal test scene without reading success columns."""

    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    roots = sorted(
        (
            repo
            / "artifacts/graspnet6d/20260819_221819_graspnet6d_vgn_lambdamart/analysis_inputs/oracle_gt_mask"
        ).glob("*/test_group_universe.parquet")
    )
    if len(roots) != 1:
        raise RuntimeError(f"expected one formal 6D test universe, found {len(roots)}")
    source = roots[0]
    frame = pd.read_parquet(source, columns=["scene_id", "group_id"])
    selected: list[pd.DataFrame] = []
    for _scene, group in frame.groupby("scene_id", sort=True):
        ordered = group.assign(
            _hash=group["group_id"].astype(str).map(stable_digest)
        ).sort_values(["_hash", "group_id"], kind="mergesort")
        if len(ordered) < 14:
            raise RuntimeError("formal 6D test scene has fewer than 14 groups")
        selected.append(ordered.head(14).drop(columns="_hash"))
    profile = pd.concat(selected, ignore_index=True)
    parity = profile.assign(
        _hash=profile["group_id"].astype(str).map(stable_digest)
    ).sort_values(["_hash", "group_id"], kind="mergesort").head(20)
    payload = {
        "selection_uses_outcomes": False,
        "selection_uses_runtime": False,
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source),
        "profile_6d": {
            "count": int(len(profile)),
            "scene_count": int(profile["scene_id"].nunique()),
            "groups_per_scene": 14,
            "group_ids": profile["group_id"].astype(str).tolist(),
            "scene_ids": profile["scene_id"].astype(str).tolist(),
            "selection": "first 14 by SHA-256(group_id) within each formal test scene",
        },
        "parity_6d": {
            "count": int(len(parity)),
            "group_ids": parity["group_id"].astype(str).tolist(),
            "selection": "first 20 by SHA-256(group_id)",
        },
    }
    path = run_dir / "runtime_full/profile_subset_manifest_6d.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("6D runtime subset manifest is immutable and changed")
    else:
        atomic_json(path, payload)
    return payload


def build_combined_subset_manifest(repo: Path, run_dir: Path) -> dict[str, Any]:
    four = build_four_d_subset_manifest(repo, run_dir)
    six = build_six_d_subset_manifest(repo, run_dir)
    payload = {
        "selection_uses_outcomes": False,
        "selection_uses_runtime": False,
        **{key: value for key, value in four.items() if key.startswith(("profile_", "parity_"))},
        **{key: value for key, value in six.items() if key.startswith(("profile_", "parity_"))},
        "source_manifests": {
            "4d": {
                "path": four["source_manifest"],
                "sha256": four["source_manifest_sha256"],
            },
            "6d": {
                "path": six["source_manifest"],
                "sha256": six["source_manifest_sha256"],
            },
        },
    }
    path = require_run_dir(repo, run_dir) / "runtime_full/profile_subset_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("combined runtime subset manifest is immutable and changed")
    else:
        atomic_json(path, payload)
    return payload


def validate_required_stages(
    frame: pd.DataFrame,
    *,
    required: Sequence[str],
    route: str,
    method: str,
) -> None:
    selected = frame[(frame["route"] == route) & (frame["method"] == method)]
    present = set(selected["stage"].astype(str))
    missing = set(required) - present
    if missing:
        raise AssertionError(f"{route}/{method} missing stages: {sorted(missing)}")
    if (selected["elapsed_ns"].astype(np.int64) < 0).any():
        raise AssertionError(f"{route}/{method} contains a negative stage time")


def _rss_tree(process: psutil.Process) -> tuple[int, int]:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass
    total = 0
    live = 0
    for child in processes:
        try:
            total += int(child.memory_info().rss)
            live += 1
        except psutil.Error:
            continue
    return total, live


def run_monitored_command(
    command: Sequence[str],
    *,
    cwd: Path,
    output_dir: Path,
    route: str,
    mode: str,
    repetition: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a fresh worker and sample parent+recursive-child RSS every 5 ms."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / f"{route}_{mode}_{repetition}.stdout.log"
    stderr_path = output_dir / f"{route}_{mode}_{repetition}.stderr.log"
    start_wall = time.time_ns()
    start = time.perf_counter_ns()
    samples: list[dict[str, Any]] = []
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        process = psutil.Process(child.pid)
        while child.poll() is None:
            rss, process_count = _rss_tree(process)
            samples.append(
                {
                    "route": route,
                    "mode": mode,
                    "repetition": repetition,
                    "timestamp_ns": time.time_ns(),
                    "elapsed_since_spawn_ns": time.perf_counter_ns() - start,
                    "rss_total_bytes": rss,
                    "recursive_process_count": process_count,
                    "child_pid": child.pid,
                }
            )
            time.sleep(MEMORY_POLL_SECONDS)
        return_code = child.wait()
    elapsed = time.perf_counter_ns() - start
    samples_frame = pd.DataFrame(samples)
    sample_path = output_dir / f"{route}_{mode}_{repetition}.memory.parquet"
    atomic_frame(sample_path, samples_frame)
    result = {
        "route": route,
        "mode": mode,
        "repetition": repetition,
        "command": list(command),
        "spawn_wall_time_ns": start_wall,
        "elapsed_ns": int(elapsed),
        "return_code": int(return_code),
        "peak_total_rss_bytes": (
            int(samples_frame["rss_total_bytes"].max()) if len(samples_frame) else None
        ),
        "memory_sample_count": int(len(samples_frame)),
        "memory_poll_interval_ms": MEMORY_POLL_SECONDS * 1000,
        "memory_samples_path": str(sample_path.resolve()),
        "memory_samples_sha256": sha256_file(sample_path),
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
    }
    atomic_json(output_dir / f"{route}_{mode}_{repetition}.monitor.json", result)
    return result
