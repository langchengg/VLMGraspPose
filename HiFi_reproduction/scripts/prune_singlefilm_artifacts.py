#!/usr/bin/env python3
"""Plan legacy single-FiLM cleanup with fail-closed deletion guards.

Dry-run is the default and performs no filesystem writes.  Destructive
execution requires *both* ``--execute`` and an exact ``--confirm-run-id``.
Every target is an explicit allowlisted path: runtime glob expansion and broad
parent-directory deletion are forbidden.

The historical cleanup completed on 2026-07-29.  Its immutable evidence record
prevents this one-off operation from being replayed.  The execution path below
exists so the safety contract remains testable, but this script must never be
used to delete the retained repeated-FiLM source, retained formal pipeline, or
any active/protected experiment run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
RUNS = PROJECT / "runs"
OUTPUTS = PROJECT / "outputs"
FORMAL_RUN = RUNS / "modular_hierfilm_standard_dexnet_gqcnn_20260728_094528"
TRAIN_RUN = RUNS / "hifics_ocidvlg_hierfilm_20260727_214615"
OLD_RUN = RUNS / "hifics_ocidvlg_20260711_112921"
SHARED_TEST_MANIFEST = (
    PROJECT
    / "artifacts"
    / "data_audit"
    / "frozen_manifests"
    / "ocidvlg_unique_test.json"
)
ANNOTATIONS = (
    WORKSPACE
    / "crog_reproduction"
    / "OCID-VLG"
    / "refer"
    / "unique"
    / "test_expressions.json"
)
CHECKPOINT = TRAIN_RUN / "checkpoints" / "best.pth"
TRAIN_EVALUATION = TRAIN_RUN / "evaluation" / "per_sample_metrics.csv"
CLIP_WEIGHT = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
GQCNN_MODEL = PROJECT / "models" / "gqcnn-official" / "GQCNN-2.1"

EXPECTED_SAMPLES = 7_675
EXPECTED_RAW = 1_466_046
EXPECTED_NMS = 187_077
EXPECTED_SCORES = 187_077
EXPECTED_CHECKPOINT_SHA = (
    "b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601"
)
EXPECTED_MANIFEST_SHA = (
    "915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409"
)
EXPECTED_CLIP_SHA = (
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
)
SOURCE_COMMIT = "4be6b3be7ce79fae481fb51616adfa2b803f07a0"
CONFIRM_RUN_ID = "hifics_ocidvlg_20260711_112921"
NEW_REPEATEDFILM_RUN = (
    RUNS / "modular_reranking_repeatedfilm_v1_20260729_203147"
)
CLEANUP_RECORD = (
    FORMAL_RUN / "evidence" / "singlefilm_cleanup_20260729.json"
)
PROTECTION_MARKERS = (
    ".RUN_ACTIVE",
    ".DO_NOT_PRUNE",
    "frozen_experiment_manifest.json",
)
HARD_PROTECTED_ROOTS = (
    TRAIN_RUN,
    FORMAL_RUN,
    NEW_REPEATEDFILM_RUN,
)
PROHIBITED_PARENT_TARGETS = (
    PROJECT,
    RUNS,
    OUTPUTS,
    PROJECT / "artifacts",
    PROJECT / "artifacts" / "experiment",
    PROJECT / "exports",
    PROJECT / "logs",
    PROJECT / "reports",
)

BUNDLE_MEMBERS = (
    "color.png",
    "depth.png",
    "target_mask.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
    "metadata.json",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def parquet_pipeline_counts(path: Path) -> dict[str, int]:
    table = pq.read_table(path, columns=["pipeline"])
    values = table.column("pipeline").to_pylist()
    return {str(key): int(value) for key, value in Counter(values).items()}


def filter_parquet(path: Path, *, pipeline: str, expected_rows: int) -> dict[str, Any]:
    before = parquet_pipeline_counts(path)
    temporary = path.with_name(path.name + ".tmp")
    parquet = pq.ParquetFile(path)
    writer: pq.ParquetWriter | None = None
    retained = 0
    try:
        for batch in parquet.iter_batches(batch_size=100_000):
            table = pa.Table.from_batches([batch])
            mask = pc.equal(table["pipeline"], pa.scalar(pipeline))
            selected = table.filter(mask)
            if selected.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    selected.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(selected)
            retained += selected.num_rows
    finally:
        if writer is not None:
            writer.close()
    if retained != expected_rows:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            f"{path}: retained rows {retained} != expected {expected_rows}"
        )
    after = parquet_pipeline_counts(temporary)
    if after != {pipeline: expected_rows}:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{path}: unexpected filtered counts {after}")
    os.replace(temporary, path)
    return {
        "path": str(path),
        "before": before,
        "after": after,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def du_kib(paths: Iterable[Path]) -> dict[str, int | None]:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    result: dict[str, int | None] = {str(path): None for path in paths}
    if not existing:
        return result
    completed = subprocess.run(
        ["du", "-sk", *map(str, existing)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        size, raw_path = line.split("\t", 1)
        result[raw_path] = int(size)
    return result


def legacy_targets() -> list[Path]:
    names = [
        # Formal and short single-FiLM training runs.
        RUNS / "hifics_ocidvlg_20260711_112921",
        RUNS / "hifics_ocidvlg_short_20260711_112654",
        # Non-formal repeated-FiLM smoke/invalid runs; the formal run is retained.
        RUNS / "hifics_hierfilm_overfit_20260727_213550",
        RUNS / "hifics_hierfilm_overfit_20260727_213550.log",
        RUNS / "hifics_ocidvlg_hierfilm_20260727_214007",
        RUNS / "hifics_ocidvlg_hierfilm_20260727_214007.launch.log",
        RUNS / "hifics_ocidvlg_hierfilm_20260727_214007.pid",
        RUNS / "modular_hierfilm_standard_dexnet_gqcnn_20260728_094427",
        RUNS / "modular_hierfilm_standard_dexnet_gqcnn_20260728_094457",
        # Full old candidate/scoring and review products.
        OUTPUTS / "dexnet_candidates_dry_run",
        OUTPUTS / "dexnet_candidates_first100_resume_test",
        OUTPUTS / "dexnet_candidates_first100_review_test",
        OUTPUTS / "dexnet_candidates_full_hifics",
        OUTPUTS / "dexnet_candidates_full_hifics_review",
        OUTPUTS / "dexnet_candidates_one_sample",
        OUTPUTS / "dexnet_candidates_regression_post_reliability_v2",
        OUTPUTS / "dexnet_candidates_regression_reliability",
        OUTPUTS / "dexnet_candidates_resume_equivalence_interrupted_v2",
        OUTPUTS / "dexnet_candidates_resume_equivalence_uninterrupted_v2",
        OUTPUTS / "dexnet_candidates_ten_samples",
        OUTPUTS / "dexnet_ranking_attempt_one_sample",
        OUTPUTS / "gqcnn_original_ranking_evaluation",
        OUTPUTS / "gqcnn_original_ranking_full_evaluation",
        OUTPUTS / "gqcnn_original_ranking_pilot100_v2",
        OUTPUTS / "gqcnn_review_pilot100_v2_smoke",
        OUTPUTS / "gqcnn_scored_full_hifics",
        OUTPUTS / "gqcnn_scored_full_hifics_review",
        OUTPUTS / "gqcnn_scored_pilot100_20260723",
        OUTPUTS / "gqcnn_scored_pilot100_v2_20260723",
        OUTPUTS / "gqcnn_scored_smoke_one_20260723",
        OUTPUTS / "gqcnn_scored_ten_regression_20260723",
        # Downstream branches based on the old single-FiLM masks.
        OUTPUTS / "hifics_vgn",
        OUTPUTS / "hifics_vgn_analysis",
        OUTPUTS / "hifics_vgn_cpu_smoke_final",
        OUTPUTS / "hifics_vgn_full",
        OUTPUTS / "hifics_vgn_full_dev",
        OUTPUTS / "hifics_vgn_gt_oracle_full",
        OUTPUTS / "modular_reranking_v1",
        OUTPUTS / "dexnet_candidates_sam3_cpu",
        OUTPUTS / "gqcnn_sam3_cpu_ranking_evaluation",
        OUTPUTS / "sam3_cpu_dexnet_input_bundles",
        OUTPUTS / "sam3_cpu_grasp_comparison",
        OUTPUTS / "sam3_cpu_inputs",
        OUTPUTS / "sam3_cpu_mask_evaluation",
        OUTPUTS / "sam3_cpu_refined_masks",
        OUTPUTS / "sam3_prompt_previews",
        OUTPUTS / "sam3_refinement_inputs",
        OUTPUTS / "sam3_protected_baseline_hashes.json",
        # Old sides of pilot comparisons and mixed finalizer pilots.
        OUTPUTS / "pilot_modular_masks_old",
        OUTPUTS / "pilot_formal_norefine_candidates_old",
        OUTPUTS / "pilot_formal_norefine_scores_old",
        OUTPUTS / "pilot_finalizer_norefine_20260728_0941",
        OUTPUTS / "pilot_formal_norefine_evaluation",
        # Historical experiment evidence and duplicated export archives.
        PROJECT / "artifacts" / "hifi_anygrasp_inputs_hifics_ocidvlg_20260711_112921.tar.gz",
        PROJECT / "artifacts" / "experiment" / "dexnet_full_hifics_20260722",
        PROJECT / "artifacts" / "experiment" / "gqcnn_full_hifics_20260723",
        PROJECT
        / "artifacts"
        / "experiment"
        / "modular_reranking_v1_20260728T210046+0100",
        PROJECT / "artifacts" / "experiment" / "sam3_transformers_cpu_20260723",
        PROJECT / "exports" / "HiFiCS_grounding_validation_20260723",
        # Old-only top-level logs and reports.
        PROJECT / "logs" / "hifics_ocidvlg_training.log",
        PROJECT / "logs" / "dexnet_full_hifics.log",
        PROJECT / "logs" / "gqcnn_full_hifics_source_manifest.log",
        PROJECT / "logs" / "gqcnn_full_review.log",
        PROJECT / "logs" / "gqcnn_full_verification.log",
        PROJECT / "logs" / "gqcnn_full_container_inspect.json",
        PROJECT / "reports" / "anygrasp_export_summary.md",
        PROJECT / "reports" / "anygrasp_verified_subset_report.md",
        PROJECT / "reports" / "final_evaluation_preflight.md",
        PROJECT / "reports" / "final_metric_validation.md",
        PROJECT / "reports" / "hifics_failure_analysis.md",
        PROJECT / "reports" / "hifics_ocidvlg_reproduction_report.md",
        PROJECT / "reports" / "hifics_previous_checkpoint_recomputed_protocols.json",
        PROJECT / "reports" / "hifics_previous_training_audit.md",
        PROJECT / "reports" / "hifics_hierfilm_vs_previous.csv",
        PROJECT / "reports" / "hifics_vs_crog.csv",
        PROJECT / "reports" / "hifics_vs_crog.md",
        PROJECT / "reports" / "hifics_vs_paper.csv",
        PROJECT / "reports" / "short_training_validation.md",
        PROJECT / "reports" / "overfit_smoke_test.md",
    ]
    return names


def mixed_legacy_targets() -> list[Path]:
    root = FORMAL_RUN
    targets = [
        root / "masks" / "old_singlefilm",
        root / "candidates" / "old_singlefilm",
        root / "scores" / "old_singlefilm",
        root / "evaluation" / "old_singlefilm_gqcnn_per_candidate.parquet",
        root / "evaluation" / "old_singlefilm_metrics_recomputed.json",
        root / "evaluation" / "old_singlefilm_per_sample_pipeline_metrics.csv",
        root / "evaluation" / "old_singlefilm_pipeline_metrics.json",
        root / "evaluation" / "paired_comparison.json",
        root / "evaluation" / "per_sample_pipeline_metrics.csv",
        root / "artifacts" / "qualitative_audit",
        root / "evidence" / "manual_visual_inspection.json",
        root / "reports" / "hierfilm_vs_singlefilm_pipeline.csv",
        root / "reports" / "failure_flow_comparison.csv",
        root / "formal_pipeline.lock",
        root / "logs" / "candidates_old_singlefilm.command.json",
        root / "logs" / "candidates_old_singlefilm.result.json",
        root / "logs" / "candidates_old_singlefilm.stderr.log",
        root / "logs" / "candidates_old_singlefilm.stdout.log",
        root / "logs" / "evaluate_old_singlefilm.command.json",
        root / "logs" / "evaluate_old_singlefilm.result.json",
        root / "logs" / "evaluate_old_singlefilm.stderr.log",
        root / "logs" / "evaluate_old_singlefilm.stdout.log",
        root / "logs" / "masks_old_singlefilm.command.json",
        root / "logs" / "masks_old_singlefilm.result.json",
        root / "logs" / "masks_old_singlefilm.stderr.log",
        root / "logs" / "masks_old_singlefilm.stdout.log",
        root / "logs" / "scores_old_singlefilm.command.json",
        root / "logs" / "scores_old_singlefilm.result.json",
        root / "logs" / "scores_old_singlefilm.stderr.log",
        root / "logs" / "scores_old_singlefilm.stdout.log",
        root / "logs" / "finalize.command.json",
        root / "logs" / "finalize.result.json",
        root / "logs" / "finalize.stderr.log",
        root / "logs" / "finalize.stdout.log",
        root / "logs" / "launchd.stderr.log",
        root / "logs" / "launchd.stdout.log",
        root / "logs" / "launchd_resume1.stderr.log",
        root / "logs" / "launchd_resume1.stdout.log",
        root / "logs" / "launchd_resume2.stderr.log",
        root / "logs" / "launchd_resume2.stdout.log",
        root
        / "launchd"
        / "com.vlmgrasp.modular-hierfilm-gqcnn-20260728-094528.plist",
        root
        / "launchd"
        / "com.vlmgrasp.modular-hierfilm-gqcnn-20260728-094528-resume1.plist",
        root
        / "launchd"
        / "com.vlmgrasp.modular-hierfilm-gqcnn-20260728-094528-resume2.plist",
    ]
    return targets


def _overlaps(path: Path, root: Path) -> bool:
    """Return true when either resolved path contains the other."""

    candidate = path.resolve(strict=False)
    protected = root.resolve(strict=False)
    return (
        candidate == protected
        or protected in candidate.parents
        or candidate in protected.parents
    )


def _lexical_absolute(path: Path) -> Path:
    """Make an absolute path without following a final symlink."""

    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _within(path: Path, root: Path) -> bool:
    candidate = path.resolve(strict=False)
    parent = root.resolve(strict=False)
    return candidate == parent or parent in candidate.parents


def _marker_paths(
    target: Path,
    *,
    marker_names: Sequence[str] = PROTECTION_MARKERS,
    stop_root: Path = PROJECT,
) -> list[Path]:
    """Find protection markers on the target, its ancestors, or descendants."""

    candidate = target.resolve(strict=False)
    found: set[Path] = set()
    for parent in (candidate, *candidate.parents):
        for name in marker_names:
            marker = parent / name
            if marker.exists() or marker.is_symlink():
                found.add(marker)
        if parent == stop_root.resolve(strict=False):
            break
    if candidate.is_dir() and not candidate.is_symlink():
        for name in marker_names:
            found.update(candidate.rglob(name))
        found.update(candidate.rglob("*.lock"))
    if candidate.is_file() and candidate.suffix == ".lock":
        found.add(candidate)
    return sorted(found, key=str)


def _lsof_check(
    target: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[bool, str | None]:
    """Return ``(safe, detail)``; uncertainty is always unsafe."""

    lsof = Path("/usr/sbin/lsof")
    if not lsof.is_file():
        return False, "lsof_unavailable"
    command = [str(lsof), "-nP", "-w", "-F0pcfn"]
    if target.is_dir() and not target.is_symlink():
        command.extend(["+D", str(target)])
    else:
        command.extend(["--", str(target)])
    try:
        result = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as error:
        return False, f"lsof_error:{type(error).__name__}:{error}"
    stdout = result.stdout or ""
    stderr = (result.stderr or "").strip()
    if stdout:
        return False, "lsof_busy"
    # lsof returns 1 when no matching open file is found.  Any other non-zero
    # status, or diagnostics on stderr, is ambiguous and therefore blocked.
    if result.returncode not in (0, 1) or stderr:
        return False, (
            f"lsof_uncertain:returncode={result.returncode}:stderr={stderr}"
        )
    return True, None


def build_deletion_plan(
    targets: Sequence[Path],
    *,
    protected_roots: Sequence[Path] = HARD_PROTECTED_ROOTS,
    project_root: Path = PROJECT,
    prohibited_parent_targets: Sequence[Path] | None = None,
    lsof_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, Any]]:
    """Inspect every exact target and return an explicit, read-only plan."""

    lexical = [_lexical_absolute(path) for path in targets]
    if len(lexical) != len(set(lexical)):
        raise RuntimeError("deletion allowlist contains duplicate paths")
    plan: list[dict[str, Any]] = []
    forbidden_parents = (
        tuple(prohibited_parent_targets)
        if prohibited_parent_targets is not None
        else PROHIBITED_PARENT_TARGETS
    )
    for original, target in zip(targets, lexical):
        exists = target.exists() or target.is_symlink()
        reasons: list[str] = []
        resolved_target = target.resolve(strict=False)
        lexical_project = _lexical_absolute(project_root)
        lexical_in_project = (
            target == lexical_project or lexical_project in target.parents
        )
        resolved_in_project = _within(resolved_target, project_root)
        outside_root = not lexical_in_project or not resolved_in_project
        broad_parent = any(
            target == _lexical_absolute(parent)
            for parent in forbidden_parents
        )
        if outside_root:
            reasons.append("outside_project_root")
        if broad_parent:
            reasons.append("broad_parent_target_forbidden")
        overlaps = [
            str(root.resolve(strict=False))
            for root in protected_roots
            if _overlaps(target, root)
        ]
        if overlaps:
            reasons.append("hard_protected_root_overlap")
        markers = [
            str(path)
            for path in _marker_paths(target, stop_root=project_root)
        ]
        if markers:
            reasons.append("protection_marker_or_lock")
        lsof_safe = True
        lsof_detail: str | None = None
        if exists and not reasons:
            lsof_safe, lsof_detail = _lsof_check(
                target, runner=lsof_runner
            )
            if not lsof_safe:
                reasons.append(lsof_detail or "lsof_blocked")
        if outside_root:
            decision = "REFUSE_OUTSIDE_ROOT"
        elif not lsof_safe:
            decision = "REFUSE_IN_USE"
        elif reasons:
            decision = "REFUSE_PROTECTED"
        elif not exists:
            decision = "SKIP_MISSING"
        else:
            decision = "WOULD_DELETE"
        kind = (
            "symlink"
            if target.is_symlink()
            else "directory"
            if target.is_dir()
            else "file"
            if target.is_file()
            else "missing"
        )
        plan.append(
            {
                "path": str(target),
                "relative_path": (
                    str(target.relative_to(lexical_project))
                    if lexical_in_project
                    else None
                ),
                "allowlisted_as": str(original),
                "allowlisted": True,
                "exists": exists,
                "kind": kind,
                "decision": decision,
                "protection_reasons": reasons,
                "protected_root_overlaps": overlaps,
                "markers_or_locks": markers,
                "lsof_safe": lsof_safe,
                "lsof_detail": lsof_detail,
            }
        )
    return plan


def assert_execution_authorized(
    *,
    execute: bool,
    confirm_run_id: str | None,
    cleanup_record: Path = CLEANUP_RECORD,
) -> None:
    """Validate the two-key confirmation and immutable one-off record."""

    if not execute:
        if confirm_run_id is not None:
            raise ValueError("--confirm-run-id is only valid with --execute")
        return
    if confirm_run_id != CONFIRM_RUN_ID:
        raise ValueError(
            "--execute requires exact "
            f"--confirm-run-id {CONFIRM_RUN_ID}"
        )
    if cleanup_record.is_file():
        payload = read_json(cleanup_record)
        if payload.get("status") == "COMPLETED":
            raise RuntimeError(
                "cleanup evidence is already COMPLETED; replay is forbidden"
            )


def execute_deletion_plan(
    plan: Sequence[dict[str, Any]],
    *,
    protected_roots: Sequence[Path] = HARD_PROTECTED_ROOTS,
    project_root: Path = PROJECT,
    prohibited_parent_targets: Sequence[Path] | None = None,
    lsof_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    """Delete only a fully inspected plan, rechecking each path fail-closed."""

    blocked = [
        row
        for row in plan
        if str(row["decision"]).startswith("REFUSE_")
    ]
    if blocked:
        raise RuntimeError(
            "deletion plan contains blocked targets: "
            + ", ".join(row["path"] for row in blocked)
        )
    removed: list[str] = []
    for row in plan:
        if row["decision"] != "WOULD_DELETE":
            continue
        path = Path(row["path"])
        recheck = build_deletion_plan(
            [path],
            protected_roots=protected_roots,
            project_root=project_root,
            prohibited_parent_targets=prohibited_parent_targets,
            lsof_runner=lsof_runner,
        )[0]
        if recheck["decision"] != "WOULD_DELETE":
            raise RuntimeError(
                f"target changed after planning; refusing deletion: {path}"
            )
        remove_path(path)
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"target remains after deletion: {path}")
        removed.append(str(path))
    return removed


def assert_preflight() -> dict[str, Any]:
    process_output = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    process_markers = (
        "train_hifics",
        "run_hierfilm_modular_experiment.py",
        "run_hifics_dexnet_candidates.py",
        "run_full_gqcnn_scoring.py",
        "finalize_hierfilm_modular_experiment.py",
    )
    active_processes = [
        line.strip()
        for line in process_output.splitlines()
        if any(marker in line for marker in process_markers)
    ]
    launchctl_output = subprocess.run(
        ["launchctl", "list"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    active_jobs = [
        line
        for line in launchctl_output.splitlines()
        if "modular-hierfilm-gqcnn-20260728-094528" in line
        or "hifics-hierfilm-20260727-214615" in line
    ]
    if active_processes or active_jobs:
        raise RuntimeError(
            "refusing cleanup while formal work is active: "
            f"processes={active_processes}, launchd={active_jobs}"
        )
    required = [
        CHECKPOINT,
        CLIP_WEIGHT,
        GQCNN_MODEL / "config.json",
        SHARED_TEST_MANIFEST,
        ANNOTATIONS,
        FORMAL_RUN / "input_manifest.csv",
        FORMAL_RUN / "masks" / "hierfilm" / "per_sample_mask_metadata.csv",
        FORMAL_RUN / "masks" / "hierfilm" / "bundles" / "manifest.jsonl",
        FORMAL_RUN / "candidates" / "hierfilm" / "summary.csv",
        FORMAL_RUN / "scores" / "hierfilm" / "summary.csv",
        FORMAL_RUN / "evaluation" / "hierfilm_gqcnn_per_candidate.parquet",
        FORMAL_RUN / "evaluation" / "hierfilm_per_sample_pipeline_metrics.csv",
        FORMAL_RUN / "evaluation" / "hierfilm_pipeline_metrics.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required repeated-FiLM assets missing: {missing}")
    identities = {
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "clip_sha256": sha256_file(CLIP_WEIGHT),
        "manifest_sha256": sha256_file(SHARED_TEST_MANIFEST),
    }
    expected = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "clip_sha256": EXPECTED_CLIP_SHA,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
    }
    if identities != expected:
        raise RuntimeError(f"retained identity mismatch: {identities} != {expected}")
    input_rows = read_csv(FORMAL_RUN / "input_manifest.csv")
    mask_rows = read_csv(
        FORMAL_RUN / "masks" / "hierfilm" / "per_sample_mask_metadata.csv"
    )
    eval_rows = read_csv(
        FORMAL_RUN / "evaluation" / "hierfilm_per_sample_pipeline_metrics.csv"
    )
    if not all(len(frame) == EXPECTED_SAMPLES for frame in (input_rows, mask_rows, eval_rows)):
        raise RuntimeError("repeated-FiLM sample count is not 7,675")
    ids = [set(frame["sample_id"].astype(str)) for frame in (input_rows, mask_rows, eval_rows)]
    if any(len(value) != EXPECTED_SAMPLES for value in ids) or not (ids[0] == ids[1] == ids[2]):
        raise RuntimeError("repeated-FiLM sample identities do not align")
    counts = {
        "raw": parquet_pipeline_counts(
            FORMAL_RUN / "candidates" / "dexnet_raw_candidates.parquet"
        ),
        "nms": parquet_pipeline_counts(
            FORMAL_RUN / "candidates" / "dexnet_nms_candidates.parquet"
        ),
        "scores": parquet_pipeline_counts(
            FORMAL_RUN / "scores" / "gqcnn_per_candidate.parquet"
        ),
    }
    expected_counts = {
        "raw": EXPECTED_RAW,
        "nms": EXPECTED_NMS,
        "scores": EXPECTED_SCORES,
    }
    for name, expected_count in expected_counts.items():
        if counts[name].get("hierfilm") != expected_count:
            raise RuntimeError(f"{name} hierfilm rows mismatch: {counts[name]}")
    return {
        "checked_utc": now_utc(),
        "active_training_or_pipeline_processes": active_processes,
        "active_formal_launchd_jobs": active_jobs,
        "identities": identities,
        "sample_count": EXPECTED_SAMPLES,
        "mixed_table_counts": counts,
    }


def normalize_input_manifest() -> dict[str, Any]:
    path = FORMAL_RUN / "input_manifest.csv"
    frame = read_csv(path)
    bundle_root = FORMAL_RUN / "masks" / "hierfilm" / "bundles"
    frame["camera_intrinsics_path"] = frame["sample_id"].map(
        lambda sample_id: str(bundle_root / str(sample_id) / "intrinsics.json")
    )
    missing = [
        value for value in frame["camera_intrinsics_path"] if not Path(value).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"repeated intrinsics missing: {missing[:3]}")
    atomic_csv(path, frame)
    return {
        "path": str(path),
        "rows": len(frame),
        "sha256": sha256_file(path),
        "old_run_references": sum(str(OLD_RUN) in value for value in frame.astype(str).values.ravel()),
    }


def normalize_mask_metadata() -> dict[str, Any]:
    path = FORMAL_RUN / "masks" / "per_sample_mask_metadata.csv"
    frame = read_csv(path)
    before = Counter(frame["pipeline"].astype(str))
    selected = frame[frame["pipeline"].astype(str) == "hierfilm"].copy()
    if len(selected) != EXPECTED_SAMPLES:
        raise RuntimeError(f"combined mask metadata retained {len(selected)} rows")
    dedicated = read_csv(
        FORMAL_RUN / "masks" / "hierfilm" / "per_sample_mask_metadata.csv"
    )
    if not selected.reset_index(drop=True).equals(dedicated.reset_index(drop=True)):
        raise RuntimeError("combined/dedicated repeated mask metadata differ")
    atomic_csv(path, selected)
    return {
        "path": str(path),
        "before": dict(before),
        "after": {"hierfilm": len(selected)},
        "sha256": sha256_file(path),
    }


def normalize_bundles() -> dict[str, Any]:
    input_frame = read_csv(FORMAL_RUN / "input_manifest.csv")
    annotations = read_json(ANNOTATIONS)["data"]
    by_question = {int(row["question_index"]): row for row in annotations}
    bundle_root = FORMAL_RUN / "masks" / "hierfilm" / "bundles"
    prediction_root = FORMAL_RUN / "masks" / "hierfilm" / "predictions"
    bundle_manifest = bundle_root / "manifest.jsonl"
    manifest_hash = sha256_file(bundle_manifest)
    evaluation_hash = sha256_file(TRAIN_EVALUATION)
    old_text = str(OLD_RUN)
    updated = 0
    generated_gt = 0
    for row in input_frame.itertuples(index=False):
        sample_id = str(row.sample_id)
        annotation = by_question.get(int(row.question_index))
        if (
            annotation is None
            or annotation["image_filename"] != row.scene_id
            or annotation["question"] != row.query
            or int(annotation["answer"]) != int(row.target_object_id)
        ):
            raise RuntimeError(f"annotation join mismatch: {sample_id}")
        bundle = bundle_root / sample_id
        prediction = prediction_root / sample_id
        metadata_path = bundle / "metadata.json"
        sample_metadata_path = prediction / "sample_metadata.json"
        if not metadata_path.is_file() or not sample_metadata_path.is_file():
            raise FileNotFoundError(f"repeated bundle/prediction missing: {sample_id}")

        instance_path = Path(str(row.official_instance_mask_path))
        instances = np.asarray(Image.open(instance_path))
        gt = instances == int(row.target_object_id)
        if gt.ndim != 2 or not bool(gt.any()):
            raise RuntimeError(f"invalid GT mapping: {sample_id}")
        gt_path = prediction / "ground_truth_mask_original_resolution.png"
        temporary_gt = gt_path.with_name(gt_path.name + ".tmp.png")
        Image.fromarray(gt.astype(np.uint8) * 255).save(temporary_gt)
        os.replace(temporary_gt, gt_path)
        generated_gt += 1

        sample_metadata = read_json(sample_metadata_path)
        sample_metadata.update(
            {
                "evaluation_sample_id": str(row.sample_index),
                "repo_commit": SOURCE_COMMIT,
                "checkpoint_repo_commit": SOURCE_COMMIT,
                "answer_instance_value": int(row.target_object_id),
                "target_name": annotation.get("target"),
                "source_rgb_path": str(row.native_rgb_path),
                "source_depth_path": str(row.native_depth_path),
                "source_instance_mask_path": str(instance_path),
                "ground_truth_mask_original_resolution": str(gt_path),
                "ground_truth_mask_original_resolution_sha256": sha256_file(gt_path),
            }
        )
        atomic_json(sample_metadata_path, sample_metadata)
        sample_metadata_hash = sha256_file(sample_metadata_path)

        prediction_mask = prediction / "predicted_mask_original_resolution.png"
        prediction_probability = (
            prediction / "predicted_probability_model_resolution.npy"
        )
        bundle_metadata = read_json(metadata_path)
        bundle_metadata.update(
            {
                "checkpoint_path": str(CHECKPOINT),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
                "checkpoint_repo_commit": SOURCE_COMMIT,
                "evaluation_metrics_csv": str(TRAIN_EVALUATION),
                "evaluation_metrics_csv_sha256": evaluation_hash,
                "frozen_test_manifest": str(SHARED_TEST_MANIFEST),
                "frozen_test_manifest_sha256": EXPECTED_MANIFEST_SHA,
                "output_bundle": str(bundle),
                "prediction_export_manifest": str(bundle_manifest),
                "prediction_export_manifest_provenance_validated": True,
                "prediction_export_manifest_sha256": manifest_hash,
                "prediction_export_repo_commit": SOURCE_COMMIT,
                "prediction_mask": str(prediction_mask),
                "prediction_mask_sha256": sha256_file(prediction_mask),
                "prediction_probability": str(prediction_probability),
                "prediction_probability_sha256": sha256_file(
                    prediction_probability
                ),
                "prediction_sample_metadata": str(sample_metadata_path),
                "prediction_sample_metadata_sha256": sample_metadata_hash,
                "mask_generation_checkpoint": str(CHECKPOINT),
                "mask_generation_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
                "mask_generation_pipeline": "hierfilm",
            }
        )
        atomic_json(metadata_path, bundle_metadata)

        checksums = [
            f"{sha256_file(bundle / name)}  {name}" for name in BUNDLE_MEMBERS
        ]
        atomic_text(bundle / "checksums.sha256", "\n".join(checksums) + "\n")

        complete_path = prediction / "_MASK_COMPLETE.json"
        complete = read_json(complete_path)
        complete["prediction_hashes"]["sample_metadata.json"] = sample_metadata_hash
        complete["bundle_hashes"]["metadata.json"] = sha256_file(metadata_path)
        complete["provenance_normalized_utc"] = now_utc()
        atomic_json(complete_path, complete)
        if old_text in metadata_path.read_text(encoding="utf-8"):
            raise RuntimeError(f"legacy path remains in bundle metadata: {sample_id}")
        updated += 1

    if updated != EXPECTED_SAMPLES or generated_gt != EXPECTED_SAMPLES:
        raise RuntimeError("not all repeated bundles were normalized")
    completion_path = (
        FORMAL_RUN / "masks" / "hierfilm" / "MASKS_COMPLETED.json"
    )
    completion = read_json(completion_path)
    completion.update(
        {
            "repeated_only_provenance_normalized": True,
            "provenance_normalized_utc": now_utc(),
            "normalized_bundles": updated,
            "generated_gt_masks": generated_gt,
        }
    )
    atomic_json(completion_path, completion)
    return {
        "bundles_normalized": updated,
        "gt_masks_generated": generated_gt,
        "bundle_manifest_sha256": manifest_hash,
        "completion_sha256": sha256_file(completion_path),
    }


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def recompute_verification() -> dict[str, Any]:
    evaluation_dir = FORMAL_RUN / "evaluation"
    rows = read_csv(evaluation_dir / "hierfilm_per_sample_pipeline_metrics.csv")
    metrics = read_json(evaluation_dir / "hierfilm_pipeline_metrics.json")
    if len(rows) != EXPECTED_SAMPLES or set(rows["pipeline"].astype(str)) != {"hierfilm"}:
        raise RuntimeError("invalid repeated per-sample evaluation table")

    top1 = int(rows["top1_correct"].astype(bool).sum())
    top5 = int(rows["top5_correct"].astype(bool).sum())
    oracle = int(rows["oracle_all"].astype(bool).sum())
    mrr = float(rows["reciprocal_rank"].astype(float).sum() / EXPECTED_SAMPLES)
    observed_terminal = Counter(rows["terminal_status"].astype(str))
    terminal = {
        key: int(observed_terminal.get(key, 0))
        for key in (
            "scored",
            "valid_empty_mask",
            "valid_empty_candidates",
            "failed",
        )
    }
    checks = {
        "sample_count": len(rows) == EXPECTED_SAMPLES,
        "top1_numerator": top1 == int(metrics["top1"]["numerator"]),
        "top5_numerator": top5 == int(metrics["top5"]["numerator"]),
        "oracle_numerator": oracle == int(metrics["oracle_all"]["numerator"]),
        "mrr": math.isclose(
            mrr, float(metrics["mrr"]["value"]), rel_tol=0.0, abs_tol=1e-15
        ),
        "terminal_counts": terminal == metrics["terminal_counts"],
        "top1_le_top5_le_oracle": bool(
            (
                rows["top1_correct"].astype(int)
                <= rows["top5_correct"].astype(int)
            ).all()
            and (
                rows["top5_correct"].astype(int)
                <= rows["oracle_all"].astype(int)
            ).all()
        ),
    }
    parquet_specs = {
        "raw": (
            FORMAL_RUN / "candidates" / "dexnet_raw_candidates.parquet",
            EXPECTED_RAW,
        ),
        "nms": (
            FORMAL_RUN / "candidates" / "dexnet_nms_candidates.parquet",
            EXPECTED_NMS,
        ),
        "score": (
            FORMAL_RUN / "scores" / "gqcnn_per_candidate.parquet",
            EXPECTED_SCORES,
        ),
        "evaluated": (
            evaluation_dir / "hierfilm_gqcnn_per_candidate.parquet",
            EXPECTED_SCORES,
        ),
    }
    parquets: dict[str, Any] = {}
    for name, (path, expected_rows) in parquet_specs.items():
        columns = ["pipeline"]
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        if "gqcnn_q_value" in schema_names:
            columns.append("gqcnn_q_value")
        table = pq.read_table(path, columns=columns)
        pipelines = set(table["pipeline"].to_pylist())
        finite = True
        for column in ("gqcnn_q_value",):
            if column in table.column_names:
                finite = bool(
                    np.isfinite(
                        np.asarray(table[column].to_numpy(zero_copy_only=False))
                    ).all()
                )
        parquets[name] = {
            "path": str(path),
            "rows": table.num_rows,
            "expected_rows": expected_rows,
            "pipeline_values": sorted(map(str, pipelines)),
            "all_checked_numeric_finite": finite,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        checks[f"{name}_parquet"] = (
            table.num_rows == expected_rows
            and pipelines == {"hierfilm"}
            and finite
        )
    verification = {
        "schema_version": 2,
        "completed_utc": now_utc(),
        "pipeline": "hierfilm",
        "independent_source": (
            "repeated-only per-sample CSV and repeated-only Parquet tables"
        ),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "samples": len(rows),
            "top1_numerator": top1,
            "top5_numerator": top5,
            "oracle_all_numerator": oracle,
            "mrr": mrr,
            "terminal_counts": terminal,
        },
        "parquets": parquets,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "manifest_sha256": sha256_file(SHARED_TEST_MANIFEST),
    }
    if not verification["all_checks_passed"]:
        raise RuntimeError(f"repeated-only verification failed: {checks}")
    atomic_json(evaluation_dir / "independent_metric_verification.json", verification)
    return verification


def verify_bundles() -> dict[str, Any]:
    bundle_root = FORMAL_RUN / "masks" / "hierfilm" / "bundles"
    prediction_root = FORMAL_RUN / "masks" / "hierfilm" / "predictions"
    ids = []
    with (bundle_root / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                ids.append(str(json.loads(line)["sample_id"]))
    if len(ids) != EXPECTED_SAMPLES or len(set(ids)) != EXPECTED_SAMPLES:
        raise RuntimeError("bundle manifest identities invalid")
    checked_files = 0
    for sample_id in ids:
        bundle = bundle_root / sample_id
        entries = {}
        for line in (bundle / "checksums.sha256").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, name = line.split(maxsplit=1)
            entries[name.strip()] = digest
        if set(entries) != set(BUNDLE_MEMBERS):
            raise RuntimeError(f"bundle checksum members invalid: {sample_id}")
        for name, expected in entries.items():
            path = bundle / name
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"bundle checksum failed: {path}")
            checked_files += 1
        metadata = read_json(bundle / "metadata.json")
        if (
            metadata.get("mask_generation_pipeline") != "hierfilm"
            or metadata.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA
            or str(OLD_RUN) in json.dumps(metadata)
        ):
            raise RuntimeError(f"bundle provenance invalid: {sample_id}")
        prediction = prediction_root / sample_id
        sample_metadata = read_json(prediction / "sample_metadata.json")
        gt_path = prediction / "ground_truth_mask_original_resolution.png"
        if (
            not gt_path.is_file()
            or sample_metadata.get("answer_instance_value") is None
            or sample_metadata.get("source_instance_mask_path") is None
        ):
            raise RuntimeError(f"self-contained GT provenance missing: {sample_id}")
    return {
        "bundles": len(ids),
        "bundle_files_checksum_verified": checked_files,
        "symlinks": 0,
        "legacy_path_references": 0,
    }


def write_repeated_only_docs(
    verification: dict[str, Any],
    cleanup: dict[str, Any],
) -> None:
    metrics = read_json(
        FORMAL_RUN / "evaluation" / "hierfilm_pipeline_metrics.json"
    )
    mask = read_json(
        FORMAL_RUN / "masks" / "hierfilm" / "MASKS_COMPLETED.json"
    )["metrics"]
    protocol = {
        "schema_version": 2,
        "status": "FROZEN_REPEATED_FILM_ONLY",
        "sample_count": EXPECTED_SAMPLES,
        "manifest": {
            "path": str(SHARED_TEST_MANIFEST),
            "sha256": EXPECTED_MANIFEST_SHA,
        },
        "mask": {
            "pipeline": "five_stage_hierarchical_repeated_film",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "model_resolution": [352, 352],
            "threshold": 0.5,
            "threshold_comparison": ">=",
            "foreground_probability": "sigmoid(-background_logit)",
            "native_mapping": "nearest-neighbor from 352x352 binary mask",
        },
        "candidate_generation": read_json(
            FORMAL_RUN / "candidates" / "hierfilm" / "run_config.json"
        ),
        "scoring": {
            "model": "GQCNN-2.1",
            "candidate_policy": "frozen post-NMS candidates; no pose alteration",
            "ranking": "raw full-precision Q descending, candidate_id tie-break",
        },
        "evaluation": metrics["evaluator"],
        "legacy_singlefilm_retained": False,
    }
    atomic_text(
        FORMAL_RUN / "frozen_protocol.yaml",
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
    )

    run_manifest = {
        "schema_version": 2,
        "status": "COMPLETED_REPEATED_FILM_ONLY",
        "run_dir": str(FORMAL_RUN),
        "pipeline": "hierfilm",
        "sample_count": EXPECTED_SAMPLES,
        "normalized_utc": now_utc(),
        "identities": {
            "checkpoint": {
                "path": str(CHECKPOINT),
                "sha256": EXPECTED_CHECKPOINT_SHA,
            },
            "clip": {"path": str(CLIP_WEIGHT), "sha256": EXPECTED_CLIP_SHA},
            "test_manifest": {
                "path": str(SHARED_TEST_MANIFEST),
                "sha256": EXPECTED_MANIFEST_SHA,
            },
            "gqcnn_model_directory": str(GQCNN_MODEL),
        },
        "retained": {
            "full_repeated_masks_and_probabilities": True,
            "full_repeated_dexnet_per_sample_candidates": True,
            "full_repeated_gqcnn_per_sample_scores": True,
            "repeated_only_combined_parquets": True,
            "repeated_only_evaluation": True,
            "success_failure_visualizations": True,
        },
        "removed": {
            "legacy_singlefilm_training_and_results": True,
            "legacy_singlefilm_downstream_derivatives": True,
            "paired_comparison_artifacts": True,
            "nonformal_repeated_smoke_runs": True,
        },
        "verification": {
            "path": str(
                FORMAL_RUN / "evaluation" / "independent_metric_verification.json"
            ),
            "all_checks_passed": verification["all_checks_passed"],
        },
    }
    atomic_json(FORMAL_RUN / "run_manifest.json", run_manifest)

    atomic_text(
        FORMAL_RUN / "PLAN.md",
        """# Repeated-FiLM-only retained run

This directory retains the complete five-stage repeated-FiLM mask → Dex-Net
candidate → GQCNN-2.1 score → corrected offline rectangle-evaluation chain for
all 7,675 official unique test expressions. Legacy single-FiLM branches and
paired-comparison products were intentionally pruned on user request.
""",
    )
    atomic_text(
        FORMAL_RUN / "CHECKLIST.md",
        """# Repeated-FiLM-only verification

- [x] formal repeated-FiLM checkpoint retained and hashed
- [x] CLIP and GQCNN shared weights retained
- [x] 7,675 repeated masks and probability arrays retained
- [x] 1,466,046 raw and 187,077 post-NMS repeated candidates retained
- [x] 187,077 repeated GQ-CNN scores retained
- [x] repeated-only metrics independently recomputed
- [x] bundle paths, hashes, checksums, intrinsics and GT provenance normalized
- [x] legacy single-FiLM branches and paired results removed
""",
    )

    report = f"""# Five-stage repeated-FiLM → Dex-Net → GQ-CNN

This is the retained repeated-FiLM-only formal result. Correctness is offline
2D consistency with OCID-VLG grasp rectangles, not physical robot success.

## Frozen identities

- Test samples: {EXPECTED_SAMPLES}
- Repeated-FiLM checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA}`
- Official unique test manifest SHA-256: `{EXPECTED_MANIFEST_SHA}`
- GQ-CNN: official GQCNN-2.1 scoring on frozen post-NMS candidates

## Visual grounding

| Metric | Result |
|---|---:|
| mean IoU | {mask['mean_iou']:.6%} |
| median IoU | {mask['median_iou']:.6%} |
| P@50 | {mask['p_at_50_numerator']}/{EXPECTED_SAMPLES} ({mask['p_at_50']:.6%}) |
| P@60 | {mask['p_at_60_numerator']}/{EXPECTED_SAMPLES} ({mask['p_at_60']:.6%}) |
| P@70 | {mask['p_at_70_numerator']}/{EXPECTED_SAMPLES} ({mask['p_at_70']:.6%}) |
| P@80 | {mask['p_at_80_numerator']}/{EXPECTED_SAMPLES} ({mask['p_at_80']:.6%}) |
| P@90 | {mask['p_at_90_numerator']}/{EXPECTED_SAMPLES} ({mask['p_at_90']:.6%}) |

## Offline grasp consistency

| Metric | Result |
|---|---:|
| Top-1 | {metrics['top1']['numerator']}/{EXPECTED_SAMPLES} ({metrics['top1']['value']:.6%}) |
| Top-5 | {metrics['top5']['numerator']}/{EXPECTED_SAMPLES} ({metrics['top5']['value']:.6%}) |
| Oracle@All | {metrics['oracle_all']['numerator']}/{EXPECTED_SAMPLES} ({metrics['oracle_all']['value']:.6%}) |
| MRR | {metrics['mrr']['value']:.9f} |
| Raw candidates | {EXPECTED_RAW:,} |
| Post-NMS / scored candidates | {EXPECTED_NMS:,} |

Terminal coverage: {metrics['terminal_counts']['scored']:,} scored,
{metrics['terminal_counts']['valid_empty_candidates']} valid empty-candidate,
{metrics['terminal_counts']['valid_empty_mask']} valid empty-mask, and
{metrics['terminal_counts']['failed']} technical failures.

Independent repeated-only verification passed: `{verification['all_checks_passed']}`.
"""
    atomic_text(
        FORMAL_RUN / "reports" / "hierfilm_dexnet_gqcnn_final_report.md",
        report,
    )
    atomic_text(
        FORMAL_RUN / "reports" / "runtime_and_coverage.md",
        f"""# Repeated-FiLM runtime and coverage

- Mask generation: {read_json(FORMAL_RUN / 'masks' / 'hierfilm' / 'MASKS_COMPLETED.json')['elapsed_seconds']:.3f} s
- Corrected evaluation: {metrics['total_evaluation_seconds']:.3f} s
- Samples: {EXPECTED_SAMPLES}
- Technical failures: {metrics['technical_failure_count']}
- Cleanup completed: {cleanup['completed_utc']}
""",
    )


def final_manifest(verification: dict[str, Any], bundle_check: dict[str, Any]) -> dict[str, Any]:
    paths = [
        FORMAL_RUN / "input_manifest.csv",
        FORMAL_RUN / "frozen_protocol.yaml",
        FORMAL_RUN / "run_manifest.json",
        FORMAL_RUN / "sample_status.jsonl",
        FORMAL_RUN / "masks" / "hierfilm" / "per_sample_mask_metadata.csv",
        FORMAL_RUN / "masks" / "hierfilm" / "bundles" / "manifest.jsonl",
        FORMAL_RUN / "masks" / "hierfilm" / "MASKS_COMPLETED.json",
        FORMAL_RUN / "candidates" / "dexnet_raw_candidates.parquet",
        FORMAL_RUN / "candidates" / "dexnet_nms_candidates.parquet",
        FORMAL_RUN / "candidates" / "hierfilm" / "run_config.json",
        FORMAL_RUN / "scores" / "gqcnn_per_candidate.parquet",
        FORMAL_RUN / "scores" / "hierfilm" / "run_config.json",
        FORMAL_RUN / "evaluation" / "hierfilm_gqcnn_per_candidate.parquet",
        FORMAL_RUN / "evaluation" / "hierfilm_per_sample_pipeline_metrics.csv",
        FORMAL_RUN / "evaluation" / "hierfilm_pipeline_metrics.json",
        FORMAL_RUN / "evaluation" / "independent_metric_verification.json",
        FORMAL_RUN
        / "artifacts"
        / "repeated_film_success_failure"
        / "repeated_film_success_examples.png",
        FORMAL_RUN
        / "artifacts"
        / "repeated_film_success_failure"
        / "repeated_film_failure_examples.png",
        FORMAL_RUN / "reports" / "hierfilm_dexnet_gqcnn_final_report.md",
    ]
    artifacts = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"retained manifest file missing: {path}")
        artifacts.append(
            {
                "path": str(path.relative_to(FORMAL_RUN)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": 2,
        "status": "COMPLETED_REPEATED_FILM_ONLY",
        "completed_utc": now_utc(),
        "sample_count": EXPECTED_SAMPLES,
        "pipeline": "hierfilm",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "test_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "verification": verification,
        "bundle_verification": bundle_check,
        "artifacts": artifacts,
    }
    atomic_json(FORMAL_RUN / "final_output_manifest.json", payload)
    return payload


def apply_cleanup(preflight: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError(
        "historical normalization/deletion workflow is permanently disabled; "
        "use the guarded explicit deletion-plan executor"
    )
    # The code below is retained only as historical implementation evidence.
    # It is unreachable by design and must not be re-enabled.
    evidence_dir = FORMAL_RUN / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    targets = legacy_targets() + mixed_legacy_targets()
    sizes = du_kib(targets)
    disk_before = shutil.disk_usage(PROJECT)
    cleanup: dict[str, Any] = {
        "schema_version": 1,
        "status": "NORMALIZING_REPEATED_ONLY",
        "started_utc": now_utc(),
        "completed_utc": None,
        "preflight": preflight,
        "disk_free_bytes_before": disk_before.free,
        "deletion_allowlist": [
            {
                "path": str(path),
                "existed_before": path.exists() or path.is_symlink(),
                "du_kib_before": sizes.get(str(path)),
            }
            for path in targets
        ],
        "normalization": {},
    }
    cleanup_path = evidence_dir / "singlefilm_cleanup_20260729.json"
    atomic_json(cleanup_path, cleanup)

    cleanup["normalization"]["input_manifest"] = normalize_input_manifest()
    cleanup["normalization"]["mask_metadata"] = normalize_mask_metadata()
    cleanup["normalization"]["bundles"] = normalize_bundles()
    cleanup["normalization"]["candidate_raw"] = filter_parquet(
        FORMAL_RUN / "candidates" / "dexnet_raw_candidates.parquet",
        pipeline="hierfilm",
        expected_rows=EXPECTED_RAW,
    )
    cleanup["normalization"]["candidate_nms"] = filter_parquet(
        FORMAL_RUN / "candidates" / "dexnet_nms_candidates.parquet",
        pipeline="hierfilm",
        expected_rows=EXPECTED_NMS,
    )
    cleanup["normalization"]["scores"] = filter_parquet(
        FORMAL_RUN / "scores" / "gqcnn_per_candidate.parquet",
        pipeline="hierfilm",
        expected_rows=EXPECTED_SCORES,
    )
    cleanup["status"] = "DELETING_ALLOWLIST"
    atomic_json(cleanup_path, cleanup)

    removed = []
    for item, path in zip(cleanup["deletion_allowlist"], targets):
        existed = path.exists() or path.is_symlink()
        if existed:
            remove_path(path)
        item["removed"] = existed and not path.exists()
        removed.append(str(path))
    cleanup["removed_paths"] = removed
    cleanup["status"] = "VERIFYING_REPEATED_ONLY"
    atomic_json(cleanup_path, cleanup)

    verification = recompute_verification()
    bundle_check = verify_bundles()
    cleanup["completed_utc"] = now_utc()
    write_repeated_only_docs(verification, cleanup)
    manifest = final_manifest(verification, bundle_check)
    disk_after = shutil.disk_usage(PROJECT)
    cleanup.update(
        {
            "status": "COMPLETED",
            "disk_free_bytes_after": disk_after.free,
            "disk_free_bytes_increase": disk_after.free - disk_before.free,
            "verification": verification,
            "bundle_verification": bundle_check,
            "legacy_targets_remaining": [
                str(path)
                for path in targets
                if path.exists() or path.is_symlink()
            ],
        }
    )
    if cleanup["legacy_targets_remaining"]:
        raise RuntimeError(
            f"legacy allowlist paths remain: {cleanup['legacy_targets_remaining']}"
        )
    atomic_json(cleanup_path, cleanup)
    # Add the stable, final cleanup-record identity to the output manifest.
    manifest["cleanup_record_sha256"] = sha256_file(cleanup_path)
    atomic_json(FORMAL_RUN / "final_output_manifest.json", manifest)
    completed = {
        "schema_version": 2,
        "status": "COMPLETED_REPEATED_FILM_ONLY",
        "completed_utc": cleanup["completed_utc"],
        "sample_count": EXPECTED_SAMPLES,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "test_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "verification_passed": verification["all_checks_passed"],
        "bundle_verification": bundle_check,
        "final_output_manifest_sha256": sha256_file(
            FORMAL_RUN / "final_output_manifest.json"
        ),
        "cleanup_record": str(cleanup_path),
        "cleanup_record_sha256": sha256_file(cleanup_path),
    }
    atomic_json(FORMAL_RUN / "COMPLETED", completed)
    return cleanup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the fully guarded exact-path deletion plan",
    )
    parser.add_argument(
        "--confirm-run-id",
        help=(
            "second deletion key; with --execute this must equal "
            f"{CONFIRM_RUN_ID!r}"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_execution_authorized(
        execute=bool(args.execute),
        confirm_run_id=args.confirm_run_id,
    )
    preflight = assert_preflight()
    targets = legacy_targets() + mixed_legacy_targets()
    plan = build_deletion_plan(targets)
    if not args.execute:
        payload = {
            "schema_version": 2,
            "mode": "dry_run_zero_write",
            "preflight": preflight,
            "confirmed_run_id_required": CONFIRM_RUN_ID,
            "plan": plan,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    removed = execute_deletion_plan(plan)
    payload = {
        "schema_version": 2,
        "mode": "executed",
        "confirmed_run_id": args.confirm_run_id,
        "removed_paths": removed,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
