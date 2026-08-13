"""Predeclared R0-R7 method/budget/job universe for the D1 extension."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.training import FORMAL_SEEDS

from .execution import load_content_manifest
from .provenance import load_source_closure
from .rules import R1_FORMULA


NEURAL_METHODS = {
    "R2": "bce",
    "R3": "ranknet",
    "R4": "listwise",
    "R6": "jacquard_margin_ranknet",
}
R7_ELIGIBLE = ("R3", "R5", "R6")
HIDDEN_GRID = ((64, 32), (128, 64))
DROPOUT_GRID = (0.0, 0.1)
LAMBDAMART_GRID = tuple(
    {"num_leaves": leaves, "learning_rate": learning_rate, "n_estimators": 200}
    for leaves, learning_rate in itertools.product((15, 31), (0.03, 0.05))
)
R1_GRID = (
    {"name": "q_plus_center", "beta_center": 0.5, "beta_rect": 0.0, "beta_jaw": 0.0},
    {"name": "q_plus_rectangle", "beta_center": 0.0, "beta_rect": 0.5, "beta_jaw": 0.0},
    {"name": "q_plus_jaw", "beta_center": 0.0, "beta_rect": 0.0, "beta_jaw": 0.5},
    {
        "name": "q_plus_all_soft_support",
        "beta_center": 0.5,
        "beta_rect": 0.5,
        "beta_jaw": 0.5,
    },
)
PRIMARY_PLAN_RELATIVE_PATH = "configs/d1_primary_matrix_plan.json"
PRIMARY_PLAN_REGISTRY_RELATIVE = Path("configs/primary_plans")
PRIMARY_PLAN_POINTER_RELATIVE = Path("configs/d1_primary_matrix_plan_active.json")
PRIMARY_PYTHON_RELATIVE_PATH = "HiFi_reproduction/.venv-grasp4dof/bin/python"


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"D1 primary-plan source is not regular: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _plan_run_root(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if source.parent.name == "primary_plans" and source.parent.parent.name == "configs":
        return source.parent.parent.parent
    if source.parent.name == "configs":
        return source.parent.parent
    raise RuntimeError(f"D1 primary plan path is outside its registry: {source}")


def _primary_development_sources(root: Path) -> dict[str, Any]:
    development: dict[str, Any] = {}
    for split in ("train", "validation"):
        candidate_path = root / f"02_candidates/{split}/manifest.json"
        raw_path = root / f"03_features/{split}/top5/matched_common_raw/manifest.json"
        final_path = root / f"03_features/{split}/top5/T2_matched_common/manifest.json"
        labels_path = root / f"03_features/{split}/top5/labels/manifest.json"
        candidate = load_content_manifest(
            candidate_path,
            name=f"D1 primary {split} candidates",
            statuses=("COMPLETE",),
        )
        raw = load_content_manifest(
            raw_path, name=f"D1 primary {split} raw features", statuses=("COMPLETE",)
        )
        final = load_content_manifest(
            final_path,
            name=f"D1 primary {split} final features",
            statuses=("COMPLETE",),
        )
        labels = load_content_manifest(
            labels_path, name=f"D1 primary {split} labels", statuses=("COMPLETE",)
        )
        if any(
            value.get("candidate_test_labels_read") is not False
            for value in (candidate, raw, final, labels)
        ):
            raise PermissionError(
                "D1 primary plan development source violates Test isolation"
            )
        candidate_artifacts = _mapping(
            candidate.get("artifacts"), name=f"D1 primary {split} candidate artifacts"
        )
        for required in ("top5", "candidate_hashes"):
            if not isinstance(candidate_artifacts.get(required), Mapping):
                raise RuntimeError(f"D1 primary {split} candidates miss {required}")
        entry = {
            "denominator": _record(root / f"01_manifests/d1_paired_{split}.parquet"),
            "candidate_manifest": _record(candidate_path),
            "candidate_pool": dict(candidate_artifacts["top5"]),
            "candidate_hashes": dict(candidate_artifacts["candidate_hashes"]),
            "raw_feature_manifest": _record(raw_path),
            "raw_feature_artifacts": _mapping(
                raw.get("artifacts"), name=f"D1 primary {split} raw artifacts"
            ),
            "final_feature_manifest": _record(final_path),
            "final_feature_artifacts": _mapping(
                final.get("artifacts"), name=f"D1 primary {split} final artifacts"
            ),
            "development_label_manifest": _record(labels_path),
            "development_label_artifact": _mapping(
                labels.get("artifact"), name=f"D1 primary {split} label artifact"
            ),
        }
        verify_artifact_records_recursive(
            entry,
            name=f"D1 primary {split} immutable input closure",
            require_at_least_one=True,
        )
        development[split] = entry
    return development


def primary_source_closure(
    run_dir: str | Path,
    *,
    python_path: str | Path,
    tool_paths: Sequence[Path],
) -> dict[str, Any]:
    """Bind every byte that can affect a primary cell before planning."""

    root = Path(run_dir).expanduser().resolve()
    closure_path, closure = load_source_closure(root)
    if (
        closure.get("canonical_snapshot") != "A"
        or closure.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(
            "D1 primary plan requires the active label-free Snapshot A closure"
        )
    calibration_path = root / "05_calibration/top5/calibration_manifest.json"
    calibration = load_content_manifest(
        calibration_path, name="D1 primary selected calibration", statuses=("COMPLETE",)
    )
    if calibration.get("candidate_test_labels_read") is not False or not str(
        calibration.get("selected_method", "")
    ):
        raise RuntimeError("D1 primary selected calibration contract differs")
    folds_manifest_path = root / "04_splits/split_leakage_audit.json"
    folds_manifest = load_content_manifest(
        folds_manifest_path,
        name="D1 primary split audit",
        statuses=("PASS", "PASS_WITH_SEQUENCE_OVERLAP_LIMITATION"),
    )
    folds_path = root / "04_splits/fold_assignments.parquet"
    if folds_manifest.get("fold_assignments") != _record(folds_path):
        # Older manifests may omit byte count while retaining exact path/hash.
        expected = _record(folds_path)
        observed = _mapping(
            folds_manifest.get("fold_assignments"), name="D1 primary folds record"
        )
        if any(observed.get(key) != expected[key] for key in ("path", "sha256")):
            raise RuntimeError("D1 primary split audit/fold binding differs")
    sources = {
        "source_closure": _record(closure_path),
        "development": _primary_development_sources(root),
        "folds": {
            "manifest": _record(folds_manifest_path),
            "assignments": _record(folds_path),
        },
        "selected_calibration": {
            "manifest": _record(calibration_path),
            "selected_method": str(calibration["selected_method"]),
            "sources": _mapping(
                calibration.get("sources"), name="D1 primary calibration sources"
            ),
            "artifacts": _mapping(
                calibration.get("artifacts"), name="D1 primary calibration artifacts"
            ),
        },
        "environment": {
            "python_executable": _record(python_path),
            "bootstrap_environment": _record(root / "environment.txt"),
            "native_thread_limit": 1,
        },
        "code": [
            _record(path)
            for path in sorted(
                {Path(path).expanduser().resolve() for path in tool_paths}, key=str
            )
        ],
    }
    verify_artifact_records_recursive(
        sources, name="D1 primary exact source closure", require_at_least_one=True
    )
    return sources


def planned_cell_configuration(
    *,
    method: str,
    seed: int,
    mode: str,
    held_fold: int | None,
    hidden_dims: tuple[int, int] = (64, 32),
    dropout: float = 0.1,
    num_leaves: int = 31,
    tree_learning_rate: float = 0.05,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    alpha: float = 0.5,
    temperature: float = 1.0,
    beta: float = 1.0,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 1024,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """Build the exact predeclared configuration for one primary cell."""

    common = {
        "route": "D1",
        "pool": "top5",
        "track": "T2_matched_common",
        "method": method,
        "seed": int(seed),
        "mode": mode,
        "held_fold": held_fold,
    }
    if method in NEURAL_METHODS:
        return {
            **common,
            "encoder": "d1_residual_mlp",
            "loss": NEURAL_METHODS[method],
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
            "temperature": float(temperature),
            "beta": float(beta),
            "optimizer": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "alpha": float(alpha),
            "epochs": int(epochs),
            "patience": int(patience),
            "batch_size": int(batch_size),
            "gradient_clip_norm": 5.0,
        }
    if method == "R5":
        return {
            **common,
            "encoder": "lambdamart",
            "loss": "lambdarank",
            "num_leaves": int(num_leaves),
            "learning_rate": float(tree_learning_rate),
            "n_estimators": int(n_estimators),
        }
    raise ValueError(f"method is not a learned D1 primary method: {method}")


def _job(
    identifier_payload: dict[str, Any], configuration: dict[str, Any]
) -> dict[str, Any]:
    identifier = canonical_sha256(identifier_payload)[:16]
    return {
        "job_id": identifier,
        "worker_argv": [
            "-m",
            "tools.d1_reranking.train_primary_cell",
            "--job-id",
            identifier,
            "--resume",
        ],
        "configuration": configuration,
    }


def primary_plan(
    *,
    tool_paths: tuple[Path, ...],
    run_dir: str | Path | None = None,
    python_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the immutable Top5/T2 method and execution universe."""

    common_neural = {
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "alpha": 0.5,
        "epochs": 100,
        "patience": 10,
        "batch_size": 1024,
        "gradient_clip_norm": 5.0,
    }
    jobs: list[dict[str, Any]] = []
    for method, loss in NEURAL_METHODS.items():
        for hidden, dropout, seed, mode_fold in itertools.product(
            HIDDEN_GRID,
            DROPOUT_GRID,
            FORMAL_SEEDS,
            ("validation", 0, 1, 2, 3, 4),
        ):
            mode = "validation" if mode_fold == "validation" else "oof"
            fold = None if mode == "validation" else int(mode_fold)
            configuration = planned_cell_configuration(
                method=method,
                seed=seed,
                mode=mode,
                held_fold=fold,
                hidden_dims=hidden,
                dropout=dropout,
            )
            jobs.append(_job(configuration, configuration))
    for trial in LAMBDAMART_GRID:
        for seed, mode_fold in itertools.product(
            FORMAL_SEEDS, ("validation", 0, 1, 2, 3, 4)
        ):
            mode = "validation" if mode_fold == "validation" else "oof"
            fold = None if mode == "validation" else int(mode_fold)
            configuration = planned_cell_configuration(
                method="R5",
                seed=seed,
                mode=mode,
                held_fold=fold,
                num_leaves=trial["num_leaves"],
                tree_learning_rate=trial["learning_rate"],
            )
            jobs.append(_job(configuration, configuration))
    method_registry = {
        "R0": {"name": "native_q", "score": "native_score_raw"},
        "R1": {
            "name": "calibrated_q_plus_soft_target_support",
            "formula": R1_FORMULA,
            "trials": list(R1_GRID),
        },
        **{
            method: {
                "name": f"residual_mlp_{loss}",
                "loss": loss,
                "shared_architecture_grid": {
                    "hidden_dims": [list(value) for value in HIDDEN_GRID],
                    "dropout": list(DROPOUT_GRID),
                },
                "shared_budget": common_neural,
            }
            for method, loss in NEURAL_METHODS.items()
        },
        "R5": {
            "name": "lambdamart_lambdarank",
            "trials": [
                {"num_leaves": leaves, "learning_rate": rate, "n_estimators": 200}
                for leaves, rate in itertools.product((15, 31), (0.03, 0.05))
            ],
        },
        "R7": {
            "name": "validation_selected_expected_gain_gate",
            "eligible_ungated_methods": list(R7_ELIGIBLE),
            "fallback": "NO_GO_NATIVE",
        },
    }
    if run_dir is None:
        # Retained only for pure method-universe unit tests. Execution loaders
        # reject this unbound form; the public plan CLI always supplies a run.
        sources: Any = [_record(path) for path in tool_paths]
        source_signature: str | None = None
    else:
        if python_path is None:
            raise ValueError(
                "D1 bound primary plan requires an exact Python executable"
            )
        sources = primary_source_closure(
            run_dir, python_path=python_path, tool_paths=tool_paths
        )
        source_signature = canonical_sha256(sources)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "status": "PLANNED",
        "execution_authorized": False,
        "route": "D1",
        "primary_contract": {"pool": "top5", "track": "T2_matched_common"},
        "formal_seeds": list(FORMAL_SEEDS),
        "outer_folds": 5,
        "fold_local_preprocessing": True,
        "fold_local_calibration": {
            "required": True,
            "method_family": "selected on official Validation, parameters refit on each cell fit partition",
            "global_train_oof_base_logit_for_ranker_cells": False,
        },
        "method_registry": method_registry,
        "equal_validation_trial_budget": {
            "R2": 4,
            "R3": 4,
            "R4": 4,
            "R5": 4,
            "R6": 4,
        },
        "r7_selection": {
            "eligible": list(R7_ELIGIBLE),
            "trial_selection": (
                "mean of the three seed scores per candidate; select the trial by "
                "Validation J@1, MRR@5, nDCG@5, then canonical trial key"
            ),
            "seed_policy": "fixed three-seed score ensemble; best-seed selection forbidden",
            "oof_policy": "concatenate each seed's five held-fold predictions before ensembling",
            "metric_order": [
                "J@1 desc",
                "MRR@5 desc",
                "nDCG@5 desc",
                "method code asc",
            ],
            "test_metrics_used": False,
        },
        "jobs": jobs,
        "job_count": len(jobs),
        "job_ids_sha256": canonical_sha256([job["job_id"] for job in jobs]),
        "input_closure_bound": run_dir is not None,
        "candidate_test_labels_read": False,
        "sources": sources,
    }
    if source_signature is not None:
        plan["source_signature_sha256"] = source_signature
    plan["content_sha256"] = canonical_sha256(plan)
    return plan


def load_primary_plan(path: str | Path) -> dict[str, Any]:
    """Load and rebuild a bound primary plan from current source bytes."""

    source = Path(path).expanduser().resolve()
    plan = load_content_manifest(
        source, name="D1 primary matrix plan", statuses=("PLANNED",)
    )
    sources = _mapping(plan.get("sources"), name="D1 primary plan sources")
    if (
        plan.get("input_closure_bound") is not True
        or plan.get("candidate_test_labels_read") is not False
        or plan.get("job_count") != 360
        or plan.get("job_ids_sha256")
        != canonical_sha256([job.get("job_id") for job in plan.get("jobs", [])])
        or plan.get("source_signature_sha256") != canonical_sha256(sources)
    ):
        raise RuntimeError("D1 primary bound plan contract differs")
    environment = _mapping(
        sources.get("environment"), name="D1 primary plan environment"
    )
    executable = verified_artifact_path(
        _mapping(environment.get("python_executable"), name="D1 primary Python record"),
        name="D1 primary Python executable",
    )
    code = sources.get("code")
    if not isinstance(code, Sequence) or isinstance(code, (str, bytes)):
        raise RuntimeError("D1 primary plan code inventory differs")
    tools = tuple(
        verified_artifact_path(
            _mapping(record, name="D1 primary code record"),
            name="D1 primary code",
        )
        for record in code
    )
    expected = primary_plan(
        tool_paths=tools,
        run_dir=_plan_run_root(source),
        python_path=executable,
    )
    if plan != expected:
        raise RuntimeError("D1 primary plan/source universe differs")
    return plan


def write_primary_plan(
    path: str | Path,
    *,
    run_dir: str | Path,
    python_path: str | Path,
    tool_paths: tuple[Path, ...],
) -> dict[str, Any]:
    plan = primary_plan(tool_paths=tool_paths, run_dir=run_dir, python_path=python_path)
    atomic_json(path, plan)
    return plan


def publish_primary_plan(
    run_dir: str | Path,
    *,
    python_path: str | Path,
    tool_paths: Sequence[Path],
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    """Publish one immutable primary plan and advance its active pointer."""

    from .primary_execution import exclusive_json
    from .run import assert_writable_prelock

    root = Path(run_dir).expanduser().resolve()
    assert_writable_prelock(root)
    expected = primary_plan(
        tool_paths=tuple(tool_paths), run_dir=root, python_path=python_path
    )
    plan_id = str(expected["content_sha256"])[:20]
    destination = root / PRIMARY_PLAN_REGISTRY_RELATIVE / f"{plan_id}.json"
    if destination.exists():
        existing = load_primary_plan(destination)
        if existing != expected:
            raise RuntimeError("immutable D1 primary plan differs")
        if not resume:
            raise FileExistsError(f"D1 primary plan already exists: {destination}")
    else:
        exclusive_json(destination, expected)
    pointer: dict[str, Any] = {
        "schema_version": 1,
        "status": "PLANNED_POINTER",
        "active_plan": _record(destination),
        "job_count": 360,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    assert_writable_prelock(root)
    atomic_json(root / PRIMARY_PLAN_POINTER_RELATIVE, pointer)
    return destination, expected


def load_active_primary_plan(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Load the active immutable plan while retaining legacy fixture support."""

    root = Path(run_dir).expanduser().resolve()
    pointer_path = root / PRIMARY_PLAN_POINTER_RELATIVE
    if not pointer_path.exists():
        legacy = root / PRIMARY_PLAN_RELATIVE_PATH
        return legacy, load_content_manifest(
            legacy, name="legacy D1 primary matrix plan", statuses=("PLANNED",)
        )
    pointer = load_content_manifest(
        pointer_path,
        name="D1 active primary-plan pointer",
        statuses=("PLANNED_POINTER",),
    )
    plan_path = verified_artifact_path(
        _mapping(pointer.get("active_plan"), name="D1 active primary-plan record"),
        name="D1 active primary plan",
    )
    if plan_path.parent != (root / PRIMARY_PLAN_REGISTRY_RELATIVE).resolve():
        raise RuntimeError("D1 active primary plan is outside its registry")
    plan = load_primary_plan(plan_path)
    if (
        pointer.get("active_plan") != _record(plan_path)
        or pointer.get("job_count") != 360
        or pointer.get("candidate_test_labels_read") is not False
        or pointer.get("test_inputs_referenced") is not False
    ):
        raise RuntimeError("D1 active primary-plan pointer differs")
    return plan_path, plan


__all__ = [
    "PRIMARY_PLAN_POINTER_RELATIVE",
    "PRIMARY_PLAN_REGISTRY_RELATIVE",
    "PRIMARY_PLAN_RELATIVE_PATH",
    "PRIMARY_PYTHON_RELATIVE_PATH",
    "load_active_primary_plan",
    "load_primary_plan",
    "planned_cell_configuration",
    "primary_plan",
    "primary_source_closure",
    "publish_primary_plan",
    "write_primary_plan",
]
