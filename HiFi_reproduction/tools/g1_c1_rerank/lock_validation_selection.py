#!/usr/bin/env python3
"""Freeze every validation-selected model and formal prediction route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (str(REPOSITORY_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json  # noqa: E402
from src.grasping.g1_c1_safe_rerank.contracts import sha256_file  # noqa: E402
from tools.g1_c1_rerank.run_local_matrix import (  # noqa: E402
    BACKENDS,
    MANUAL_METHODS,
    METHODS,
    POOLS,
    SEEDS,
)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _artifact(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Create the primary lock exactly once without a check-then-replace race."""

    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"primary lock already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _tree_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({value.resolve() for value in paths}, key=str):
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    base = args.base_run.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    lock_root = run / "08_lock"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / "PRIMARY_METHOD_LOCK.json"
    if lock_path.exists():
        raise FileExistsError(f"primary lock already exists: {lock_path}")
    selection = json.loads((run / "07_validation" / "VALIDATION_SELECTION.json").read_text(encoding="utf-8"))
    gate_selection = json.loads((run / "07_validation" / "GATE_SELECTION.json").read_text(encoding="utf-8"))
    router_selection = json.loads((run / "07_validation" / "cross_backend" / "ROUTER_SELECTION.json").read_text(encoding="utf-8"))
    union_rows_path = run / "07_validation" / "cross_backend" / "CROSS_BACKEND_ORACLE_AND_UNION.csv"
    import pandas as pd

    union_rows = pd.read_csv(union_rows_path)
    validation_selection_artifacts = [
        _artifact(run / "07_validation" / "VALIDATION_SELECTION.json"),
        _artifact(run / "07_validation" / "LOSS_SELECTION.json"),
        _artifact(run / "07_validation" / "ENCODER_SELECTION.json"),
        _artifact(run / "07_validation" / "R5_TEMPERATURE_TRAIN_CV.csv"),
        _artifact(run / "07_validation" / "VALIDATION_MATRIX.csv"),
        _artifact(run / "07_validation" / "VALIDATION_ENSEMBLE_BOOTSTRAP.csv"),
        _artifact(run / "07_validation" / "GATE_SELECTION.json"),
        _artifact(run / "07_validation" / "cross_backend" / "ROUTER_SELECTION.json"),
        _artifact(union_rows_path),
    ]
    source_manifest_artifacts = [
        _artifact(base / "manifests" / f"{split}_samples.parquet")
        for split in ("train", "validation", "test")
    ]
    # Hash the canonical formal-label sources as opaque bytes before lock.  No
    # semantic Test columns are opened here; prepare_labels verifies these
    # hashes immediately before its one permitted label transaction.
    source_label_artifacts = [
        *[
            _artifact(base / "formal_test" / backend / "per_candidate_predictions.parquet")
            for backend in ("G1", "C1")
        ],
        _artifact(base / "manifests" / "test_labels.parquet"),
    ]
    audit_artifacts = [
        _artifact(run / "AUDIT_INVENTORY.json"),
        _artifact(run / "audit" / "source_baselines.parquet"),
        _artifact(run / "audit" / "frozen_pool_inventory.json"),
        _artifact(run / "03_splits" / "split_leakage_audit.json"),
        _artifact(run / "03_splits" / "fold_assignments.parquet"),
        _artifact(run / "02_features" / "FEATURE_STATUS.json"),
        _artifact(run / "00_audit" / "smoke_matrix" / "SMOKE_MATRIX.json"),
    ]
    final_matrix_provenance = run / "00_audit" / "FINAL_MATRIX_CODE_PROVENANCE.json"
    if not final_matrix_provenance.is_file():
        raise FileNotFoundError("FINAL_MATRIX_CODE_PROVENANCE.json is mandatory")
    provenance = json.loads(final_matrix_provenance.read_text(encoding="utf-8"))
    if provenance.get("status") != "PASS":
        raise RuntimeError("final matrix code provenance is not PASS")
    for relative, expected_sha in provenance.get("trainable_code_snapshot", {}).items():
        source = REPOSITORY_ROOT / relative
        if not source.is_file() or sha256_file(source) != str(expected_sha):
            raise RuntimeError(f"trainable code drift after matrix start: {source}")
    provenance_artifacts = provenance.get("matrix_artifacts", [])
    if not provenance_artifacts:
        raise RuntimeError("final matrix provenance lacks artifact inventory")
    for entry in provenance_artifacts:
        path = Path(str(entry.get("path", ""))).expanduser().resolve()
        try:
            path.relative_to(run)
        except ValueError as error:
            raise RuntimeError(f"matrix artifact outside run: {path}") from error
        if "quarantine" in path.parts or not path.is_file() or sha256_file(path) != str(entry.get("sha256")):
            raise RuntimeError(f"ineligible or drifted matrix artifact: {path}")
    counts = provenance.get("completion_counts", {})
    if int(counts.get("main_matrix_models", -1)) != len(BACKENDS) * len(POOLS) * len(METHODS) * len(SEEDS):
        raise RuntimeError("final matrix provenance main model count is incomplete")
    if int(counts.get("r5_temperature_cv_models", -1)) != len(BACKENDS) * len(POOLS) * 5 * 3:
        raise RuntimeError("R5 temperature Train-CV model count is incomplete")
    audit_artifacts.extend(
        _artifact(process_provenance)
        for process_provenance in sorted(
            (run / "00_audit").glob("*CODE_PROVENANCE.json")
        )
    )
    learned_union = union_rows.loc[union_rows["method"].astype(str).ne("oracle")]
    selected_union = learned_union.sort_values(
        ["j_at_1", "harmful", "switch_rate", "method"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).iloc[0].to_dict()

    selected_methods: dict[str, Any] = {
        "selection_source": "official validation only",
        "test_labels_accessed": False,
        "backend": {},
        "cross_backend": {
            "router": router_selection["selected"],
            "union": selected_union,
        },
    }
    locked_models: list[dict[str, Any]] = []
    ranker_calibrators: list[dict[str, Any]] = []
    for backend in BACKENDS:
        record = selection["backend"][backend.upper()]
        method = str(record["primary_ungated_method"])
        pool = str(record["primary_pool"])
        primary_gate = gate_selection["backend"][backend.upper()]["primary_gate"]
        if primary_gate is None:
            raise RuntimeError(f"{backend}: primary method lacks an OOF gate")
        gate_kind = str(primary_gate["gate_kind"])
        safe = primary_gate["safe_lcb"]
        deploy = bool(primary_gate["deploy_safe_lcb"])
        selected_methods["backend"][backend.upper()] = {
            **record,
            "primary_encoder": (
                "candidate_aligned_crop_cnn"
                if method == "r13_crop_cnn"
                else METHODS[method]["kind"]
            ),
            "primary_gated_method": f"{method}+{gate_kind}" if deploy else "baseline_no_switch",
            "primary_gate_kind": gate_kind,
            "primary_gate_operating_point": safe,
            "gate_deployed": deploy,
        }
        for seed in SEEDS:
            locked_models.append(_artifact(run / "05_models" / backend / pool / f"{method}_seed{seed}.joblib"))
        gate_root = run / "07_validation" / "gates" / backend / pool / method / gate_kind
        locked_models.append(_artifact(gate_root / "gate.joblib"))
        locked_models.append(_artifact(run / "04_calibration" / "ranker" / backend / pool / method / "calibrator.pkl"))
        locked_models.append(_artifact(run / "04_calibration" / backend / "calibrator.pkl"))

    # Lock every validation-matrix model so formal all-method comparisons cannot
    # silently train or replace a method after test labels are opened.
    for backend in BACKENDS:
        for pool in POOLS:
            for method in [*METHODS, "r13_crop_cnn"]:
                for seed in SEEDS:
                    artifact = _artifact(run / "05_models" / backend / pool / f"{method}_seed{seed}.joblib")
                    if artifact not in locked_models:
                        locked_models.append(artifact)
                calibrator_path = (
                    run / "04_calibration" / "ranker" / backend / pool / method / "calibrator.pkl"
                )
                if calibrator_path.is_file():
                    calibrator_artifact = _artifact(calibrator_path)
                    ranker_calibrators.append(calibrator_artifact)
                    if calibrator_artifact not in locked_models:
                        locked_models.append(calibrator_artifact)
    for pool in POOLS:
        for seed in SEEDS:
            locked_models.append(_artifact(run / "05_models" / "pooled" / pool / f"seed{seed}.joblib"))
    for union_name in ("union_concat", "union_nms"):
        for pool in POOLS:
            for method in ("union_mlp", "union_deepsets", "union_gnn"):
                for seed in SEEDS:
                    locked_models.append(_artifact(run / "05_models" / "cross_backend" / union_name / pool / method / f"seed{seed}.joblib"))
    router_kind = str(router_selection["selected"]["kind"])
    locked_models.append(_artifact(run / "07_validation" / "cross_backend" / "router" / router_kind / "model.joblib"))

    candidate_artifacts = [
        _artifact(run / "data" / f"frozen_{backend}_{split}_candidates.parquet")
        for backend in BACKENDS
        for split in ("train", "validation", "test")
    ]
    feature_artifacts = [
        _artifact(run / "02_features" / split / backend / f"{pool}_features.parquet")
        for backend in BACKENDS
        for split in ("train", "validation", "test")
        for pool in POOLS
    ]
    feature_artifacts.extend(
        _artifact(run / "02_features" / split / backend / "cross_backend_evidence.parquet")
        for backend in BACKENDS
        for split in ("train", "validation", "test")
    )
    # R13 consumes the immutable crop tensors directly at formal inference.
    # Lock every shard and index/manifest file, not just the COMPLETE marker.
    for backend in BACKENDS:
        for split in ("train", "validation", "test"):
            crop_root = run / "02_features" / split / backend / "candidate_crops_64"
            crop_files = sorted(path for path in crop_root.rglob("*") if path.is_file())
            if not crop_files:
                raise FileNotFoundError(f"missing R13 crop store: {crop_root}")
            feature_artifacts.extend(_artifact(path) for path in crop_files)
    code_paths = [
        path
        for root in (
            PROJECT_ROOT / "src" / "grasping" / "g1_c1_safe_rerank",
            PROJECT_ROOT / "tools" / "g1_c1_rerank",
            REPOSITORY_ROOT / "reranking",
        )
        for path in root.rglob("*.py")
        if path.is_file()
    ]
    evaluator_path = PROJECT_ROOT / "src" / "grasping" / "g1_c1_safe_rerank" / "evaluation.py"
    code_sha = _tree_hash(code_paths)
    code_artifacts = [_artifact(path) for path in sorted(code_paths)]
    (lock_root / "code_sha256.txt").write_text(code_sha + "\n", encoding="utf-8")
    (lock_root / "evaluator_sha256.txt").write_text(sha256_file(evaluator_path) + "\n", encoding="utf-8")

    train_total = len(pd.read_parquet(base / "manifests" / "train_samples.parquet", columns=["sample_id"]))
    fold_sha = sha256_file(run / "03_splits" / "fold_assignments.parquet")
    manual_alphas: dict[str, dict[str, dict[str, float]]] = {}
    manual_cv_artifacts: list[dict[str, Any]] = []
    controlled_temperature_artifacts: list[dict[str, Any]] = []
    for backend in BACKENDS:
        manual_alphas[backend] = {}
        for pool in POOLS:
            temperature_root = run / "07_validation" / "r5_temperature_cv" / backend / pool
            temperature_csv = temperature_root / "R5_TEMPERATURE_TRAIN_CV.csv"
            temperature_json = temperature_root / "R5_TEMPERATURE_TRAIN_CV.json"
            temperature_selection = json.loads(temperature_json.read_text(encoding="utf-8"))
            if (
                temperature_selection.get("status") != "PASS"
                or temperature_selection.get("selection_scope") != "nested 5-fold scene-grouped Train OOF"
                or int(temperature_selection.get("folds", -1)) != 5
                or int(temperature_selection.get("train_total", -1)) != train_total
                or temperature_selection.get("fold_assignments_sha256") != fold_sha
                or float(temperature_selection.get("selected_temperature", float("nan"))) not in {0.5, 1.0, 2.0}
            ):
                raise RuntimeError(f"R5 temperature Train-CV evidence invalid: {backend}/{pool}")
            selected_temperature = float(temperature_selection["selected_temperature"])
            for seed in SEEDS:
                metric_path = run / "07_validation" / "metrics" / backend / pool / f"r5_mlp_listwise_seed{seed}.json"
                model_metadata_path = run / "05_models" / backend / pool / f"r5_mlp_listwise_seed{seed}.json"
                metric_record = json.loads(metric_path.read_text(encoding="utf-8"))
                model_record = json.loads(model_metadata_path.read_text(encoding="utf-8"))
                if (
                    float(metric_record.get("temperature", float("nan"))) != selected_temperature
                    or float(model_record.get("temperature", float("nan"))) != selected_temperature
                    or metric_record.get("objective") != "listwise"
                    or model_record.get("objective") != "listwise"
                ):
                    raise RuntimeError(f"R5 formal model differs from Train-CV temperature: {backend}/{pool}/seed{seed}")
            loss_path = run / "07_validation" / "loss_selection" / backend / f"{pool}.json"
            loss_selection = json.loads(loss_path.read_text(encoding="utf-8"))
            if (
                loss_selection.get("status") != "LOCKED_FROM_OFFICIAL_VALIDATION"
                or loss_selection.get("selected_objective") not in {"bce", "ranknet", "listwise"}
                or Path(str(loss_selection.get("r5_temperature_source", ""))).resolve() != temperature_json.resolve()
            ):
                raise RuntimeError(f"controlled loss lock invalid: {backend}/{pool}")
            controlled_temperature_artifacts.extend(
                [_artifact(temperature_csv), _artifact(temperature_json), _artifact(loss_path)]
            )
            rows = pd.read_csv(
                run / "07_validation" / "tables" / f"{backend}_{pool}_metrics.csv"
            )
            manual_alphas[backend][pool] = {}
            for method in MANUAL_METHODS:
                selected_alpha = rows.loc[
                    rows["method"].astype(str).eq(method)
                    & pd.to_numeric(rows["seed"], errors="coerce").eq(17)
                ]
                if len(selected_alpha) != 1:
                    raise RuntimeError(
                        f"R1 alpha row is not unique: {backend}/{pool}/{method}"
                    )
                alpha = float(selected_alpha.iloc[0]["alpha"])
                cv_root = run / "07_validation" / "r1_train_cv" / backend / pool
                cv_path = cv_root / f"{method}.csv"
                selection_path = cv_root / f"{method}_selection.json"
                cv_selection = json.loads(selection_path.read_text(encoding="utf-8"))
                if (
                    cv_selection.get("status") != "PASS"
                    or cv_selection.get("selection_scope") != "5-fold scene-grouped held-fold Train CV"
                    or int(cv_selection.get("folds", -1)) != 5
                    or int(cv_selection.get("total", -1)) != train_total
                    or cv_selection.get("fold_assignments_sha256") != fold_sha
                    or float(cv_selection.get("selected_alpha", float("nan"))) != alpha
                    or not cv_selection.get("feature_directions")
                ):
                    raise RuntimeError(f"R1 Train-CV lock evidence invalid: {backend}/{pool}/{method}")
                manual_cv_artifacts.extend([_artifact(cv_path), _artifact(selection_path)])
                manual_alphas[backend][pool][method] = alpha
    validation_selection_artifacts.extend(manual_cv_artifacts)
    validation_selection_artifacts.extend(controlled_temperature_artifacts)
    listwise_temperature = json.loads(
        (run / "07_validation" / "LOSS_SELECTION.json").read_text(encoding="utf-8")
    )
    prediction_plan = {
        "backends": list(BACKENDS),
        "pools": list(POOLS),
        "seeds": list(SEEDS),
        "matrix_methods": [*METHODS, "r13_crop_cnn"],
        "manual_methods": list(MANUAL_METHODS),
        "manual_alphas": manual_alphas,
        "controlled_loss_selection": listwise_temperature,
        "primary": selected_methods,
        "cross_backend_routes": ["router", "union_concat", "union_nms"],
        "pooled_backend_conditioned": {
            "method": "r14_pooled_backend_conditioned",
            "pools": list(POOLS),
            "seeds": list(SEEDS),
        },
        "statistical_families": {
            "formal_intra_backend_all_locked_methods": (
                len(BACKENDS)
                * len(POOLS)
                * (len(MANUAL_METHODS) + len(METHODS) + 2)
                + len(BACKENDS)
            ),
            "formal_cross_backend_locked_methods": 1 + 2 * len(POOLS) * 3,
        },
        "formal_test_repetitions": 1,
        "post_test_reselection": False,
    }
    prediction_plan_path = lock_root / "FORMAL_PREDICTION_PLAN.json"
    atomic_json(prediction_plan_path, prediction_plan)
    artifacts = {
        "selected_methods": selected_methods,
        "selected_features": {backend: record["primary_feature_set"] for backend, record in selected_methods["backend"].items()},
        "selected_hyperparameters": {
            "seeds": list(SEEDS),
            "neural": {"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "residual_alpha": 0.5},
            "bootstrap_draws": 10_000,
            "manual_alphas": manual_alphas,
            "controlled_loss_selection": listwise_temperature,
        },
        "calibration_manifest": [_artifact(run / "04_calibration" / backend / "calibrator.pkl") for backend in BACKENDS],
        "gate_thresholds": {backend: record["primary_gate_operating_point"] for backend, record in selected_methods["backend"].items()},
        "candidate_manifests": candidate_artifacts,
    }
    for name, payload in artifacts.items():
        atomic_json(lock_root / f"{name}.json", payload if isinstance(payload, dict) else {"artifacts": payload})
    declaration = [
        "# Primary Method Declaration",
        "",
        "Selection used official Validation only; no reranking test labels were read.",
        "",
    ]
    for backend, record in selected_methods["backend"].items():
        declaration.extend(
            [
                f"## {backend}",
                "",
                f"- Ungated: `{record['primary_ungated_method']}`",
                f"- Gated: `{record['primary_gated_method']}`",
                f"- Pool: `{record['primary_pool']}`",
                f"- Feature set: `{record['primary_feature_set']}`",
                f"- Loss/encoder: `{record['primary_loss']}` / `{record['primary_encoder']}`",
                f"- Gate: `{record['primary_gate_kind']}`, deployed={record['gate_deployed']}",
                "",
            ]
        )
    (lock_root / "PRIMARY_METHOD_DECLARATION.md").write_text("\n".join(declaration) + "\n", encoding="utf-8")
    lock_support_artifacts = [
        _artifact(lock_root / f"{name}.json") for name in artifacts
    ]
    lock_support_artifacts.extend(
        [
            _artifact(lock_root / "PRIMARY_METHOD_DECLARATION.md"),
            _artifact(lock_root / "code_sha256.txt"),
            _artifact(lock_root / "evaluator_sha256.txt"),
        ]
    )
    lock = {
        "status": "LOCKED",
        "base_run": str(base),
        "selection_scope": "official validation only",
        "test_labels_accessed": False,
        "selected_methods": selected_methods,
        "locked_models": locked_models,
        "candidate_artifacts": candidate_artifacts,
        "feature_artifacts": feature_artifacts,
        "code_artifacts": code_artifacts,
        "validation_selection_artifacts": validation_selection_artifacts,
        "source_manifest_artifacts": source_manifest_artifacts,
        "source_label_artifacts": source_label_artifacts,
        "audit_artifacts": audit_artifacts,
        "ranker_calibrators": ranker_calibrators,
        "lock_support_artifacts": lock_support_artifacts,
        "prediction_plan": _artifact(prediction_plan_path),
        "evaluator": _artifact(evaluator_path),
        "code_sha256": code_sha,
        "git_head": os.popen(f"git -C {REPOSITORY_ROOT} rev-parse HEAD").read().strip(),
    }
    _exclusive_json(lock_path, lock)
    atomic_json(lock_root / "FORMAL_TEST_LOCK.json", lock)
    print(json.dumps({"status": "LOCKED", "path": str(lock_path), "sha256": sha256_file(lock_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
