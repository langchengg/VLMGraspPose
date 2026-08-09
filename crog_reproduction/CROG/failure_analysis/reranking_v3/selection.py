from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .experiment_config import SELECTION_GRID, selection_contract
from .feature_store import FeatureCatalog, fit_streaming_normalizers, load_label_lookup, prior_array_to_lookup
from .metrics import paired_switch_metrics, ranking_metrics, strip_metric_arrays
from .oof import predict_ranker_streaming, save_predictions, train_ranker_streaming
from .artifacts import code_fingerprint
from .schema import artifact_identity, atomic_write_json, canonical_json, read_jsonl, sha256_bytes, sha256_file
from .splits import verify_v3_split_manifest
from .statistics import cluster_bootstrap_difference
from .v2_prior import load_v2_oof_prior, load_v2_validation_prior


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fields = list(rows[0]) if rows else []
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["sample_id"]): row for row in payload["rows"]}


def _v2_rankings(path: str | Path, predictions: dict[str, Any]) -> np.ndarray:
    lookup={}
    for record in read_jsonl(path):
        sample_id=str(record["sample_id"])
        if sample_id in lookup: raise ValueError(f"duplicate locked V2 prediction: {sample_id}")
        lookup[sample_id]=record
    rows = []
    for sample_id, candidate_ids in zip(predictions["sample_ids"], predictions["candidate_ids"], strict=True):
        record = lookup.get(str(sample_id))
        if record is None:
            raise ValueError(f"locked V2 prediction missing {sample_id}")
        order_ids = list(map(str, record["candidate_order"]))
        index = {str(value): position for position, value in enumerate(candidate_ids)}
        try:
            rows.append([index[value] for value in order_ids])
        except KeyError as error:
            raise ValueError(f"V2 candidate identity differs for {sample_id}") from error
    return np.asarray(rows, dtype=np.int64)


def _labels_in_prediction_order(labels: dict[str, np.ndarray], sample_ids: np.ndarray) -> np.ndarray:
    return np.stack([labels[str(value)] for value in sample_ids]).astype(np.float32)


def run_selection_grid(
    *,
    train_catalog: FeatureCatalog,
    select_catalog: FeatureCatalog,
    train_ids: set[str],
    select_ids: set[str],
    train_labels_path: str | Path,
    validation_labels_path: str | Path,
    v2_root: str | Path,
    v2_validation_predictions: str | Path,
    v3_split_manifest: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    seed: int = 20260801,
    epochs: int = 5,
    batch_size: int = 16,
    resume: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=resume)
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if contract != selection_contract():
        raise ValueError("predeclared selection contract differs from executable grid")
    split_payload=verify_v3_split_manifest(v3_split_manifest)
    missing_train=train_ids-set(train_catalog.locations)
    if missing_train: raise ValueError(f"training feature catalog is missing cohort rows: {sorted(missing_train)[:5]}")
    select_catalog.assert_exact_ids(select_ids)
    run_spec={
        "schema_version":"3.0.0","kind":"v3_selection_run_spec","status":"frozen_before_fit",
        "code_fingerprint":code_fingerprint(),"contract":artifact_identity(contract_path),
        "v3_split":artifact_identity(v3_split_manifest),"split_content_sha256":split_payload["source_v2_split_manifest"]["sha256"],
        "train_features":train_catalog.provenance(),"select_features":select_catalog.provenance(),
        "train_labels":artifact_identity(train_labels_path),"select_labels":artifact_identity(validation_labels_path),
        "v2_validation_predictions":artifact_identity(v2_validation_predictions),
        "seed":int(seed),"epochs":int(epochs),"batch_size":int(batch_size),"device":str(device),
    }
    run_spec["content_sha256"]=sha256_bytes(canonical_json(run_spec).encode())
    run_spec_path=output/"RUN_SPEC.json"
    if run_spec_path.exists():
        if not resume or json.loads(run_spec_path.read_text())!=run_spec: raise ValueError("selection resume run-spec mismatch")
    else: atomic_write_json(run_spec_path,run_spec)
    train_labels = load_label_lookup(train_labels_path, allowed_ids=train_ids, candidate_identity=train_catalog.candidate_identity(train_ids))
    select_labels = load_label_lookup(validation_labels_path, allowed_ids=select_ids, candidate_identity=select_catalog.candidate_identity(select_ids))
    ordered_train = sorted(train_ids); ordered_select = sorted(select_ids)
    train_prior_values, train_valid = load_v2_oof_prior(v2_root, ordered_train)
    select_prior_values, select_valid = load_v2_validation_prior(v2_root, ordered_select)
    if not train_valid.all() or not select_valid.all():
        raise ValueError("provenance-valid V2 prior coverage is incomplete")
    train_priors = prior_array_to_lookup(np.asarray(ordered_train), train_prior_values)
    select_priors = prior_array_to_lookup(np.asarray(ordered_select), select_prior_values)
    normalizers = fit_streaming_normalizers(train_catalog, allowed_ids=train_ids, priors=train_priors)
    split_rows = _manifest_rows(v3_split_manifest)
    results = []
    for config_index, config in enumerate(SELECTION_GRID):
        model_path = output / "models" / f"{config_index:02d}_{config['name']}_seed{seed}.pt"
        train_summary = train_ranker_streaming(
            catalog=train_catalog, train_ids=train_ids, labels=train_labels, priors=train_priors,
            config=config, output_path=model_path, seed=seed, device=device, epochs=epochs,
            batch_size=batch_size, normalizers=normalizers, resume=resume,
        )
        prediction_path = output / "predictions" / f"{config_index:02d}_{config['name']}.npz"
        if prediction_path.exists() and resume:
            with np.load(prediction_path) as payload:
                predictions = {key: np.asarray(payload[key]) for key in payload.files}
            predictions["checkpoint_sha256"] = train_summary["checkpoint"]["sha256"]
            predictions["producing_seed"] = seed
        else:
            predictions = predict_ranker_streaming(
                catalog=select_catalog, sample_ids=select_ids, priors=select_priors,
                checkpoint_path=model_path, device=device, batch_size=max(batch_size, 32),
            )
            save_predictions(prediction_path, predictions, scope="v3_select", configuration=config["name"])
        labels = _labels_in_prediction_order(select_labels, predictions["sample_ids"])
        ranking = np.argsort(-predictions["scores"], axis=1, kind="stable")
        q_ranking = np.argsort(-predictions["q"], axis=1, kind="stable")
        v2_ranking = _v2_rankings(v2_validation_predictions, predictions)
        metrics = ranking_metrics(labels, ranking, probabilities=predictions["probabilities"])
        vs_q = paired_switch_metrics(labels, q_ranking, ranking)
        vs_v2 = paired_switch_metrics(labels, v2_ranking, ranking)
        frames = np.asarray([split_rows[str(value)]["frame_id"] for value in predictions["sample_ids"]])
        scenes = np.asarray([split_rows[str(value)]["sequence_id"] for value in predictions["sample_ids"]])
        bootstrap_iterations=int(contract["selection_bootstrap_iterations"])
        frame_ci = cluster_bootstrap_difference(vs_v2["reference_correct_mask"], vs_v2["challenger_correct_mask"], frames, iterations=bootstrap_iterations, seed=seed)
        scene_ci = cluster_bootstrap_difference(vs_v2["reference_correct_mask"], vs_v2["challenger_correct_mask"], scenes, iterations=bootstrap_iterations, seed=seed+1)
        results.append({
            "config_index": config_index, "configuration": config["name"], "use_depth": bool(config["use_depth"]),
            "hidden_dim": int(config["hidden_dim"]), "alpha": float(config["alpha"]),
            "correct": metrics["correct"], "j_at_1": metrics["j_at_1"], "oracle_at_5": metrics["oracle_at_5"],
            "delta_vs_q": vs_q["delta_j_at_1"], "delta_vs_v2": vs_v2["delta_j_at_1"],
            "recovered_vs_v2": vs_v2["recovered"], "harmful_vs_v2": vs_v2["harmful"],
            "switch_coverage_vs_v2": vs_v2["switch_coverage"],
            "outcome_changing_precision_vs_v2": vs_v2["outcome_changing_precision"],
            "frame_ci_lower": frame_ci["ci95"][0], "frame_ci_upper": frame_ci["ci95"][1],
            "scene_ci_lower": scene_ci["ci95"][0], "scene_ci_upper": scene_ci["ci95"][1],
            "candidate_brier": metrics["candidate_brier"], "candidate_nll": metrics["candidate_nll"], "ece": metrics["ece"],
            "parameter_count": train_summary["parameters"], "checkpoint_sha256": train_summary["checkpoint"]["sha256"],
            "prediction_sha256": sha256_file(prediction_path),
        })
    # Exact order declared in the immutable pre-training contract: corrected
    # delta first, then the frame-cluster lower bound and conservative criteria.
    selected = max(results, key=lambda value: (
        value["delta_vs_v2"], value["frame_ci_lower"], value["scene_ci_lower"], -value["harmful_vs_v2"],
        -value["switch_coverage_vs_v2"], value["outcome_changing_precision_vs_v2"] or 0.0,
        -value["parameter_count"], not value["use_depth"], -value["config_index"],
    ))
    baseline_metrics = {
        "q_only": strip_metric_arrays(ranking_metrics(labels, q_ranking)),
        "v2_locked_primary": strip_metric_arrays(ranking_metrics(labels, v2_ranking)),
    }
    _atomic_csv(output / "results_validation.csv", results)
    summary = {
        "schema_version": "3.0.0", "kind": "v3_architecture_selection", "status": "complete",
        "selection_contract_sha256": sha256_file(contract_path), "scope": "v3_select",
        "run_spec":artifact_identity(run_spec_path),
        "seed": int(seed), "epochs": int(epochs), "batch_size": int(batch_size), "device": device,
        "configuration_count": len(results), "selected": selected, "results": results,
        "baselines": baseline_metrics,
        "train_feature_provenance": train_catalog.provenance(), "select_feature_provenance": select_catalog.provenance(),
        "labels_read": [str(Path(train_labels_path).resolve()), str(Path(validation_labels_path).resolve())],
        "formal_test_read": False,
    }
    atomic_write_json(output / "selection_summary.json", summary)
    return summary
