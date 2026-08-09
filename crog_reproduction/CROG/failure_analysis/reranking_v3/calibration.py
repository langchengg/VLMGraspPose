from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np

from .schema import atomic_write_json, read_jsonl


def risk_coverage_curve(correct: np.ndarray, confidence: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(correct, dtype=bool)
    value = np.asarray(confidence, dtype=np.float64)
    if y.shape != value.shape:
        raise ValueError("risk/coverage arrays differ in shape")
    order = np.argsort(-value, kind="stable")
    ordered = y[order]
    cumulative_error = np.cumsum(~ordered)
    points = []
    for count in np.unique(np.linspace(1, len(y), min(len(y), 101), dtype=int)):
        points.append({"coverage": float(count / len(y)), "risk": float(cumulative_error[count - 1] / count), "count": int(count), "threshold": float(value[order[count - 1]])})
    return points


def _candidate_ranks(
    values: np.ndarray, *, fcer_scores: np.ndarray, q_ranks: np.ndarray,
    candidate_ids: np.ndarray, baseline_indices: np.ndarray,
) -> np.ndarray:
    result = np.empty(len(baseline_indices), dtype=np.int64)
    for row, baseline in enumerate(baseline_indices):
        candidates = [value for value in range(5) if value != int(baseline)]
        result[row] = min(
            candidates,
            key=lambda candidate: (
                -float(values[row, candidate]),
                -float(fcer_scores[row, candidate]),
                float(q_ranks[row, candidate]),
                str(candidate_ids[row, candidate]),
            ),
        )
    return result


def decide_safe_gate(
    *,
    baseline_indices: np.ndarray,
    gate_probabilities: np.ndarray,
    seed_gate_probabilities: np.ndarray | None,
    uncertainty: np.ndarray,
    valid_fraction: np.ndarray,
    fcer_scores: np.ndarray,
    candidate_ids: np.ndarray,
    harm_cost: float,
    threshold: float,
    uncertainty_kappa: float,
    consensus: int,
    minimum_valid_fraction: float,
    q_ranks: np.ndarray,
    seed_selected_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Pure calibration/deployment decision with deterministic candidate ordering."""
    probabilities = np.asarray(gate_probabilities, dtype=np.float64)
    baseline = np.asarray(baseline_indices, dtype=np.int64)
    if baseline.ndim != 1 or not len(baseline):
        raise ValueError("gate baseline indices must be a non-empty vector")
    u = np.asarray(uncertainty, dtype=np.float64)
    valid = np.asarray(valid_fraction, dtype=np.float64)
    scores = np.asarray(fcer_scores, dtype=np.float64)
    ids = np.asarray(candidate_ids).astype(str)
    expected = (len(baseline), 5)
    if probabilities.shape != (*expected, 3):
        raise ValueError("gate probabilities must be [samples,5,3]")
    if any(value.shape != expected for value in (u, valid, scores)) or ids.shape != expected:
        raise ValueError("gate decision candidate inputs must be [samples,5]")
    if np.any((baseline < 0) | (baseline >= 5)):
        raise ValueError("gate baseline indices are out of range")
    if any(len(set(row)) != 5 for row in ids.tolist()):
        raise ValueError("gate candidate IDs must be unique per sample")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("gate probabilities are invalid")
    if not np.isfinite(u).all() or not np.isfinite(scores).all():
        raise ValueError("gate uncertainty/FCER scores must be finite")
    if not np.isfinite(valid).all() or np.any(valid < 0.0) or np.any(valid > 1.0):
        raise ValueError("gate valid fractions must be in [0,1]")
    minimum = float(minimum_valid_fraction)
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum valid fraction must be in [0,1]")
    required_consensus = int(consensus)
    ranks = np.asarray(q_ranks, dtype=np.float64)
    if ranks.shape != expected or not np.isfinite(ranks).all():
        raise ValueError("q ranks must be finite [samples,5]")

    rows = np.arange(len(baseline))
    gain = probabilities[..., 0] - float(harm_cost) * probabilities[..., 1]
    safe_gain = gain - float(uncertainty_kappa) * u
    safe_gain[rows, baseline] = -np.inf
    proposed = _candidate_ranks(
        safe_gain, fcer_scores=scores, q_ranks=ranks,
        candidate_ids=ids, baseline_indices=baseline,
    )
    best = safe_gain[rows, proposed]

    if seed_gate_probabilities is not None:
        seed_probabilities = np.asarray(seed_gate_probabilities, dtype=np.float64)
        if seed_probabilities.ndim != 4 or seed_probabilities.shape[1:] != probabilities.shape:
            raise ValueError("seed gate probabilities must be [seeds,samples,5,3]")
        if not np.isfinite(seed_probabilities).all() or np.any(seed_probabilities < 0.0) or np.any(seed_probabilities > 1.0):
            raise ValueError("seed gate probabilities are invalid")
        seed_gain = seed_probabilities[..., 0] - float(harm_cost) * seed_probabilities[..., 1]
        seed_safe_gain = seed_gain - float(uncertainty_kappa) * u[None, ...]
        seed_safe_gain[:, rows, baseline] = -np.inf
        seed_selected = np.stack([
            _candidate_ranks(
                value, fcer_scores=scores, q_ranks=ranks,
                candidate_ids=ids, baseline_indices=baseline,
            )
            for value in seed_safe_gain
        ], axis=1)
    elif seed_selected_indices is not None:
        seed_selected = np.asarray(seed_selected_indices, dtype=np.int64)
        if seed_selected.ndim != 2 or seed_selected.shape[0] != len(baseline):
            raise ValueError("seed selected indices must be [samples,seeds]")
        if np.any((seed_selected < 0) | (seed_selected >= 5)):
            raise ValueError("seed selected indices are out of range")
    else:
        raise ValueError("gate decision requires seed probabilities or seed selections")
    if required_consensus < 1 or required_consensus > seed_selected.shape[1]:
        raise ValueError("gate consensus is out of range")
    observed_consensus = (seed_selected == proposed[:, None]).sum(axis=1)
    coverage_ok = valid[rows, proposed] >= minimum
    switched = (
        (best > float(threshold))
        & (observed_consensus >= required_consensus)
        & coverage_ok
    )
    selected = np.where(switched, proposed, baseline)
    return {
        "selected_indices": selected,
        "proposed_indices": proposed,
        "safe_gain": safe_gain,
        "best_safe_gain": best,
        "seed_selected_indices": seed_selected,
        "consensus": observed_consensus,
        "coverage_ok": coverage_ok,
        "switched": switched,
    }


def tune_safe_gate(
    *,
    labels: np.ndarray,
    baseline_indices: np.ndarray,
    gate_probabilities: np.ndarray,
    uncertainty: np.ndarray,
    candidate_ids: np.ndarray,
    q_ranks: np.ndarray,
    valid_fraction: np.ndarray | None = None,
    fcer_scores: np.ndarray | None = None,
    seed_selected_indices: np.ndarray | None = None,
    seed_gate_probabilities: np.ndarray | None = None,
    harm_costs: tuple[float, ...] = (2.0, 5.0, 10.0),
    thresholds: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2),
    kappas: tuple[float, ...] = (0.0, 0.5, 1.0),
    consensus_values: tuple[int, ...] = (2, 3),
    minimum_valid_fractions: tuple[float, ...] = (1.0,),
) -> dict[str, Any]:
    y = np.asarray(labels)
    baseline = np.asarray(baseline_indices, dtype=np.int64)
    probabilities = np.asarray(gate_probabilities, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)
    seeds = None if seed_selected_indices is None else np.asarray(seed_selected_indices, dtype=np.int64)
    seed_probabilities = None if seed_gate_probabilities is None else np.asarray(seed_gate_probabilities, dtype=np.float64)
    if seed_probabilities is not None and (seed_probabilities.ndim != 4 or seed_probabilities.shape[1:] != probabilities.shape):
        raise ValueError("seed gate probabilities must be [seeds,samples,candidates,3]")
    if seeds is None and seed_probabilities is None:
        raise ValueError("gate calibration requires seed selections or per-seed gate probabilities")
    rows = np.arange(len(y))
    expected = (len(y), 5)
    if y.shape != expected or baseline.shape != (len(y),) or not len(y):
        raise ValueError("gate calibration labels/baseline must be non-empty [samples,5]/[samples]")
    if not all((harm_costs, thresholds, kappas, consensus_values, minimum_valid_fractions)):
        raise ValueError("gate calibration grid dimensions must be non-empty")
    valid = np.ones(expected, dtype=np.float64) if valid_fraction is None else np.asarray(valid_fraction, dtype=np.float64)
    scores = np.zeros(expected, dtype=np.float64) if fcer_scores is None else np.asarray(fcer_scores, dtype=np.float64)
    ids = np.asarray(candidate_ids).astype(str)
    ranks = np.asarray(q_ranks, dtype=np.float64)
    baseline_correct = y[rows, baseline] > 0.5
    trials = []
    for harm_cost in harm_costs:
        for kappa in kappas:
            for threshold in thresholds:
                for consensus_required in consensus_values:
                    for minimum_valid_fraction in minimum_valid_fractions:
                        decision = decide_safe_gate(
                            baseline_indices=baseline,
                            gate_probabilities=probabilities,
                            seed_gate_probabilities=seed_probabilities,
                            seed_selected_indices=seeds,
                            uncertainty=u,
                            valid_fraction=valid,
                            fcer_scores=scores,
                            candidate_ids=ids,
                            q_ranks=ranks,
                            harm_cost=harm_cost,
                            threshold=threshold,
                            uncertainty_kappa=kappa,
                            consensus=consensus_required,
                            minimum_valid_fraction=minimum_valid_fraction,
                        )
                        selected = decision["selected_indices"]
                        selected_correct = y[rows, selected] > 0.5
                        recovered = int(((~baseline_correct) & selected_correct).sum())
                        harmful = int((baseline_correct & (~selected_correct)).sum())
                        switched = selected != baseline
                        changed = recovered + harmful
                        trials.append({"harm_cost": harm_cost, "threshold": threshold, "uncertainty_kappa": kappa, "consensus": consensus_required, "minimum_valid_fraction": minimum_valid_fraction, "selected_indices": selected, "delta": float(selected_correct.mean() - baseline_correct.mean()), "recovered": recovered, "harmful": harmful, "switch_coverage": float(switched.mean()), "outcome_changing_precision": None if changed == 0 else float(recovered / changed)})
    best = max(trials, key=lambda value: (value["delta"], -value["harmful"], -value["switch_coverage"], value["outcome_changing_precision"] or 0.0, value["minimum_valid_fraction"], -value["harm_cost"], -value["threshold"]))
    return {key: value for key, value in best.items() if key != "selected_indices"} | {"selected_indices": best["selected_indices"], "trial_count": len(trials)}


def calibrate_gate_bundle(
    *, bundle_path: str | Path, labels_path: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Join corrected calibration labels only at calibration time and lock a finite-grid policy."""
    with np.load(bundle_path) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    required = {
        "sample_ids", "candidate_ids", "candidate_checksums", "q_ranks",
        "baseline_indices", "gate_probabilities", "seed_gate_probabilities",
        "uncertainty_score_std", "uncertainty_valid_fraction", "scores",
    }
    missing_fields = sorted(required - set(arrays))
    if missing_fields:
        raise ValueError(f"calibration bundle is missing identity/gate fields: {missing_fields}")
    ids = list(map(str, arrays["sample_ids"]))
    if len(set(ids)) != len(ids):
        raise ValueError("calibration bundle contains duplicate sample IDs")
    candidate_ids = np.asarray(arrays["candidate_ids"]).astype(str)
    if candidate_ids.shape != (len(ids), 5):
        raise ValueError("calibration bundle candidate IDs must be [samples,5]")
    if any(len(set(row)) != 5 for row in candidate_ids.tolist()):
        raise ValueError("calibration bundle contains duplicate candidate IDs")
    candidate_checksums = np.asarray(arrays["candidate_checksums"]).astype(str)
    q_ranks = np.asarray(arrays["q_ranks"], dtype=np.float64)
    if candidate_checksums.shape != (len(ids), 5):
        raise ValueError("calibration bundle candidate checksums must be [samples,5]")
    if np.any(candidate_checksums == ""):
        raise ValueError("calibration bundle contains empty candidate checksums")
    if q_ranks.shape != (len(ids), 5) or not np.isfinite(q_ranks).all():
        raise ValueError("calibration bundle q ranks must be finite [samples,5]")
    if any(
        sorted(row) != [0, 1, 2, 3, 4]
        for row in q_ranks.astype(np.int64).tolist()
    ) or not np.array_equal(q_ranks, q_ranks.astype(np.int64)):
        raise ValueError("calibration bundle q ranks must be an explicit permutation of 0..4")
    expected_by_sample = {
        sample_id: dict(zip(row_ids, row_checksums, strict=True))
        for sample_id, row_ids, row_checksums in zip(
            ids, candidate_ids.tolist(), candidate_checksums.tolist(), strict=True
        )
    }
    label_lookup: dict[str, np.ndarray] = {}
    for record in read_jsonl(labels_path):
        sample_id = str(record["sample_id"])
        if sample_id not in expected_by_sample:
            continue
        if sample_id in label_lookup:
            raise ValueError(f"duplicate calibration label sample ID: {sample_id}")
        candidates: dict[str, tuple[float, str]] = {}
        for value in record["candidate_labels"]:
            candidate_id = str(value["candidate_id"])
            if candidate_id in candidates:
                raise ValueError(
                    f"duplicate calibration label candidate ID: {sample_id}/{candidate_id}"
                )
            checksum = str(value.get("candidate_checksum", ""))
            candidates[candidate_id] = (float(value["candidate_correct"]), checksum)
        expected = expected_by_sample[sample_id]
        if len(candidates) != 5 or set(candidates) != set(expected):
            raise ValueError(f"calibration label candidate IDs differ: {sample_id}")
        for candidate_id, checksum in expected.items():
            if candidates[candidate_id][1] != checksum:
                raise ValueError(
                    f"calibration label candidate checksum differs: {sample_id}/{candidate_id}"
                )
        label_lookup[sample_id] = np.asarray(
            [candidates[candidate_id][0] for candidate_id in expected], dtype=np.float32
        )
    missing = set(ids) - set(label_lookup)
    if missing:
        raise ValueError(f"calibration label artifact is missing IDs: {sorted(missing)[:5]}")
    labels = np.stack([label_lookup[value] for value in ids])
    result = tune_safe_gate(
        labels=labels, baseline_indices=arrays["baseline_indices"],
        gate_probabilities=arrays["gate_probabilities"],
        uncertainty=arrays["uncertainty_score_std"], seed_selected_indices=None,
        seed_gate_probabilities=arrays["seed_gate_probabilities"],
        valid_fraction=arrays["uncertainty_valid_fraction"],
        fcer_scores=arrays["scores"], candidate_ids=arrays["candidate_ids"],
        q_ranks=q_ranks,
    )
    selected = np.asarray(result.pop("selected_indices"), dtype=np.int64)
    rows = np.arange(len(ids)); baseline = arrays["baseline_indices"].astype(np.int64)
    baseline_correct = labels[rows, baseline] > .5; selected_correct = labels[rows, selected] > .5
    payload = {
        "schema_version":"3.0.0","kind":"v3_calibrated_safe_gate_policy","status":"locked_on_calibration",
        **result, "sample_count":len(ids), "baseline_correct":int(baseline_correct.sum()),
        "selected_correct":int(selected_correct.sum()), "delta":float(selected_correct.mean()-baseline_correct.mean()),
        "recovered":int(((~baseline_correct)&selected_correct).sum()), "harmful":int((baseline_correct&(~selected_correct)).sum()),
        "labels_read":str(Path(labels_path).resolve()), "formal_test_read":False,
    }
    atomic_write_json(output_path,payload); return payload
