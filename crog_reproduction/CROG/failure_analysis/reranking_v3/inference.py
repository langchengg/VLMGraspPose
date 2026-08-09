from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from failure_analysis.reranking_v2.schema import atomic_write_jsonl, read_jsonl

from .calibration import decide_safe_gate
from .gate_training import gate_extras, predict_gate_ensemble
from .schema import artifact_identity, atomic_write_json, sha256_file


def _bundle_candidate_identity(bundle: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = np.asarray(bundle["sample_ids"]).astype(str)
    candidate_ids = np.asarray(bundle["candidate_ids"]).astype(str)
    candidate_checksums = np.asarray(bundle["candidate_checksums"]).astype(str)
    q_ranks = np.asarray(bundle["q_ranks"], dtype=np.float64)
    expected = (len(sample_ids), 5)
    if len(sample_ids) == 0 or len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("gate bundle sample identity must be non-empty and unique")
    if candidate_ids.shape != expected or any(len(set(row)) != 5 for row in candidate_ids.tolist()):
        raise ValueError("gate bundle candidate IDs must be unique [samples,5]")
    if candidate_checksums.shape != expected or np.any(candidate_checksums == ""):
        raise ValueError("gate bundle candidate checksums must be non-empty [samples,5]")
    if q_ranks.shape != expected or not np.isfinite(q_ranks).all():
        raise ValueError("gate bundle q ranks must be finite [samples,5]")
    if any(
        sorted(row) != [0, 1, 2, 3, 4]
        for row in q_ranks.astype(np.int64).tolist()
    ) or not np.array_equal(q_ranks, q_ranks.astype(np.int64)):
        raise ValueError("gate bundle q ranks must be an explicit permutation of 0..4")
    return candidate_ids, candidate_checksums, q_ranks.astype(np.int64)


def _align_v2_baseline(
    *, sample_ids: np.ndarray, candidate_ids: np.ndarray, v2_predictions_path: str | Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    requested = list(map(str, sample_ids))
    if len(set(requested)) != len(requested):
        raise ValueError("FCER inference cohort contains duplicate sample IDs")
    candidates = np.asarray(candidate_ids).astype(str)
    if candidates.shape != (len(requested), 5):
        raise ValueError("FCER candidate IDs must be [samples,5]")
    if any(len(set(row)) != 5 for row in candidates.tolist()):
        raise ValueError("FCER candidate pool contains duplicate IDs")
    lookup: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(v2_predictions_path):
        sample_id = str(value["sample_id"])
        if sample_id in lookup:
            raise ValueError(f"duplicate V2 baseline prediction: {sample_id}")
        lookup[sample_id] = value
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"V2 baseline prediction is missing FCER samples: {missing[:5]}")
    baseline = np.empty(len(sample_ids), dtype=np.int64)
    records = []
    for index, sample_id in enumerate(requested):
        record = lookup[sample_id]
        order = list(map(str, record["candidate_order"]))
        frozen_ids = candidates[index].tolist()
        if len(order) != 5 or len(set(order)) != 5 or set(order) != set(frozen_ids):
            raise ValueError(f"V2 candidate pool differs from FCER: {sample_id}")
        selected_id = str(record["selection"]["selected_candidate_id"])
        if order[0] != selected_id:
            raise ValueError(f"V2 selected candidate differs from ranking top: {sample_id}")
        matches = np.flatnonzero(candidates[index] == selected_id)
        if len(matches) != 1:
            raise ValueError(f"V2 selected candidate identity mismatch: {sample_id}")
        baseline[index] = int(matches[0]); records.append(record)
    return baseline, records


def prepare_gate_inference(
    *, ensemble_path: str | Path, uncertainty_path: str | Path,
    v2_predictions_path: str | Path, gate_checkpoint_paths: Sequence[str | Path],
    device: str = "auto",
) -> dict[str, Any]:
    with np.load(ensemble_path) as payload:
        ensemble = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(uncertainty_path) as payload:
        uncertainty = {key: np.asarray(payload[key]) for key in payload.files}
    required_identity = {"sample_ids", "candidate_ids", "candidate_checksums", "q_ranks"}
    missing_identity = sorted(required_identity - set(ensemble))
    if missing_identity:
        raise ValueError(f"FCER ensemble is missing frozen candidate identity: {missing_identity}")
    sample_ids = ensemble["sample_ids"].astype(str)
    candidate_ids, candidate_checksums, q_ranks = _bundle_candidate_identity(ensemble)
    if list(map(str, uncertainty["sample_ids"])) != list(sample_ids):
        raise ValueError("uncertainty and ensemble sample order differ")
    if not np.array_equal(uncertainty["candidate_ids"].astype(str), candidate_ids):
        raise ValueError("uncertainty changed frozen candidate identity")
    baseline, v2_records = _align_v2_baseline(
        sample_ids=sample_ids, candidate_ids=candidate_ids, v2_predictions_path=v2_predictions_path,
    )
    scores = ensemble["scores"].mean(axis=0); probabilities = ensemble["probabilities"].mean(axis=0)
    residual = ensemble["residual"].mean(axis=0); embeddings = ensemble["embeddings"].astype(np.float32).mean(axis=0)
    extras = gate_extras(scores=scores, probabilities=probabilities, q=ensemble["q"], residual=residual, baseline_indices=baseline)
    gate = predict_gate_ensemble(
        embeddings=embeddings, extras=extras, baseline_indices=baseline,
        checkpoint_paths=list(gate_checkpoint_paths), device=device,
    )
    return {
        "sample_ids": sample_ids, "candidate_ids": candidate_ids,
        "candidate_checksums": candidate_checksums, "q_ranks": q_ranks,
        "q": ensemble["q"],
        "scores": scores, "seed_scores": ensemble["scores"], "probabilities": probabilities,
        "seed_probabilities": ensemble["probabilities"], "any_probability": ensemble["any_probability"].mean(axis=0),
        "embeddings": embeddings, "token_attention": ensemble["token_attention"].mean(axis=0),
        "baseline_indices": baseline, "v2_records": v2_records,
        "gate_probabilities": gate["probabilities"], "seed_gate_probabilities": gate["seed_probabilities"],
        "uncertainty": uncertainty,
    }


def apply_locked_gate(bundle: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, np.ndarray]:
    candidate_ids, _, q_ranks = _bundle_candidate_identity(bundle)
    return decide_safe_gate(
        baseline_indices=np.asarray(bundle["baseline_indices"]),
        gate_probabilities=np.asarray(bundle["gate_probabilities"]),
        seed_gate_probabilities=np.asarray(bundle["seed_gate_probabilities"]),
        uncertainty=np.asarray(bundle["uncertainty"]["score_std"]),
        valid_fraction=np.asarray(bundle["uncertainty"]["valid_fraction"]),
        fcer_scores=np.asarray(bundle["scores"]),
        candidate_ids=candidate_ids,
        q_ranks=q_ranks,
        harm_cost=float(policy["harm_cost"]),
        threshold=float(policy["threshold"]),
        uncertainty_kappa=float(policy["uncertainty_kappa"]),
        consensus=int(policy["consensus"]),
        minimum_valid_fraction=float(policy.get("minimum_valid_fraction", 1.0)),
    )


def write_v3_predictions(
    *, bundle: Mapping[str, Any], policy: Mapping[str, Any], output_dir: str | Path,
    gate_checkpoint_paths: Sequence[str | Path], method: str = "v3_locked_primary",
) -> dict[str, Any]:
    method_name = str(method).strip()
    if not method_name:
        raise ValueError("formal ranking method must be non-empty")
    output = Path(output_dir)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    gate = apply_locked_gate(bundle, policy); rows = []
    sample_ids = np.asarray(bundle["sample_ids"]).astype(str)
    candidate_ids, candidate_checksums, q_ranks = _bundle_candidate_identity(bundle)
    scores = np.asarray(bundle["scores"]); q = np.asarray(bundle["q"]); uncertainty = np.asarray(bundle["uncertainty"]["score_std"])
    probabilities = np.asarray(bundle["probabilities"]); token_attention = np.asarray(bundle["token_attention"])
    for row_index, sample_id in enumerate(sample_ids):
        selected = int(gate["selected_indices"][row_index]); ids = candidate_ids[row_index]
        remaining = sorted(
            (value for value in range(5) if value != selected),
            key=lambda value: (
                -float(scores[row_index, value]),
                float(q_ranks[row_index, value]),
                str(ids[value]),
            ),
        )
        order = [selected, *remaining]
        attention = token_attention[row_index]
        evidence = []
        for candidate_index in range(5):
            weights = attention[candidate_index] if attention.ndim == 2 and attention.shape[1] else np.asarray([])
            evidence.append({"top_token_index": None if not len(weights) else int(np.argmax(weights)), "top_token_attention": None if not len(weights) else float(np.max(weights))})
        gains = gate["safe_gain"][row_index].astype(float); gains[int(bundle["baseline_indices"][row_index])] = 0.0
        rows.append({
            "schema_version": "3.0.0", "kind": "v3_locked_ranking", "sample_id": sample_id,
            "method": method_name,
            "candidate_order": ids[order].tolist(), "candidate_scores": scores[row_index].astype(float).tolist(),
            "candidate_checksum_ids": ids.tolist(),
            "candidate_checksums": candidate_checksums[row_index].tolist(),
            "candidate_q_rank_ids": ids.tolist(),
            "candidate_q_ranks": q_ranks[row_index].astype(int).tolist(),
            "candidate_probability_ids": ids.tolist(),
            "candidate_correctness_probabilities": probabilities[row_index].astype(float).tolist(),
            "gate_gains": gains.tolist(),
            "uncertainty": uncertainty[row_index].astype(float).tolist(), "candidate_evidence": evidence,
            "selection": {
                "anchor": "v2_locked_primary", "baseline_index": int(bundle["baseline_indices"][row_index]),
                "proposed_index": int(gate["proposed_indices"][row_index]), "selected_index": selected,
                "selected_candidate_id": str(ids[selected]), "switched": bool(gate["switched"][row_index]),
                "ensemble_consensus": int(gate["consensus"][row_index]), "gain_lower_bound": float(gate["best_safe_gain"][row_index]),
                "fallback_reason": None if gate["switched"][row_index] else ("feature_coverage" if not gate["coverage_ok"][row_index] else "gate_threshold_or_consensus"),
            },
        })
    path = output / "predictions.jsonl"; atomic_write_jsonl(path, rows)
    summary = {
        "schema_version": "3.0.0", "kind": "v3_locked_predictions", "status": "complete",
        "row_count": len(rows), "unique_sample_count": len({value["sample_id"] for value in rows}),
        "unique_candidate_count": len(rows) * 5, "missing_count": 0,
        "fallback_count": int((~gate["switched"]).sum()), "switch_count": int(gate["switched"].sum()),
        "method": method_name, "policy": dict(policy), "prediction": artifact_identity(path),
        "gate_checkpoint_sha256": [sha256_file(value) for value in gate_checkpoint_paths],
        "candidate_identity_unchanged": True,
        "candidate_checksums_persisted": True, "q_ranks_persisted": True,
        "ranking_tie_break": "score_desc_then_q_rank_asc_then_candidate_id_asc",
        "labels_read": False,
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def save_gate_bundle(path: str | Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Persist label-free calibration inputs plus frozen candidate identity."""
    output = Path(path)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(bundle[key]) for key in (
        "sample_ids", "candidate_ids", "candidate_checksums", "q_ranks", "q", "scores", "seed_scores", "probabilities", "seed_probabilities",
        "any_probability", "token_attention", "gate_probabilities",
        "seed_gate_probabilities", "baseline_indices",
    )}
    for key in ("score_std", "valid_fraction", "ranking_consistency", "ensemble_disagreement"):
        arrays[f"uncertainty_{key}"] = np.asarray(bundle["uncertainty"][key])
    _bundle_candidate_identity(arrays)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    try: np.savez_compressed(temporary, **arrays); os.replace(temporary, output)
    finally: temporary.unlink(missing_ok=True)
    manifest = {"schema_version":"3.0.0","kind":"v3_label_free_gate_bundle","status":"complete","row_count":len(arrays["sample_ids"]),"artifact":artifact_identity(output),"candidate_checksums_persisted":True,"q_ranks_persisted":True,"token_attention_persisted":True,"labels_read":False}
    atomic_write_json(output.with_suffix(".manifest.json"), manifest); return manifest
