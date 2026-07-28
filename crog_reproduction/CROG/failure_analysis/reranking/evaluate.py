import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .geometry import assert_candidate_set_unchanged
from .labels import label_map, validate_label_candidate_join
from .rankers import q_only_matches_q_rank, rank_candidates
from .schema import INFERENCE_FEATURE_ALLOWLIST, canonical_json, read_jsonl


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _cluster_bootstrap_delta(outcomes, seed=17, iterations=2000):
    by_scene = defaultdict(list)
    for item in outcomes:
        by_scene[str(item["scene_id"])].append(item)
    scenes = sorted(by_scene)
    if not scenes:
        return [None, None]
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        numerator = denominator = 0
        for scene in sampled:
            group = by_scene[str(scene)]
            numerator += sum(int(item["reranked_success"]) - int(item["original_success"]) for item in group)
            denominator += len(group)
        deltas[index] = numerator / max(1, denominator)
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def _outcome_categories(item):
    categories = []
    if not item["candidate_count"]:
        categories.append("candidate_generation_failure")
    if item["predicted_mask_area"] == 0:
        categories.append("segmentation_failure")
    if item["depth_failure"]:
        categories.append("depth_or_pcd_failure")
    if item["border_candidate"]:
        categories.append("border_candidate")
    if item["candidate_count"] < 5:
        categories.append("low_candidate_count")
    if item["valid_candidate_count"] > 1:
        categories.append("ambiguous_gt")

    if item["original_success"] and item["reranked_success"]:
        categories.append("stable_success")
    elif not item["original_success"] and item["reranked_success"]:
        categories.append("recovered")
    elif item["original_success"] and not item["reranked_success"]:
        categories.append("harmful_flip")
    elif item["oracle_success"]:
        categories.append("missed_recovery")
    else:
        categories.append("unrecoverable")
    return categories


def evaluate_feature_records(
    feature_records,
    label_records,
    *,
    ranker,
    tuned_weights=None,
    mlp_scorer=None,
    seed=17,
    bootstrap_iterations=2000,
):
    labels_by_id = {str(record["sample_id"]): record for record in label_records}
    outcomes = []
    missing = defaultdict(int)
    mismatch_sample_ids = []
    total_candidates = 0
    for feature_record in feature_records:
        sample_id = str(feature_record["sample_id"])
        if sample_id not in labels_by_id:
            raise ValueError(f"missing labels for sample {sample_id}")
        label_record = labels_by_id[sample_id]
        validate_label_candidate_join(feature_record, label_record)
        candidates = feature_record.get("candidates", [])
        before = [dict(candidate) for candidate in candidates]
        if ranker == "q_only" and not q_only_matches_q_rank(candidates):
            mismatch_sample_ids.append(feature_record["sample_id"])
        ranked = rank_candidates(
            candidates,
            ranker,
            tuned_weights=tuned_weights,
            mlp_scorer=mlp_scorer,
        )
        assert_candidate_set_unchanged(before, ranked)
        validity = label_map(label_record)
        original_success = bool(candidates and validity[candidates[0]["candidate_id"]])
        reranked_success = bool(ranked and validity[ranked[0]["candidate_id"]])
        oracle_before = bool(any(validity.get(candidate["candidate_id"], False) for candidate in candidates))
        oracle_after = bool(any(validity.get(candidate["candidate_id"], False) for candidate in ranked))
        if oracle_before != oracle_after:
            raise AssertionError(f"Oracle@5 changed for sample {sample_id}")
        first_valid = next(
            (index + 1 for index, candidate in enumerate(ranked) if validity[candidate["candidate_id"]]),
            None,
        )
        total_candidates += len(candidates)
        for candidate in candidates:
            for name in INFERENCE_FEATURE_ALLOWLIST:
                feature = candidate.get("features", {}).get(name, {})
                if feature.get("value") is None or float(feature.get("reliability", 0.0) or 0.0) <= 0.0:
                    missing[name] += 1
        image_support = [
            candidate.get("features", {}).get("image_support", {}).get("value")
            for candidate in candidates
        ]
        depth_reliabilities = [
            float(candidate.get("features", {}).get("depth_mad_m", {}).get("reliability", 0.0) or 0.0)
            for candidate in candidates
        ]
        item = {
            "sample_id": feature_record["sample_id"],
            "scene_id": feature_record.get("scene_id", sample_id),
            "candidate_count": len(candidates),
            "valid_candidate_count": int(sum(validity.values())),
            "original_success": original_success,
            "reranked_success": reranked_success,
            "oracle_success": oracle_before,
            "same_top1": bool(
                (not candidates and not ranked)
                or (
                    candidates
                    and ranked
                    and candidates[0]["candidate_id"] == ranked[0]["candidate_id"]
                )
            ),
            "first_valid_rank": first_valid,
            "original_top1_candidate_id": candidates[0]["candidate_id"] if candidates else None,
            "reranked_top1_candidate_id": ranked[0]["candidate_id"] if ranked else None,
            "predicted_mask_area": int(feature_record.get("predicted_mask_area", 0)),
            "depth_failure": bool(candidates and not any(depth_reliabilities)),
            "border_candidate": any(value is not None and value < 1.0 for value in image_support),
            "candidates": [],
            "visualization_index": {
                "image_path": feature_record.get("image_path"),
                "predicted_mask": feature_record.get("predicted_mask_rle"),
            },
        }
        rerank_by_id = {candidate["candidate_id"]: candidate for candidate in ranked}
        for candidate in candidates:
            reranked = rerank_by_id[candidate["candidate_id"]]
            output_candidate = dict(candidate)
            output_candidate["new_rank"] = reranked["rerank_rank"]
            output_candidate["rerank_score"] = reranked["rerank_score"]
            output_candidate["score_decomposition"] = reranked["score_decomposition"]
            output_candidate["evaluation"] = {"candidate_valid": validity[candidate["candidate_id"]]}
            item["candidates"].append(output_candidate)
        outcomes.append(item)

    sample_count = len(outcomes)
    original_count = sum(item["original_success"] for item in outcomes)
    reranked_count = sum(item["reranked_success"] for item in outcomes)
    oracle_count = sum(item["oracle_success"] for item in outcomes)
    recovered = sum(not item["original_success"] and item["reranked_success"] for item in outcomes)
    harmful = sum(item["original_success"] and not item["reranked_success"] for item in outcomes)
    if reranked_count - original_count != recovered - harmful:
        raise AssertionError("J@1 delta does not equal recovered - harmful_flip")
    first_valid_ranks = [item["first_valid_rank"] for item in outcomes if item["first_valid_rank"]]
    eligible = sum(not item["original_success"] and item["oracle_success"] for item in outcomes)
    discordant = recovered + harmful
    mcnemar_p = float(binomtest(recovered, discordant, p=0.5).pvalue) if discordant else 1.0
    original_j1 = _safe_div(original_count, sample_count) or 0.0
    reranked_j1 = _safe_div(reranked_count, sample_count) or 0.0
    oracle = _safe_div(oracle_count, sample_count) or 0.0
    headroom = oracle - original_j1
    summary = {
        "ranker": ranker,
        "sample_count": sample_count,
        "candidate_count": total_candidates,
        "original_success_count": original_count,
        "reranked_success_count": reranked_count,
        "oracle_success_count": oracle_count,
        "original_j1": original_j1,
        "reranked_j1": reranked_j1,
        "oracle_at_5": oracle,
        "j1_absolute_delta_percentage_points": 100.0 * (reranked_j1 - original_j1),
        "top1_consistency": _safe_div(sum(item["same_top1"] for item in outcomes), sample_count),
        "first_valid_grasp_mean_rank": float(np.mean(first_valid_ranks)) if first_valid_ranks else None,
        "first_valid_grasp_median_rank": float(np.median(first_valid_ranks)) if first_valid_ranks else None,
        "mrr": _safe_div(sum(1.0 / rank for rank in first_valid_ranks), sample_count),
        "hit_at_1": reranked_j1,
        "hit_at_3": _safe_div(sum(item["first_valid_rank"] is not None and item["first_valid_rank"] <= 3 for item in outcomes), sample_count),
        "hit_at_5": oracle,
        "ranking_changed_count": sum(not item["same_top1"] for item in outcomes),
        "recovered": recovered,
        "harmful_flip": harmful,
        "missed_recovery": sum(
            not item["original_success"] and item["oracle_success"] and not item["reranked_success"]
            for item in outcomes
        ),
        "unrecoverable": sum(not item["original_success"] and not item["oracle_success"] for item in outcomes),
        "net_gain": recovered - harmful,
        "eligible": eligible,
        "recovery_rate": _safe_div(recovered, eligible),
        "harm_rate": _safe_div(harmful, original_count),
        "headroom": headroom,
        "remaining_headroom": oracle - reranked_j1,
        "headroom_achieved": _safe_div(reranked_j1 - original_j1, headroom),
        "mcnemar_exact_pvalue": mcnemar_p,
        "scene_frame_bootstrap_delta_95ci": _cluster_bootstrap_delta(
            outcomes, seed=seed, iterations=bootstrap_iterations
        ),
        "feature_missing_rates": {
            name: _safe_div(missing.get(name, 0), total_candidates)
            for name in INFERENCE_FEATURE_ALLOWLIST
        },
        "q_only_q_rank_mismatch_sample_ids": mismatch_sample_ids,
    }
    if ranker == "q_only" and mismatch_sample_ids:
        raise AssertionError(
            f"q_only did not reproduce q_rank for {len(mismatch_sample_ids)} samples"
        )

    categories = defaultdict(list)
    for item in outcomes:
        for category in _outcome_categories(item):
            categories[category].append(item["sample_id"])
    return summary, outcomes, dict(sorted(categories.items()))


def evaluate_paths(
    features_path,
    labels_path,
    *,
    ranker,
    limit=None,
    tuned_weights=None,
    mlp_scorer=None,
    seed=17,
    bootstrap_iterations=2000,
):
    features = list(read_jsonl(features_path, limit=limit))
    sample_ids = {str(record["sample_id"]) for record in features}
    labels = [
        record for record in read_jsonl(labels_path) if str(record["sample_id"]) in sample_ids
    ]
    return evaluate_feature_records(
        features,
        labels,
        ranker=ranker,
        tuned_weights=tuned_weights,
        mlp_scorer=mlp_scorer,
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
    )


def write_evaluation(
    output_dir,
    summary,
    outcomes,
    categories,
    *,
    overwrite=False,
    run_manifest=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "samples": output_dir / "per_sample.jsonl",
        "cases": output_dir / "case_index.json",
        "manifest": output_dir / "run_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("evaluation output exists; pass --overwrite explicitly")
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with paths["samples"].open("w", encoding="utf-8") as handle:
        for item in outcomes:
            handle.write(canonical_json(item) + "\n")
    paths["cases"].write_text(json.dumps(categories, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if run_manifest is not None:
        paths["manifest"].write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return paths
