from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .api import MODEL_IDS
from .critic_features import p3_hard_rule
from .manifest import file_identity
from .policy import full_denominator_metrics
from .renderer import PerturbationVariant
from .runner import _atomic_json, _parquet


STABILITY_CONTRACT = {
    "required_variants": sorted(
        variant.value
        for variant in PerturbationVariant
        if variant is not PerturbationVariant.ORIGINAL
    ),
    "minimum_hard_rule_consistency": 0.90,
    "maximum_direction_flip_rate": 0.10,
    "minimum_valid_pair_coverage": 0.98,
}


def _preference_score(parsed: dict[str, Any] | None) -> float:
    if not parsed:
        return float("-inf")
    return float(
        (parsed.get("challenger_target_alignment", 0) - parsed.get("baseline_target_alignment", 0))
        + (parsed.get("challenger_contact_geometry", 0) - parsed.get("baseline_contact_geometry", 0))
        + 0.5 * (parsed.get("challenger_width_compatibility", 0) - parsed.get("baseline_width_compatibility", 0))
        - (parsed.get("challenger_collision_risk", 0) - parsed.get("baseline_collision_risk", 0))
    )


def evaluate_diagnostic_phase(
    run_dir: str | Path,
    phase: str,
    *,
    perturbation_phase: str | None = None,
) -> dict[str, Any]:
    root = Path(run_dir); phase_dir = root / phase
    responses = pq.read_table(phase_dir / "pairwise_responses.parquet").to_pylist()
    evaluation = pq.read_table(phase_dir / "evaluation_manifest.parquet").to_pylist()
    expected_original_keys = {
        (str(row["sample_id"]), str(row["challenger_candidate_id"]), str(model))
        for row in evaluation
        for model in MODEL_IDS
    }
    original_rows = [row for row in responses if str(row.get("variant")) == "original"]
    actual_original_keys = {
        (str(row["sample_id"]), str(row["challenger_candidate_id"]), str(row["model_id"]))
        for row in original_rows
    }
    if len(actual_original_keys) != len(original_rows) or actual_original_keys != expected_original_keys:
        raise ValueError("diagnostic original response coverage is incomplete or duplicated")
    if perturbation_phase is not None:
        perturbation_rows = pq.read_table(
            root / perturbation_phase / "pairwise_responses.parquet"
        ).to_pylist()
        if any(str(row.get("variant")) == "original" for row in perturbation_rows):
            raise ValueError("separate perturbation phase must not repeat original requests")
        original_keys = actual_original_keys
        expected_variants = {
            variant.value
            for variant in PerturbationVariant
            if variant is not PerturbationVariant.ORIGINAL
        }
        actual_keys = {
            (
                str(row["sample_id"]),
                str(row["challenger_candidate_id"]),
                str(row["model_id"]),
                str(row["variant"]),
            )
            for row in perturbation_rows
        }
        expected_keys = {
            (*key, variant) for key in original_keys for variant in expected_variants
        }
        if len(actual_keys) != len(perturbation_rows) or actual_keys != expected_keys:
            raise ValueError("diagnostic perturbation response coverage is incomplete or duplicated")
        responses.extend(perturbation_rows)
    labels = {(row["sample_id"], row["challenger_candidate_id"]): row for row in evaluation}
    outcomes = []
    for response in responses:
        label = labels[(response["sample_id"], response["challenger_candidate_id"])]
        parsed = response.get("parsed")
        switch = p3_hard_rule(parsed)
        baseline_correct = bool(label["corrected_baseline_correct"])
        challenger_correct = bool(label["corrected_challenger_correct"])
        outcomes.append({
            "sample_id": response["sample_id"], "challenger_candidate_id": response["challenger_candidate_id"],
            "model_id": response["model_id"], "variant": response["variant"], "status": response["status"],
            "cohort": label["cohort"], "query_type": label["query_type"],
            "baseline_correct": baseline_correct, "challenger_correct": challenger_correct,
            "selected_correct": challenger_correct if switch else baseline_correct,
            "switched": switch, "critic_preferred": bool(parsed and parsed.get("decision") == "PREFER_CHALLENGER"),
            "evidence_reliable": bool(parsed and parsed.get("evidence_reliable")),
            "valid_response": bool(response["status"] == "SUCCEEDED" and parsed is not None),
            "preference_score": _preference_score(parsed),
        })
    result: dict[str, Any] = {
        "phase": phase,
        "models": {},
        "stability_contract": STABILITY_CONTRACT,
    }
    for model in sorted({row["model_id"] for row in outcomes}):
        original = [row for row in outcomes if row["model_id"] == model and row["variant"] == "original"]
        by_sample: dict[str, list[dict[str, Any]]] = {}
        for row in original:
            by_sample.setdefault(str(row["sample_id"]), []).append(row)
        sample_outcomes = []
        for sample_id, candidates in by_sample.items():
            baseline_correct = bool(candidates[0]["baseline_correct"])
            eligible = [row for row in candidates if row["switched"]]
            selected = sorted(
                eligible,
                key=lambda row: (-float(row["preference_score"]), str(row["challenger_candidate_id"])),
            )[0] if eligible else None
            sample_outcomes.append({
                "sample_id": sample_id, "model_id": model,
                "baseline_correct": baseline_correct,
                "selected_correct": baseline_correct if selected is None else bool(selected["challenger_correct"]),
                "switched": selected is not None,
                "selected_candidate_id": "candidate_0" if selected is None else selected["challenger_candidate_id"],
                "fallback": all(row["status"] != "SUCCEEDED" for row in candidates),
                "cohort": candidates[0]["cohort"], "query_type": candidates[0]["query_type"],
            })
        metrics = full_denominator_metrics(
            sample_outcomes, baseline_key="baseline_correct", selected_key="selected_correct"
        )
        cohorts = {}
        for name in sorted({row["cohort"] for row in original}):
            rows = [row for row in original if row["cohort"] == name]
            cohorts[name] = {
                "pairs": len(rows), "critic_prefer_rate": sum(row["critic_preferred"] for row in rows) / len(rows),
                "hard_rule_switch_rate": sum(row["switched"] for row in rows) / len(rows),
                "beneficial_pairs_preferred": sum(row["critic_preferred"] and not row["baseline_correct"] and row["challenger_correct"] for row in rows),
                "harmful_pairs_preferred": sum(row["critic_preferred"] and row["baseline_correct"] and not row["challenger_correct"] for row in rows),
            }
        # Stability compares every available perturbation with the original pair.
        original_by_pair = {(row["sample_id"], row["challenger_candidate_id"]): row for row in original}
        stability = {}
        for variant in sorted({row["variant"] for row in outcomes if row["model_id"] == model} - {"original"}):
            rows = [row for row in outcomes if row["model_id"] == model and row["variant"] == variant]
            paired = [(original_by_pair[(row["sample_id"], row["challenger_candidate_id"])], row) for row in rows if (row["sample_id"], row["challenger_candidate_id"]) in original_by_pair]
            valid_paired = [
                (original_row, variant_row)
                for original_row, variant_row in paired
                if original_row["valid_response"] and variant_row["valid_response"]
            ]
            stability[variant] = {
                "pairs": len(paired),
                "valid_pairs": len(valid_paired),
                "valid_pair_coverage": len(valid_paired) / len(paired) if paired else 0.0,
                "critic_choice_consistency": sum(a["critic_preferred"] == b["critic_preferred"] for a, b in valid_paired) / len(valid_paired) if valid_paired else None,
                "hard_rule_consistency": sum(a["switched"] == b["switched"] for a, b in valid_paired) / len(valid_paired) if valid_paired else None,
                "direction_flip_rate": sum(a["critic_preferred"] != b["critic_preferred"] for a, b in valid_paired) / len(valid_paired) if valid_paired else None,
            }
        result["models"][model] = {
            "corrected_hard_rule": metrics, "cohorts": cohorts, "stability": stability,
            "status_counts": dict(Counter(row["status"] for row in original)),
        }
        result["models"][model]["stability_passed"] = bool(
            set(stability) == set(STABILITY_CONTRACT["required_variants"])
            and all(
                value["hard_rule_consistency"] is not None
                and value["valid_pair_coverage"]
                >= STABILITY_CONTRACT["minimum_valid_pair_coverage"]
                and value["hard_rule_consistency"]
                >= STABILITY_CONTRACT["minimum_hard_rule_consistency"]
                and value["direction_flip_rate"] is not None
                and value["direction_flip_rate"]
                <= STABILITY_CONTRACT["maximum_direction_flip_rate"]
                for value in stability.values()
            )
        )
        result["models"][model]["sample_denominator"] = len(sample_outcomes)
        result.setdefault("sample_outcomes", []).extend(sample_outcomes)
    _parquet(phase_dir / "pairwise_outcomes.parquet", outcomes)
    _parquet(phase_dir / "sample_outcomes.parquet", result.pop("sample_outcomes", []))
    source_manifest = json.loads(
        (phase_dir / "inference_manifest.json").read_text(encoding="utf-8")
    )
    evidence_paths = {
        "source_inference_sidecar": phase_dir / "INFERENCE_MANIFEST_IDENTITY.json",
        "source_inference_manifest": phase_dir / "inference_manifest.json",
        "source_sample_manifest": phase_dir / "sample_manifest.json",
        "source_evaluation_manifest": phase_dir / "evaluation_manifest.parquet",
        "source_responses": phase_dir / "pairwise_responses.parquet",
        "source_features": Path(str(source_manifest["feature_file"])),
        "pairwise_outcomes": phase_dir / "pairwise_outcomes.parquet",
        "sample_outcomes": phase_dir / "sample_outcomes.parquet",
        "evaluator_source": Path(__file__),
    }
    if perturbation_phase is not None:
        perturbation_dir = root / perturbation_phase
        evidence_paths.update({
            "perturbation_inference_sidecar": perturbation_dir / "INFERENCE_MANIFEST_IDENTITY.json",
            "perturbation_inference_manifest": perturbation_dir / "inference_manifest.json",
            "perturbation_responses": perturbation_dir / "pairwise_responses.parquet",
        })
    result["diagnostic_evidence_bindings"] = {
        name: file_identity(path)
        for name, path in evidence_paths.items()
        if path.is_file()
    }
    _atomic_json(phase_dir / "DIAGNOSTIC_RESULTS.json", result)
    return result
