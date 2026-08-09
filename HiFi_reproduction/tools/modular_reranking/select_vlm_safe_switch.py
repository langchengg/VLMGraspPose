#!/usr/bin/env python3
"""Select a validation-only VLM safe switch with an explicit never-switch arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
    local_vlm_preregistration_payload,
    validate_artifact_identity,
    validate_config_identity,
    validate_vlm_summary_runtime_binding,
)
from src.grasping.reranking_v1.local_vlm import (  # noqa: E402
    validate_effective_result_record,
)

VLM_VISUAL_METHOD = "repeatedfilm_local_vlm_visual"
VLM_METADATA_METHOD = "repeatedfilm_local_vlm_visual_metadata"
VLM_SAFE_SWITCH_METHOD = "repeatedfilm_local_vlm_safe_switch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument(
        "--per-sample",
        type=Path,
        required=True,
        help="Complete validation universe, including valid-empty samples",
    )
    parser.add_argument("--vlm-results", type=Path, required=True)
    parser.add_argument("--vlm-summary", type=Path, required=True)
    parser.add_argument("--vlm-runtime-metrics", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--geometry-risk-column", required=True)
    parser.add_argument("--geometry-risk-threshold", type=float, required=True)
    parser.add_argument("--harmful-rate-limit", type=float, default=0.01)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("VLM safe-switch config must be a mapping")
    validate_config_identity(config)
    preregistered = local_vlm_preregistration_payload(config)
    if (
        args.geometry_risk_column
        != preregistered["geometry_risk_column"]
        or args.geometry_risk_threshold
        != preregistered["geometry_risk_threshold"]
    ):
        raise ValueError(
            "VLM safe-switch geometry disagrees with preregistered config"
        )
    if args.method not in {VLM_VISUAL_METHOD, VLM_METADATA_METHOD}:
        raise ValueError("safe-switch source must be a registered local VLM variant")
    if not 0.0 <= args.harmful_rate_limit <= 1.0:
        raise ValueError("harmful-rate-limit must be in [0,1]")
    candidate_path = args.per_candidate.resolve()
    sample_path = args.per_sample.resolve()
    results_path = args.vlm_results.resolve()
    summary_path = args.vlm_summary.resolve()
    runtime_path = args.vlm_runtime_metrics.resolve()
    candidates = pd.read_parquet(candidate_path)
    sample_universe = pd.read_parquet(sample_path)
    if "sample_id" not in sample_universe:
        raise ValueError("per-sample table requires sample_id")
    universe_ids = sample_universe["sample_id"].astype(str)
    if universe_ids.duplicated().any():
        raise ValueError("per-sample table has duplicate sample IDs")
    if (
        "split" not in sample_universe
        or set(sample_universe["split"].astype(str)) - {"val", "validation"}
        or "split" not in candidates
        or set(candidates["split"].astype(str)) - {"val", "validation"}
    ):
        raise ValueError("VLM safe-switch selection may consume validation only")
    universe_set = set(universe_ids)
    candidate_sample_ids = set(candidates["sample_id"].astype(str))
    if not candidate_sample_ids <= universe_set:
        raise ValueError("candidate table contains samples outside validation universe")
    required = {
        "sample_id",
        "candidate_id",
        "original_gqcnn_rank",
        "candidate_positive",
        args.geometry_risk_column,
    }
    if missing := sorted(required - set(candidates)):
        raise ValueError(f"candidate table missing columns: {missing}")
    results = _jsonl(results_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime_metrics = json.loads(runtime_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        summary, context="VLM safe-switch validation summary"
    )
    validate_artifact_identity(
        runtime_metrics, context="VLM safe-switch validation runtime"
    )
    eligible_count = sum(
        bool(row.get("eligible_for_vlm")) for row in results
    )
    validate_vlm_summary_runtime_binding(
        summary,
        runtime_metrics,
        results_path=results_path,
        expected_split="validation",
        expected_formal_mode=False,
        expected_sample_count=len(sample_universe),
        expected_eligible_count=eligible_count,
        expected_max_output_tokens=int(
            preregistered["max_output_tokens"]
        ),
        context="VLM safe-switch validation run",
    )
    if len({row["sample_id"] for row in results}) != len(results):
        raise ValueError("VLM results have duplicate sample IDs")
    result_ids = {str(row["sample_id"]) for row in results}
    if result_ids != set(universe_ids):
        raise ValueError(
            "VLM results must cover the complete validation per-sample universe"
        )
    by_sample = {sample_id: group for sample_id, group in candidates.groupby("sample_id")}
    examples: list[dict] = []
    ranking_by_sample: dict[str, list[str]] = {}
    for row in results:
        sample_id = str(row["sample_id"])
        ranking_by_sample[sample_id] = [
            str(item["candidate_id"]) for item in row.get("ranking", [])
        ]
        group = by_sample.get(sample_id)
        if group is None or group.empty:
            if (
                row.get("eligible_for_vlm") is not False
                or row.get("selected_candidate_id") is not None
                or row.get("ranking") != []
                or row.get("skip_reason") != "valid_empty_no_vlm_call"
                or bool(row.get("http_call_performed"))
            ):
                raise ValueError(f"invalid valid-empty VLM result: {sample_id}")
            examples.append(
                {
                    "sample_id": sample_id,
                    "old_candidate_id": None,
                    "new_candidate_id": None,
                    "old_correct": False,
                    "new_correct": False,
                    "switch_outcome": "neutral",
                    "confidence": 0.0,
                    "geometry_safe": False,
                    "eligible": False,
                }
            )
            continue
        top5 = group.loc[group["original_gqcnn_rank"].astype(int) <= 5].sort_values(
            ["original_gqcnn_rank", "candidate_id"], kind="mergesort"
        )
        q_ids = top5["candidate_id"].astype(str).tolist()
        validate_effective_result_record(
            row,
            candidate_ids=q_ids,
            original_top1_candidate_id=q_ids[0],
        )
        if ranking_by_sample[sample_id] != [
            item["candidate_id"] for item in row["ranking"]
        ]:
            raise AssertionError("VLM ranking normalization changed IDs")
        old = top5.iloc[0]
        selected_id = row.get("selected_candidate_id")
        selected = group.loc[group["candidate_id"].astype(str) == str(selected_id)]
        valid_selected = len(selected) == 1
        new = selected.iloc[0] if valid_selected else old
        old_correct = bool(old["candidate_positive"])
        new_correct = bool(new["candidate_positive"])
        outcome = (
            "beneficial"
            if not old_correct and new_correct
            else "harmful"
            if old_correct and not new_correct
            else "neutral"
        )
        geometry_safe = (
            float(new[args.geometry_risk_column])
            <= args.geometry_risk_threshold
        )
        eligible = (
            valid_selected
            and not bool(row.get("fallback"))
            and not bool(row.get("abstain"))
            and str(selected_id) != str(old["candidate_id"])
            and geometry_safe
        )
        examples.append(
            {
                "sample_id": sample_id,
                "old_candidate_id": str(old["candidate_id"]),
                "new_candidate_id": str(new["candidate_id"]),
                "old_correct": old_correct,
                "new_correct": new_correct,
                "switch_outcome": outcome,
                "confidence": float(row.get("confidence", 0.0)),
                "geometry_safe": geometry_safe,
                "eligible": eligible,
            }
        )
    frame = pd.DataFrame(examples)
    numeric_thresholds = sorted(frame.loc[frame["eligible"], "confidence"].unique())
    arms: list[tuple[str, float | None]] = [
        ("threshold", float(value)) for value in numeric_thresholds
    ]
    arms.append(("never_switch", None))
    sweep = []
    total = max(len(sample_universe), 1)
    decisions_by_arm: dict[tuple[str, float | None], pd.Series] = {}
    for kind, threshold in arms:
        switched = (
            pd.Series(False, index=frame.index)
            if kind == "never_switch"
            else frame["eligible"] & (frame["confidence"] >= float(threshold))
        )
        decisions_by_arm[(kind, threshold)] = switched
        recovered = int((switched & frame["switch_outcome"].eq("beneficial")).sum())
        harmful = int((switched & frame["switch_outcome"].eq("harmful")).sum())
        switch_count = int(switched.sum())
        outcome_changes = recovered + harmful
        sweep.append(
            {
                "threshold_kind": kind,
                "threshold": threshold,
                "recovered": recovered,
                "harmful": harmful,
                "net_gain": recovered - harmful,
                "harmful_rate": harmful / total,
                "harmful_rate_all_samples": harmful / total,
                "coverage": switch_count / total,
                "decision_precision": recovered / switch_count if switch_count else 0.0,
                "outcome_precision": (
                    recovered / outcome_changes if outcome_changes else 0.0
                ),
                "eligible_under_harm_limit": (
                    harmful / total <= args.harmful_rate_limit
                ),
            }
        )
    sweep_frame = pd.DataFrame(sweep)
    eligible_arms = sweep_frame.loc[sweep_frame["eligible_under_harm_limit"]].copy()
    eligible_arms["_never"] = eligible_arms["threshold_kind"].eq("never_switch")
    selected = eligible_arms.sort_values(
        [
            "net_gain",
            "harmful",
            "recovered",
            "_never",
            "outcome_precision",
            "coverage",
            "threshold",
        ],
        ascending=[False, True, False, False, False, True, False],
        na_position="first",
    ).iloc[0]
    selected_key = (
        str(selected["threshold_kind"]),
        None if pd.isna(selected["threshold"]) else float(selected["threshold"]),
    )
    switched = decisions_by_arm[selected_key]
    frame["switch_applied"] = switched
    frame["selected_candidate_id"] = frame["old_candidate_id"]
    frame.loc[switched, "selected_candidate_id"] = frame.loc[
        switched, "new_candidate_id"
    ]
    predictions = []
    for row in frame.itertuples(index=False):
        group = by_sample.get(row.sample_id)
        if group is None:
            continue
        q_ids = (
            group.loc[group["original_gqcnn_rank"] <= 5]
            .sort_values(["original_gqcnn_rank", "candidate_id"])["candidate_id"]
            .astype(str)
            .tolist()
        )
        vlm_ids = ranking_by_sample[row.sample_id]
        order = vlm_ids if row.switch_applied else q_ids
        if set(order) != set(q_ids):
            raise ValueError(f"{row.sample_id} safe-switch changed the Top-5 pool")
        predictions.extend(
            {
                "sample_id": row.sample_id,
                "candidate_id": candidate_id,
                "method": VLM_SAFE_SWITCH_METHOD,
                "protocol": "gqcnn_top5",
                "rank": rank,
            }
            for rank, candidate_id in enumerate(order, start=1)
        )
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(root / "validation_decisions.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(root / "predictions.parquet", index=False)
    sweep_frame.drop(columns=[], errors="ignore").to_csv(
        root / "threshold_sweep.csv", index=False
    )
    selection = {
        key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in selected.drop(labels=["_never"]).to_dict().items()
    }
    selection.update(
        {
            **identity_payload(),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "method": VLM_SAFE_SWITCH_METHOD,
            "source_method": args.method,
            "source_variant": (
                "visual_metadata" if args.method == VLM_METADATA_METHOD else "visual"
            ),
            "selection_split": "validation",
            "harmful_rate_limit": args.harmful_rate_limit,
            "harmful_rate_denominator": "all_validation_samples",
            "all_validation_sample_count": len(sample_universe),
            "nonempty_validation_sample_count": len(by_sample),
            "geometry_risk_column": args.geometry_risk_column,
            "geometry_risk_threshold": args.geometry_risk_threshold,
            "geometry_semantics": preregistered["geometry_semantics"],
            "explicit_never_switch_arm_included": True,
            "inputs": {
                "per_candidate": str(candidate_path),
                "per_candidate_sha256": sha256_file(candidate_path),
                "per_sample": str(sample_path),
                "per_sample_sha256": sha256_file(sample_path),
                "vlm_results": str(results_path),
                "vlm_results_sha256": sha256_file(results_path),
                "vlm_summary": str(summary_path),
                "vlm_summary_sha256": sha256_file(summary_path),
                "vlm_runtime_metrics": str(runtime_path),
                "vlm_runtime_metrics_sha256": sha256_file(runtime_path),
            },
            "vlm_model_name": summary.get("model_name"),
            "vlm_model_digest": summary.get("model_digest"),
            "stable_session_contract_sha256": summary.get(
                "stable_session_contract_sha256"
            ),
            "vlm_ollama_version": summary.get("ollama_version"),
            "vlm_runtime_local_only_passed": runtime_metrics.get(
                "local_only_runtime_passed"
            ),
        }
    )
    (root / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
