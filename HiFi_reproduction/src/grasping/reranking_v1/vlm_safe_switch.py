"""Pure, GT-free application of a validation-locked local-VLM safe switch."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from .artifact_contract import LOCAL_VLM_GEOMETRY_SEMANTICS
from .local_vlm import validate_effective_result_record


def apply_vlm_safe_switch_policy(
    candidates: pd.DataFrame,
    results: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild formal predictions and decisions without using ground truth."""

    if selection.get("geometry_semantics") != LOCAL_VLM_GEOMETRY_SEMANTICS:
        raise ValueError(
            "VLM safe-switch geometry must remain a visible-surface proxy"
        )
    required = {
        "sample_id",
        "candidate_id",
        "candidate_identity_sha256",
        "original_gqcnn_rank",
        str(selection["geometry_risk_column"]),
    }
    if missing := sorted(required - set(candidates.columns)):
        raise ValueError(f"VLM safe-switch candidate columns missing: {missing}")
    by_sample = {
        str(sample_id): group
        for sample_id, group in candidates.groupby("sample_id", sort=False)
    }
    predictions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    geometry_column = str(selection["geometry_risk_column"])
    for result in results:
        sample_id = str(result["sample_id"])
        group = by_sample.get(sample_id)
        if group is None or group.empty:
            if bool(result.get("eligible_for_vlm", False)):
                raise ValueError(
                    f"VLM claims unknown/empty sample is eligible: {sample_id}"
                )
            continue
        top5 = group.loc[group["original_gqcnn_rank"] <= 5].sort_values(
            ["original_gqcnn_rank", "candidate_id"], kind="mergesort"
        )
        q_ids = top5["candidate_id"].astype(str).tolist()
        validated_ranking = validate_effective_result_record(
            result,
            candidate_ids=q_ids,
            original_top1_candidate_id=q_ids[0],
        )
        ranking = [item.candidate_id for item in validated_ranking]
        if set(ranking) != set(q_ids) or len(ranking) != len(set(ranking)):
            raise ValueError(
                f"VLM result changed or duplicated Top-5 pool: {sample_id}"
            )
        selected_id = result.get("selected_candidate_id")
        selected = top5.loc[
            top5["candidate_id"].astype(str) == str(selected_id)
        ]
        valid = len(selected) == 1
        geometry_safe = (
            valid
            and float(selected.iloc[0][geometry_column])
            <= float(selection["geometry_risk_threshold"])
        )
        threshold_kind = str(selection["threshold_kind"])
        threshold_pass = (
            False
            if threshold_kind == "never_switch"
            else float(result.get("confidence", 0.0))
            >= float(selection["threshold"])
        )
        switch = (
            valid
            and not bool(result.get("fallback"))
            and not bool(result.get("abstain"))
            and str(selected_id) != q_ids[0]
            and geometry_safe
            and threshold_pass
        )
        order = ranking if switch else q_ids
        identity = top5.set_index("candidate_id")[
            "candidate_identity_sha256"
        ]
        predictions.extend(
            {
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "candidate_identity_sha256": str(identity.loc[candidate_id]),
                "method": str(selection["method"]),
                "protocol": "gqcnn_top5",
                "rank": rank,
            }
            for rank, candidate_id in enumerate(order, start=1)
        )
        decisions.append(
            {
                "sample_id": sample_id,
                "old_candidate_id": q_ids[0],
                "proposed_candidate_id": selected_id,
                "selected_candidate_id": order[0],
                "confidence": float(result.get("confidence", 0.0)),
                "geometry_safe": bool(geometry_safe),
                "switch_applied": bool(switch),
                "fallback": bool(result.get("fallback")),
                "abstain": bool(result.get("abstain")),
                "GT_columns_used": False,
            }
        )
    return pd.DataFrame(predictions), pd.DataFrame(decisions)
