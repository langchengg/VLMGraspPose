from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .schema import PairwiseDecision, ReasonCode


def critic_feature_vector(
    outputs: Mapping[str, Mapping[str, object] | None],
    *,
    model_order: Sequence[str] = ("gemini-robotics-er-2-preview", "gemini-3.6-flash"),
) -> tuple[list[str], np.ndarray]:
    """Encode model reports as observations; missing/failure is explicit."""

    names: list[str] = []; values: list[float] = []
    score_fields = (
        "baseline_target_alignment", "challenger_target_alignment",
        "baseline_contact_geometry", "challenger_contact_geometry",
        "baseline_collision_risk", "challenger_collision_risk",
        "baseline_width_compatibility", "challenger_width_compatibility",
    )
    for model in model_order:
        prefix = "er2" if "robotics" in model else "flash"
        output = outputs.get(model)
        names.append(f"{prefix}_missing"); values.append(float(output is None))
        for decision in PairwiseDecision:
            names.append(f"{prefix}_decision_{decision.value.lower()}")
            values.append(float(output is not None and output.get("decision") == decision.value))
        names.append(f"{prefix}_evidence_reliable")
        values.append(float(bool(output and output.get("evidence_reliable", False))))
        for field in score_fields:
            names.append(f"{prefix}_{field}")
            values.append(float(output.get(field, 0.0)) if output is not None else 0.0)
        reasons = set(output.get("reason_codes", [])) if output is not None else set()
        for reason in ReasonCode:
            names.append(f"{prefix}_reason_{reason.value.lower()}")
            values.append(float(reason.value in reasons))
    return names, np.asarray(values, dtype=np.float64)


def p3_hard_rule(output: Mapping[str, object] | None) -> bool:
    if output is None or output.get("decision") != PairwiseDecision.PREFER_CHALLENGER.value:
        return False
    if not bool(output.get("evidence_reliable", False)):
        return False
    target_delta = float(output["challenger_target_alignment"]) - float(output["baseline_target_alignment"])
    contact_delta = float(output["challenger_contact_geometry"]) - float(output["baseline_contact_geometry"])
    collision_delta = float(output["challenger_collision_risk"]) - float(output["baseline_collision_risk"])
    width_delta = float(output["challenger_width_compatibility"]) - float(output["baseline_width_compatibility"])
    return target_delta >= 0.15 and contact_delta >= 0.10 and collision_delta <= 0.0 and width_delta >= -0.05

