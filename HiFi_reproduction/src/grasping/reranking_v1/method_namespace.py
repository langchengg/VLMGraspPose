"""Public method names for the retained repeated-FiLM experiment lineage."""

from __future__ import annotations

from typing import Final


PUBLIC_METHOD_NAMES: Final[dict[str, str]] = {
    "q_only": "repeatedfilm_gqcnn_q_only",
    "q_softmask_rule": "repeatedfilm_q_softmask_rule",
    "geometry_gated_q": "repeatedfilm_geometry_gated_q",
    "regularized_linear_ranker": "repeatedfilm_regularized_linear_ranker",
    "pairwise_ranker": "repeatedfilm_pairwise_ranker",
    "multi_positive_listwise_ranker": (
        "repeatedfilm_multi_positive_listwise_ranker"
    ),
    "residual_mlp": "repeatedfilm_residual_mlp",
    "residual_mlp_safe_switch": "repeatedfilm_residual_mlp_safe_switch",
    "set_aware_residual": "repeatedfilm_set_aware_residual",
    "q_top5": "repeatedfilm_gqcnn_q_top5",
    "tabular_residual_top5": "repeatedfilm_tabular_residual_top5",
    "setrank_top5": "repeatedfilm_setrank_top5",
}

FULL_NMS_BASELINE: Final[str] = PUBLIC_METHOD_NAMES["q_only"]
GQCNN_TOP5_BASELINE: Final[str] = PUBLIC_METHOD_NAMES["q_top5"]

FULL_NMS_INTERNAL_METHODS: Final[tuple[str, ...]] = (
    "q_only",
    "q_softmask_rule",
    "geometry_gated_q",
    "regularized_linear_ranker",
    "pairwise_ranker",
    "multi_positive_listwise_ranker",
    "residual_mlp",
    "set_aware_residual",
    "residual_mlp_safe_switch",
)
GQCNN_TOP5_INTERNAL_ALIASES: Final[dict[str, str]] = {
    "q_only": "q_top5",
    "residual_mlp": "tabular_residual_top5",
    "set_aware_residual": "setrank_top5",
}
TABULAR_FORMAL_METHOD_PROTOCOLS: Final[tuple[tuple[str, str], ...]] = (
    *(
        ("full_nms", PUBLIC_METHOD_NAMES[method])
        for method in FULL_NMS_INTERNAL_METHODS
    ),
    *(
        ("gqcnn_top5", PUBLIC_METHOD_NAMES[alias])
        for alias in GQCNN_TOP5_INTERNAL_ALIASES.values()
    ),
)
LOCAL_VLM_RAW_METHODS: Final[frozenset[str]] = frozenset(
    {
        "repeatedfilm_local_vlm_visual",
        "repeatedfilm_local_vlm_visual_metadata",
    }
)
LOCAL_VLM_SAFE_SWITCH_METHOD: Final[str] = (
    "repeatedfilm_local_vlm_safe_switch"
)


def public_method_name(method: str) -> str:
    """Map an internal implementation key to its lineage-explicit name."""

    value = str(method)
    return PUBLIC_METHOD_NAMES.get(value, value)


def public_method_names(methods: list[str] | tuple[str, ...]) -> list[str]:
    return [public_method_name(method) for method in methods]


def expected_formal_method_protocols(
    selected_raw_vlm_method: str,
) -> tuple[tuple[str, str], ...]:
    """Return the exact tabular + validation-selected VLM formal method set."""

    selected = str(selected_raw_vlm_method)
    if selected not in LOCAL_VLM_RAW_METHODS:
        raise ValueError(
            "formal raw VLM must be a public repeated-FiLM VLM method"
        )
    return (
        *TABULAR_FORMAL_METHOD_PROTOCOLS,
        ("gqcnn_top5", selected),
        ("gqcnn_top5", LOCAL_VLM_SAFE_SWITCH_METHOD),
    )
