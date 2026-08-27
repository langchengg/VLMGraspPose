"""Frozen lifecycle and baseline-reconciliation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


SCIENTIFIC_NAME = "Post-formal GT-mask stage-replacement counterfactual diagnostic"


class RunState(StrEnum):
    """Monotonic lifecycle for the independent counterfactual run."""

    P0_AUDIT = "P0_AUDIT"
    P1_BASELINE_REPLAY_PASS = "P1_BASELINE_REPLAY_PASS"
    P2_GT_MAPPING_PASS = "P2_GT_MAPPING_PASS"
    P3_PROTOCOL_LOCKED = "P3_PROTOCOL_LOCKED"
    P4_C1_PILOT_PASS = "P4_C1_PILOT_PASS"
    P5_C1_FULL_COMPLETE = "P5_C1_FULL_COMPLETE"
    # P6 and every downstream state predate the corrected C1-before-G1 order.
    # P5B inserts the required G1 closure without renumbering those public
    # lifecycle values or invalidating existing D1/postprocess consumers.
    P5B_G1_FULL_COMPLETE = "P5B_G1_FULL_COMPLETE"
    P6_D1_COUNTERFACTUAL_COMPLETE = "P6_D1_COUNTERFACTUAL_COMPLETE"
    P7_TAXONOMY_COMPLETE = "P7_TAXONOMY_COMPLETE"
    P8_STATISTICS_COMPLETE = "P8_STATISTICS_COMPLETE"
    P9_GALLERIES_COMPLETE = "P9_GALLERIES_COMPLETE"
    P10_INDEPENDENT_RECOMPUTE_PASS = "P10_INDEPENDENT_RECOMPUTE_PASS"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


RUN_STATE_ORDER = {
    state.value: index
    for index, state in enumerate(
        (
            RunState.P0_AUDIT,
            RunState.P1_BASELINE_REPLAY_PASS,
            RunState.P2_GT_MAPPING_PASS,
            RunState.P3_PROTOCOL_LOCKED,
            RunState.P4_C1_PILOT_PASS,
            RunState.P5_C1_FULL_COMPLETE,
            RunState.P5B_G1_FULL_COMPLETE,
            RunState.P6_D1_COUNTERFACTUAL_COMPLETE,
            RunState.P7_TAXONOMY_COMPLETE,
            RunState.P8_STATISTICS_COMPLETE,
            RunState.P9_GALLERIES_COMPLETE,
            RunState.P10_INDEPENDENT_RECOMPUTE_PASS,
            RunState.COMPLETE,
        )
    )
}


RUN_DIRECTORIES = (
    "00_audit",
    "01_protocol_lock",
    "02_sample_manifest",
    "03_gt_mask_registry",
    "04_predicted_replay",
    "05_gtmask_inputs",
    "06_gtmask_predictions",
    "07_candidate_tables",
    "08_metrics",
    "09_failure_taxonomy",
    "10_statistics",
    "11_stratified_analysis",
    "12_case_selection",
    "13_figures",
    "14_galleries",
    "15_reports",
    "16_independent_recompute",
    "logs",
    "configs",
    "tests",
)


SOURCE_LOCK_EXPECTATIONS = {
    "unified": "4b52eac6494e59a0f902792b794c3cca02824bf7a36bed4699b569986383f793",
    "d1": "bc678841cdb30e9ee562fd71c0749c18822bfad8c92450b38e936bdedbd27970",
}


@dataclass(frozen=True)
class BaselineTarget:
    """Integer authority used only after metrics are recomputed from outcomes."""

    sample_count: int
    native_correct: int
    oracle_top5: int
    oracle_all: int
    no_output: int
    no_positive_full_pool: int
    positive_only_below_top5: int
    final_correct: int
    oracle_top10: int | None = None
    top10_selected_correct: int | None = None
    all_selected_correct: int | None = None
    no_positive_includes_no_output: bool = False


# These are reconciliation authorities, never metric inputs.  Numerators are
# preferred to rounded report decimals so replay is exact at N=7,675.
BASELINE_TARGETS = {
    "g1": BaselineTarget(
        sample_count=7675,
        native_correct=3647,
        oracle_top5=4469,
        oracle_all=4475,
        no_output=41,
        no_positive_full_pool=3159,
        positive_only_below_top5=6,
        final_correct=4347,
    ),
    "c1": BaselineTarget(
        sample_count=7675,
        native_correct=3363,
        oracle_top5=4512,
        oracle_all=4519,
        no_output=13,
        no_positive_full_pool=3143,
        positive_only_below_top5=7,
        final_correct=4317,
    ),
    "d1": BaselineTarget(
        sample_count=7675,
        native_correct=2525,
        oracle_top5=4527,
        oracle_top10=5241,
        oracle_all=6021,
        no_output=108,
        no_positive_full_pool=1654,
        positive_only_below_top5=1494,
        final_correct=3954,
        top10_selected_correct=4410,
        all_selected_correct=4957,
        no_positive_includes_no_output=True,
    ),
}


REQUIRED_PROTOCOL_BINDINGS = (
    "source_locks",
    "source_code",
    "configs",
    "baseline_replay",
    "sample_manifest",
    "gt_grasp_source",
    "gt_mask_registry",
    "mapping_qa",
    "route_contracts",
    "resize_rules",
    "evaluator",
    "taxonomy",
    "statistics",
    "case_selection",
)
