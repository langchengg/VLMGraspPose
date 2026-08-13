"""Static, label-free command DAG for the D1 P0--P17 protocol.

The catalog is intentionally declarative.  It does not inspect a run directory,
open Test data, or execute any command.  Its purpose is to keep the documented
execution order honest: every edge names a producer, every heavy node names its
fresh resource-gate scope, and known missing producers remain explicit blockers.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CommandNode:
    """One aggregate command (possibly a declared split/pool fan-out)."""

    node_id: str
    stage: str
    description: str
    command: str | None
    dependencies: tuple[str, ...]
    outputs: tuple[str, ...]
    gate_scope: str | None = None
    lifecycle_after: str | None = None
    fanout: str | None = None
    blocked_reason: str | None = None

    @property
    def executable(self) -> bool:
        return self.command is not None and self.blocked_reason is None


def _node(
    node_id: str,
    stage: str,
    description: str,
    tool: str | None,
    dependencies: tuple[str, ...],
    outputs: tuple[str, ...],
    *,
    args: str = "--run-dir {RUN_DIR}",
    gate_scope: str | None = None,
    lifecycle_after: str | None = None,
    fanout: str | None = None,
    blocked_reason: str | None = None,
) -> CommandNode:
    command = None if tool is None else f"{{PYTHON}} tools/d1_reranking/{tool} {args}"
    return CommandNode(
        node_id=node_id,
        stage=stage,
        description=description,
        command=command,
        dependencies=dependencies,
        outputs=outputs,
        gate_scope=gate_scope,
        lifecycle_after=lifecycle_after,
        fanout=fanout,
        blocked_reason=blocked_reason,
    )


def command_nodes() -> tuple[CommandNode, ...]:
    """Return the frozen aggregate command catalog in dependency order."""

    n = _node
    return (
        n(
            "bootstrap",
            "P0",
            "Create the D1 run skeleton and bind the frozen unified run/snapshot/evaluator.",
            "bootstrap.py",
            (),
            (
                "manifest.json",
                "pipeline_status.json",
                "run_ledger.sqlite",
                "00_audit/SOURCE_RUN_IMMUTABILITY_BEFORE.json",
            ),
            args=(
                "--run-dir {RUN_DIR} --unified-run {UNIFIED_RUN} --snapshot-a {SNAPSHOT_A} "
                "--evaluator {EVALUATOR}"
            ),
            lifecycle_after="AUDIT",
        ),
        n(
            "source_closure",
            "P1",
            "Reconcile exact source, pairing, evaluator, and label-isolation closure.",
            "reconcile_sources.py",
            ("bootstrap",),
            (
                "00_audit/source_closures/<closure_id>.json",
                "configs/d1_active_source_closure.json",
            ),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "candidate_policy",
            "P2",
            "Freeze the candidate-generation resource policy.",
            "write_candidate_resource_policy.py",
            ("source_closure",),
            (
                "configs/candidate_resource_policies/<policy_id>.json",
                "configs/d1_candidate_resource_policy.json",
            ),
        ),
        n(
            "candidate_gate",
            "P2",
            "Run the required three continuous five-minute candidate resource windows.",
            "audit_resources.py",
            ("candidate_policy",),
            ("00_audit/resource_gates/<gate_id>.json", "00_audit/RESOURCE_AUDIT.json"),
            args="--run-dir {RUN_DIR} --scope candidates --owner {OWNER}",
            gate_scope="candidates",
        ),
        n(
            "candidates",
            "P2-P3",
            "Materialize exact Train/Validation/Test Top5, Top10, and AllNMS pools.",
            "build_candidates.py",
            ("candidate_gate",),
            (
                "02_candidates/train/manifest.json",
                "02_candidates/validation/manifest.json",
                "02_candidates/test_manifest.json",
                "02_candidates/candidate_registry.json",
            ),
            args="--run-dir {RUN_DIR} --split all --resume",
            gate_scope="candidates",
            lifecycle_after="CANDIDATES_FROZEN",
        ),
        n(
            "development_labels",
            "P4",
            "Build development-only candidate labels for every Train/Validation pool.",
            "build_dev_labels.py",
            ("candidates",),
            (
                "03_features/{train,validation}/{top5,top10,allnms}/labels/manifest.json",
            ),
            args="--run-dir {RUN_DIR} --split {DEV_SPLIT} --pool {POOL} --resume",
            fanout="DEV_SPLIT=train,validation x POOL=top5,top10,allnms (6 commands)",
        ),
        n(
            "folds",
            "P7",
            "Copy and verify the frozen five-fold development split.",
            "build_splits.py",
            ("source_closure",),
            (
                "04_splits/fold_assignments.parquet",
                "04_splits/split_leakage_audit.json",
            ),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "feature_plan",
            "P4-P6",
            "Freeze the exact nine raw matched-common extraction jobs.",
            "plan_feature_extraction.py",
            ("candidates",),
            (
                "configs/feature_plans/<plan_id>.json",
                "configs/d1_feature_extraction_plan_active.json",
            ),
            args="--run-dir {RUN_DIR} --python {PYTHON} --resume",
        ),
        n(
            "feature_policy",
            "P4-P6",
            "Freeze the raw-feature extraction resource policy.",
            "write_feature_resource_policy.py",
            ("feature_plan",),
            (
                "configs/feature_resource_policies/<policy_id>.json",
                "configs/d1_feature_resource_policy.json",
            ),
        ),
        n(
            "feature_gate",
            "P4-P6",
            "Run the required three continuous five-minute raw-feature resource windows.",
            "audit_resources.py",
            ("feature_policy",),
            ("00_audit/resource_gates/<gate_id>.json",),
            args="--run-dir {RUN_DIR} --scope features --owner {OWNER}",
            gate_scope="features",
        ),
        n(
            "feature_authority",
            "P4-P6",
            "Create immutable execution authority for the nine raw-feature jobs.",
            "authorize_feature_extraction.py",
            ("feature_gate",),
            ("configs/feature_executions/<execution_id>/execution.json",),
            args="--run-dir {RUN_DIR} --gate-manifest {FEATURE_GATE} --owner {OWNER}",
            gate_scope="features",
        ),
        n(
            "feature_matrix",
            "P4-P6",
            "Serially execute or fresh-gated resume the exact nine raw-feature jobs.",
            "run_feature_extraction_matrix.py",
            ("feature_authority",),
            (
                "configs/d1_feature_execution.json",
                "03_features/{split}/{pool}/matched_common_raw/manifest.json",
            ),
            args="--run-dir {RUN_DIR} --python {PYTHON}",
            gate_scope="features",
        ),
        n(
            "feature_finalize",
            "P4-P6",
            "Replay the exact nine-job/result bijection and publish feature readiness.",
            "finalize_feature_extraction.py",
            ("feature_matrix",),
            ("configs/d1_feature_execution_manifest.json",),
            lifecycle_after="FEATURES_READY",
        ),
        n(
            "t1_features",
            "P5",
            "Project the label-free native T1 feature track.",
            "project_native_features.py",
            ("feature_finalize",),
            ("03_features/{split}/{pool}/T1_native_available/manifest.json",),
            args="--run-dir {RUN_DIR} --split {SPLIT} --pool {POOL} --resume",
            fanout="SPLIT=train,validation,test x POOL=top5,top10,allnms (9 commands)",
        ),
        n(
            "calibration_fit",
            "P8",
            "Fit strict fold-local Train/Validation calibrators for each pool.",
            "fit_calibration.py",
            ("development_labels", "folds", "feature_finalize"),
            ("05_calibration/{pool}/calibration_manifest.json",),
            args="--run-dir {RUN_DIR} --pool {POOL} --resume",
            fanout="POOL=top5,top10,allnms (3 commands)",
        ),
        n(
            "calibration_test",
            "P8",
            "Apply each frozen calibrator to label-free Test candidates.",
            "apply_calibration.py",
            ("calibration_fit",),
            ("05_calibration/{pool}/test_calibration_manifest.json",),
            args="--run-dir {RUN_DIR} --pool {POOL} --resume",
            fanout="POOL=top5,top10,allnms (3 commands)",
        ),
        n(
            "t2_features",
            "P5",
            "Finalize matched-common T2 features for all split/pool cells.",
            "finalize_matched_common.py",
            ("feature_finalize", "calibration_fit", "calibration_test"),
            ("03_features/{split}/{pool}/T2_matched_common/manifest.json",),
            args="--run-dir {RUN_DIR} --split {SPLIT} --pool {POOL} --resume",
            fanout="SPLIT=train,validation,test x POOL=top5,top10,allnms (9 commands)",
        ),
        n(
            "t3_features",
            "P5",
            "Assemble route-rich T3 features for all split/pool cells.",
            "assemble_route_rich_features.py",
            ("t1_features", "t2_features"),
            ("03_features/{split}/{pool}/T3_route_rich/manifest.json",),
            args="--run-dir {RUN_DIR} --split {SPLIT} --pool {POOL} --resume",
            fanout="SPLIT=train,validation,test x POOL=top5,top10,allnms (9 commands)",
        ),
        n(
            "primary_plan",
            "P9",
            "Freeze the 360-cell R2--R6 Top5/T2 development matrix.",
            "plan_primary_matrix.py",
            ("t2_features", "development_labels", "folds"),
            ("configs/d1_primary_matrix_plan_active.json",),
            args="--run-dir {RUN_DIR} --python {PYTHON} --resume",
        ),
        n(
            "primary_policy",
            "P9",
            "Freeze the primary-matrix resource policy.",
            "write_resource_policy.py",
            ("primary_plan",),
            ("configs/d1_primary_resource_gate_policy_active.json",),
        ),
        n(
            "primary_gate",
            "P9",
            "Run the required three continuous five-minute primary resource windows.",
            "audit_resources.py",
            ("primary_policy",),
            ("00_audit/resource_gates/<gate_id>.json",),
            args="--run-dir {RUN_DIR} --scope primary --owner {OWNER}",
            gate_scope="primary",
        ),
        n(
            "primary_authority",
            "P9",
            "Authorize the fixed serial CPU primary matrix.",
            "authorize_primary_execution.py",
            ("primary_gate",),
            ("configs/primary_matrix_executions/<execution_id>/execution.json",),
            args="--run-dir {RUN_DIR} --gate-manifest {PRIMARY_GATE} --owner {OWNER}",
            gate_scope="primary",
        ),
        n(
            "primary_matrix",
            "P9",
            "Execute the fixed 360-cell primary matrix.",
            "run_primary_matrix.py",
            ("primary_authority",),
            (
                "06_oof/primary_cells/<cell_key>/manifest.json",
                "07_validation/primary_cells/<cell_key>/manifest.json",
                "07_validation/primary_matrix_execution.json",
            ),
            args="--run-dir {RUN_DIR} --python {PYTHON}",
            gate_scope="primary",
            lifecycle_after="VALIDATION_SCREEN",
        ),
        n(
            "r0_r1",
            "P9",
            "Produce R0/R1 baselines and the complete R0--R7 Validation table inputs.",
            "run_r0_r1.py",
            ("t2_features", "development_labels", "folds"),
            (
                "07_validation/r0_r1_selection.json",
                "07_validation/tables/r1_trials.csv",
            ),
        ),
        n(
            "primary_selection",
            "P9",
            "Select the frozen Validation winner from R3/R5/R6.",
            "select_primary_ranker.py",
            ("primary_matrix", "r0_r1"),
            (
                "07_validation/selected_primary_ungated.json",
                "07_validation/tables/selected_primary_ungated.csv",
            ),
        ),
        n(
            "gate_inputs",
            "P11",
            "Build OOF/Validation expected-gain gate inputs.",
            "prepare_gate_inputs.py",
            ("primary_selection",),
            ("07_validation/gate_inputs/manifest.json",),
        ),
        n(
            "gate_grid",
            "P11",
            "Freeze the 108-point R7 gate grid.",
            "plan_gate_grid.py",
            ("gate_inputs",),
            ("configs/d1_gate_grid.json",),
        ),
        n(
            "gate_selection",
            "P11",
            "Replay and select the Validation R7 operating point.",
            "select_gate.py",
            ("gate_grid",),
            (
                "07_validation/gate/d1/gate_selection.json",
                "07_validation/tables/gate_operating_point.csv",
            ),
        ),
        n(
            "top5_test_ranker",
            "P13",
            "Apply the selected Top5 ranker to Test without labels.",
            "apply_selected_test_ranker.py",
            ("primary_selection", "t2_features"),
            ("08_lock/label_free_test_rankers/d1/manifest.json",),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "top5_test_gate",
            "P13",
            "Apply the selected Top5 R7 gate to Test without labels.",
            "apply_locked_test_gate.py",
            ("gate_selection", "top5_test_ranker"),
            ("08_lock/label_free_test_gates/d1/manifest.json",),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "k_plan",
            "P10",
            "Freeze the exact 54-cell Top10/AllNMS K-sensitivity matrix.",
            "plan_k_sensitivity.py",
            (
                "primary_selection",
                "t2_features",
                "t3_features",
                "development_labels",
                "folds",
            ),
            ("configs/d1_k_sensitivity_plan.json",),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "k_policy",
            "P10",
            "Freeze the K-sensitivity resource policy.",
            "write_k_sensitivity_resource_policy.py",
            ("k_plan",),
            ("configs/d1_k_resource_policy.json",),
        ),
        n(
            "k_gate",
            "P10",
            "Run the required three continuous five-minute K-sensitivity windows.",
            "audit_resources.py",
            ("k_policy",),
            ("00_audit/resource_gates/<gate_id>.json",),
            args="--run-dir {RUN_DIR} --scope k_sensitivity --owner {OWNER}",
            gate_scope="k_sensitivity",
        ),
        n(
            "k_authority",
            "P10",
            "Create immutable authority for the serial 54-cell K matrix.",
            "authorize_k_sensitivity_execution.py",
            ("k_gate",),
            ("configs/k_sensitivity_executions/<execution_id>/execution.json",),
            args="--run-dir {RUN_DIR} --gate-manifest {K_GATE} --owner {OWNER}",
            gate_scope="k_sensitivity",
        ),
        n(
            "k_matrix",
            "P10",
            "Execute or fresh-gated resume the serial 54-cell K matrix.",
            "run_k_sensitivity_matrix.py",
            ("k_authority",),
            ("11_k_sensitivity/execution_manifest.json",),
            args="--run-dir {RUN_DIR} --python {PYTHON}",
            gate_scope="k_sensitivity",
        ),
        n(
            "k_selection",
            "P10",
            "Select K sensitivity results and publish the comparison table.",
            "select_k_sensitivity.py",
            ("k_matrix",),
            (
                "11_k_sensitivity/selection_manifest.json",
                "11_k_sensitivity/k_comparison.csv",
            ),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "k_gates",
            "P10-P13",
            "Build scenario-specific Top10 and AllNMS R7 gates.",
            "build_k_scenario_gates.py",
            ("k_selection",),
            ("11_k_sensitivity/scenario_gates/manifest.json",),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "k_formal_inputs",
            "P13-P14",
            "Build five normalized D1 formal inputs (Top5 variants, Top10, AllNMS).",
            "build_d1_formal_inputs.py",
            ("top5_test_ranker", "top5_test_gate", "k_gates", "t2_features"),
            (
                "08_lock/formal_inputs/d1_top5_r0/manifest.json",
                "08_lock/formal_inputs/d1_top5_r7_ungated/manifest.json",
                "08_lock/formal_inputs/d1_top5_r7_gated/manifest.json",
                "08_lock/formal_inputs/d1_top10_locked/manifest.json",
                "08_lock/formal_inputs/d1_allnms_locked/manifest.json",
            ),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "p12_plan",
            "P12",
            "Freeze the four-route router/Top20/T4 plan from explicit named sources.",
            "plan_four_route_extension.py",
            ("top5_test_ranker", "top5_test_gate", "t3_features"),
            ("configs/d1_four_route_extension_plan.json",),
            args=(
                "--run-dir {RUN_DIR} --three-route-run {THREE_ROUTE_RUN} "
                "--producer-spec {P12_PRODUCER_SPEC} {P12_THREE_SOURCES} {P12_D1_SOURCES}"
            ),
        ),
        n(
            "p12_policy",
            "P12",
            "Freeze the exact plan/code-bound P12 Validation resource policy.",
            "run_four_route_validation.py",
            ("p12_plan",),
            ("configs/d1_four_route_validation_resource_policy.json",),
            args="write-policy --run-dir {RUN_DIR}",
        ),
        n(
            "p12_gate",
            "P12",
            "Run the required three continuous five-minute P12 Validation resource gate.",
            "audit_resources.py",
            ("p12_policy",),
            ("00_audit/resource_gates/<gate_id>.json", "00_audit/RESOURCE_AUDIT.json"),
            args="--run-dir {RUN_DIR} --scope four_route_validation --owner {OWNER}",
            gate_scope="four_route_validation",
        ),
        n(
            "p12_authorize",
            "P12",
            "Authorize one immutable P12 Validation execution from the fresh gate.",
            "run_four_route_validation.py",
            ("p12_gate",),
            ("configs/d1_four_route_validation_execution.json",),
            args=(
                "authorize --run-dir {RUN_DIR} --gate-manifest {P12_GATE_MANIFEST} "
                "--owner {OWNER}"
            ),
        ),
        n(
            "p12_validation",
            "P12",
            "Run the Validation-only router/union/T4 producer matrix.",
            "run_four_route_validation.py",
            ("p12_authorize",),
            (
                "13_four_route_extension/validation_router_union_manifest.json",
                "13_four_route_extension/router/router_selection_manifest.json",
                "13_four_route_extension/union/selected_union_ranker.json",
                "03_features/{train,validation}/top5/T4_four_route_consensus/manifest.json",
            ),
            args="run --run-dir {RUN_DIR}",
            gate_scope="four_route_validation",
        ),
        n(
            "ablation_plan",
            "P10",
            "Freeze evidence-track and feature-family ablations, including T4.",
            "plan_ablation.py",
            ("p12_validation", "primary_selection"),
            ("configs/d1_ablation_plan_v1.json",),
        ),
        n(
            "ablation_policy",
            "P10",
            "Freeze the versioned P10 ablation resource policy.",
            "write_ablation_resource_policy.py",
            ("ablation_plan",),
            ("configs/ablation_resource_policies/<policy_id>.json",),
        ),
        n(
            "ablation_gate",
            "P10",
            "Run the required three continuous five-minute P10 ablation gate.",
            "audit_resources.py",
            ("ablation_policy",),
            ("00_audit/resource_gates/<gate_id>.json",),
            args="--run-dir {RUN_DIR} --scope ablation --owner {OWNER}",
            gate_scope="ablation",
        ),
        n(
            "ablation_authority",
            "P10",
            "Authorize one immutable serial P10 ablation execution.",
            "authorize_ablation_execution.py",
            ("ablation_gate",),
            ("configs/ablation_executions/<execution_id>/execution.json",),
            args="--run-dir {RUN_DIR} --gate-manifest {ABLATION_GATE} --owner {OWNER}",
            gate_scope="ablation",
        ),
        n(
            "ablation_matrix",
            "P10",
            "Execute or fresh-gated resume the serial P10 ablation matrix.",
            "run_ablation_matrix.py",
            ("ablation_authority",),
            ("configs/d1_ablation_execution.json",),
            args="--run-dir {RUN_DIR} --python {PYTHON}",
            gate_scope="ablation",
        ),
        n(
            "ablation_selection",
            "P10",
            "Replay the P10 matrix and publish evidence-track/feature-family tables.",
            "select_ablation_with_schema_adapter.py",
            ("ablation_matrix",),
            (
                "12_feature_ablation/feature_ablation.csv",
                "07_validation/tables/evidence_track_ablation.csv",
            ),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "p12_test_inputs",
            "P12-P13",
            "Assemble exact label-free four-route router, Top20, and Test T4 inputs.",
            "assemble_four_route_test_inputs.py",
            ("p12_validation", "top5_test_gate", "t3_features"),
            (
                "13_four_route_extension/test_inputs/manifest.json",
                "03_features/test/top5/T4_four_route_consensus/manifest.json",
            ),
            args=(
                "--run-dir {RUN_DIR} --three-route-run {THREE_ROUTE_RUN} --resume"
            ),
        ),
        n(
            "p12_test",
            "P12-P13",
            "Apply the locked four-route router/Top20 union to Test label-free.",
            "apply_locked_four_route_test.py",
            ("p12_test_inputs",),
            (
                "08_lock/formal_inputs/four_route_crog_default_router/manifest.json",
                "08_lock/formal_inputs/top20_union/manifest.json",
            ),
            args=(
                "--run-dir {RUN_DIR} "
                "--router-test-input {RUN_DIR}/13_four_route_extension/test_inputs/router/test_label_free.parquet "
                "--union-test-features {RUN_DIR}/13_four_route_extension/test_inputs/union/test_top20.parquet "
                "--resume"
            ),
        ),
        n(
            "reference_inputs",
            "P13-P14",
            "Normalize frozen three-route router and Top15 reference outputs.",
            "assemble_reference_systems.py",
            ("source_closure",),
            (
                "08_lock/formal_inputs/three_route_crog_default_router_reference/manifest.json",
                "08_lock/formal_inputs/top15_union_reference/manifest.json",
            ),
            args=(
                "--run-dir {RUN_DIR} --three-route-decisions {THREE_ROUTE_DECISIONS} "
                "--top15-decisions {TOP15_DECISIONS} --top15-ranking {TOP15_RANKING} --resume"
            ),
        ),
        n(
            "ranker_contributions",
            "P13",
            "Replay the locked three-seed R5 ensemble into label-free native LightGBM contributions.",
            "build_ranker_contributions.py",
            ("top5_test_ranker",),
            (
                "08_lock/postformal_sources/r5_candidate_contributions_manifest.json",
                "08_lock/postformal_sources/r5_candidate_contributions.parquet",
            ),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "postformal_sources",
            "P13",
            "Build fixed label-free covariates and runtime manifests for later P15 use.",
            "prepare_postformal_sources.py",
            ("top5_test_gate", "t3_features", "ranker_contributions"),
            (
                "configs/d1_postformal_sources.json",
                "08_lock/postformal_sources/sample_covariates.parquet",
                "08_lock/postformal_sources/runtime/{component}.json",
                "08_lock/postformal_sources/ranker_contributions.json",
            ),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "lightweight_audits",
            "P13",
            "Replay evaluator and source-bound candidate/preprocessing/geometry audits.",
            "assemble_lightweight_audits.py",
            ("source_closure", "candidates"),
            (
                "00_audit/EVALUATOR_REPLAY.md",
                "00_audit/EVALUATOR_HASH.json",
                "00_audit/CANDIDATE_GENERATION_REPLAY.md",
                "00_audit/GQCNN_PREPROCESSING_AUDIT.md",
                "00_audit/CANDIDATE_GEOMETRY_VISUAL_CHECK.pdf",
                "00_audit/D1_LIGHTWEIGHT_AUDITS.json",
            ),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "validation_evidence_tables",
            "P13",
            "Replay and assemble the nine fixed Validation-only evidence tables.",
            "assemble_validation_evidence_tables.py",
            (
                "r0_r1",
                "primary_selection",
                "gate_selection",
                "k_selection",
                "ablation_selection",
                "p12_validation",
            ),
            (
                "configs/d1_validation_evidence_tables.json",
                "07_validation/tables/r0_r7_full.csv",
            ),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "postformal_evidence",
            "P13",
            "Predeclare exact Validation tables and label-free postformal inputs.",
            "write_postformal_evidence.py",
            ("validation_evidence_tables", "postformal_sources"),
            ("configs/d1_postformal_evidence.json",),
            args=(
                "--run-dir {RUN_DIR} --q-saturation-threshold {Q_THRESHOLD} "
                "--mask-quality-threshold {MASK_THRESHOLD} --resume"
            ),
        ),
        n(
            "statistics_config",
            "P13",
            "Freeze formal statistical tests and the bootstrap seed.",
            "write_statistics_config.py",
            ("source_closure",),
            ("configs/d1_statistics_config.json",),
            args="--run-dir {RUN_DIR} --bootstrap-seed {BOOTSTRAP_SEED} --resume",
        ),
        n(
            "prelock",
            "P13",
            "Replay all required semantics and publish label-free readiness.",
            "assemble_prelock.py",
            (
                "k_formal_inputs",
                "ablation_selection",
                "p12_test",
                "reference_inputs",
                "postformal_evidence",
                "lightweight_audits",
                "statistics_config",
            ),
            ("08_lock/PRELOCK_READINESS.json", "08_lock/PRIMARY_METHOD_DECLARATION.md"),
            args="--run-dir {RUN_DIR} --resume",
            lifecycle_after="PRELOCK_LABEL_FREE",
        ),
        n(
            "formal_plan",
            "P14",
            "Assemble the fixed nine-system formal evaluation plan.",
            "assemble_formal_evaluation_plan.py",
            ("prelock",),
            ("08_lock/FORMAL_EVALUATION_PLAN.json",),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "formal_lock",
            "P14",
            "Create the immutable formal Test lock.",
            "create_formal_lock.py",
            ("formal_plan",),
            ("08_lock/FORMAL_TEST_LOCK.json", "08_lock/FORMAL_TEST_LOCK.sha256"),
            args="--run-dir {RUN_DIR} --evaluation-plan {FORMAL_EVALUATION_PLAN}",
            lifecycle_after="FORMAL_LOCKED",
        ),
        n(
            "formal_execute",
            "P14",
            "Execute the exactly-once locked formal Test transaction.",
            "run_formal_test_once.py",
            ("formal_lock",),
            (
                "09_formal_test/FORMAL_TEST_EXECUTION.json",
                "09_formal_test/formal_test_manifest.json",
            ),
            lifecycle_after="FORMAL_EXECUTED",
        ),
        n(
            "independent_recompute",
            "P17",
            "Independently recompute the formal metrics from locked bytes.",
            "independent_recompute.py",
            ("formal_execute",),
            ("17_independent_recompute/INDEPENDENT_RECOMPUTE.json",),
            args="--run-dir {RUN_DIR} --resume",
        ),
        n(
            "postformal",
            "P15",
            "Build postformal tables, figures, failures, cases, and reports.",
            "build_postformal.py",
            ("independent_recompute", "postformal_evidence"),
            ("16_reports/D1_POSTFORMAL_MANIFEST.json",),
            lifecycle_after="POSTFORMAL",
        ),
        n(
            "finalize",
            "P16-P17",
            "Run fail-closed final inventory checks and publish COMPLETE.",
            "finalize_d1_run.py",
            ("postformal",),
            ("FINAL_RUN_LOCK.json", "COMPLETE"),
            lifecycle_after="COMPLETE",
        ),
    )


PROMPT_ARTIFACT_GAPS: tuple[dict[str, str], ...] = ()


def validate_catalog(
    nodes: Iterable[CommandNode] | None = None,
) -> tuple[CommandNode, ...]:
    """Validate identifiers, edges, commands, gate declarations, and acyclicity."""

    catalog = tuple(command_nodes() if nodes is None else nodes)
    by_id = {node.node_id: node for node in catalog}
    if len(by_id) != len(catalog):
        raise ValueError("D1 command DAG contains duplicate node identifiers")
    for node in catalog:
        missing = sorted(set(node.dependencies).difference(by_id))
        if missing:
            raise ValueError(
                f"D1 command DAG {node.node_id} has missing dependencies: {missing}"
            )
        if node.gate_scope and node.command and "audit_resources.py" in node.command:
            if f"--scope {node.gate_scope}" not in node.command:
                raise ValueError(
                    f"D1 gate node {node.node_id} scope differs from its command"
                )
    topological_order(catalog)
    return catalog


def topological_order(
    nodes: Iterable[CommandNode] | None = None,
) -> tuple[CommandNode, ...]:
    """Return a stable topological order and reject cycles."""

    catalog = tuple(command_nodes() if nodes is None else nodes)
    by_id = {node.node_id: node for node in catalog}
    indegree = {node.node_id: len(node.dependencies) for node in catalog}
    children: dict[str, list[str]] = defaultdict(list)
    for node in catalog:
        for dependency in node.dependencies:
            if dependency not in by_id:
                raise ValueError(f"D1 command DAG misses dependency {dependency}")
            children[dependency].append(node.node_id)
    ready = [node.node_id for node in catalog if indegree[node.node_id] == 0]
    ordered: list[CommandNode] = []
    while ready:
        current = ready.pop(0)
        ordered.append(by_id[current])
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(ordered) != len(catalog):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree)
        raise ValueError(f"D1 command DAG contains a dependency cycle: {cyclic}")
    return tuple(ordered)


def audit_catalog(repo_root: str | Path) -> dict[str, Any]:
    """Compare declared runnable nodes with the currently present local CLIs."""

    root = Path(repo_root).expanduser().resolve()
    nodes = validate_catalog()
    missing_tools: list[dict[str, str]] = []
    for node in nodes:
        if node.command is None:
            continue
        token = node.command.split("tools/d1_reranking/", 1)[-1].split(" ", 1)[0]
        tool = root / "tools" / "d1_reranking" / token
        if not tool.is_file():
            missing_tools.append({"node_id": node.node_id, "tool": str(tool)})
    return {
        "schema_version": 1,
        "status": "BLOCKED"
        if any(node.blocked_reason for node in nodes) or missing_tools
        else "READY",
        "node_count": len(nodes),
        "executable_node_count": sum(node.executable for node in nodes),
        "blocked_node_count": sum(node.blocked_reason is not None for node in nodes),
        "heavy_gate_scopes": sorted(
            {node.gate_scope for node in nodes if node.gate_scope}
        ),
        "missing_tools": missing_tools,
        "prompt_artifact_gaps": list(PROMPT_ARTIFACT_GAPS),
        "nodes": [asdict(node) | {"executable": node.executable} for node in nodes],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    """Render an operator-facing, dependency-ordered runbook."""

    lines = [
        "# D1 reranking command DAG",
        "",
        "> Generated from the local CLI catalog. This document does not execute a gate, open Test labels, or mutate a run.",
        "",
        f"Catalog status: **{audit['status']}**. Nodes: {audit['node_count']}; "
        f"runnable: {audit['executable_node_count']}; blocked: {audit['blocked_node_count']}.",
        "",
        "## Required operator variables",
        "",
        "`RUN_DIR`, `PYTHON`, `UNIFIED_RUN`, `SNAPSHOT_A`, `EVALUATOR`, `OWNER`, "
        "resource-gate paths, completed three-route inputs, P12 named sources, and fixed statistical thresholds.",
        "",
        "## Dependency-ordered commands",
        "",
    ]
    for index, raw in enumerate(audit["nodes"], start=1):
        dependencies = ", ".join(raw["dependencies"]) or "none"
        state = "BLOCKED" if raw["blocked_reason"] else "RUNNABLE"
        lines.extend(
            [
                f"### {index}. {raw['node_id']} ({raw['stage']}; {state})",
                "",
                raw["description"],
                "",
                f"Depends on: `{dependencies}`.",
            ]
        )
        if raw["gate_scope"]:
            lines.append(
                f"Fresh-gate scope: `{raw['gate_scope']}`; serial CPU execution is required for heavy work."
            )
        if raw["fanout"]:
            lines.append(f"Fan-out: {raw['fanout']}.")
        if raw["lifecycle_after"]:
            lines.append(f"Lifecycle after success: `{raw['lifecycle_after']}`.")
        if raw["command"]:
            lines.extend(["", "```text", raw["command"], "```"])
        if raw["outputs"]:
            lines.append(
                "Outputs: " + ", ".join(f"`{item}`" for item in raw["outputs"]) + "."
            )
        if raw["blocked_reason"]:
            lines.append(f"Blocker: {raw['blocked_reason']}.")
        lines.append("")
    lines.extend(["## Missing exact prompt artifacts", ""])
    for gap in audit["prompt_artifact_gaps"]:
        lines.append(f"- **{gap['stage']} — {gap['artifact']}**: {gap['reason']}.")
    if audit["missing_tools"]:
        lines.extend(["", "## Missing referenced CLIs", ""])
        for missing in audit["missing_tools"]:
            lines.append(f"- `{missing['node_id']}`: `{missing['tool']}`")
    lines.extend(
        [
            "",
            "## Safety boundary",
            "",
            "Do not execute downstream nodes through a blocked dependency. Heavy work requires the exact active policy, a fresh 3 x 5 minute gate for its scope, the repository-wide lease, serial CPU children, and a live resource recheck before each job or resume. Formal Test remains exactly once and follows PRELOCK_LABEL_FREE.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "CommandNode",
    "PROMPT_ARTIFACT_GAPS",
    "audit_catalog",
    "command_nodes",
    "render_markdown",
    "topological_order",
    "validate_catalog",
]
