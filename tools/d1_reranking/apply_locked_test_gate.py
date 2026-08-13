"""Apply the Validation-locked D1 expected-gain gate to label-free Test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.contracts import assert_label_free_parquet_schema  # noqa: E402
from d1_reranking.gate_inputs import (  # noqa: E402
    build_label_free_test_gate_input_frame,
)
from d1_reranking.gate_validation import (  # noqa: E402
    validate_gate_selection_semantics,
)
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.gate import (  # noqa: E402
    ConservativeTransitionModel,
    GateEvidence,
    GateOperatingPoint,
    SAFE_GATE_FEATURE_COLUMNS,
    gate_switch_mask,
)
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.test_access_guard import append_access_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _normalize_no_output_seed_votes(inputs: pd.DataFrame) -> pd.DataFrame:
    """Encode the absence of a seed vote only for exact no-output rows."""

    result = inputs.copy()
    counts = pd.to_numeric(result["candidate_count"], errors="raise").astype(int)
    votes = pd.to_numeric(
        result["seed_challenger_votes"], errors="raise"
    ).astype(float)
    bearing = counts.gt(0)
    if votes.loc[bearing].isna().any() or not np.isfinite(
        votes.loc[bearing].to_numpy(float)
    ).all():
        raise RuntimeError("D1 candidate-bearing Test row lacks finite seed votes")
    unexpected_no_output = votes.loc[~bearing].notna() & votes.loc[~bearing].ne(0.0)
    if unexpected_no_output.any():
        raise RuntimeError("D1 no-output Test row declares nonzero seed votes")
    normalized = votes.fillna(0.0)
    if (
        not np.isfinite(normalized.to_numpy(float)).all()
        or normalized.lt(0.0).any()
        or normalized.gt(3.0).any()
        or not normalized.eq(np.floor(normalized)).all()
    ):
        raise RuntimeError("D1 Test seed votes are outside the integer range 0..3")
    result["seed_challenger_votes"] = normalized.astype(int)
    return result


def _evidence(inputs: pd.DataFrame) -> GateEvidence:
    return GateEvidence(
        score_margin=inputs["score_margin"].to_numpy(float),
        challenger_reliability=inputs["challenger_reliability"].to_numpy(float),
        perturbation_stability=inputs["perturbation_stability"].to_numpy(float),
        seed_challenger_votes=inputs["seed_challenger_votes"].to_numpy(float),
        candidate_id_unchanged=inputs["candidate_id_unchanged"].to_numpy(bool),
        geometry_hash_unchanged=inputs["geometry_hash_unchanged"].to_numpy(bool),
        challenger_exists=inputs["challenger_exists"].to_numpy(bool),
    )


def _grid_points(grid: dict[str, object]) -> tuple[dict[str, object], ...]:
    if grid.get("minimum_seed_votes") != 2:
        raise RuntimeError("D1 gate grid seed consensus differs")
    points = tuple(
        asdict(GateOperatingPoint(lam, utility, margin, reliability, stability))
        for lam in grid["lambda_harm"]  # type: ignore[union-attr]
        for utility in grid["utility_thresholds"]  # type: ignore[union-attr]
        for margin in grid["score_margin_thresholds"]  # type: ignore[union-attr]
        for reliability in grid["reliability_thresholds"]  # type: ignore[union-attr]
        for stability in grid["stability_thresholds"]  # type: ignore[union-attr]
    )
    if len(points) != 108 or len({canonical_sha256(point) for point in points}) != 108:
        raise RuntimeError("D1 gate grid must contain 108 unique operating points")
    return points


def _validate_locked_selection(
    gate: dict[str, object], grid: dict[str, object]
) -> GateOperatingPoint | None:
    selection = gate.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeError("D1 gate selection payload is absent")
    if selection.get("status") != gate.get("decision"):
        raise RuntimeError("D1 gate top-level/nested decisions differ")
    trials = selection.get("trials")
    if not isinstance(trials, list) or len(trials) != 108:
        raise RuntimeError("D1 gate selection lacks the exact trial inventory")
    grid_points = _grid_points(grid)
    grid_keys = {canonical_sha256(point) for point in grid_points}
    observed: list[tuple[dict[str, object], dict[str, object]]] = []
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict) or not isinstance(
            trial.get("operating_point"), dict
        ):
            raise RuntimeError(f"D1 gate trial {index} is invalid")
        point = trial["operating_point"]
        if canonical_sha256(point) not in grid_keys:
            raise RuntimeError(f"D1 gate trial {index} is outside the locked grid")
        metrics = np.asarray(
            [
                trial.get("bootstrap_lower_bound"),
                trial.get("mean_delta"),
                trial.get("harmful"),
                trial.get("switch_rate"),
            ],
            dtype=float,
        )
        if not np.isfinite(metrics).all():
            raise RuntimeError(
                f"D1 gate trial {index} has non-finite selection metrics"
            )
        observed.append((point, trial))
    if {canonical_sha256(point) for point, _trial in observed} != grid_keys:
        raise RuntimeError("D1 gate trials do not cover the locked grid exactly")
    best_point, best_trial = max(
        observed,
        key=lambda item: (
            float(item[1]["bootstrap_lower_bound"]),
            float(item[1]["mean_delta"]),
            -int(item[1]["harmful"]),
            -float(item[1]["switch_rate"]),
        ),
    )
    selected = selection.get("selected_operating_point")
    if float(best_trial["bootstrap_lower_bound"]) <= 0.0:
        if selection.get("status") != "NO_GO_NATIVE" or selected is not None:
            raise RuntimeError("D1 gate NO_GO decision differs from locked trials")
        return None
    if selection.get("status") != "GO" or selected != best_point:
        raise RuntimeError("D1 gate selected point differs from locked trial tie-break")
    return GateOperatingPoint(**best_point)


def _apply(
    selection: dict[str, object],
    model: ConservativeTransitionModel,
    inputs: pd.DataFrame,
) -> tuple[pd.DataFrame, GateOperatingPoint | None]:
    # This is the shared inference boundary used by both the canonical Top5
    # applicator and the K-sensitivity formal-input producer.  Keep the exact
    # no-output encoding here so callers cannot accidentally bypass it.
    inputs = _normalize_no_output_seed_votes(inputs)
    configured = tuple(selection.get("configuration", {}).get("feature_columns", ()))  # type: ignore[union-attr]
    if configured != tuple(SAFE_GATE_FEATURE_COLUMNS):
        raise RuntimeError("D1 gate selection/Test feature schemas differ")
    if (
        tuple(model.feature_names_ or ())
        != tuple(selection.get("transition_model", {}).get("feature_names", ()))  # type: ignore[union-attr]
        or tuple(model.feature_names_ or ()) != configured
    ):
        raise RuntimeError("D1 gate model/Test feature order differs")
    matrix = inputs.loc[:, SAFE_GATE_FEATURE_COLUMNS].to_numpy(float)
    probability_recover, probability_harm = model.predict_probabilities(matrix)
    point_payload = selection.get("selection", {}).get("selected_operating_point")  # type: ignore[union-attr]
    if point_payload is None:
        if selection.get("decision") != "NO_GO_NATIVE":
            raise RuntimeError("D1 gate has no point but is not NO_GO_NATIVE")
        point = None
        switches = np.zeros(len(inputs), dtype=bool)
        utility = np.full(len(inputs), np.nan)
    else:
        if selection.get("decision") != "GO" or not isinstance(point_payload, dict):
            raise RuntimeError("D1 gate operating-point decision is inconsistent")
        point = GateOperatingPoint(**point_payload)
        switches = gate_switch_mask(
            probability_recover, probability_harm, _evidence(inputs), point
        )
        utility = probability_recover - point.lambda_harm * probability_harm
    native_ids = inputs["native_candidate_id"].fillna("").astype(str).to_numpy()
    challenger_ids = inputs["challenger_candidate_id"].fillna("").astype(str).to_numpy()
    native_geometry = inputs["native_geometry_sha256"].fillna("").astype(str).to_numpy()
    challenger_geometry = (
        inputs["challenger_geometry_sha256"].fillna("").astype(str).to_numpy()
    )
    native_identity = inputs["native_identity_sha256"].fillna("").astype(str).to_numpy()
    challenger_identity = (
        inputs["challenger_identity_sha256"].fillna("").astype(str).to_numpy()
    )
    decisions = pd.DataFrame(
        {
            "sample_id": inputs["sample_id"].astype(str),
            "prediction_source": "test_label_free",
            "probability_recover": probability_recover,
            "probability_harm": probability_harm,
            "utility": utility,
            "switch": switches,
            "candidate_count": inputs["candidate_count"].astype(int),
            "native_candidate_id": native_ids,
            "challenger_candidate_id": challenger_ids,
            "selected_candidate_id": np.where(switches, challenger_ids, native_ids),
            "native_identity_sha256": native_identity,
            "challenger_identity_sha256": challenger_identity,
            "selected_identity_sha256": np.where(
                switches, challenger_identity, native_identity
            ),
            "native_geometry_sha256": native_geometry,
            "challenger_geometry_sha256": challenger_geometry,
            "selected_geometry_sha256": np.where(
                switches, challenger_geometry, native_geometry
            ),
        }
    )
    no_output = decisions["candidate_count"].eq(0)
    identity_columns = [
        "native_candidate_id",
        "challenger_candidate_id",
        "selected_candidate_id",
        "native_geometry_sha256",
        "challenger_geometry_sha256",
        "selected_geometry_sha256",
        "native_identity_sha256",
        "challenger_identity_sha256",
        "selected_identity_sha256",
    ]
    if not decisions.loc[no_output, identity_columns].eq("").all().all():
        raise RuntimeError("D1 Test gate does not preserve no-output samples")
    if decisions.loc[no_output, "switch"].any():
        raise RuntimeError("D1 Test gate switches a no-output sample")
    return decisions, point


def run(run_dir: Path, *, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    gate_path = root / "07_validation" / "gate" / "d1" / "gate_selection.json"
    gate = load_content_manifest(
        gate_path, name="D1 gate selection", statuses=("COMPLETE",)
    )
    validated_gate = validate_gate_selection_semantics(gate)

    # No Test manifest, schema or table may be opened before the complete
    # development-only semantic replay above succeeds.
    ranker_path = root / "08_lock" / "label_free_test_rankers" / "d1" / "manifest.json"
    feature_path = (
        root / "03_features" / "test" / "top5" / "T2_matched_common" / "manifest.json"
    )
    candidate_path = root / "02_candidates" / "test_manifest.json"
    ranker = load_content_manifest(
        ranker_path, name="D1 label-free Test ranker", statuses=("COMPLETE",)
    )
    features = load_content_manifest(
        feature_path, name="D1 label-free Test T2", statuses=("COMPLETE",)
    )
    candidates = load_content_manifest(
        candidate_path, name="D1 label-free Test candidates", statuses=("COMPLETE",)
    )
    gate_input_record = gate.get("sources", {}).get("input_manifest", {})  # type: ignore[union-attr]
    gate_input_path = verified_artifact_path(
        gate_input_record, name="D1 development gate input manifest"
    )
    gate_inputs = load_content_manifest(
        gate_input_path, name="D1 development gate inputs", statuses=("COMPLETE",)
    )
    ranker_selection = ranker.get("sources", {}).get("selection")  # type: ignore[union-attr]
    gate_selection = gate_inputs.get("sources", {}).get("primary_selection")  # type: ignore[union-attr]
    if gate_selection != ranker_selection:
        raise RuntimeError("D1 Test gate and ranker use different primary selections")
    selection_path = verified_artifact_path(
        ranker_selection if isinstance(ranker_selection, dict) else {},
        name="D1 current primary selection",
    )
    primary_selection = load_content_manifest(
        selection_path, name="D1 current primary selection", statuses=("COMPLETE",)
    )
    if ranker.get("selected_method") != primary_selection.get(
        "selected_method"
    ) or ranker.get("selected_trial_id") != primary_selection.get("selected_trial_id"):
        raise RuntimeError("D1 Test ranker differs from its primary selection")
    for name, value in (
        ("gate", gate),
        ("ranker", ranker),
        ("features", features),
        ("candidates", candidates),
        ("development gate inputs", gate_inputs),
        ("primary selection", primary_selection),
    ):
        if value.get("candidate_test_labels_read") is not False:
            raise RuntimeError(f"D1 {name} violates prelock Test-label isolation")
    verify_artifact_records_recursive(
        {
            "gate": gate,
            "ranker": ranker,
            "features": features,
            "candidates": candidates,
        },
        name="D1 Test gate sources",
        require_at_least_one=True,
    )
    model_path = validated_gate.transition_model_path
    train_oof_path = validated_gate.train_oof_path
    decision_path = verified_artifact_path(
        ranker.get("artifacts", {}).get("per_sample_decisions", {}),  # type: ignore[union-attr]
        name="D1 Test ranker decisions",
    )
    feature_table_path = verified_artifact_path(
        features.get("artifacts", {}).get("candidate_features", {}),  # type: ignore[union-attr]
        name="D1 Test T2 candidate features",
    )
    candidate_table_path = verified_artifact_path(
        candidates.get("artifacts", {}).get("top5", {}),  # type: ignore[union-attr]
        name="D1 Test Top5 candidates",
    )
    paired_path = root / "01_manifests" / "d1_paired_manifest.parquet"
    expected_ranker_sources = {
        "final_feature_manifest": artifact_record(feature_path),
        "candidate_manifest": artifact_record(candidate_path),
        "candidates": artifact_record(candidate_table_path),
        "denominator": artifact_record(paired_path),
    }
    for source_name, expected in expected_ranker_sources.items():
        if ranker.get("sources", {}).get(source_name) != expected:  # type: ignore[union-attr]
            raise RuntimeError(f"D1 Test ranker/gate {source_name} sources differ")
    grid_path = validated_gate.grid_path
    locked_point = validated_gate.selected_operating_point
    declared_point = gate.get("selection", {}).get("selected_operating_point")  # type: ignore[union-attr]
    if (None if locked_point is None else asdict(locked_point)) != declared_point:
        raise RuntimeError("D1 gate selected point semantic replay differs")
    sources = {
        "gate_selection": artifact_record(gate_path),
        "development_gate_inputs": artifact_record(gate_input_path),
        "primary_selection": artifact_record(selection_path),
        "transition_model": artifact_record(model_path),
        "gate_train_oof": artifact_record(train_oof_path),
        "ranker_application": artifact_record(ranker_path),
        "ranker_decisions": artifact_record(decision_path),
        "feature_manifest": artifact_record(feature_path),
        "candidate_features": artifact_record(feature_table_path),
        "candidate_manifest": artifact_record(candidate_path),
        "candidates": artifact_record(candidate_table_path),
        "denominator": artifact_record(paired_path),
        "grid": artifact_record(grid_path),
        "inference_code": [
            artifact_record(path)
            for path in (
                ROOT / "src/d1_reranking/contracts.py",
                ROOT / "src/d1_reranking/gate_inputs.py",
                ROOT / "src/d1_reranking/gate_validation.py",
                ROOT / "src/unified_reranking/gate.py",
                Path(__file__),
            )
        ],
    }
    signature = canonical_sha256(sources)
    output = root / "08_lock" / "label_free_test_gates" / "d1"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name="D1 Test gate application", statuses=("COMPLETE",)
        )
        if (
            resume
            and existing.get("sources") == sources
            and existing.get("source_signature_sha256") == canonical_sha256(sources)
            and existing.get("source_signature_sha256") == signature
        ):
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                },
                name="D1 Test gate resume",
                require_at_least_one=True,
            )
            append_access_log(
                root,
                {
                    "event": "prelock_label_free_test_stage",
                    "stage": "d1_gate_test_application",
                    "route": "D1",
                    "output_manifest": str(manifest_path.resolve()),
                    "output_manifest_sha256": sha256_file(manifest_path),
                    "candidate_labels_opened_as_table": False,
                    "resumed": True,
                },
            )
            return existing
        raise RuntimeError("D1 Test gate application differs or is corrupt")

    for path, name in (
        (paired_path, "Test paired denominator"),
        (decision_path, "Test ranker decisions"),
        (feature_table_path, "Test T2 features"),
        (candidate_table_path, "Test Top5 candidates"),
    ):
        assert_label_free_parquet_schema(path, name=name)
    decision_columns = [
        "sample_id",
        "native_candidate_id",
        "native_identity_sha256",
        "native_geometry_sha256",
        "selected_candidate_id",
        "selected_identity_sha256",
        "selected_geometry_sha256",
        "ensemble_score_margin",
        "seed_challenger_votes",
        "candidate_count",
        "challenger_exists",
    ]
    feature_columns = [
        "sample_id",
        "candidate_id",
        "calibrated_native_probability",
        "native_score_raw",
        "overall_feature_reliability",
        "peak_retention_rate",
        "perturbed_valid_fraction",
        "mask_reliability",
    ]
    candidate_columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
    ]
    inputs = build_label_free_test_gate_input_frame(
        paired=pd.read_parquet(paired_path, columns=["sample_id", "scene_id"]),
        ranker_decisions=pd.read_parquet(decision_path, columns=decision_columns),
        candidate_features=pd.read_parquet(feature_table_path, columns=feature_columns),
        candidates=pd.read_parquet(candidate_table_path, columns=candidate_columns),
    )
    inputs = _normalize_no_output_seed_votes(inputs)
    model = validated_gate.model
    decisions, point = _apply(gate, model, inputs)
    artifacts = {
        "inputs": artifact_record(
            atomic_parquet(inputs, output / "gate_test_inputs.parquet")
        ),
        "decisions": artifact_record(
            atomic_parquet(decisions, output / "gate_test_decisions.parquet")
        ),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "route": "D1",
        "decision": gate["decision"],
        "source_signature_sha256": signature,
        "feature_columns": list(SAFE_GATE_FEATURE_COLUMNS),
        "selected_operating_point": None if point is None else asdict(point),
        "sample_count": len(inputs),
        "switch_count": int(decisions["switch"].sum()),
        "prediction_source": "test_label_free",
        "candidate_test_labels_read": False,
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "d1_gate_test_application",
            "route": "D1",
            "output_manifest": str(manifest_path.resolve()),
            "output_manifest_sha256": sha256_file(manifest_path),
            "candidate_labels_opened_as_table": False,
            "resumed": False,
        },
    )
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    path = root / "08_lock" / "label_free_test_gates" / "d1" / "manifest.json"
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_label_free_test_gate",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R7",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, resume=args.resume)
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
