from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from failure_analysis.vlm_safe_rerank.calibration import (
    cross_fit_dual_risk,
    group_folds,
    threshold_sweep,
)
from failure_analysis.vlm_safe_rerank.critic_features import critic_feature_vector, p3_hard_rule
from failure_analysis.vlm_safe_rerank.features import (
    build_pair_evidence,
    local_feature_vector,
    ordered_candidates,
    select_challengers,
)
from failure_analysis.vlm_safe_rerank.ledger import (
    PairwiseLedger,
    legacy_pairwise_request_hash,
    pairwise_request_hash,
)
from failure_analysis.vlm_safe_rerank.local_experiment import _bootstrap_delta
from failure_analysis.vlm_safe_rerank.policy import (
    GateThresholds,
    PairGateEvidence,
    PairLabel,
    full_denominator_metrics,
    pair_label,
    protected_select,
)
from failure_analysis.vlm_safe_rerank.renderer import PerturbationVariant, render_pairwise_board
from failure_analysis.vlm_safe_rerank.schema import PairwiseDecision, parse_pairwise_response
from failure_analysis.vlm_safe_rerank.security import assert_no_ground_truth


REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "failure_analysis/reranking_outputs/v2_20260727T174412+0100/base_train/features.jsonl"


@pytest.fixture(scope="module")
def frozen_feature() -> dict:
    with FEATURES.open(encoding="utf-8") as handle:
        return json.loads(next(handle))


def _critic(decision: PairwiseDecision, **changes: object) -> PairGateEvidence:
    payload = dict(
        challenger_id="candidate_1", p_benefit=0.9, p_harm=0.01, reliability=0.9,
        challenger_hard_valid=True, critic_decision=decision,
        confirmation_decision=PairwiseDecision.PREFER_CHALLENGER, confirmation_valid=True,
    )
    payload.update(changes)
    return PairGateEvidence(**payload)


def _valid_json() -> str:
    return json.dumps({
        "decision": "KEEP_BASELINE", "evidence_reliable": True,
        "baseline_target_alignment": 0.8, "challenger_target_alignment": 0.7,
        "baseline_contact_geometry": 0.8, "challenger_contact_geometry": 0.7,
        "baseline_collision_risk": 0.1, "challenger_collision_risk": 0.2,
        "baseline_width_compatibility": 0.8, "challenger_width_compatibility": 0.7,
        "reason_codes": ["NO_CLEAR_ADVANTAGE"], "brief_rationale": "No reliable advantage."
    }, separators=(",", ":"))


def test_default_is_q_only() -> None:
    result = protected_select("candidate_0", [], GateThresholds(0.8, 0.1), query_gate_passed=True)
    assert result.selected_id == "candidate_0" and not result.switched


def test_clustered_bootstrap_short_circuits_exact_q_only_delta() -> None:
    rows = [
        {
            "group_id": f"g{index % 2}",
            "baseline_correct": bool(index % 2),
            "selected_correct": bool(index % 2),
        }
        for index in range(8)
    ]
    assert _bootstrap_delta(rows, draws=10000, seed=20260803) == {
        "lower": 0.0,
        "median": 0.0,
        "upper": 0.0,
        "draws": 10000,
    }


@pytest.mark.parametrize("reason", ["network", "schema"])
def test_technical_failure_keeps_q_only(reason: str) -> None:
    result = protected_select(
        "candidate_0", [_critic(PairwiseDecision.PREFER_CHALLENGER, terminal_failure=True)],
        GateThresholds(0.8, 0.1), query_gate_passed=True,
    )
    rows = [{"baseline": True, "selected": True, "switched": result.switched, "failure": reason}]
    assert result.selected_id == "candidate_0"
    assert full_denominator_metrics(rows, baseline_key="baseline", selected_key="selected")["total"] == 1


@pytest.mark.parametrize("decision", [PairwiseDecision.KEEP_BASELINE, PairwiseDecision.INSUFFICIENT_EVIDENCE])
def test_abstain_or_insufficient_keeps_q_only(decision: PairwiseDecision) -> None:
    result = protected_select("candidate_0", [_critic(decision)], GateThresholds(0.8, 0.1), query_gate_passed=True)
    assert result.selected_id == "candidate_0"


def test_strict_schema_rejects_fences_extra_and_long_rationale() -> None:
    assert parse_pairwise_response(_valid_json()).decision is PairwiseDecision.KEEP_BASELINE
    with pytest.raises(ValueError):
        parse_pairwise_response("```json\n" + _valid_json() + "\n```")
    row = json.loads(_valid_json()); row["unknown"] = 1
    with pytest.raises(ValueError):
        parse_pairwise_response(json.dumps(row))
    row.pop("unknown"); row["brief_rationale"] = "word " * 41
    with pytest.raises(ValueError):
        parse_pairwise_response(json.dumps(row))


def test_candidate_ids_and_q_values_are_frozen(frozen_feature: dict) -> None:
    before = [(row["candidate_id"], row["candidate_checksum"], row["q_raw"]) for row in frozen_feature["candidates"]]
    ordered = ordered_candidates(frozen_feature)
    pair = build_pair_evidence(frozen_feature, str(ordered[1]["candidate_id"]))
    after = [(row["candidate_id"], row["candidate_checksum"], row["q_raw"]) for row in frozen_feature["candidates"]]
    assert before == after
    assert pair["baseline"]["q"] == ordered[0]["q_raw"]
    assert pair["challenger"]["q"] == ordered[1]["q_raw"]


def test_pair_features_and_preselector_do_not_need_ground_truth(frozen_feature: dict) -> None:
    challenger_ids = select_challengers(frozen_feature, maximum=2)
    assert len(challenger_ids) == 2 and "candidate_0" not in challenger_ids
    pair = build_pair_evidence(frozen_feature, challenger_ids[0])
    names, values = local_feature_vector(pair)
    assert len(names) == len(values) and np.isfinite(values).all()
    assert_no_ground_truth({"evidence": pair, "instruction": frozen_feature["language_instruction"]})
    with pytest.raises(AssertionError):
        assert_no_ground_truth({"evidence": pair, "candidate_correct": True})


def test_pair_labels() -> None:
    assert pair_label(baseline_correct=False, challenger_correct=True) is PairLabel.BENEFICIAL
    assert pair_label(baseline_correct=True, challenger_correct=False) is PairLabel.HARMFUL
    assert pair_label(baseline_correct=True, challenger_correct=True) is PairLabel.NEUTRAL_BOTH_CORRECT
    assert pair_label(baseline_correct=False, challenger_correct=False) is PairLabel.NEUTRAL_BOTH_WRONG


def test_panel_swap_confirmation() -> None:
    thresholds = GateThresholds(0.8, 0.1, require_confirmation=True)
    unstable = _critic(PairwiseDecision.PREFER_CHALLENGER, confirmation_decision=PairwiseDecision.KEEP_BASELINE)
    assert not protected_select("candidate_0", [unstable], thresholds, query_gate_passed=True).switched
    assert protected_select("candidate_0", [_critic(PairwiseDecision.PREFER_CHALLENGER)], thresholds, query_gate_passed=True).switched


def test_renderer_perturbations_have_distinct_hashes(frozen_feature: dict) -> None:
    challenger = ordered_candidates(frozen_feature)[1]["candidate_id"]
    hashes = []
    png_hashes = []
    for variant in (
        PerturbationVariant.ORIGINAL, PerturbationVariant.PANEL_SWAP,
        PerturbationVariant.PANEL_SWAP_TABLE_FIXED, PerturbationVariant.DISPLAY_ID_RENAME,
    ):
        png, metadata = render_pairwise_board(frozen_feature, challenger, variant=variant)
        assert png.startswith(b"\x89PNG")
        hashes.append(metadata["renderer_contract_hash"])
        png_hashes.append(__import__("hashlib").sha256(png).hexdigest())
    assert len(set(hashes)) == 4
    assert png_hashes[1] != png_hashes[2]


def test_p3_visual_renderer_has_distinct_contract_and_no_numeric_table(frozen_feature: dict) -> None:
    challenger = ordered_candidates(frozen_feature)[1]["candidate_id"]
    p4_png, p4 = render_pairwise_board(frozen_feature, challenger, include_numeric=True)
    p3_png, p3 = render_pairwise_board(frozen_feature, challenger, include_numeric=False)
    assert p4["numeric_evidence_included"] is True
    assert p3["numeric_evidence_included"] is False
    assert p3["renderer_contract_hash"] != p4["renderer_contract_hash"]
    assert __import__("hashlib").sha256(p3_png).hexdigest() != __import__("hashlib").sha256(p4_png).hexdigest()


def test_request_cache_idempotent_and_prompt_hash_changes_key(tmp_path: Path) -> None:
    kwargs = dict(
        sample_id="s", baseline_candidate_id="c0", challenger_candidate_id="c1", model_id="m",
        protocol="P4", prompt_hash="p1", schema_hash="s1", renderer_hash="r1",
        board_sha256="b1", evidence_hash="e1", generation={"temperature": 0},
        perturbation_variant="original",
    )
    digest = pairwise_request_hash(**kwargs)
    assert digest == pairwise_request_hash(**kwargs)
    assert digest != pairwise_request_hash(**{**kwargs, "prompt_hash": "p2"})
    assert digest != pairwise_request_hash(**{**kwargs, "board_sha256": "b2"})
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        assert ledger.acquire(digest, "worker") == "ACQUIRED"
        ledger.record_attempt(digest, "SUCCEEDED", estimated_cost_usd=0.2)
        ledger.finish(digest, "SUCCEEDED", raw_response=_valid_json(), parsed_json=json.loads(_valid_json()), estimated_cost_usd=0.2)
        assert ledger.acquire(digest, "worker-2") == "CACHED"
        assert ledger.cached(digest).status == "SUCCEEDED"
        assert ledger.summary()["attempt_estimated_cost_usd"] == pytest.approx(0.2)
        assert ledger.summary()["duplicate_successful_request_hashes"] == 0


def test_paid_v1_cache_row_is_rekeyed_without_new_attempt(tmp_path: Path) -> None:
    common = dict(
        sample_id="s", baseline_candidate_id="c0", challenger_candidate_id="c1",
        model_id="m", protocol="P4", prompt_hash="p", schema_hash="s",
        renderer_hash="r", evidence_hash="e", generation={"temperature": 0},
        perturbation_variant="original",
    )
    old = legacy_pairwise_request_hash(**common)
    new = pairwise_request_hash(**common, board_sha256="b")
    with PairwiseLedger(tmp_path / "cache.sqlite") as ledger:
        assert ledger.acquire(old, "worker") == "ACQUIRED"
        ledger.record_attempt(old, "SUCCEEDED", estimated_cost_usd=.2)
        ledger.finish(old, "SUCCEEDED", raw_response=_valid_json(), parsed_json=json.loads(_valid_json()))
        assert ledger.migrate_terminal_hash(old, new, board_sha256="b")
        assert ledger.cached(old) is None
        assert ledger.cached(new) is not None
        summary = ledger.summary()
        assert summary["attempts"] == 1
        assert summary["cache_hash_migrations"] == 1
        assert summary["duplicate_successful_request_hashes"] == 0


def test_group_oof_has_no_overlap_and_dual_models() -> None:
    rng = np.random.default_rng(4)
    groups = np.asarray([f"g{i // 6}" for i in range(60)])
    values = rng.normal(size=(60, 3))
    benefit = (values[:, 0] > 1).astype(float)
    harm = (values[:, 1] > 1).astype(float)
    bundle, p_b, p_h, folds = cross_fit_dual_risk(
        values, benefit, harm, groups, feature_names=("a", "b", "c"), n_splits=5,
    )
    for group in set(groups):
        assert len(set(folds[groups == group])) == 1
    assert np.all((p_b >= 0) & (p_b <= 1)) and np.all((p_h >= 0) & (p_h <= 1))
    assert bundle.calibration_partition == "calibration"


def test_group_oof_scores_confirmation_with_same_held_out_models() -> None:
    rng = np.random.default_rng(8)
    groups = np.asarray([f"g{i // 6}" for i in range(60)])
    values = rng.normal(size=(60, 3))
    confirmation = values + rng.normal(scale=.1, size=values.shape)
    benefit = (values[:, 0] > 1).astype(float)
    harm = (values[:, 1] > 1).astype(float)
    result = cross_fit_dual_risk(
        values,
        benefit,
        harm,
        groups,
        feature_names=("a", "b", "c"),
        n_splits=5,
        secondary_values=confirmation,
    )
    assert len(result) == 6
    _, _, _, folds, confirmation_b, confirmation_h = result
    assert confirmation_b.shape == confirmation_h.shape == (60,)
    assert np.isfinite(confirmation_b).all() and np.isfinite(confirmation_h).all()
    for group in set(groups):
        assert len(set(folds[groups == group])) == 1


def test_threshold_fitted_without_validation() -> None:
    rows = [
        {"baseline_correct": False, "challenger_correct": True, "p_benefit": .9, "p_harm": .01,
         "query_score": .9, "evidence_reliable": True, "challenger_hard_valid": True,
         "confirmation_stable": True, "hard_veto": False},
        *[{"baseline_correct": True, "challenger_correct": False, "p_benefit": .1, "p_harm": .9,
           "query_score": .1, "evidence_reliable": True, "challenger_hard_valid": True,
           "confirmation_stable": True, "hard_veto": False} for _ in range(100)],
    ]
    sweep, selected = threshold_sweep(rows, tau_grid=[.8], eta_grid=[.05])
    assert selected is not None and selected["net"] == 1 and sweep[0]["eligible"]
    assert "validation" not in selected


def test_confirmation_must_pass_the_same_external_risk_gate() -> None:
    rows = [{
        "sample_id": "s",
        "baseline_correct": False,
        "challenger_correct": True,
        "p_benefit": .9,
        "p_harm": .01,
        "confirmation_p_benefit": .1,
        "confirmation_p_harm": .8,
        "query_score": .9,
        "evidence_reliable": True,
        "challenger_hard_valid": True,
        "confirmation_stable": True,
        "hard_veto": False,
    }]
    sweep, selected = threshold_sweep(rows, tau_grid=[.8], eta_grid=[.05])
    assert selected is None and sweep[0]["switches"] == 0


def test_balanced_threshold_sweep_uses_natural_sample_weights() -> None:
    common = {
        "p_benefit": .9,
        "p_harm": .01,
        "query_score": .9,
        "evidence_reliable": True,
        "challenger_hard_valid": True,
        "confirmation_stable": True,
        "hard_veto": False,
    }
    rows = [
        {**common, "sample_id": "r1", "baseline_correct": False, "challenger_correct": True, "sample_weight": 1.0},
        {**common, "sample_id": "r2", "baseline_correct": False, "challenger_correct": True, "sample_weight": 1.0},
        {**common, "sample_id": "h", "baseline_correct": True, "challenger_correct": False, "sample_weight": 100.0},
    ]
    sweep, selected = threshold_sweep(rows, tau_grid=[.8], eta_grid=[.05])
    assert selected is None
    assert sweep[0]["recovered"] == 2.0 and sweep[0]["harmful"] == 100.0


def test_hard_rule_and_critic_features_are_conservative() -> None:
    output = json.loads(_valid_json())
    output.update({
        "decision": "PREFER_CHALLENGER", "baseline_target_alignment": .5,
        "challenger_target_alignment": .8, "baseline_contact_geometry": .5,
        "challenger_contact_geometry": .7, "baseline_collision_risk": .3,
        "challenger_collision_risk": .2, "challenger_width_compatibility": .8,
    })
    assert p3_hard_rule(output)
    assert not p3_hard_rule({**output, "evidence_reliable": False})
    names, values = critic_feature_vector({"gemini-robotics-er-2-preview": output, "gemini-3.6-flash": None})
    assert len(names) == len(values) and np.isfinite(values).all()
    assert values[names.index("flash_missing")] == 1.0
