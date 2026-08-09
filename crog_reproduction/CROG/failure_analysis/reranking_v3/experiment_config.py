from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import atomic_write_json, sha256_file


ENSEMBLE_SEEDS = (20260801, 20260802, 20260803)
PERTURBATIONS = (
    {"kind": "center_x_px", "value": -4.0},
    {"kind": "center_x_px", "value": -2.0},
    {"kind": "center_x_px", "value": 2.0},
    {"kind": "center_x_px", "value": 4.0},
    {"kind": "center_y_px", "value": -4.0},
    {"kind": "center_y_px", "value": -2.0},
    {"kind": "center_y_px", "value": 2.0},
    {"kind": "center_y_px", "value": 4.0},
    {"kind": "angle_deg", "value": -10.0},
    {"kind": "angle_deg", "value": -5.0},
    {"kind": "angle_deg", "value": 5.0},
    {"kind": "angle_deg", "value": 10.0},
    {"kind": "width_scale", "value": 0.9},
    {"kind": "width_scale", "value": 0.95},
    {"kind": "width_scale", "value": 1.05},
    {"kind": "width_scale", "value": 1.1},
)

DIAGNOSTIC_ABLATIONS = {
    "leave_one_group_out": ["G2_mask", "G3_angle_confidence", "G4_width_consistency", "G5_cross_head", "G6_multiscale_latent", "G7_token_interaction", "G8_rgb_crop", "G9_depth", "G10_set_context", "G11_v2_prior", "uncertainty"],
    "latent_layers": ["c3", "c4", "c5", "fpn_pre", "decoder_1", "decoder_2", "decoder_3_post", "all_multiscale"],
    "text": ["none", "sentence_only", "token_candidate", "token_plus_sentence_global"],
    "maps": ["native_raw_and_activated", "restored_scalar_only"],
    "gate": ["full_without_gate", "q_anchored_gate", "v2_anchored_gate", "v2_anchored_gate_with_uncertainty"],
    "sensitivity": ["feature_group_permutation", "recovered_harmful_distribution", "precision_coverage"],
}


# This grid is deliberately finite and ordered.  Its order is part of the
# experiment contract and is written before any scientific model is fitted.
SELECTION_GRID: tuple[dict[str, Any], ...] = (
    {"name": "head_g0_g4", "head_groups": [0, 1, 2, 3, 4], "use_latent": False, "use_attention": False, "text_mode": "none", "use_crop": False, "use_set": False, "use_prior": False, "use_depth": False, "hidden_dim": 128, "alpha": 0.25, "dropout": 0.1},
    {"name": "head_g0_g5", "head_groups": [0, 1, 2, 3, 4, 5], "use_latent": False, "use_attention": False, "text_mode": "none", "use_crop": False, "use_set": False, "use_prior": False, "use_depth": False, "hidden_dim": 128, "alpha": 0.25, "dropout": 0.1},
    {"name": "fpn_pre_decoder", "head_groups": [0, 1, 2, 3, 4, 5], "use_latent": True, "use_attention": False, "latent_layers": [3], "text_mode": "none", "use_crop": False, "use_set": False, "use_prior": False, "use_depth": False, "hidden_dim": 128, "alpha": 0.5, "dropout": 0.1},
    {"name": "post_decoder", "head_groups": [0, 1, 2, 3, 4, 5], "use_latent": True, "use_attention": False, "latent_layers": [6], "text_mode": "none", "use_crop": False, "use_set": False, "use_prior": False, "use_depth": False, "hidden_dim": 128, "alpha": 0.5, "dropout": 0.1},
    {"name": "all_multiscale", "head_groups": [0, 1, 2, 3, 4, 5], "use_latent": True, "use_attention": True, "latent_layers": "all", "text_mode": "none", "use_crop": False, "use_set": False, "use_prior": False, "use_depth": False, "hidden_dim": 128, "alpha": 0.5, "dropout": 0.1},
    {"name": "sentence_only", "head_groups": [0, 1, 2, 3, 4, 5], "use_latent": True, "latent_layers": "all", "text_mode": "sentence", "use_crop": False, "use_set": False, "use_prior": False, "use_depth": False, "hidden_dim": 128, "alpha": 0.5, "dropout": 0.1},
    {"name": "token_cross_attention", "head_groups": [0, 1, 2, 3, 4, 5], "use_latent": True, "latent_layers": "all", "text_mode": "token", "use_crop": False, "use_set": False, "use_prior": False, "use_depth": False, "hidden_dim": 128, "alpha": 0.5, "dropout": 0.1},
    {"name": "aligned_rgb_maps", "head_groups": [0, 1, 2, 3, 4, 5], "use_latent": True, "latent_layers": "all", "text_mode": "token", "use_crop": True, "use_set": False, "use_prior": False, "use_depth": False, "hidden_dim": 128, "alpha": 0.5, "dropout": 0.1},
    {"name": "native_setrank", "head_groups": [0, 1, 2, 3, 4, 5, 10], "use_latent": True, "latent_layers": "all", "text_mode": "token", "use_crop": True, "use_set": True, "use_prior": False, "use_depth": False, "hidden_dim": 128, "alpha": 0.5, "dropout": 0.1},
    {"name": "native_plus_v2_oof", "head_groups": [0, 1, 2, 3, 4, 5, 10], "use_latent": True, "latent_layers": "all", "text_mode": "token", "use_crop": True, "use_set": True, "use_prior": True, "use_depth": False, "hidden_dim": 128, "alpha": 0.5, "dropout": 0.1},
    {"name": "native_h256", "head_groups": [0, 1, 2, 3, 4, 5, 10], "use_latent": True, "latent_layers": "all", "text_mode": "token", "use_crop": True, "use_set": True, "use_prior": True, "use_depth": False, "hidden_dim": 256, "alpha": 0.5, "dropout": 0.1},
    {"name": "rgbd_h256", "head_groups": [0, 1, 2, 3, 4, 5, 10], "use_latent": True, "latent_layers": "all", "text_mode": "token", "use_crop": True, "use_set": True, "use_prior": True, "use_depth": True, "hidden_dim": 256, "alpha": 0.5, "dropout": 0.1},
)


def selection_contract() -> dict[str, Any]:
    return {
        "schema_version": "3.0.0",
        "kind": "v3_predeclared_selection_grid",
        "primary_evaluator": "corrected_scientific",
        "anchor": "v2_locked_primary",
        "fold_count": 3,
        "fold_group": "capture_sequence",
        "fold_seed": 1702,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "maximum_complete_configurations": 12,
        "contract_revision": "width_semantics_and_explicit_g10_remediation_v1",
        "configurations": list(SELECTION_GRID),
        "loss": {"beta_abs": 1.0, "beta_pair": 0.25, "beta_any": 0.5, "beta_res": 0.01},
        "optimizer": {"name": "AdamW", "learning_rate": 0.0002, "weight_decay": 0.0001, "epochs": 5, "early_stopping_patience": 2},
        "gate_grid": {"harm_cost": [2.0, 5.0, 10.0], "threshold": [0.0, 0.05, 0.1, 0.2], "uncertainty_kappa": [0.0, 0.5, 1.0], "consensus": [2, 3]},
        "selection_rule": ["maximum corrected delta versus V2", "maximum frame-cluster bootstrap lower bound", "maximum scene-cluster sensitivity lower bound", "fewer harmful switches", "lower switch coverage", "higher outcome-changing precision", "simpler model", "CROG-native over RGB-D"],
        "selection_bootstrap_iterations": 10000,
        "perturbations": list(PERTURBATIONS),
    }


def write_selection_contract(path: str | Path) -> dict[str, Any]:
    output = Path(path)
    atomic_write_json(output, selection_contract())
    return {"path": str(output.resolve()), "sha256": sha256_file(output)}


def diagnostic_ablation_contract() -> dict[str,Any]:
    return {"schema_version":"3.0.0","kind":"v3_predeclared_diagnostic_ablation_contract","status":"frozen_before_valid_selection","seed":20260801,"primary_evaluator":"corrected_scientific","analysis_only":True,"must_not_use_lockcheck_or_test_for_selection":True,"ablations":DIAGNOSTIC_ABLATIONS}


def write_diagnostic_ablation_contract(path: str|Path) -> dict[str,Any]:
    output=Path(path); atomic_write_json(output,diagnostic_ablation_contract()); return {"path":str(output.resolve()),"sha256":sha256_file(output)}
