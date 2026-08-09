from __future__ import annotations

import csv
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration import decide_safe_gate
from .experiment_config import diagnostic_ablation_contract
from .feature_store import FeatureCatalog
from .metrics import paired_switch_metrics, ranking_metrics, strip_metric_arrays
from .oof import predict_ranker_streaming, save_predictions, train_ranker_streaming
from .schema import (
    artifact_identity,
    atomic_write_json,
    canonical_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)


LATENT_LAYER_INDEX = {
    "c3": 0,
    "c4": 1,
    "c5": 2,
    "fpn_pre": 3,
    "decoder_1": 4,
    "decoder_2": 5,
    "decoder_3_post": 6,
}

SELECTION_REUSE = {
    ("latent_layers", "fpn_pre"): "fpn_pre_decoder",
    ("latent_layers", "decoder_3_post"): "post_decoder",
    ("latent_layers", "all_multiscale"): "all_multiscale",
    ("text", "none"): "all_multiscale",
    ("text", "sentence_only"): "sentence_only",
    ("text", "token_plus_sentence_global"): "token_cross_attention",
    ("maps", "native_raw_and_activated"): "aligned_rgb_maps",
    ("maps", "restored_scalar_only"): "head_g0_g5",
    ("modality", "crog_native"): "native_h256",
    ("modality", "rgbd"): "rgbd_h256",
}


def _ids_sha256(values: Sequence[str] | set[str]) -> str:
    return sha256_bytes("\n".join(sorted(map(str, values))).encode("utf-8"))


def _array_lookup_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = __import__("hashlib").sha256()
    for sample_id in sorted(map(str, values)):
        array = np.ascontiguousarray(np.asarray(values[sample_id], dtype=np.float32))
        digest.update(sample_id.encode("utf-8")); digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii")); digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _input_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return artifact_identity(value)


def _callable_identity(value: Callable[..., Any]) -> str:
    return f"{getattr(value, '__module__', '<unknown>')}:{getattr(value, '__qualname__', getattr(value, '__name__', '<callable>'))}"


def _assert_no_forbidden_paths(paths: Sequence[str | Path | None]) -> None:
    forbidden = [str(value) for value in paths if value is not None and any(
        token in str(value).lower() for token in ("lockcheck", "formal_test", "test_labels")
    )]
    if forbidden:
        raise PermissionError(f"diagnostic runner refuses lockcheck/formal-test inputs: {forbidden}")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fields = list(rows[0]) if rows else ["configuration", "status"]
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_lookup(
    name: str,
    lookup: Mapping[str, np.ndarray],
    expected_ids: set[str],
    expected_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    observed = set(map(str, lookup))
    if observed != expected_ids:
        raise ValueError(
            f"{name} cohort mismatch: missing={sorted(expected_ids-observed)[:5]}, "
            f"extra={sorted(observed-expected_ids)[:5]}"
        )
    result: dict[str, np.ndarray] = {}
    for sample_id in expected_ids:
        value = np.asarray(lookup[sample_id], dtype=np.float32)
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"{name} has invalid values for {sample_id}: {value.shape}")
        result[sample_id] = value
    return result


def _selection_contract(
    *, summary_path: Path, summary: Mapping[str, Any], explicit_path: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    if explicit_path is not None:
        path = Path(explicit_path).resolve()
    else:
        run_spec_path = summary_path.parent / "RUN_SPEC.json"
        run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        path = Path(run_spec["contract"]["path"]).resolve()
        if artifact_identity(path) != run_spec["contract"]:
            raise ValueError("selection contract differs from the frozen selection run spec")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("kind") != "v3_predeclared_selection_grid":
        raise ValueError("invalid selection contract kind")
    if sha256_file(path) != str(summary["selection_contract_sha256"]):
        raise ValueError("selection summary and selection contract differ")
    return path, contract


def _selected_config(
    summary: Mapping[str, Any], selection_contract: Mapping[str, Any],
) -> dict[str, Any]:
    selected = summary.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("selection summary has no selected configuration")
    index = int(selected["config_index"])
    configs = list(selection_contract["configurations"])
    if index < 0 or index >= len(configs):
        raise ValueError("selected configuration index is out of range")
    config = deepcopy(configs[index])
    if str(config["name"]) != str(selected["configuration"]):
        raise ValueError("selected configuration identity differs from its contract")
    return config


def _trained_spec(
    *, category: str, component: str, config: Mapping[str, Any], reason: str,
) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["name"] = f"diagnostic_{category}_{component}".lower().replace("-", "_")
    return {
        "category": category,
        "component": component,
        "execution": "trained_diagnostic",
        "status": "pending",
        "reason": reason,
        "config": value,
    }


def _unavailable_spec(
    *, category: str, component: str, reason: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "component": component,
        "execution": "none",
        "status": "unavailable_dependency",
        "reason": reason,
        "config": None,
    }


def build_diagnostic_plan(
    *,
    selected_config: Mapping[str, Any],
    diagnostic_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the finite post-selection plan without looking at any labels."""
    if diagnostic_contract != diagnostic_ablation_contract():
        raise ValueError("diagnostic contract differs from the executable finite plan")
    base = deepcopy(dict(selected_config))
    plan: list[dict[str, Any]] = [{
        "category": "reference",
        "component": "selected_full_model",
        "execution": "reused_selection_checkpoint",
        "selection_configuration": str(base["name"]),
        "status": "pending",
        "reason": "selected validation checkpoint; no diagnostic refit",
        "config": base,
    }]

    group_operations: dict[str, Callable[[dict[str, Any]], str | None]] = {
        "G2_mask": lambda value: _remove_head_group(value, 2),
        "G3_angle_confidence": lambda value: _remove_head_group(value, 3),
        "G4_width_consistency": lambda value: _remove_head_group(value, 4),
        "G5_cross_head": lambda value: _remove_head_group(value, 5),
        "G6_multiscale_latent": _remove_multiscale_latent,
        "G7_token_interaction": _remove_token_interaction,
        "G8_rgb_crop": lambda value: _disable_flag(value, "use_crop"),
        "G9_depth": lambda value: _disable_flag(value, "use_depth"),
        "G10_set_context": _remove_set_context,
        "G11_v2_prior": lambda value: _disable_flag(value, "use_prior"),
    }
    for component in diagnostic_contract["ablations"]["leave_one_group_out"]:
        if component == "uncertainty":
            plan.append({
                "category": "leave_one_group_out", "component": component,
                "execution": "label_free_gate_evaluation", "status": "pending",
                "gate_variant": "v2_anchored_gate", "config": None,
                "reason": "evaluate the locked V2 gate with uncertainty penalty disabled",
            })
            continue
        config = deepcopy(base)
        unavailable = group_operations[component](config)
        if unavailable is not None:
            plan.append(_unavailable_spec(
                category="leave_one_group_out", component=component, reason=unavailable,
            ))
        else:
            plan.append(_trained_spec(
                category="leave_one_group_out", component=component, config=config,
                reason="fresh fit on train; evaluate select only",
            ))

    for component in diagnostic_contract["ablations"]["latent_layers"]:
        reuse_name = SELECTION_REUSE.get(("latent_layers", component))
        if reuse_name is not None:
            plan.append(_reuse_spec("latent_layers", component, reuse_name))
            continue
        config = deepcopy(base)
        config["use_latent"] = True
        config["latent_layers"] = [LATENT_LAYER_INDEX[component]]
        plan.append(_trained_spec(
            category="latent_layers", component=component, config=config,
            reason="fresh single-layer fit on train; evaluate select only",
        ))

    for component in diagnostic_contract["ablations"]["text"]:
        if component == "token_candidate":
            plan.append(_unavailable_spec(
                category="text", component=component,
                reason="the frozen token module jointly consumes candidate ROI and sentence/dynamic context; token-only removal is not identifiable",
            ))
        else:
            plan.append(_reuse_spec("text", component, SELECTION_REUSE[("text", component)]))
    for component in diagnostic_contract["ablations"]["maps"]:
        plan.append(_reuse_spec("maps", component, SELECTION_REUSE[("maps", component)]))
    plan.extend([
        _reuse_spec("modality", "crog_native", SELECTION_REUSE[("modality", "crog_native")]),
        _reuse_spec("modality", "rgbd", SELECTION_REUSE[("modality", "rgbd")]),
    ])
    for component in diagnostic_contract["ablations"]["gate"]:
        if component == "q_anchored_gate":
            plan.append(_unavailable_spec(
                category="gate", component=component,
                reason="a V2-anchored gate bundle cannot be re-anchored to q without a separately trained label-free q-anchor bundle",
            ))
        else:
            plan.append({
                "category": "gate", "component": component,
                "execution": "label_free_gate_evaluation", "status": "pending",
                "gate_variant": component, "config": None,
                "reason": "evaluate an existing label-free bundle under a supplied locked policy",
            })
    for component in diagnostic_contract["ablations"]["sensitivity"]:
        plan.append({
            "category": "sensitivity", "component": component,
            "execution": "analysis_module", "status": "delegated_analysis",
            "reason": "non-model analysis; intentionally not represented as a fitted ablation",
            "config": None,
        })
    return plan


def _reuse_spec(category: str, component: str, selection_configuration: str) -> dict[str, Any]:
    return {
        "category": category, "component": component,
        "execution": "reused_selection_checkpoint", "status": "pending",
        "selection_configuration": selection_configuration,
        "reason": "identical predeclared selection configuration; reused without refit",
        "config": None,
    }


def _remove_head_group(config: dict[str, Any], group: int) -> str | None:
    groups = list(map(int, config.get("head_groups", [])))
    if group not in groups:
        return f"G{group} is absent from the selected configuration"
    config["head_groups"] = [value for value in groups if value != group]
    return None


def _disable_flag(config: dict[str, Any], field: str) -> str | None:
    if not bool(config.get(field, False)):
        return f"{field} is absent from the selected configuration"
    config[field] = False
    return None


def _remove_multiscale_latent(config: dict[str, Any]) -> str | None:
    if not bool(config.get("use_latent", False)) and not bool(config.get("use_attention", False)):
        return "multiscale latent/attention evidence is absent from the selected configuration"
    # Token interaction may still consume the frozen ROI as its query.  This
    # disables only the standalone latent adapter and decoder-attention branch.
    config["use_latent"] = False
    config["use_attention"] = False
    return None


def _remove_token_interaction(config: dict[str, Any]) -> str | None:
    if config.get("text_mode") != "token":
        return "candidate-token interaction is absent from the selected configuration"
    config["text_mode"] = "none"
    return None


def _remove_set_context(config: dict[str, Any]) -> str | None:
    groups = list(map(int, config.get("head_groups", [])))
    has_group = 10 in groups
    has_set = bool(config.get("use_set", False))
    if not has_group and not has_set:
        return "explicit G10 relations and learned set context are both absent"
    config["head_groups"] = [value for value in groups if value != 10]
    config["use_set"] = False
    return None


def _load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _load_resumed_prediction(
    path: Path, *, checkpoint_sha256: str, seed: int,
) -> dict[str, np.ndarray]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("resumed diagnostic prediction has no completed manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("resumed diagnostic prediction manifest is incomplete")
    if artifact_identity(path) != manifest.get("prediction"):
        raise ValueError("resumed diagnostic prediction changed after completion")
    if str(manifest.get("checkpoint_sha256")) != str(checkpoint_sha256):
        raise ValueError("resumed diagnostic prediction belongs to another checkpoint")
    if int(manifest.get("producing_seed")) != int(seed):
        raise ValueError("resumed diagnostic prediction belongs to another seed")
    return _load_predictions(path)


def _ordered_prediction(
    prediction: Mapping[str, np.ndarray], *, select_ids: set[str], catalog: FeatureCatalog,
) -> dict[str, np.ndarray]:
    ids = np.asarray(prediction["sample_ids"]).astype(str)
    if len(ids) != len(select_ids) or len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != select_ids:
        raise ValueError("diagnostic prediction cohort differs from select")
    candidate_ids = np.asarray(prediction["candidate_ids"]).astype(str)
    candidate_checksums = np.asarray(prediction["candidate_checksums"]).astype(str)
    q_ranks = np.asarray(prediction["q_ranks"], dtype=np.int64)
    if candidate_ids.shape != (len(ids), 5) or candidate_checksums.shape != (len(ids), 5):
        raise ValueError("diagnostic prediction candidate identity is invalid")
    if any(sorted(row) != [0, 1, 2, 3, 4] for row in q_ranks.tolist()):
        raise ValueError("diagnostic q ranks are not permutations of 0..4")
    expected = catalog.candidate_identity(select_ids)
    for row, sample_id in enumerate(ids):
        identity = expected[sample_id]
        if candidate_ids[row].tolist() != list(map(str, identity["candidate_ids"])):
            raise ValueError(f"diagnostic changed candidate IDs: {sample_id}")
        if candidate_checksums[row].tolist() != list(map(str, identity["candidate_checksums"])):
            raise ValueError(f"diagnostic changed candidate checksums: {sample_id}")
    required = ("scores", "probabilities", "q")
    for field in required:
        value = np.asarray(prediction[field])
        if value.shape != (len(ids), 5) or not np.isfinite(value).all():
            raise ValueError(f"invalid diagnostic prediction field: {field}")
    return {key: np.asarray(value) for key, value in prediction.items()}


def _restore_legacy_selection_identity(
    prediction: Mapping[str, np.ndarray], *, catalog: FeatureCatalog,
) -> dict[str, np.ndarray]:
    """Restore identity fields omitted by selection artifacts predating hardening.

    This is deliberately limited to the two immutable identity fields that can be
    reconstructed exactly from the same frozen candidate source used by the
    selection run.  Candidate IDs must already be present and match before either
    field is restored.
    """
    restored = {key: np.asarray(value) for key, value in prediction.items()}
    missing = {"candidate_checksums", "q_ranks"} - set(restored)
    if not missing:
        return restored
    if not missing.issubset({"candidate_checksums", "q_ranks"}):  # defensive
        raise ValueError(f"unsupported missing legacy selection fields: {sorted(missing)}")
    ids = np.asarray(restored.get("sample_ids", ())).astype(str)
    candidate_ids = np.asarray(restored.get("candidate_ids", ())).astype(str)
    if candidate_ids.shape != (len(ids), 5):
        raise ValueError("legacy selection candidate identity is invalid")
    expected = catalog.candidate_identity(set(ids.tolist()))
    checksums: list[list[str]] = []
    q_ranks: list[list[int]] = []
    for row, sample_id in enumerate(ids.tolist()):
        identity = expected[sample_id]
        if candidate_ids[row].tolist() != list(map(str, identity["candidate_ids"])):
            raise ValueError(f"legacy selection changed candidate IDs: {sample_id}")
        checksums.append(list(map(str, identity["candidate_checksums"])))
        frozen = getattr(catalog, "candidate_records", {}).get(sample_id)
        if frozen is None:
            raise ValueError(
                f"legacy selection q ranks require the frozen candidate source: {sample_id}"
            )
        ranks = [int(value["q_rank"]) for value in frozen["candidates"]]
        if sorted(ranks) != [0, 1, 2, 3, 4]:
            raise ValueError(f"legacy selection q ranks are invalid: {sample_id}")
        q_ranks.append(ranks)
    restored.setdefault("candidate_checksums", np.asarray(checksums))
    restored.setdefault("q_ranks", np.asarray(q_ranks, dtype=np.int64))
    return restored


def _stable_ranking(
    scores: np.ndarray, q_ranks: np.ndarray, candidate_ids: np.ndarray,
) -> np.ndarray:
    result = np.empty(np.asarray(scores).shape, dtype=np.int64)
    for row in range(len(result)):
        result[row] = sorted(
            range(5),
            key=lambda index: (
                -float(scores[row, index]), int(q_ranks[row, index]),
                str(candidate_ids[row, index]),
            ),
        )
    return result


def _v2_rankings(
    path: str | Path, *, sample_ids: np.ndarray, candidate_ids: np.ndarray,
) -> np.ndarray:
    lookup: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        sample_id = str(record["sample_id"])
        if sample_id in lookup:
            raise ValueError(f"duplicate locked V2 prediction: {sample_id}")
        lookup[sample_id] = record
    result = np.empty((len(sample_ids), 5), dtype=np.int64)
    for row, sample_id in enumerate(map(str, sample_ids)):
        record = lookup.get(sample_id)
        if record is None:
            raise ValueError(f"locked V2 prediction is missing select sample: {sample_id}")
        order = list(map(str, record["candidate_order"]))
        frozen = candidate_ids[row].tolist()
        if len(order) != 5 or len(set(order)) != 5 or set(order) != set(frozen):
            raise ValueError(f"locked V2 candidate pool differs: {sample_id}")
        by_id = {value: index for index, value in enumerate(frozen)}
        result[row] = [by_id[value] for value in order]
    # The locked V2 validation artifact covers the complete validation set;
    # diagnostics consume only the frozen v3_select subset.
    return result


def _evaluate_prediction(
    prediction: Mapping[str, np.ndarray], *, labels: Mapping[str, np.ndarray],
    v2_predictions_path: str | Path,
) -> dict[str, Any]:
    sample_ids = np.asarray(prediction["sample_ids"]).astype(str)
    candidate_ids = np.asarray(prediction["candidate_ids"]).astype(str)
    q_ranks = np.asarray(prediction["q_ranks"], dtype=np.int64)
    scores = np.asarray(prediction["scores"], dtype=np.float64)
    probabilities = np.asarray(prediction["probabilities"], dtype=np.float64)
    y = np.stack([labels[value] for value in sample_ids]).astype(np.float64)
    ranking = _stable_ranking(scores, q_ranks, candidate_ids)
    q_ranking = np.argsort(q_ranks, axis=1, kind="stable")
    v2_ranking = _v2_rankings(
        v2_predictions_path, sample_ids=sample_ids, candidate_ids=candidate_ids,
    )
    metrics = ranking_metrics(y, ranking, probabilities=probabilities)
    vs_q = paired_switch_metrics(y, q_ranking, ranking)
    vs_v2 = paired_switch_metrics(y, v2_ranking, ranking)
    return {
        "metrics": strip_metric_arrays(metrics),
        "vs_q": strip_metric_arrays(vs_q),
        "vs_v2": strip_metric_arrays(vs_v2),
        "ranking": ranking,
        "q_ranking": q_ranking,
        "v2_ranking": v2_ranking,
    }


def _gate_prediction(
    *, bundle_path: str | Path, policy: Mapping[str, Any], variant: str,
    select_ids: set[str], catalog: FeatureCatalog, v2_predictions_path: str | Path,
) -> dict[str, np.ndarray]:
    bundle = _load_predictions(bundle_path)
    prediction = {
        key: bundle[key] for key in (
            "sample_ids", "candidate_ids", "candidate_checksums", "q_ranks", "q", "scores",
        )
    }
    prediction["probabilities"] = np.asarray(bundle["probabilities"])
    prediction = _ordered_prediction(prediction, select_ids=select_ids, catalog=catalog)
    sample_ids = prediction["sample_ids"].astype(str)
    candidate_ids = prediction["candidate_ids"].astype(str)
    v2_ranking = _v2_rankings(
        v2_predictions_path, sample_ids=sample_ids, candidate_ids=candidate_ids,
    )
    if variant == "full_without_gate":
        return prediction
    if variant not in {"v2_anchored_gate", "v2_anchored_gate_with_uncertainty"}:
        raise ValueError(f"unsupported gate diagnostic variant: {variant}")
    baseline = np.asarray(bundle["baseline_indices"], dtype=np.int64)
    if not np.array_equal(baseline, v2_ranking[:, 0]):
        raise ValueError("gate bundle baseline is not the locked V2 selection")
    uncertainty = np.asarray(bundle["uncertainty_score_std"], dtype=np.float64)
    kappa = (
        0.0 if variant == "v2_anchored_gate"
        else float(policy["uncertainty_kappa"])
    )
    decision = decide_safe_gate(
        baseline_indices=baseline,
        gate_probabilities=np.asarray(bundle["gate_probabilities"]),
        seed_gate_probabilities=np.asarray(bundle["seed_gate_probabilities"]),
        uncertainty=uncertainty,
        valid_fraction=np.asarray(bundle["uncertainty_valid_fraction"]),
        fcer_scores=np.asarray(bundle["scores"]),
        candidate_ids=candidate_ids,
        q_ranks=np.asarray(bundle["q_ranks"]),
        harm_cost=float(policy["harm_cost"]), threshold=float(policy["threshold"]),
        uncertainty_kappa=kappa, consensus=int(policy["consensus"]),
        minimum_valid_fraction=float(policy.get("minimum_valid_fraction", 1.0)),
    )
    selected = decision["selected_indices"]
    ranking = _stable_ranking(
        np.asarray(bundle["scores"]), np.asarray(bundle["q_ranks"]), candidate_ids,
    )
    for row, top in enumerate(selected):
        rest = [value for value in ranking[row] if int(value) != int(top)]
        ranking[row] = [int(top), *rest]
    # Encode the locked gate ranking as monotonic synthetic scores solely for
    # the shared evaluator.  Candidate probabilities remain model outputs.
    gated_scores = np.empty_like(np.asarray(bundle["scores"], dtype=np.float64))
    for row in range(len(ranking)):
        gated_scores[row, ranking[row]] = np.arange(5, 0, -1, dtype=np.float64)
    prediction["scores"] = gated_scores
    return prediction


def _row(
    spec: Mapping[str, Any], *, evaluation: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    prediction: Mapping[str, Any] | None = None,
    status: str | None = None, reason: str | None = None,
    parameters: int | None = None,
) -> dict[str, Any]:
    metrics = {} if evaluation is None else evaluation["metrics"]
    vs_q = {} if evaluation is None else evaluation["vs_q"]
    vs_v2 = {} if evaluation is None else evaluation["vs_v2"]
    return {
        "configuration": f"{spec['category']}:{spec['component']}",
        "category": spec["category"], "component": spec["component"],
        "execution": spec["execution"], "status": status or spec["status"],
        "reason": reason or spec["reason"],
        "sample_count": metrics.get("sample_count"), "correct": metrics.get("correct"),
        "j_at_1": metrics.get("j_at_1"), "oracle_correct": metrics.get("oracle_correct"),
        "oracle_at_5": metrics.get("oracle_at_5"),
        "delta_vs_q": vs_q.get("delta_j_at_1"),
        "delta_vs_v2": vs_v2.get("delta_j_at_1"),
        "recovered_vs_v2": vs_v2.get("recovered"),
        "harmful_vs_v2": vs_v2.get("harmful"),
        "net_recovered_vs_v2": vs_v2.get("net_recovered"),
        "switch_coverage_vs_v2": vs_v2.get("switch_coverage"),
        "outcome_changing_precision_vs_v2": vs_v2.get("outcome_changing_precision"),
        "parameter_count": parameters,
        "checkpoint_sha256": None if checkpoint is None else checkpoint["sha256"],
        "prediction_sha256": None if prediction is None else prediction["sha256"],
    }


def _record_path(output: Path, index: int, spec: Mapping[str, Any]) -> Path:
    slug = f"{index:02d}_{spec['category']}_{spec['component']}".lower()
    slug = "".join(value if value.isalnum() or value in "-_" else "_" for value in slug)
    return output / "records" / f"{slug}.json"


def _verify_resumed_record(record: Mapping[str, Any]) -> None:
    for field in ("checkpoint", "prediction"):
        identity = record.get(field)
        if identity is not None and artifact_identity(identity["path"]) != identity:
            raise ValueError(f"resumed diagnostic {field} artifact changed")


def run_validation_diagnostic_ablations(
    *,
    train_catalog: FeatureCatalog,
    select_catalog: FeatureCatalog,
    train_ids: set[str],
    select_ids: set[str],
    train_labels: Mapping[str, np.ndarray],
    select_labels: Mapping[str, np.ndarray],
    train_priors: Mapping[str, np.ndarray],
    select_priors: Mapping[str, np.ndarray],
    v2_validation_predictions: str | Path,
    selection_summary_path: str | Path,
    diagnostic_contract_path: str | Path,
    output_dir: str | Path,
    train_labels_provenance: str | Path | None = None,
    select_labels_provenance: str | Path | None = None,
    train_priors_provenance: str | Path | None = None,
    select_priors_provenance: str | Path | None = None,
    selection_contract_path: str | Path | None = None,
    gate_bundle_path: str | Path | None = None,
    gate_policy: Mapping[str, Any] | None = None,
    gate_policy_path: str | Path | None = None,
    device: str = "auto",
    seed: int = 20260801,
    epochs: int = 5,
    batch_size: int = 16,
    resume: bool = False,
    trainer: Callable[..., dict[str, Any]] = train_ranker_streaming,
    predictor: Callable[..., dict[str, Any]] = predict_ranker_streaming,
) -> dict[str, Any]:
    """Run development-selection diagnostics without reading lockcheck/test labels.

    Labels and priors are explicit in-memory joins.  This keeps model inference
    label-free and makes every label-bearing input visible in the run manifest.
    """
    # Fail closed on declared physical lineage before inspecting any label or
    # prior value supplied by the caller.
    _assert_no_forbidden_paths((
        train_labels_provenance, select_labels_provenance,
        train_priors_provenance, select_priors_provenance,
        v2_validation_predictions, selection_summary_path,
        diagnostic_contract_path, selection_contract_path,
        gate_bundle_path, gate_policy_path,
    ))
    train_ids = set(map(str, train_ids)); select_ids = set(map(str, select_ids))
    if not train_ids or not select_ids or train_ids & select_ids:
        raise ValueError("diagnostic train/select cohorts must be non-empty and disjoint")
    missing_train = train_ids - set(train_catalog.locations)
    if missing_train:
        raise ValueError(f"training catalog misses diagnostic train rows: {sorted(missing_train)[:5]}")
    select_catalog.assert_exact_ids(select_ids)
    train_labels = _validate_lookup("train labels", train_labels, train_ids, (5,))
    select_labels = _validate_lookup("select labels", select_labels, select_ids, (5,))
    train_priors = _validate_lookup("train V2 OOF priors", train_priors, train_ids, (5, 80))
    select_priors = _validate_lookup("select V2 priors", select_priors, select_ids, (5, 80))

    summary_path = Path(selection_summary_path).resolve()
    selection_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if selection_summary.get("status") != "complete" or selection_summary.get("scope") != "v3_select":
        raise ValueError("diagnostics require a completed v3_select selection summary")
    selection_contract_file, selection_grid = _selection_contract(
        summary_path=summary_path, summary=selection_summary,
        explicit_path=selection_contract_path,
    )
    selected_config = _selected_config(selection_summary, selection_grid)
    diagnostic_path = Path(diagnostic_contract_path).resolve()
    contract = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    plan = build_diagnostic_plan(selected_config=selected_config, diagnostic_contract=contract)
    policy = None if gate_policy is None else dict(gate_policy)
    if gate_policy_path is not None:
        stored_policy = json.loads(Path(gate_policy_path).read_text(encoding="utf-8"))
        if policy is None:
            policy = dict(stored_policy.get("selected", stored_policy.get("policy", stored_policy)))
    input_paths = [v2_validation_predictions, summary_path, diagnostic_path, selection_contract_file]
    input_paths.extend(value for value in (
        train_labels_provenance, select_labels_provenance,
        train_priors_provenance, select_priors_provenance,
        gate_bundle_path, gate_policy_path,
    ) if value is not None)
    _assert_no_forbidden_paths(input_paths)

    run_spec = {
        "schema_version": "3.0.0", "kind": "v3_validation_diagnostic_run_spec",
        "status": "frozen_before_fit", "scope": "v3_select_only",
        "module_sha256": sha256_file(__file__),
        "trainer": _callable_identity(trainer), "predictor": _callable_identity(predictor),
        "diagnostic_contract": artifact_identity(diagnostic_path),
        "selection_contract": artifact_identity(selection_contract_file),
        "selection_summary": artifact_identity(summary_path),
        "train_features": train_catalog.provenance(), "select_features": select_catalog.provenance(),
        "train_ids_sha256": _ids_sha256(train_ids), "select_ids_sha256": _ids_sha256(select_ids),
        "train_labels_sha256": _array_lookup_sha256(train_labels),
        "select_labels_sha256": _array_lookup_sha256(select_labels),
        "train_labels_provenance": _input_identity(train_labels_provenance),
        "select_labels_provenance": _input_identity(select_labels_provenance),
        "train_priors_sha256": _array_lookup_sha256(train_priors),
        "select_priors_sha256": _array_lookup_sha256(select_priors),
        "train_priors_provenance": _input_identity(train_priors_provenance),
        "select_priors_provenance": _input_identity(select_priors_provenance),
        "v2_validation_predictions": artifact_identity(v2_validation_predictions),
        "gate_bundle": _input_identity(gate_bundle_path),
        "gate_policy": policy,
        "gate_policy_artifact": _input_identity(gate_policy_path),
        "plan": plan, "seed": int(seed), "epochs": int(epochs),
        "batch_size": int(batch_size), "device": str(device),
        "labels_allowed": ["train", "v3_select"], "formal_test_read": False,
    }
    run_spec["content_sha256"] = sha256_bytes(canonical_json(run_spec).encode("utf-8"))
    output = Path(output_dir).resolve()
    spec_path = output / "RUN_SPEC.json"
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if not spec_path.is_file() or json.loads(spec_path.read_text(encoding="utf-8")) != run_spec:
            raise ValueError("diagnostic resume run-spec mismatch")
        complete_path = output / "feature_ablation.json"
        if complete_path.is_file():
            completed = json.loads(complete_path.read_text(encoding="utf-8"))
            for record in completed["records"]:
                _verify_resumed_record(record)
            return completed
    else:
        output.mkdir(parents=True)
        atomic_write_json(spec_path, run_spec)

    selection_results = {
        str(value["configuration"]): value for value in selection_summary["results"]
    }
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    oracle_values: set[tuple[int, float]] = set()
    for index, spec in enumerate(plan):
        record_path = _record_path(output, index, spec)
        if resume and record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("run_spec_sha256") != run_spec["content_sha256"]:
                raise ValueError("diagnostic record belongs to a different run spec")
            _verify_resumed_record(record)
            records.append(record); rows.append(record["row"])
            if record["row"]["oracle_correct"] is not None:
                oracle_values.add((record["row"]["oracle_correct"], record["row"]["oracle_at_5"]))
            continue

        checkpoint_identity = None; prediction_identity = None; parameters = None
        if spec["status"] in {"unavailable_dependency", "delegated_analysis"}:
            row = _row(spec)
        elif spec["execution"] == "trained_diagnostic":
            checkpoint_path = output / "models" / f"{index:02d}_{spec['config']['name']}_seed{seed}.pt"
            trained = trainer(
                catalog=train_catalog, train_ids=train_ids, labels=train_labels,
                priors=train_priors, config=spec["config"], output_path=checkpoint_path,
                seed=seed, device=device, epochs=epochs, batch_size=batch_size,
                normalizers=None, resume=resume,
            )
            checkpoint_identity = artifact_identity(checkpoint_path)
            if trained.get("checkpoint", {}).get("sha256") != checkpoint_identity["sha256"]:
                raise ValueError("diagnostic trainer checkpoint identity mismatch")
            parameters = int(trained.get("parameters", 0))
            prediction_path = output / "predictions" / f"{index:02d}_{spec['config']['name']}.npz"
            if resume and prediction_path.is_file():
                prediction = _load_resumed_prediction(
                    prediction_path, checkpoint_sha256=checkpoint_identity["sha256"], seed=seed,
                )
            else:
                prediction = predictor(
                    catalog=select_catalog, sample_ids=select_ids, priors=select_priors,
                    checkpoint_path=checkpoint_path, device=device,
                    batch_size=max(int(batch_size), 32),
                )
                save_predictions(
                    prediction_path, prediction, scope="v3_select_diagnostic",
                    configuration=spec["config"]["name"],
                )
            prediction_identity = artifact_identity(prediction_path)
            prediction = _ordered_prediction(prediction, select_ids=select_ids, catalog=select_catalog)
            evaluation = _evaluate_prediction(
                prediction, labels=select_labels,
                v2_predictions_path=v2_validation_predictions,
            )
            row = _row(
                spec, evaluation=evaluation, checkpoint=checkpoint_identity,
                prediction=prediction_identity, status="complete", parameters=parameters,
            )
        elif spec["execution"] == "reused_selection_checkpoint":
            selection_name = str(spec["selection_configuration"])
            selection_record = selection_results.get(selection_name)
            if selection_record is None:
                row = _row(spec, status="unavailable_dependency", reason=f"selection result is missing {selection_name}")
            else:
                selection_index = int(selection_record["config_index"])
                model_path = summary_path.parent / "models" / f"{selection_index:02d}_{selection_name}_seed{selection_summary['seed']}.pt"
                prediction_path = summary_path.parent / "predictions" / f"{selection_index:02d}_{selection_name}.npz"
                if not model_path.is_file() or not prediction_path.is_file():
                    row = _row(spec, status="unavailable_dependency", reason=f"selection artifacts are missing for {selection_name}")
                else:
                    checkpoint_identity = artifact_identity(model_path)
                    prediction_identity = artifact_identity(prediction_path)
                    if checkpoint_identity["sha256"] != str(selection_record["checkpoint_sha256"]):
                        raise ValueError(f"selection checkpoint changed: {selection_name}")
                    if prediction_identity["sha256"] != str(selection_record["prediction_sha256"]):
                        raise ValueError(f"selection prediction changed: {selection_name}")
                    prediction = _ordered_prediction(
                        _restore_legacy_selection_identity(
                            _load_predictions(prediction_path), catalog=select_catalog,
                        ),
                        select_ids=select_ids, catalog=select_catalog,
                    )
                    evaluation = _evaluate_prediction(
                        prediction, labels=select_labels,
                        v2_predictions_path=v2_validation_predictions,
                    )
                    parameters = int(selection_record["parameter_count"])
                    row = _row(
                        spec, evaluation=evaluation, checkpoint=checkpoint_identity,
                        prediction=prediction_identity, status="complete", parameters=parameters,
                    )
        elif spec["execution"] == "label_free_gate_evaluation":
            variant = str(spec["gate_variant"])
            if gate_bundle_path is None or (variant != "full_without_gate" and policy is None):
                row = _row(
                    spec, status="unavailable_dependency",
                    reason="label-free gate bundle and locked policy were not both supplied",
                )
            else:
                prediction = _gate_prediction(
                    bundle_path=gate_bundle_path, policy={} if policy is None else policy,
                    variant=variant, select_ids=select_ids, catalog=select_catalog,
                    v2_predictions_path=v2_validation_predictions,
                )
                evaluation = _evaluate_prediction(
                    prediction, labels=select_labels,
                    v2_predictions_path=v2_validation_predictions,
                )
                prediction_identity = artifact_identity(gate_bundle_path)
                row = _row(
                    spec, evaluation=evaluation, prediction=prediction_identity,
                    status="complete",
                )
        else:
            raise AssertionError(f"unknown diagnostic execution: {spec['execution']}")
        if row["oracle_correct"] is not None:
            oracle_values.add((row["oracle_correct"], row["oracle_at_5"]))
        record = {
            "schema_version": "3.0.0", "kind": "v3_diagnostic_ablation_record",
            "status": row["status"], "run_spec_sha256": run_spec["content_sha256"],
            "spec": spec, "row": row, "checkpoint": checkpoint_identity,
            "prediction": prediction_identity,
            "labels_read": [
                str(Path(train_labels_provenance).resolve()) if train_labels_provenance else f"in_memory:sha256:{run_spec['train_labels_sha256']}",
                str(Path(select_labels_provenance).resolve()) if select_labels_provenance else f"in_memory:sha256:{run_spec['select_labels_sha256']}",
            ],
            "formal_test_read": False,
        }
        atomic_write_json(record_path, record)
        records.append(record); rows.append(row)

    if len(oracle_values) > 1:
        raise AssertionError("diagnostic reranking changed the frozen candidate Oracle@5")
    checkpoint_hashes = sorted({
        str(record["checkpoint"]["sha256"]) for record in records
        if record.get("checkpoint") is not None
    })
    payload = {
        "schema_version": "3.0.0", "kind": "v3_validation_diagnostic_ablations",
        "status": "complete", "scope": "v3_select_only",
        "run_spec": artifact_identity(spec_path), "record_count": len(records),
        "records": records, "rows": rows, "checkpoint_sha256": checkpoint_hashes,
        "candidate_identity_unchanged": True, "oracle_unchanged": len(oracle_values) <= 1,
        "oracle": None if not oracle_values else {
            "correct": next(iter(oracle_values))[0], "at_5": next(iter(oracle_values))[1],
        },
        "labels_read": [
            str(Path(train_labels_provenance).resolve()) if train_labels_provenance else f"in_memory:sha256:{run_spec['train_labels_sha256']}",
            str(Path(select_labels_provenance).resolve()) if select_labels_provenance else f"in_memory:sha256:{run_spec['select_labels_sha256']}",
        ],
        "formal_test_read": False,
    }
    _atomic_csv(output / "feature_ablation.csv", rows)
    atomic_write_json(output / "feature_ablation.json", payload)
    return payload


__all__ = [
    "LATENT_LAYER_INDEX", "build_diagnostic_plan",
    "run_validation_diagnostic_ablations",
]
