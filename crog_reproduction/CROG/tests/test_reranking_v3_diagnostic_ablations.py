from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from failure_analysis.reranking_v3.diagnostic_ablations import (
    _restore_legacy_selection_identity,
    build_diagnostic_plan,
    run_validation_diagnostic_ablations,
)
from failure_analysis.reranking_v3.experiment_config import (
    diagnostic_ablation_contract,
    selection_contract,
)
from failure_analysis.reranking_v3.schema import artifact_identity, sha256_file


class _Catalog:
    def __init__(self, identities):
        self.identities = identities
        self.locations = {value: object() for value in identities}
        self.candidate_records = {
            sample_id: {
                "candidates": [
                    {"q_rank": rank} for rank in range(5)
                ]
            }
            for sample_id in identities
        }

    def assert_exact_ids(self, expected):
        if set(self.locations) != set(expected):
            raise ValueError("cohort mismatch")

    def candidate_identity(self, allowed):
        return {value: self.identities[value] for value in allowed}

    def provenance(self):
        return {"kind": "synthetic_features", "ids": sorted(self.locations)}


def _identity(sample_id: str):
    return {
        "candidate_ids": [f"{sample_id}-c{index}" for index in range(5)],
        "candidate_checksums": [f"{sample_id}-sha{index}" for index in range(5)],
    }


def _prediction(catalog, sample_ids, checkpoint_sha="checkpoint", seed=7):
    ids = np.asarray(sorted(sample_ids))
    candidate_ids = np.asarray([
        catalog.identities[value]["candidate_ids"] for value in ids
    ])
    checksums = np.asarray([
        catalog.identities[value]["candidate_checksums"] for value in ids
    ])
    q_ranks = np.tile(np.arange(5, dtype=np.int64), (len(ids), 1))
    scores = np.tile(np.asarray([0.0, 3.0, 2.0, 1.0, -1.0]), (len(ids), 1))
    return {
        "sample_ids": ids, "candidate_ids": candidate_ids,
        "candidate_checksums": checksums, "q_ranks": q_ranks,
        "scores": scores.astype(np.float32),
        "probabilities": np.full((len(ids), 5), 0.5, np.float32),
        "q": np.tile(np.linspace(0.9, 0.1, 5), (len(ids), 1)).astype(np.float32),
        "checkpoint_sha256": checkpoint_sha, "producing_seed": seed,
    }


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path):
    train_ids = {"train-a", "train-b"}
    select_ids = {"select-a", "select-b"}
    train_catalog = _Catalog({value: _identity(value) for value in train_ids})
    select_catalog = _Catalog({value: _identity(value) for value in select_ids})
    train_labels = {
        "train-a": np.asarray([1, 0, 0, 0, 0], np.float32),
        "train-b": np.asarray([0, 1, 0, 0, 0], np.float32),
    }
    select_labels = {
        "select-a": np.asarray([0, 1, 0, 0, 0], np.float32),
        "select-b": np.asarray([1, 1, 0, 0, 0], np.float32),
    }
    train_priors = {value: np.zeros((5, 80), np.float32) for value in train_ids}
    select_priors = {value: np.zeros((5, 80), np.float32) for value in select_ids}

    selection_grid = selection_contract()
    selection_contract_path = tmp_path / "selection_contract.json"
    _write_json(selection_contract_path, selection_grid)
    diagnostic_path = tmp_path / "diagnostic_contract.json"
    _write_json(diagnostic_path, diagnostic_ablation_contract())
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    model_path = selection_dir / "models" / "11_rgbd_h256_seed20260801.pt"
    model_path.parent.mkdir(); model_path.write_bytes(b"synthetic selection checkpoint")
    prediction_path = selection_dir / "predictions" / "11_rgbd_h256.npz"
    prediction_path.parent.mkdir()
    np.savez_compressed(prediction_path, **{
        key: value for key, value in _prediction(select_catalog, select_ids).items()
        if isinstance(value, np.ndarray)
    })
    summary = {
        "schema_version": "3.0.0", "kind": "v3_architecture_selection",
        "status": "complete", "scope": "v3_select", "seed": 20260801,
        "selection_contract_sha256": sha256_file(selection_contract_path),
        "selected": {"config_index": 11, "configuration": "rgbd_h256"},
        "results": [{
            "config_index": 11, "configuration": "rgbd_h256",
            "checkpoint_sha256": sha256_file(model_path),
            "prediction_sha256": sha256_file(prediction_path),
            "parameter_count": 123,
        }],
    }
    summary_path = selection_dir / "selection_summary.json"
    _write_json(summary_path, summary)
    v2_path = tmp_path / "v2_validation_predictions.jsonl"
    with v2_path.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(select_ids):
            ids = select_catalog.identities[sample_id]["candidate_ids"]
            handle.write(json.dumps({"sample_id": sample_id, "candidate_order": ids}) + "\n")
        # The locked validation artifact is intentionally a superset.
        handle.write(json.dumps({
            "sample_id": "select-not-requested",
            "candidate_order": [f"extra-{value}" for value in range(5)],
        }) + "\n")
    train_label_path = tmp_path / "train_labels.jsonl"; train_label_path.write_text("{}\n")
    select_label_path = tmp_path / "select_labels.jsonl"; select_label_path.write_text("{}\n")
    return locals()


def test_finite_plan_marks_reuse_refit_and_unavailable_dependencies():
    selected = selection_contract()["configurations"][-1]
    plan = build_diagnostic_plan(
        selected_config=selected,
        diagnostic_contract=diagnostic_ablation_contract(),
    )
    by_name = {(value["category"], value["component"]): value for value in plan}
    assert by_name[("leave_one_group_out", "G2_mask")]["execution"] == "trained_diagnostic"
    assert 2 not in by_name[("leave_one_group_out", "G2_mask")]["config"]["head_groups"]
    assert by_name[("leave_one_group_out", "G6_multiscale_latent")]["config"]["use_latent"] is False
    assert by_name[("leave_one_group_out", "G6_multiscale_latent")]["config"]["text_mode"] == "token"
    assert by_name[("latent_layers", "fpn_pre")]["selection_configuration"] == "fpn_pre_decoder"
    assert by_name[("text", "sentence_only")]["execution"] == "reused_selection_checkpoint"
    assert by_name[("text", "token_candidate")]["status"] == "unavailable_dependency"
    assert by_name[("modality", "rgbd")]["selection_configuration"] == "rgbd_h256"
    assert len({value["component"] for value in plan if value["category"] == "latent_layers"}) == 8


def test_legacy_selection_identity_is_restored_only_from_frozen_catalog():
    sample_ids = {"select-a", "select-b"}
    catalog = _Catalog({value: _identity(value) for value in sample_ids})
    legacy = _prediction(catalog, sample_ids)
    legacy.pop("candidate_checksums")
    legacy.pop("q_ranks")
    restored = _restore_legacy_selection_identity(legacy, catalog=catalog)
    assert restored["candidate_checksums"].shape == (2, 5)
    assert np.array_equal(restored["q_ranks"], np.tile(np.arange(5), (2, 1)))
    restored["candidate_ids"][0, 0] = "changed"
    with pytest.raises(ValueError, match="changed candidate IDs"):
        _restore_legacy_selection_identity(
            {key: value for key, value in restored.items() if key not in {"candidate_checksums", "q_ranks"}},
            catalog=catalog,
        )


def test_runner_refits_each_true_model_ablation_and_resumes_exactly(tmp_path):
    data = _fixture(tmp_path)
    train_calls = []

    def trainer(**kwargs):
        train_calls.append((kwargs["config"]["name"], set(kwargs["train_ids"])))
        path = Path(kwargs["output_path"]); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(kwargs["config"]["name"].encode())
        return {"checkpoint": artifact_identity(path), "parameters": 17, "history": []}

    def predictor(**kwargs):
        return _prediction(
            kwargs["catalog"], kwargs["sample_ids"],
            checkpoint_sha=sha256_file(kwargs["checkpoint_path"]), seed=20260801,
        )

    common = dict(
        train_catalog=data["train_catalog"], select_catalog=data["select_catalog"],
        train_ids=data["train_ids"], select_ids=data["select_ids"],
        train_labels=data["train_labels"], select_labels=data["select_labels"],
        train_priors=data["train_priors"], select_priors=data["select_priors"],
        v2_validation_predictions=data["v2_path"],
        selection_summary_path=data["summary_path"],
        selection_contract_path=data["selection_contract_path"],
        diagnostic_contract_path=data["diagnostic_path"],
        train_labels_provenance=data["train_label_path"],
        select_labels_provenance=data["select_label_path"],
        output_dir=tmp_path / "diagnostics", device="cpu", epochs=1,
        batch_size=2, trainer=trainer, predictor=predictor,
    )
    result = run_validation_diagnostic_ablations(**common)
    trained_records = [
        value for value in result["records"]
        if value["spec"]["execution"] == "trained_diagnostic"
    ]
    assert len(train_calls) == len(trained_records)
    assert all(ids == data["train_ids"] for _, ids in train_calls)
    assert all(value["row"]["sample_count"] == 2 for value in trained_records)
    assert all(value["row"]["oracle_correct"] == 2 for value in trained_records)
    assert result["candidate_identity_unchanged"] is True
    assert result["oracle_unchanged"] is True and result["formal_test_read"] is False
    assert result["labels_read"] == [
        str(data["train_label_path"].resolve()), str(data["select_label_path"].resolve()),
    ]
    assert (tmp_path / "diagnostics" / "feature_ablation.csv").is_file()
    before = len(train_calls)
    resumed = run_validation_diagnostic_ablations(**common, resume=True)
    assert resumed == result and len(train_calls) == before
    changed = dict(common)
    changed["train_priors"] = dict(data["train_priors"])
    changed["train_priors"]["train-a"] = np.ones((5, 80), np.float32)
    with pytest.raises(ValueError, match="run-spec mismatch"):
        run_validation_diagnostic_ablations(**changed, resume=True)


def test_gate_diagnostics_use_existing_label_free_bundle(tmp_path):
    data = _fixture(tmp_path)
    sample_ids = np.asarray(sorted(data["select_ids"]))
    base = _prediction(data["select_catalog"], data["select_ids"])
    count = len(sample_ids)
    gate_probabilities = np.zeros((count, 5, 3), np.float32)
    gate_probabilities[..., 2] = 1.0
    gate_probabilities[:, 1, 0] = 0.9
    gate_probabilities[:, 1, 2] = 0.0
    gate_bundle = tmp_path / "select_gate_bundle.npz"
    np.savez_compressed(
        gate_bundle,
        **{key: base[key] for key in (
            "sample_ids", "candidate_ids", "candidate_checksums", "q_ranks", "q",
            "scores", "probabilities",
        )},
        baseline_indices=np.zeros(count, np.int64),
        gate_probabilities=gate_probabilities,
        seed_gate_probabilities=np.repeat(gate_probabilities[None], 3, axis=0),
        uncertainty_score_std=np.zeros((count, 5), np.float32),
        uncertainty_valid_fraction=np.ones((count, 5), np.float32),
    )
    calls = []

    def trainer(**kwargs):
        calls.append(kwargs["config"]["name"])
        path = Path(kwargs["output_path"]); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"diagnostic")
        return {"checkpoint": artifact_identity(path), "parameters": 1, "history": []}

    def predictor(**kwargs):
        return _prediction(
            kwargs["catalog"], kwargs["sample_ids"],
            checkpoint_sha=sha256_file(kwargs["checkpoint_path"]), seed=20260801,
        )

    result = run_validation_diagnostic_ablations(
        train_catalog=data["train_catalog"], select_catalog=data["select_catalog"],
        train_ids=data["train_ids"], select_ids=data["select_ids"],
        train_labels=data["train_labels"], select_labels=data["select_labels"],
        train_priors=data["train_priors"], select_priors=data["select_priors"],
        v2_validation_predictions=data["v2_path"],
        selection_summary_path=data["summary_path"],
        selection_contract_path=data["selection_contract_path"],
        diagnostic_contract_path=data["diagnostic_path"], output_dir=tmp_path / "gate_run",
        gate_bundle_path=gate_bundle,
        gate_policy={"harm_cost": 2.0, "threshold": 0.0, "uncertainty_kappa": 1.0, "consensus": 3, "minimum_valid_fraction": 1.0},
        device="cpu", epochs=1, batch_size=2, trainer=trainer, predictor=predictor,
    )
    by_name = {value["row"]["configuration"]: value["row"] for value in result["records"]}
    assert by_name["gate:full_without_gate"]["status"] == "complete"
    assert by_name["gate:v2_anchored_gate"]["status"] == "complete"
    assert by_name["gate:v2_anchored_gate_with_uncertainty"]["status"] == "complete"
    assert by_name["gate:q_anchored_gate"]["status"] == "unavailable_dependency"
    assert by_name["gate:v2_anchored_gate"]["j_at_1"] == 1.0


def test_runner_refuses_lockcheck_label_provenance(tmp_path):
    data = _fixture(tmp_path)
    forbidden = tmp_path / "lockcheck_labels.jsonl"; forbidden.write_text("{}\n")
    with pytest.raises(PermissionError, match="refuses lockcheck"):
        run_validation_diagnostic_ablations(
            train_catalog=data["train_catalog"], select_catalog=data["select_catalog"],
            train_ids=data["train_ids"], select_ids=data["select_ids"],
            train_labels=data["train_labels"], select_labels=data["select_labels"],
            train_priors=data["train_priors"], select_priors=data["select_priors"],
            v2_validation_predictions=data["v2_path"],
            selection_summary_path=data["summary_path"],
            selection_contract_path=data["selection_contract_path"],
            diagnostic_contract_path=data["diagnostic_path"],
            select_labels_provenance=forbidden, output_dir=tmp_path / "never_created",
        )
