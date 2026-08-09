from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from failure_analysis.reranking_v3 import formal_backend
from failure_analysis.reranking_v3.formal_backend import (
    build_formal_source_file_manifest,
    run_formal_inference,
    run_independent_dual_track_evaluation,
)
from failure_analysis.reranking_v3.schema import artifact_identity, read_jsonl


METHODS = (
    "q_only",
    "v2_locked_primary",
    "v3_full_head_scalar_gate",
    "v3_fcer_native",
    "v3_fcer_rgbd",
    "v3_locked_primary",
)


def _write(path: Path, value: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _write_json(path: Path, value: Any) -> Path:
    return _write(path, json.dumps(value, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _write(
        path,
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
    )


def _candidate_rows(*, with_sources: bool = False, root: Path | None = None) -> list[dict[str, Any]]:
    rows = []
    for source_id in (0, 1):
        row: dict[str, Any] = {
            "split": "test",
            "sample_id": source_id,
            "candidates": [
                {
                    "candidate_id": f"candidate_{index}",
                    "candidate_checksum": f"checksum-{source_id}-{index}",
                    "q_raw": 0.9 - index * 0.1,
                }
                for index in range(5)
            ],
        }
        if with_sources:
            assert root is not None
            row["image_path"] = str(_write(root / f"rgb-{source_id}.png"))
            row["depth_path"] = str(_write(root / f"depth-{source_id}.png"))
        rows.append(row)
    return rows


def _sample_id(source_id: int) -> str:
    return f"multiple:test:{source_id:08d}"


def _v2_rows() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": _sample_id(source_id),
            "candidate_order": [f"candidate_{index}" for index in (1, 0, 2, 3, 4)],
            "selection": {
                "selected_candidate_id": "candidate_1",
                "selected_index": 1,
            },
            "candidate_correctness_probabilities": [0.2, 0.8, 0.1, 0.1, 0.1],
        }
        for source_id in (0, 1)
    ]


def _feature_manifest(root: Path) -> Path:
    index = _write_jsonl(
        root / "index.jsonl",
        [
            {
                "sample_id": _sample_id(source_id),
                "shard": "shard_00000.npz",
                "offset": source_id,
            }
            for source_id in (0, 1)
        ],
    )
    schema = _write_json(root / "feature_schema.json", {"synthetic": True})
    shard = _write(root / "shards/shard_00000.npz", "synthetic\n")
    return _write_json(
        root / "artifact_manifest.json",
        {
            "status": "complete",
            "artifact_type": "fullchain_candidate_features",
            "outputs": [artifact_identity(value) for value in (index, schema, shard)],
        },
    )


def _locked_manifest(paths: list[Path], *, evaluators: dict[str, Path] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "formal_methods": list(METHODS),
        "locked_artifacts": [artifact_identity(value) for value in paths],
    }
    if evaluators is not None:
        result["artifacts"] = {
            "evaluators": {
                name: artifact_identity(path) for name, path in evaluators.items()
            }
        }
    return result


def _attach_evaluation_descriptor(
    locked: dict[str, Any], *, candidate: Path, root: Path
) -> None:
    corrected = _write(root / "scope-corrected.jsonl", "synthetic\n")
    legacy = _write(root / "scope-legacy.jsonl", "synthetic\n")
    descriptor = _write_json(
        root / "evaluation-descriptor.json",
        {
            "schema_version": "3.0.0",
            "kind": "v3_formal_evaluation_descriptor",
            "status": "frozen_before_evaluation",
            "cohorts": {
                scope: {
                    "candidate": artifact_identity(candidate),
                    "corrected_labels": artifact_identity(corrected),
                    "legacy_labels": artifact_identity(legacy),
                }
                for scope in ("lockcheck", "test")
            },
            "formal_methods": list(METHODS),
            "evaluator_callback": (
                "failure_analysis.reranking_v3.formal_backend:"
                "run_independent_dual_track_evaluation"
            ),
            "bootstrap_iterations": 10_000,
            "seed": 20260801,
        },
    )
    descriptor_identity = artifact_identity(descriptor)
    locked["locked_artifacts"].append(descriptor_identity)
    locked.setdefault("artifacts", {})["evaluation_descriptor"] = descriptor_identity
    locked["evaluator_callback"] = (
        "failure_analysis.reranking_v3.formal_backend:"
        "run_independent_dual_track_evaluation"
    )


def _inference_fixture(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    candidate = _write_jsonl(tmp_path / "candidates.jsonl", _candidate_rows())
    feature_manifest = _feature_manifest(tmp_path / "features")
    v2 = _write_jsonl(tmp_path / "v2.jsonl", _v2_rows())
    prior = tmp_path / "prior.npz"
    np.savez_compressed(
        prior,
        sample_ids=np.asarray([_sample_id(0), _sample_id(1)]),
        prior=np.zeros((2, 5, 80), np.float32),
        valid=np.ones(2, bool),
    )
    native_checkpoints = [tmp_path / f"native-model-{index}.pt" for index in range(3)]
    rgbd_checkpoints = [tmp_path / f"rgbd-model-{index}.pt" for index in range(3)]
    for path in native_checkpoints:
        torch.save({"status": "complete", "config": {"use_depth": False}}, path)
    for path in rgbd_checkpoints:
        torch.save({"status": "complete", "config": {"use_depth": True}}, path)
    gates = [_write(tmp_path / f"gate-{index}.pt") for index in range(3)]
    policy = _write_json(
        tmp_path / "policy.json",
        {
            "harm_cost": 5.0,
            "threshold": 0.1,
            "uncertainty_kappa": 1.0,
            "consensus": 2,
            "minimum_valid_fraction": 1.0,
        },
    )
    descriptor = _write_json(
        tmp_path / "descriptor.json",
        {
            "schema_version": "3.0.0",
            "kind": "v3_formal_inference_descriptor",
            "scopes": ["lockcheck", "test"],
            "candidate_input": "candidate_features",
            "feature_source": {
                "mode": "reuse",
                "artifact_manifest_inputs": ["feature_manifest"],
            },
            "v2_prior_input": "v2_prior",
            "v2_ranking_input": "v2_ranking",
            "methods": {
                method: {
                    "checkpoint_inputs": [
                        f"{'rgbd' if method == 'v3_fcer_rgbd' else 'native'}_model_{index}"
                        for index in range(3)
                    ],
                    "gate_checkpoint_inputs": [f"gate_{index}" for index in range(3)],
                    "policy_input": "policy",
                    **(
                        {"missing_depth_fallback_method": "v3_fcer_native"}
                        if method == "v3_fcer_rgbd" else {}
                    ),
                }
                for method in METHODS
                if method not in {"q_only", "v2_locked_primary"}
            },
        },
    )
    inputs = {
        "inference_descriptor": descriptor,
        "candidate_features": candidate,
        "feature_manifest": feature_manifest,
        "v2_prior": prior,
        "v2_ranking": v2,
        "policy": policy,
        **{f"native_model_{index}": value for index, value in enumerate(native_checkpoints)},
        **{f"rgbd_model_{index}": value for index, value in enumerate(rgbd_checkpoints)},
        **{f"gate_{index}": value for index, value in enumerate(gates)},
    }
    locked = _locked_manifest(list(inputs.values()))
    locked["artifacts"] = {
        "candidate": artifact_identity(candidate),
        "checkpoints": {
            **{
                f"native_seed_{index}": artifact_identity(value)
                for index, value in enumerate(native_checkpoints)
            },
            **{
                f"rgbd_seed_{index}": artifact_identity(value)
                for index, value in enumerate(rgbd_checkpoints)
            },
        },
        "gate": {
            "policy": artifact_identity(policy),
            **{
                f"seed_{index}": artifact_identity(value)
                for index, value in enumerate(gates)
            },
        },
        "v2": {
            "prior": artifact_identity(prior),
            "ranking": artifact_identity(v2),
        },
    }
    _attach_evaluation_descriptor(locked, candidate=candidate, root=tmp_path)
    return inputs, locked


def test_formal_inference_reuses_verified_shards_and_materializes_each_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, locked = _inference_fixture(tmp_path)
    checked_depth_modes: list[tuple[str, ...]] = []
    real_depth_check = formal_backend.checkpoint_ensemble_uses_depth

    def depth_check_spy(paths: list[Path]) -> bool:
        checked_depth_modes.append(tuple(path.name for path in paths))
        return real_depth_check(paths)

    class FakeCatalog:
        def __init__(self, *args: Any, **kwargs: Any):
            self.args = args
            self.kwargs = kwargs

        def assert_exact_ids(self, values: set[str]) -> None:
            assert values == {_sample_id(0), _sample_id(1)}

    def fake_predict(**kwargs: Any) -> dict[str, Any]:
        path = Path(kwargs["output_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, sample_ids=np.asarray([_sample_id(0), _sample_id(1)]))
        return {"status": "complete"}

    def fake_uncertainty(**kwargs: Any) -> dict[str, Any]:
        path = Path(kwargs["output_path"])
        np.savez_compressed(path, score_std=np.zeros((2, 5), np.float32))
        return {"status": "complete"}

    def fake_prepare(**kwargs: Any) -> dict[str, Any]:
        return {"synthetic": True}

    def fake_write(**kwargs: Any) -> dict[str, Any]:
        output = Path(kwargs["output_dir"])
        rows = [
            {
                "sample_id": _sample_id(source_id),
                "method": kwargs["method"],
                "candidate_order": [f"candidate_{index}" for index in range(5)],
            }
            for source_id in (0, 1)
        ]
        path = _write_jsonl(output / "predictions.jsonl", rows)
        return {"prediction": artifact_identity(path)}

    monkeypatch.setattr(formal_backend, "FeatureCatalog", FakeCatalog)
    monkeypatch.setattr(formal_backend, "predict_final_ensemble", fake_predict)
    monkeypatch.setattr(
        formal_backend, "score_perturbation_ensemble_streaming", fake_uncertainty
    )
    monkeypatch.setattr(formal_backend, "prepare_gate_inference", fake_prepare)
    monkeypatch.setattr(formal_backend, "write_v3_predictions", fake_write)
    monkeypatch.setattr(
        formal_backend, "checkpoint_ensemble_uses_depth", depth_check_spy
    )

    paths = run_formal_inference(
        scope="test",
        input_artifacts=inputs,
        output_dir=tmp_path / "output",
        locked_manifest=locked,
    )
    assert len(paths) == len(METHODS) - 1
    assert len(set(paths)) == len(METHODS) - 1
    by_method = {
        next(read_jsonl(path))["method"]: path
        for path in paths
    }
    assert set(by_method) == set(METHODS) - {"q_only"}
    assert all(
        row["method"] == "v2_locked_primary"
        for row in read_jsonl(by_method["v2_locked_primary"])
    )
    assert len(checked_depth_modes) == 4


def test_formal_inference_rejects_a_changed_feature_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, locked = _inference_fixture(tmp_path)
    _write(tmp_path / "features/shards/shard_00000.npz", "changed\n")
    monkeypatch.setattr(formal_backend, "FeatureCatalog", object)
    with pytest.raises(ValueError, match="feature output .* changed"):
        run_formal_inference(
            scope="lockcheck",
            input_artifacts=inputs,
            output_dir=tmp_path / "output",
            locked_manifest=locked,
        )


@pytest.mark.parametrize(
    ("method", "fallback", "message"),
    (
        (
            "v3_fcer_native",
            "v3_locked_primary",
            "Native method v3_fcer_native must not declare a depth fallback",
        ),
        (
            "v3_fcer_rgbd",
            "v3_locked_primary",
            "may fall back only to v3_fcer_native or V2",
        ),
    ),
)
def test_formal_inference_enforces_native_and_rgbd_fallback_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    fallback: str,
    message: str,
) -> None:
    inputs, locked = _inference_fixture(tmp_path)
    descriptor_path = inputs["inference_descriptor"]
    descriptor = json.loads(descriptor_path.read_text())
    descriptor["methods"][method]["missing_depth_fallback_method"] = fallback
    _write_json(descriptor_path, descriptor)
    replacement = artifact_identity(descriptor_path)
    locked["locked_artifacts"] = [
        replacement if identity["path"] == replacement["path"] else identity
        for identity in locked["locked_artifacts"]
    ]

    class FakeCatalog:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        def assert_exact_ids(self, values: set[str]) -> None:
            assert values == {_sample_id(0), _sample_id(1)}

    monkeypatch.setattr(formal_backend, "FeatureCatalog", FakeCatalog)
    with pytest.raises(ValueError, match=message):
        run_formal_inference(
            scope="test",
            input_artifacts=inputs,
            output_dir=tmp_path / "output",
            locked_manifest=locked,
        )


def test_fresh_extraction_requires_exact_locked_rgb_depth_file_manifest(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    rows = _candidate_rows(with_sources=True, root=source_root)
    candidate = _write_jsonl(source_root / "features.jsonl", rows)
    metadata = _write_json(source_root / "metadata.json", {"output_config": {"batch_size": 16}})
    split = _write_json(tmp_path / "split.json", {})
    config = _write(tmp_path / "crog.yaml")
    checkpoint = _write(tmp_path / "crog.pt")
    # Omitting all depth files must be rejected before the extractor is called.
    source_manifest = _write_json(
        tmp_path / "source-files.json",
        {
            "schema_version": "3.0.0",
            "kind": "v3_formal_source_file_manifest",
            "status": "complete",
            "candidate_artifact": artifact_identity(candidate),
            "source_file_count": 2,
            "files": [artifact_identity(row["image_path"]) for row in rows],
        },
    )
    descriptor = _write_json(
        tmp_path / "descriptor.json",
        {
            "schema_version": "3.0.0",
            "kind": "v3_formal_inference_descriptor",
            "scopes": ["test"],
            "candidate_input": "candidate_features",
            "feature_source": {
                "mode": "extract",
                "source_metadata_input": "source_metadata",
                "source_file_manifest_input": "source_file_manifest",
                "split_manifest_input": "split",
                "crog_config_input": "crog_config",
                "crog_checkpoint_input": "crog_checkpoint",
            },
            "v2_prior_input": "v2_prior",
            "v2_ranking_input": "v2_ranking",
            "methods": {
                method: {}
                for method in METHODS
                if method not in {"q_only", "v2_locked_primary"}
            },
        },
    )
    prior = tmp_path / "prior.npz"
    np.savez_compressed(
        prior,
        sample_ids=np.asarray([_sample_id(0), _sample_id(1)]),
        prior=np.zeros((2, 5, 80), np.float32),
        valid=np.ones(2, bool),
    )
    inputs = {
        "inference_descriptor": descriptor,
        "candidate_features": candidate,
        "source_metadata": metadata,
        "source_file_manifest": source_manifest,
        "split": split,
        "crog_config": config,
        "crog_checkpoint": checkpoint,
        "v2_prior": prior,
        "v2_ranking": _write_jsonl(tmp_path / "v2.jsonl", _v2_rows()),
    }
    locked = _locked_manifest(list(inputs.values()))
    locked["artifacts"] = {
        "candidate": artifact_identity(candidate),
        "v2": {
            "prior": artifact_identity(prior),
            "ranking": artifact_identity(inputs["v2_ranking"]),
        },
    }
    _attach_evaluation_descriptor(locked, candidate=candidate, root=tmp_path)
    with pytest.raises(ValueError, match="source-file manifest differs"):
        run_formal_inference(
            scope="test",
            input_artifacts=inputs,
            output_dir=tmp_path / "output",
            locked_manifest=locked,
        )


def test_source_file_manifest_builder_binds_exact_candidate_rgb_and_depth(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows(with_sources=True, root=tmp_path / "source")
    candidate = _write_jsonl(tmp_path / "source/features.jsonl", rows)
    output = tmp_path / "source-file-manifest.json"
    result = build_formal_source_file_manifest(
        candidate_artifact=candidate, output_path=output
    )
    assert result["source_file_count"] == 4
    assert {Path(value["path"]) for value in result["files"]} == {
        Path(row[field]).resolve()
        for row in rows
        for field in ("image_path", "depth_path")
    }
    assert result["labels_read"] is False


def _label_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source_id, feature in enumerate(candidate_rows):
        rows.append(
            {
                "sample_id": _sample_id(source_id),
                "split": "test",
                "source_sample_id": source_id,
                "frame_id": f"frame-{source_id}",
                "sequence_id": "sequence-a",
                "candidate_labels": [
                    {
                        "candidate_id": value["candidate_id"],
                        "candidate_checksum": value["candidate_checksum"],
                        "candidate_correct": index in {0, 1},
                    }
                    for index, value in enumerate(feature["candidates"])
                ],
            }
        )
    return rows


def test_independent_callback_runs_exact_dual_track_and_second_implementation(
    tmp_path: Path,
) -> None:
    candidate_rows = _candidate_rows()
    candidate = _write_jsonl(tmp_path / "candidates.jsonl", candidate_rows)
    corrected = _write_jsonl(tmp_path / "corrected.jsonl", _label_rows(candidate_rows))
    legacy = _write_jsonl(tmp_path / "legacy.jsonl", _label_rows(candidate_rows))
    corrected_impl = _write(tmp_path / "corrected_evaluator.py", "# corrected\n")
    legacy_impl = _write(tmp_path / "legacy_evaluator.py", "# legacy\n")
    v2 = _write_jsonl(
        tmp_path / "v2.jsonl",
        [dict(value, method="v2_locked_primary") for value in _v2_rows()],
    )
    v3_rankings = {
        method: _write_jsonl(
            tmp_path / f"{method}.jsonl",
            [
                {
                    "sample_id": _sample_id(source_id),
                    "method": method,
                    "candidate_order": [f"candidate_{index}" for index in range(5)],
                }
                for source_id in (0, 1)
            ],
        )
        for method in METHODS
        if method not in {"q_only", "v2_locked_primary"}
    }
    locked = _locked_manifest(
        [candidate, corrected_impl, legacy_impl, v2, *v3_rankings.values()],
        evaluators={
            "corrected_scientific": corrected_impl,
            "legacy_official_compatibility": legacy_impl,
        },
    )
    paths = run_independent_dual_track_evaluation(
        candidate_artifact=candidate,
        method_rankings={
            "q_only": None,
            "v2_locked_primary": v2,
            **v3_rankings,
        },
        evaluation_artifacts={
            "corrected_labels": corrected,
            "legacy_labels": legacy,
        },
        output_dir=tmp_path / "evaluation",
        locked_manifest=locked,
    )
    assert len(paths) == 6
    summary = json.loads((tmp_path / "evaluation/dual_track/summary.json").read_text())
    assert summary["bootstrap_iterations"] == 10_000
    assert summary["methods"] == list(METHODS)
    independent = json.loads(
        (tmp_path / "evaluation/independent_recomputation.json").read_text()
    )
    assert independent["all_correct_counts_exact"] is True
    assert independent["all_oracle_counts_exact"] is True
