from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from reranking.publish_modular_development import (
    PublishError,
    publish_modular_development,
    verify_modular_development_publication,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _publication_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, tuple[Path, ...], tuple[Path, ...]]:
    feature_root = tmp_path / "features"
    candidate_root = tmp_path / "candidates"
    feature_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    feature_path = feature_root / "per_candidate.parquet"
    candidate_path = candidate_root / "nms_candidates.parquet"
    output_path = tmp_path / "published" / "per_candidate.parquet"
    pd.DataFrame(
        {
            "sample_id": ["q2", "q1"],
            "candidate_id": ["b", "a"],
            "scene_id": ["scene-2", "scene-1"],
            "split": ["val", "train"],
            "q_raw": [0.2, 0.8],
            "candidate_positive": [0, 1],
        }
    ).to_parquet(feature_path, index=False)
    pd.DataFrame(
        {
            "sample_id": ["q1", "q2"],
            "candidate_id": ["a", "b"],
            "scene_id": ["scene-1", "scene-2"],
            "center_u_px": [10.0, 20.0],
            "center_v_px": [30.0, 40.0],
            "center_depth_m": [0.5, 0.6],
            "angle_rad": [0.1, 0.2],
            "width_m": [0.03, 0.04],
            "width_px": [30.0, 40.0],
        }
    ).to_parquet(candidate_path, index=False)
    per_sample = feature_root / "per_sample.parquet"
    pd.DataFrame(
        {
            "sample_id": ["q1", "q2"],
            "candidate_count": [1, 1],
            "positive_candidate_count": [1, 0],
        }
    ).to_parquet(per_sample, index=False)
    (feature_root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETED",
                "candidate_count": 2,
                "sample_count": 2,
                "per_candidate_sha256": _sha256(feature_path),
                "per_sample_sha256": _sha256(per_sample),
                "prediction_manifest_sha256": "1" * 64,
                "frozen_split_manifest_sha256": "2" * 64,
                "query_metadata_sha256": "3" * 64,
            }
        ),
        encoding="utf-8",
    )
    (candidate_root / "run_config.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "candidate_stage_artifacts": {
                    "nms": {
                        "path": str(candidate_path.resolve()),
                        "rows": 2,
                        "sha256": _sha256(candidate_path),
                    }
                },
                "counts": {"samples": 2, "nms_candidates": 2},
            }
        ),
        encoding="utf-8",
    )
    universe_root = tmp_path / "universe"
    universe_root.mkdir()
    train_universe = universe_root / "train.jsonl"
    val_universe = universe_root / "val.jsonl"
    train_universe.write_text(
        json.dumps(
            {"sample_id": "q1", "split": "train", "scene_id": "scene-1"}
        )
        + "\n",
        encoding="utf-8",
    )
    val_universe.write_text(
        json.dumps(
            {"sample_id": "q2", "split": "val", "scene_id": "scene-2"}
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        feature_path,
        candidate_path,
        output_path,
        (train_universe, val_universe),
        (per_sample,),
    )


def _publish(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    feature, candidate, output, universes, count_evidence = _publication_inputs(
        tmp_path
    )
    manifest = publish_modular_development(
        [feature],
        [candidate],
        output,
        query_universe_paths=universes,
        candidate_count_evidence_paths=count_evidence,
    )
    return feature, candidate, output, manifest


def _rewrite_with_valid_payload_digest(path: Path, manifest: dict[str, object]) -> None:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_payload_sha256"
    }
    manifest["manifest_payload_sha256"] = _json_sha256(payload)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_publisher_manifest_commits_inputs_coverage_keys_and_geometry(
    tmp_path: Path,
) -> None:
    feature, candidate, output, manifest = _publish(tmp_path)

    assert manifest["schema_version"] == 3
    assert manifest["output_sha256"] == _sha256(output)
    assert manifest["rows"] == 2
    assert manifest["queries"] == 2
    assert manifest["coverage"]["query"]["count"] == 2
    assert manifest["coverage"]["split"] == {
        "available": True,
        "complete": True,
        "assigned_query_count": 2,
        "unassigned_query_count": 0,
        "values": ["train", "val"],
        "value_count": 2,
        "assignment_sha256": manifest["coverage"]["split"][
            "assignment_sha256"
        ],
        "evidence": manifest["coverage"]["split"]["evidence"],
    }
    assert manifest["coverage"]["scene_source"]["complete"] is True
    assert manifest["development_query_universe"]["query_count"] == 2
    assert manifest["development_query_universe"]["zero_candidate_queries"] == 0
    assert {
        item["path"]
        for item in manifest["development_query_universe"][
            "query_universe_inputs"
        ]
    } == {
        str((tmp_path / "universe" / "train.jsonl").resolve()),
        str((tmp_path / "universe" / "val.jsonl").resolve()),
    }
    assert manifest["development_query_universe"][
        "candidate_count_evidence_inputs"
    ][0]["sha256"] == _sha256(feature.parent / "per_sample.parquet")
    assert manifest["coverage"]["input_source"]["companion_manifest_count"] == 2
    assert manifest["feature_inputs"][0]["path"] == str(feature.resolve())
    assert manifest["candidate_inputs"][0]["path"] == str(candidate.resolve())
    assert manifest["geometry_recovery_protocol"]["source_to_output_columns"] == {
        "center_u_px": "x_px",
        "center_v_px": "y_px",
        "center_depth_m": "z_m",
        "angle_rad": "angle_rad",
        "width_m": "width_m",
        "width_px": "width_px",
    }
    assert verify_modular_development_publication(output) == manifest


def test_verifier_rejects_missing_companion_manifest(tmp_path: Path) -> None:
    _, _, output, _ = _publish(tmp_path)
    output.with_name(output.name + ".manifest.json").unlink()

    with pytest.raises(PublishError, match="companion manifest.*missing"):
        verify_modular_development_publication(output)


def test_verifier_rejects_tampered_output(tmp_path: Path) -> None:
    _, _, output, _ = _publish(tmp_path)
    tampered = pd.read_parquet(output)
    tampered.loc[0, "q_raw"] = 999.0
    tampered.to_parquet(output, index=False)

    with pytest.raises(PublishError, match="not an exact reconstruction"):
        verify_modular_development_publication(output)


def test_verifier_rejects_tampered_input(tmp_path: Path) -> None:
    feature, _, output, _ = _publish(tmp_path)
    tampered = pd.read_parquet(feature)
    tampered.loc[0, "q_raw"] = 999.0
    tampered.to_parquet(feature, index=False)

    with pytest.raises(PublishError, match="companion manifest SHA-256 disagrees"):
        verify_modular_development_publication(output)


@pytest.mark.parametrize("coverage_key", ["input_source", "split", "query"])
def test_verifier_rejects_manifest_omitting_source_split_or_query_coverage(
    tmp_path: Path, coverage_key: str
) -> None:
    _, _, output, manifest = _publish(tmp_path)
    manifest_path = output.with_name(output.name + ".manifest.json")
    tampered = deepcopy(manifest)
    del tampered["coverage"][coverage_key]
    _rewrite_with_valid_payload_digest(manifest_path, tampered)

    with pytest.raises(PublishError, match="coverage differs"):
        verify_modular_development_publication(output)


@pytest.mark.parametrize("kind", ["feature", "candidate"])
def test_publisher_rejects_duplicate_candidate_keys(tmp_path: Path, kind: str) -> None:
    feature, candidate, output, universes, count_evidence = _publication_inputs(
        tmp_path
    )
    path = feature if kind == "feature" else candidate
    duplicate = pd.read_parquet(path)
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    duplicate.to_parquet(path, index=False)

    with pytest.raises(PublishError, match="duplicate candidate keys"):
        publish_modular_development(
            [feature],
            [candidate],
            output,
            query_universe_paths=universes,
            candidate_count_evidence_paths=count_evidence,
        )


def test_publisher_rejects_query_omitted_from_only_one_source(tmp_path: Path) -> None:
    feature, candidate, output, universes, count_evidence = _publication_inputs(
        tmp_path
    )
    candidates = pd.read_parquet(candidate)
    candidates.iloc[[0]].to_parquet(candidate, index=False)
    config_path = candidate.parent / "run_config.json"
    config = json.loads(config_path.read_text())
    config["candidate_stage_artifacts"]["nms"].update(
        {
            "rows": 1,
            "sha256": _sha256(candidate),
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(PublishError, match="candidate-count evidence"):
        publish_modular_development(
            [feature],
            [candidate],
            output,
            query_universe_paths=universes,
            candidate_count_evidence_paths=count_evidence,
        )


@pytest.mark.parametrize(
    ("kind", "manifest_name"),
    [("feature", "dataset_manifest.json"), ("candidate", "run_config.json")],
)
def test_publisher_rejects_noncompleted_companion_manifest(
    tmp_path: Path, kind: str, manifest_name: str
) -> None:
    feature, candidate, output, universes, count_evidence = _publication_inputs(
        tmp_path
    )
    source = feature if kind == "feature" else candidate
    manifest_path = source.parent / manifest_name
    companion = json.loads(manifest_path.read_text(encoding="utf-8"))
    companion["status"] = "RUNNING"
    manifest_path.write_text(json.dumps(companion), encoding="utf-8")

    with pytest.raises(PublishError, match="not COMPLETED"):
        publish_modular_development(
            [feature],
            [candidate],
            output,
            query_universe_paths=universes,
            candidate_count_evidence_paths=count_evidence,
        )


def test_publisher_rejects_query_deleted_from_both_candidate_sources(
    tmp_path: Path,
) -> None:
    feature, candidate, output, universes, count_evidence = _publication_inputs(
        tmp_path
    )
    feature_frame = pd.read_parquet(feature).query("sample_id != 'q2'")
    candidate_frame = pd.read_parquet(candidate).query("sample_id != 'q2'")
    feature_frame.to_parquet(feature, index=False)
    candidate_frame.to_parquet(candidate, index=False)
    feature_manifest_path = feature.parent / "dataset_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    feature_manifest.update(
        {"candidate_count": 1, "per_candidate_sha256": _sha256(feature)}
    )
    feature_manifest_path.write_text(json.dumps(feature_manifest), encoding="utf-8")
    candidate_manifest_path = candidate.parent / "run_config.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidate_manifest["candidate_stage_artifacts"]["nms"].update(
        {"rows": 1, "sha256": _sha256(candidate)}
    )
    candidate_manifest_path.write_text(json.dumps(candidate_manifest), encoding="utf-8")

    with pytest.raises(PublishError, match="candidate-count evidence"):
        publish_modular_development(
            [feature],
            [candidate],
            output,
            query_universe_paths=universes,
            candidate_count_evidence_paths=count_evidence,
        )


def test_publisher_accepts_explicit_zero_candidate_query(tmp_path: Path) -> None:
    feature, candidate, output, universes, count_evidence = _publication_inputs(
        tmp_path
    )
    with universes[0].open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"sample_id": "q0", "split": "train", "scene_id": "scene-0"}
            )
            + "\n"
        )
    evidence = pd.read_parquet(count_evidence[0])
    evidence = pd.concat(
        [evidence, pd.DataFrame([{"sample_id": "q0", "candidate_count": 0}])],
        ignore_index=True,
    )
    evidence.to_parquet(count_evidence[0], index=False)
    feature_manifest_path = feature.parent / "dataset_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    feature_manifest.update(
        {
            "sample_count": 3,
            "per_sample_sha256": _sha256(count_evidence[0]),
        }
    )
    feature_manifest_path.write_text(json.dumps(feature_manifest), encoding="utf-8")
    candidate_manifest_path = candidate.parent / "run_config.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidate_manifest["counts"]["samples"] = 3
    candidate_manifest_path.write_text(json.dumps(candidate_manifest), encoding="utf-8")

    manifest = publish_modular_development(
        [feature],
        [candidate],
        output,
        query_universe_paths=universes,
        candidate_count_evidence_paths=count_evidence,
    )

    assert manifest["rows"] == 2
    assert manifest["queries"] == 3
    assert manifest["queries_with_candidates"] == 2
    assert manifest["development_query_universe"]["zero_candidate_queries"] == 1
    assert verify_modular_development_publication(output) == manifest


def test_verifier_rejects_query_universe_source_tamper(tmp_path: Path) -> None:
    feature, candidate, output, universes, count_evidence = _publication_inputs(
        tmp_path
    )
    publish_modular_development(
        [feature],
        [candidate],
        output,
        query_universe_paths=universes,
        candidate_count_evidence_paths=count_evidence,
    )
    universes[0].write_text(
        json.dumps(
            {"sample_id": "q1", "split": "train", "scene_id": "tampered"}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PublishError, match="query-universe scene_id"):
        verify_modular_development_publication(output)


def test_verifier_rejects_candidate_count_evidence_tamper(tmp_path: Path) -> None:
    feature, candidate, output, universes, count_evidence = _publication_inputs(
        tmp_path
    )
    publish_modular_development(
        [feature],
        [candidate],
        output,
        query_universe_paths=universes,
        candidate_count_evidence_paths=count_evidence,
    )
    evidence = pd.read_parquet(count_evidence[0])
    evidence.loc[evidence["sample_id"].eq("q1"), "candidate_count"] = 0
    evidence.to_parquet(count_evidence[0], index=False)

    with pytest.raises(PublishError, match="SHA-256 disagrees"):
        verify_modular_development_publication(output)
