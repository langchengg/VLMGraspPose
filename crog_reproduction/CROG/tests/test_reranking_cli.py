import json
import subprocess
import sys

import pytest

from failure_analysis.reranking.exporter import _load_training_calibration
from failure_analysis.reranking.cli import _load_provenance_weights
from failure_analysis.reranking.schema import (
    COMMIT_FILENAME,
    implementation_sha256,
    read_jsonl,
    recover_committed_jsonl_prefix,
)
from failure_analysis.reranking.geometry import geometry_checksum


@pytest.mark.parametrize(
    "command", ["build-features", "build-labels", "evaluate", "train-mlp"]
)
def test_cli_help(command):
    result = subprocess.run(
        [sys.executable, "-m", "failure_analysis.reranking.cli", command, "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--limit" in result.stdout
    assert "--resume" in result.stdout


def test_evaluate_resume_rejects_different_ranker(tmp_path):
    features = tmp_path / "features.jsonl"
    labels = tmp_path / "labels.jsonl"
    features.write_text(
        json.dumps(
            {
                "sample_id": "empty",
                "scene_id": "frame-empty",
                "split": "val",
                "candidates": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps({"sample_id": "empty", "candidate_labels": []}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "evaluation"
    base = [
        sys.executable,
        "-m",
        "failure_analysis.reranking.cli",
        "evaluate",
        "--features",
        str(features),
        "--labels",
        str(labels),
        "--output",
        str(output),
        "--bootstrap-iterations",
        "2",
    ]
    created = subprocess.run(
        [*base, "--ranker", "legacy"], text=True, capture_output=True, check=False
    )
    assert created.returncode == 0, created.stderr
    refused = subprocess.run(
        [*base, "--ranker", "q_only", "--resume"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "fingerprint mismatch" in refused.stderr


def test_calibration_and_tuned_weights_require_split_provenance(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"calibration": {"tau_variance_m": 0.01}}))
    with pytest.raises(ValueError, match="source_split='train'"):
        _load_training_calibration(calibration)
    calibration.write_text(
        json.dumps(
            {
                "calibration": {"tau_variance_m": 0.01},
                "provenance": {"source_split": "train"},
            }
        )
    )
    values, provenance = _load_training_calibration(calibration)
    assert values["tau_variance_m"] == 0.01
    assert provenance["source_split"] == "train"
    assert provenance["file_sha256"]

    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"weights": {"q": 1.0}}))
    with pytest.raises(ValueError, match="source_split='val'"):
        _load_provenance_weights(weights)
    weights.write_text(
        json.dumps(
            {"weights": {"q": 1.0}, "provenance": {"source_split": "val"}}
        )
    )
    assert _load_provenance_weights(weights) == {"q": 1.0}


def test_implementation_fingerprint_covers_untracked_source_content(tmp_path):
    source = tmp_path / "failure_analysis" / "reranking" / "new_ranker.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = implementation_sha256(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = implementation_sha256(tmp_path)
    assert first != second


def test_train_resume_rejects_changed_seed(tmp_path):
    features_path = tmp_path / "train_features.jsonl"
    labels_path = tmp_path / "train_labels.jsonl"
    features = []
    labels = []
    for index in range(4):
        candidate = {
            "candidate_id": "candidate_0",
            "legacy_rank": 0,
            "q_rank": 0,
            "row": 10,
            "col": 20,
            "cx": 20.0,
            "cy": 10.0,
            "angle_rad": 0.0,
            "angle_deg": 0.0,
            "width_px": 30.0,
            "height_px": 20,
            "polygon": [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]],
            "q_raw": 0.8,
            "legacy_grasp": [20.0, 10.0, 30.0, 20, 0.0],
            "features": {
                "q": {"value": 0.8, "reliability": 1.0, "missing_reason": None}
            },
        }
        candidate["candidate_checksum"] = geometry_checksum(candidate)
        sample_id = f"train-{index}"
        features.append(
            {
                "sample_id": sample_id,
                "scene_id": f"frame-{index}",
                "split": "train",
                "candidates": [candidate],
            }
        )
        labels.append(
            {
                "sample_id": sample_id,
                "candidate_labels": [
                    {
                        "candidate_id": "candidate_0",
                        "candidate_checksum": candidate["candidate_checksum"],
                        "candidate_valid": bool(index % 2),
                    }
                ],
            }
        )
    features_path.write_text(
        "".join(json.dumps(item) + "\n" for item in features), encoding="utf-8"
    )
    labels_path.write_text(
        "".join(json.dumps(item) + "\n" for item in labels), encoding="utf-8"
    )
    model = tmp_path / "ranker.pt"
    base = [
        sys.executable,
        "-m",
        "failure_analysis.reranking.cli",
        "train-mlp",
        "--features",
        str(features_path),
        "--labels",
        str(labels_path),
        "--output",
        str(model),
        "--epochs",
        "1",
        "--patience",
        "1",
    ]
    created = subprocess.run(base, text=True, capture_output=True, check=False)
    assert created.returncode == 0, created.stderr
    refused = subprocess.run(
        [*base, "--seed", "19", "--resume"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "fingerprint mismatch" in refused.stderr


def test_resume_journal_trims_only_uncommitted_tail(tmp_path):
    names = ("features.jsonl", "labels.jsonl", "predictions.jsonl")
    for name in names:
        (tmp_path / name).write_text(
            json.dumps({"sample_id": "committed"})
            + "\n"
            + json.dumps({"sample_id": "dangling"})
            + "\n",
            encoding="utf-8",
        )
    with (tmp_path / names[0]).open("a", encoding="utf-8") as handle:
        handle.write('{"sample_id": "half')
    (tmp_path / COMMIT_FILENAME).write_text(
        json.dumps({"sample_id": "committed"}) + "\n", encoding="utf-8"
    )
    with (tmp_path / COMMIT_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write('{"sample_id": "half')
    assert recover_committed_jsonl_prefix(tmp_path, names) == ["committed"]
    assert all(
        [record["sample_id"] for record in read_jsonl(tmp_path / name)]
        == ["committed"]
        for name in names
    )
    assert [record["sample_id"] for record in read_jsonl(tmp_path / COMMIT_FILENAME)] == [
        "committed"
    ]
