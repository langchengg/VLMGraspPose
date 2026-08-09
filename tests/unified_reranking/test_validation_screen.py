from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.unified_reranking.select_validation_screen import (
    _assert_reported_metrics,
    _verified_current_cell,
    method_code,
)
from unified_reranking.hashing import canonical_sha256, sha256_file


def test_validation_method_codes_preserve_controlled_factor() -> None:
    assert method_code({"method": "jaw_support"}, rule=True) == "R1_jaw_support"
    assert method_code({"encoder": "linear", "loss": "ranknet"}, rule=False) == "R2_linear_ranknet"
    assert method_code({"encoder": "mlp", "loss": "bce"}, rule=False) == "R3_mlp_bce"
    assert method_code({"encoder": "mlp", "loss": "ranknet"}, rule=False) == "R4_mlp_ranknet"
    assert method_code({"encoder": "mlp", "loss": "listwise"}, rule=False) == "R5_mlp_listwise"
    assert method_code({"encoder": "lambdamart", "loss": "lambdarank"}, rule=False) == "R6_lambdamart"
    assert (
        method_code(
            {"encoder": "mlp", "loss": "jacquard_margin_ranknet"}, rule=False
        )
        == "R7_mlp_jacquard_margin"
    )


def _cell(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    source = tmp_path / "source.parquet"
    output = tmp_path / "predictions.parquet"
    source.write_bytes(b"source")
    output.write_bytes(b"predictions")
    configuration = {
        "encoder": "mlp",
        "loss": "ranknet",
        "mode": "validation",
        "seed": 42,
        "source_identity": {
            "train_features_sha256": sha256_file(source),
            "train_labels_sha256": "b" * 64,
            "folds_sha256": "c" * 64,
        },
    }
    key = canonical_sha256(configuration)[:16]
    manifest_path = tmp_path / key / "manifest.json"
    manifest_path.parent.mkdir()
    value: dict[str, object] = {
        "status": "COMPLETE",
        "configuration": configuration,
        "cell_key": key,
        "feature_columns": ["native_score_raw"],
        "feature_schema_sha256": canonical_sha256(("native_score_raw",)),
        "sources": {
            "features": {"path": str(source), "sha256": sha256_file(source)}
        },
        "artifacts": {
            "predictions": {"path": str(output), "sha256": sha256_file(output)}
        },
    }
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    return value, manifest_path, source, output


def test_validation_screen_ignores_source_drift_but_rejects_output_tampering(
    tmp_path: Path,
) -> None:
    value, path, source, output = _cell(tmp_path)
    assert _verified_current_cell(value, path, is_rule=False)
    source.write_bytes(b"changed source")
    assert not _verified_current_cell(value, path, is_rule=False)
    source.write_bytes(b"source")
    output.write_bytes(b"tampered predictions")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _verified_current_cell(value, path, is_rule=False)


def test_validation_screen_rejects_metrics_only_manifest_tampering(
    tmp_path: Path,
) -> None:
    metrics = {"j_at_1": 0.5, "mrr_at_5": 0.7}
    _assert_reported_metrics(metrics, dict(metrics), tmp_path / "cell.json")
    with pytest.raises(RuntimeError, match="metrics do not match"):
        _assert_reported_metrics(
            {**metrics, "j_at_1": 0.9}, metrics, tmp_path / "cell.json"
        )
