import hashlib
import json
from pathlib import Path

import pandas as pd

from unified_reranking.feature_cache import common_asset_records, valid_common_shard_manifest
from unified_reranking.hashing import canonical_sha256, sha256_file


def _asset(path: Path, payload: bytes) -> tuple[str, str]:
    path.write_bytes(payload)
    return str(path), sha256_file(path)


def test_common_asset_identity_rejects_referenced_byte_drift(tmp_path: Path):
    rgb, rgb_sha = _asset(tmp_path / "rgb", b"rgb")
    depth, depth_sha = _asset(tmp_path / "depth", b"depth")
    mask, mask_sha = _asset(tmp_path / "mask", b"mask")
    probability, probability_sha = _asset(tmp_path / "probability", b"probability")
    row = {
        "sample_id": "s",
        "language": "pick item",
        "language_sha256": hashlib.sha256(b"pick item").hexdigest(),
        "source_rgb_path": rgb,
        "source_rgb_sha256": rgb_sha,
        "source_depth_path": depth,
        "source_depth_sha256": depth_sha,
        "predicted_mask_path": mask,
        "predicted_mask_sha256": mask_sha,
        "predicted_probability_path": probability,
        "predicted_probability_sha256": probability_sha,
    }
    assert common_asset_records([row], {})[0]["sample_id"] == "s"
    Path(rgb).write_bytes(b"drift")
    try:
        common_asset_records([row], {})
    except RuntimeError as error:
        assert "asset hash drift" in str(error)
    else:
        raise AssertionError("referenced RGB drift was accepted")


def test_common_resume_rejects_tampered_child_artifact(tmp_path: Path):
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    frames = {
        "candidate_features": pd.DataFrame({"sample_id": ["s"], "candidate_id": ["c"], "x": [1.0]}),
        "candidate_relations": pd.DataFrame(
            {
                "sample_id": ["s"],
                "source_candidate_id": ["c"],
                "target_candidate_id": ["d"],
                "x": [1.0],
            }
        ),
        "sample_context": pd.DataFrame({"sample_id": ["s"], "x": [1.0]}),
    }
    key_columns = {
        "candidate_features": ["sample_id", "candidate_id"],
        "candidate_relations": ["sample_id", "source_candidate_id", "target_candidate_id"],
        "sample_context": ["sample_id"],
    }
    records = {}
    for name, frame in frames.items():
        path = chunk / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        keys = frame[key_columns[name]].astype(str).sort_values(key_columns[name]).to_dict("records")
        records[name] = {
            "sha256": sha256_file(path),
            "rows": len(frame),
            "key_sha256": canonical_sha256(keys),
        }
    marker = chunk / "manifest.json"
    marker.write_text(json.dumps({"status": "COMPLETE", "signature": "v1", "artifacts": records}))
    assert valid_common_shard_manifest(marker, {"signature": "v1"})
    frames["candidate_features"].assign(x=9.0).to_parquet(
        chunk / "candidate_features.parquet", index=False
    )
    assert not valid_common_shard_manifest(marker, {"signature": "v1"})
