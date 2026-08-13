from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from d1_reranking import lightweight_audits
from unified_reranking.hashing import canonical_sha256, sha256_file


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _content(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_five_lightweight_audits_are_source_bound_and_resume_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).resolve().parents[2]
    evaluator_source = (
        repository
        / "runs/fair_unified_reranking_20260809_103012/configs/canonical_evaluator.py"
    )
    evaluator = tmp_path / "configs/canonical_evaluator.py"
    evaluator.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(evaluator_source, evaluator)
    artifact = tmp_path / "candidate.bin"
    artifact.write_bytes(b"synthetic candidate artifact")
    for relative in lightweight_audits.CANDIDATE_MANIFESTS.values():
        _content(
            tmp_path / relative,
            {
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "source_signature_sha256": "a" * 64,
                "artifacts": {
                    pool: _record(artifact) for pool in ("top5", "top10", "allnms")
                },
            },
        )
    closure_path = _content(
        tmp_path / "closure.json",
        {
            "status": "PASS",
            "test": {"model_contract": [{"checkpoint_sha256": "b" * 64}]},
            "artifacts": {"evaluator": _record(evaluator)},
        },
    )
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        lightweight_audits,
        "load_source_closure",
        lambda _root: (closure_path, closure),
    )

    result = lightweight_audits.assemble_lightweight_audits(tmp_path)
    assert result["status"] == "PASS"
    assert set(result["artifacts"]) == set(lightweight_audits.AUDIT_PATHS)
    assert all(
        (tmp_path / path).is_file() for path in lightweight_audits.AUDIT_PATHS.values()
    )
    assert (
        tmp_path / lightweight_audits.AUDIT_PATHS["candidate_geometry_visual"]
    ).stat().st_size > 1000
    evaluator_hash = json.loads(
        (tmp_path / lightweight_audits.AUDIT_PATHS["evaluator_hash"]).read_text(
            encoding="utf-8"
        )
    )
    assert all(row["status"] == "PASS" for row in evaluator_hash["replay"])
    assert (
        lightweight_audits.assemble_lightweight_audits(tmp_path, resume=True) == result
    )
    artifact.write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        lightweight_audits.assemble_lightweight_audits(tmp_path, resume=True)
