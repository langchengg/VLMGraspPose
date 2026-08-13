from __future__ import annotations

from pathlib import Path

from d1_reranking.feature_source_adapter import (
    AUDIT_TOOL_RELATIVE,
    _frozen_audit_record,
)


def test_frozen_feature_audit_record_must_be_unique() -> None:
    path = str((Path(__file__).resolve().parents[2] / AUDIT_TOOL_RELATIVE).resolve())
    plan = {"sources": {"code": [{"path": path, "sha256": "a" * 64}]}}
    assert _frozen_audit_record(plan) == {"path": path, "sha256": "a" * 64}
