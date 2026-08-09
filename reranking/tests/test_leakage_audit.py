from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from reranking.leakage_audit import (
    LeakageAuditError,
    audit_development_test_identities,
    build_identity_rows,
    verify_leakage_audit_bundle,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_rows(
    identities: list[tuple[str, str, str, str, str]], *, split: str
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": query_id,
                "scene_id": scene_id,
                "frame_id": frame_id,
                "source_rgb_sha256": rgb_sha256,
                "source_depth_sha256": depth_sha256,
                "split": split,
            }
            for query_id, scene_id, frame_id, rgb_sha256, depth_sha256 in identities
        ]
    )


def _identities(
    identities: list[tuple[str, str, str, str, str]], *, split: str
) -> pd.DataFrame:
    return build_identity_rows(
        _source_rows(identities, split=split), dataset="route", split=split
    )


def test_build_identity_rows_hashes_files_and_collapses_candidate_rows(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    rgb.write_bytes(b"real-rgb-bytes")
    depth.write_bytes(b"real-depth-bytes")
    source = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "scene_id": ["scene-a", "scene-a"],
            "candidate_id": ["c1", "c2"],
            "rgb_path": [rgb.name, rgb.name],
            "depth_path": [depth.name, depth.name],
        }
    )

    rows = build_identity_rows(
        source, dataset="modular", split="train", base_dir=tmp_path
    )

    assert len(rows) == 1
    assert rows.loc[0, "frame_id"] == "scene-a"
    assert rows.loc[0, "rgb_sha256"] == _digest(rgb.read_bytes())
    assert rows.loc[0, "depth_sha256"] == _digest(depth.read_bytes())
    assert len(rows.loc[0, "identity_row_sha256"]) == 64


def test_build_identity_rows_accepts_real_manifest_aliases_and_can_verify_files(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "rgb.bin"
    depth = tmp_path / "depth.bin"
    rgb.write_bytes(b"rgb")
    depth.write_bytes(b"depth")
    source = pd.DataFrame(
        {
            "sample_id": ["q-real"],
            "scene_id": ["OCID/scene,result.png"],
            "source_rgb_path": [str(rgb)],
            "source_depth_path": [str(depth)],
            "source_rgb_sha256": [_digest(b"rgb")],
            "source_depth_sha256": [_digest(b"depth")],
        }
    )

    rows = build_identity_rows(
        source,
        dataset="modular",
        split="test",
        verify_declared_hashes=True,
    )
    assert rows.loc[0, "query_id"] == "q-real"

    source.loc[0, "source_rgb_sha256"] = "0" * 64
    with pytest.raises(LeakageAuditError, match="disagrees with file"):
        build_identity_rows(
            source,
            dataset="modular",
            split="test",
            verify_declared_hashes=True,
        )


def test_clean_audit_uses_concrete_fold_rows_and_writes_verifiable_bundle(
    tmp_path: Path,
) -> None:
    development = _identities(
        [
            ("dev-a", "scene-a", "frame-a", "1" * 64, "2" * 64),
            ("dev-b", "scene-b", "frame-b", "3" * 64, "4" * 64),
        ],
        split="development",
    )
    test = _identities(
        [("test-a", "scene-t", "frame-t", "5" * 64, "6" * 64)],
        split="test",
    )
    fit_partitions = {
        0: pd.DataFrame(
            {
                "query_id": ["dev-a", "dev-a"],
                "scene_id": ["scene-a", "scene-a"],
                "frame_id": ["frame-a", "frame-a"],
                "candidate_id": ["a", "b"],
            }
        ),
        1: pd.DataFrame(
            {
                "query_id": ["dev-b"],
                "scene_id": ["scene-b"],
                "frame_id": ["frame-b"],
            }
        ),
    }

    audit = audit_development_test_identities(
        development,
        test,
        fit_partitions=fit_partitions,
        expected_fit_partitions=[0, 1],
    )

    assert audit.passed
    assert audit.summary["fit_scope_strings_trusted"] is False
    assert audit.summary["fit_partition_count"] == 2
    assert audit.summary["fit_query_identity_count"] == 2
    assert set(audit.fold_fit_identities["evidence_source"]) == {
        "actual_fit_scene_frame"
    }
    reordered = audit_development_test_identities(
        development.iloc[::-1],
        test,
        fit_partitions={1: fit_partitions[1], 0: fit_partitions[0].iloc[::-1]},
        expected_fit_partitions=[1, 0],
    )
    assert reordered.summary["audit_digest_sha256"] == audit.summary["audit_digest_sha256"]

    written = audit.write_bundle(tmp_path / "bundle")
    checksum_lines = written[-1].read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 7
    for line in checksum_lines:
        expected, name = line.split("  ", maxsplit=1)
        assert expected == _digest((written[-1].parent / name).read_bytes())
    summary = json.loads((written[-1].parent / "summary.json").read_text())
    assert summary["audit_digest_sha256"] == audit.summary["audit_digest_sha256"]
    markdown = (written[-1].parent / "LEAKAGE_AUDIT.md").read_text()
    assert "**Overall result: PASS**" in markdown
    assert "| **development total** | **all** | **2** |" in markdown
    assert "| **test total** | **all** | **1** |" in markdown
    assert "- Fit partition completeness: **PASS**" in markdown
    assert "`identity_rows_sha256`" in markdown
    reordered.write_bundle(tmp_path / "reordered-bundle")
    assert markdown == (tmp_path / "reordered-bundle" / "LEAKAGE_AUDIT.md").read_text()
    verified = verify_leakage_audit_bundle(written[-1].parent)
    assert verified.summary == audit.summary


def test_source_overlap_reports_all_five_entity_kinds() -> None:
    shared = [("same-query", "same-scene", "same-frame", "a" * 64, "b" * 64)]
    development = _identities(shared, split="development")
    test = _identities(shared, split="test")

    audit = audit_development_test_identities(development, test)

    assert not audit.passed
    assert audit.summary["identity_overlap_counts"] == {
        "query_id": 1,
        "scene_id": 1,
        "frame_identity": 1,
        "rgb_sha256": 1,
        "depth_sha256": 1,
    }
    assert set(audit.identity_overlaps["identity_type"]) == {
        "query_id",
        "scene_id",
        "frame_identity",
        "rgb_sha256",
        "depth_sha256",
    }


def test_fit_scanner_catches_test_query_and_group_from_actual_membership() -> None:
    development = _identities(
        [("dev", "dev-scene", "dev-frame", "1" * 64, "2" * 64)],
        split="development",
    )
    test = _identities(
        [("held-out", "test-scene", "test-frame", "3" * 64, "4" * 64)],
        split="test",
    )

    audit = audit_development_test_identities(
        development,
        test,
        fit_partitions={
            "seed42-fold0": pd.DataFrame(
                {
                    "query_id": ["held-out"],
                    "scene_id": ["test-scene"],
                    "frame_id": ["test-frame"],
                }
            )
        },
    )

    assert not audit.passed
    assert audit.summary["fit_overlap_counts"] == {
        "query_id": 1,
        "group_identity": 1,
    }
    assert audit.fold_fit_identities.loc[0, "identity_source"] == "test"
    assert not bool(audit.fold_fit_identities.loc[0, "is_development_identity"])


def test_fit_scanner_rejects_scope_string_and_detects_group_mismatch() -> None:
    development = _identities(
        [("dev", "scene", "frame", "1" * 64, "2" * 64)],
        split="development",
    )
    test = _identities(
        [("test", "other", "other", "3" * 64, "4" * 64)], split="test"
    )

    with pytest.raises(LeakageAuditError, match="not a scope string"):
        audit_development_test_identities(
            development,
            test,
            fit_partitions={0: "development_train_fold_only"},
        )

    audit = audit_development_test_identities(
        development,
        test,
        fit_partitions={
            0: pd.DataFrame(
                {
                    "query_id": ["dev"],
                    "scene_id": ["wrong-scene"],
                    "frame_id": ["wrong-frame"],
                }
            )
        },
    )
    assert not audit.passed
    assert audit.summary["fit_resolution_error_count"] == 1
    assert audit.fit_resolution_errors.loc[0, "error"] == (
        "fit_group_disagrees_with_source_identity"
    )


def test_expected_fold_coverage_is_machine_checked() -> None:
    development = _identities(
        [("dev", "scene", "frame", "1" * 64, "2" * 64)],
        split="development",
    )
    test = _identities(
        [("test", "other", "other", "3" * 64, "4" * 64)], split="test"
    )

    audit = audit_development_test_identities(
        development,
        test,
        fit_partitions={0: ["dev"]},
        expected_fit_partitions=[0, 1],
    )

    assert not audit.passed
    assert audit.fit_resolution_errors.to_dict(orient="records") == [
        {"fit_partition": "1", "error": "missing_fit_partition"}
    ]


def test_source_only_bundle_is_explicit_and_fit_requirement_fails_closed(
    tmp_path: Path,
) -> None:
    development = _identities(
        [("dev", "scene", "frame", "1" * 64, "2" * 64)],
        split="development",
    )
    test = _identities(
        [("test", "other", "other", "3" * 64, "4" * 64)], split="test"
    )
    source_only = audit_development_test_identities(development, test)
    assert source_only.passed
    assert source_only.summary["fit_membership_audited"] is False
    source_only.write_bundle(tmp_path)

    with pytest.raises(LeakageAuditError, match="no complete concrete"):
        verify_leakage_audit_bundle(tmp_path)
    assert verify_leakage_audit_bundle(tmp_path, require_fit_evidence=False).passed


def test_bundle_verifier_rejects_tampering(tmp_path: Path) -> None:
    development = _identities(
        [("dev", "scene", "frame", "1" * 64, "2" * 64)],
        split="development",
    )
    test = _identities(
        [("test", "other", "other", "3" * 64, "4" * 64)], split="test"
    )
    audit = audit_development_test_identities(
        development, test, fit_partitions={0: ["dev"]}
    )
    audit.write_bundle(tmp_path)
    with (tmp_path / "fold_fit_identities.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(LeakageAuditError, match="checksum mismatch"):
        verify_leakage_audit_bundle(tmp_path)


def test_failure_markdown_has_locatable_overlap_and_partition_details(
    tmp_path: Path,
) -> None:
    shared = [("same", "scene", "frame", "1" * 64, "2" * 64)]
    development = _identities(shared, split="development")
    test = _identities(shared, split="test")
    audit = audit_development_test_identities(
        development,
        test,
        fit_partitions={
            "fold0": pd.DataFrame(
                {
                    "query_id": ["same"],
                    "scene_id": ["scene"],
                    "frame_id": ["frame"],
                }
            )
        },
        expected_fit_partitions=["fold0", "fold1"],
    )
    audit.write_bundle(tmp_path)

    markdown = (tmp_path / "LEAKAGE_AUDIT.md").read_text(encoding="utf-8")
    assert "**Overall result: FAIL**" in markdown
    for identity_type in (
        "query_id",
        "scene_id",
        "frame_identity",
        "rgb_sha256",
        "depth_sha256",
    ):
        assert f"| `{identity_type}` | 1 | FAIL |" in markdown
    assert "| route | query_id | same |" in markdown
    assert "| fold0 | 1 | 1 | 2 | 0 | FAIL |" in markdown
    assert "| fold1 | 0 | 0 | 0 | 1 | FAIL |" in markdown
    assert "missing_fit_partition" in markdown
    assert (
        verify_leakage_audit_bundle(tmp_path, require_fit_evidence=False).passed
        is False
    )


def test_verifier_rejects_markdown_tampering_even_with_updated_file_checksum(
    tmp_path: Path,
) -> None:
    development = _identities(
        [("dev", "scene", "frame", "1" * 64, "2" * 64)],
        split="development",
    )
    test = _identities(
        [("test", "other", "other", "3" * 64, "4" * 64)], split="test"
    )
    audit = audit_development_test_identities(
        development, test, fit_partitions={0: ["dev"]}
    )
    audit.write_bundle(tmp_path)
    markdown_path = tmp_path / "LEAKAGE_AUDIT.md"
    markdown_path.write_text("# forged pass\n", encoding="utf-8")
    checksum_path = tmp_path / "artifacts.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text(
        "\n".join(
            f"{_digest(markdown_path.read_bytes())}  LEAKAGE_AUDIT.md"
            if line.endswith("  LEAKAGE_AUDIT.md")
            else line
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(LeakageAuditError, match="not reproducible"):
        verify_leakage_audit_bundle(tmp_path)
