from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from gtmask_counterfactual.d1_source_view import (
    D1SourceViewError,
    build_d1_source_view,
    d1_source_view_record,
    production_source_files,
    verify_d1_execution_binding,
    verify_d1_source_view,
    write_d1_execution_binding,
)
from gtmask_counterfactual.d1_adapter import ORACLE_MASK_SOURCE
from gtmask_counterfactual.io import sha256_file


_BOOTSTRAP_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/gtmask_counterfactual/d1_oracle_candidate_bootstrap.py"
)
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "_test_gtmask_oracle_bootstrap", _BOOTSTRAP_PATH
)
assert _BOOTSTRAP_SPEC is not None and _BOOTSTRAP_SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
sys.modules[_BOOTSTRAP_SPEC.name] = bootstrap
_BOOTSTRAP_SPEC.loader.exec_module(bootstrap)


def test_oracle_bootstrap_metadata_contract_matches_parent_adapter() -> None:
    assert bootstrap.ORACLE_MASK_SOURCE == ORACLE_MASK_SOURCE


def test_source_view_is_content_addressed_exact_and_non_symlink(tmp_path: Path) -> None:
    first = tmp_path / "candidate.py"
    second = tmp_path / "scorer.py"
    first.write_text("CANDIDATE = 1\n", encoding="utf-8")
    second.write_text("SCORER = 2\n", encoding="utf-8")
    manifest = build_d1_source_view(
        tmp_path / "view",
        source_files={
            "scripts/candidate.py": first,
            "scripts/scorer.py": second,
        },
    )
    value = verify_d1_source_view(manifest)
    assert bootstrap.verify_source_view(
        manifest.parent, manifest_sha256=sha256_file(manifest)
    )["content_sha256"] == value["content_sha256"]
    assert value["file_count"] == 2
    assert not manifest.parent.joinpath("scripts/candidate.py").is_symlink()
    assert build_d1_source_view(
        tmp_path / "view",
        source_files={
            "scripts/candidate.py": first,
            "scripts/scorer.py": second,
        },
    ) == manifest

    manifest.parent.joinpath("scripts/scorer.py").write_text(
        "SCORER = 3\n", encoding="utf-8"
    )
    with pytest.raises(D1SourceViewError, match="member differs"):
        verify_d1_source_view(manifest)
    with pytest.raises(bootstrap.OracleBootstrapError, match="member differs"):
        bootstrap.verify_source_view(
            manifest.parent, manifest_sha256=sha256_file(manifest)
        )


def test_source_view_rejects_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(D1SourceViewError, match="unsafe"):
        build_d1_source_view(
            tmp_path / "view", source_files={"../escape.py": source}
        )


def test_bootstrap_allows_only_explicit_content_exact_mount_relocation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_d1_source_view(
        tmp_path / "view", source_files={"scripts/source.py": source}
    )
    relocated = tmp_path / "workspace"
    shutil.copytree(manifest.parent, relocated)
    relocated_manifest = relocated / manifest.name

    with pytest.raises(bootstrap.OracleBootstrapError, match="identity differs"):
        bootstrap.verify_source_view(
            relocated,
            manifest_sha256=sha256_file(relocated_manifest),
        )
    verified = bootstrap.verify_source_view(
        relocated,
        manifest_sha256=sha256_file(relocated_manifest),
        allow_relocated_source_view=True,
    )
    assert verified["source_identity_sha256"] == json.loads(
        manifest.read_text(encoding="utf-8")
    )["source_identity_sha256"]


def test_source_view_rejects_unlisted_member_and_recomputed_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_d1_source_view(
        tmp_path / "view", source_files={"scripts/source.py": source}
    )
    manifest.parent.joinpath("scripts/injected.py").write_text(
        "INJECTED = True\n", encoding="utf-8"
    )
    with pytest.raises(D1SourceViewError, match="inventory differs"):
        verify_d1_source_view(manifest)
    with pytest.raises(bootstrap.OracleBootstrapError, match="inventory differs"):
        bootstrap.verify_source_view(
            manifest.parent, manifest_sha256=sha256_file(manifest)
        )

    manifest.parent.joinpath("scripts/injected.py").unlink()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["source_identity_sha256"] = "0" * 64
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = bootstrap.canonical_sha256(unsigned)
    manifest.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(D1SourceViewError, match="identity differs"):
        verify_d1_source_view(manifest)
    with pytest.raises(bootstrap.OracleBootstrapError, match="identity differs"):
        bootstrap.verify_source_view(
            manifest.parent, manifest_sha256=sha256_file(manifest)
        )


def test_execution_binding_independently_revalidates_source_and_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_d1_source_view(
        tmp_path / "view", source_files={"scripts/source.py": source}
    )
    output = tmp_path / "run_config.json"
    output.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    binding = write_d1_execution_binding(
        tmp_path / "SOURCE_VIEW_EXECUTION_MANIFEST.json",
        stage="test/scoring",
        source_view_manifest=manifest,
        artifacts={"run_config": output},
    )
    verified = verify_d1_execution_binding(binding)
    assert verified["source_view"] == d1_source_view_record(manifest)
    assert verified["source_view"]["file_count"] == 1
    assert verified["source_view"]["members"] == {
        "scripts/source.py": {
            "bytes": len("VALUE = 1\n"),
            "sha256": sha256_file(manifest.parent / "scripts/source.py"),
        }
    }

    output.write_text('{"status":"CHANGED"}\n', encoding="utf-8")
    with pytest.raises(D1SourceViewError, match="bound artifact differs"):
        verify_d1_execution_binding(binding)

    output.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    manifest.parent.joinpath("scripts/source.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(D1SourceViewError, match="member differs"):
        verify_d1_execution_binding(binding)


def test_production_source_view_includes_oracle_bootstrap_and_transitive_runtime() -> None:
    files = production_source_files()
    required = {
        "scripts/gtmask_oracle_candidate_bootstrap.py",
        "scripts/run_hifics_dexnet_candidates.py",
        "scripts/run_full_gqcnn_scoring.py",
        "scripts/score_existing_dexnet_candidates.py",
        "src/grasping/camera_geometry.py",
        "src/grasping/dexnet_scoring.py",
        "src/grasping/grasp_visualization.py",
        "third_party/gqcnn-official/gqcnn/grasping/image_grasp_sampler.py",
    }
    assert required.issubset(files)
    assert all(path.is_file() and not path.is_symlink() for path in files.values())
    assert sha256_file(
        files["scripts/gtmask_oracle_candidate_bootstrap.py"]
    ) == "4300544854c78fe42aee7458dc91e15030d09f460e4c17128dfd3296b1a0f9b1"
