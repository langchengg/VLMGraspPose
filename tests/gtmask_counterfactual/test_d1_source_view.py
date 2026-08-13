from __future__ import annotations

from pathlib import Path

import pytest

from gtmask_counterfactual.d1_source_view import (
    D1SourceViewError,
    build_d1_source_view,
    production_source_files,
    verify_d1_source_view,
)
from gtmask_counterfactual.io import sha256_file


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


def test_source_view_rejects_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(D1SourceViewError, match="unsafe"):
        build_d1_source_view(
            tmp_path / "view", source_files={"../escape.py": source}
        )


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
    ) == "3412992989cf459c73f9d36f926d64356d95f0e1b52b42c56429c1bc66159fd6"
