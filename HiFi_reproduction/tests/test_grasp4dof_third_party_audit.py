from __future__ import annotations

import json
import subprocess
from collections import OrderedDict
from pathlib import Path

import pytest

from tools.grasp4dof import audit_third_party as audit


ROOT = Path(__file__).resolve().parents[1]
GR_REPO = ROOT / "third_party_src/grconvnet"
GG_REPO = ROOT / "third_party_src/ggcnn"
GG_CHECKPOINTS = ROOT / "third_party_src/checkpoints/ggcnn2"


def test_real_pinned_sources_and_checkpoints_load_strictly() -> None:
    gr_manifest, gr_model = audit.audit_grconvnet(GR_REPO)
    gg_manifest, gg_model = audit.audit_ggcnn2(GG_REPO, GG_CHECKPOINTS)

    assert gr_manifest["git"]["clean"] is True
    assert gr_manifest["pinned_commit"] == audit.GR_COMMIT
    assert gr_manifest["license"]["spdx"] == "BSD-3-Clause"
    assert len(gr_manifest["checkpoints"]) == 3
    assert all(row["strict_load_success"] for row in gr_manifest["checkpoints"])
    assert gr_model.conv1.in_channels == 4

    assert gg_manifest["git"]["clean"] is True
    assert gg_manifest["pinned_commit"] == audit.GG_COMMIT
    assert gg_manifest["architecture"]["bilinear_upsampling_layers"] == 2
    assert gg_manifest["architecture"]["transposed_convolution_layers"] == 0
    state_row = next(
        row for row in gg_manifest["checkpoints"] if row["checkpoint_id"] == "state_dict"
    )
    assert state_row["strict_load_success"] is True
    assert gg_model.features[0].in_channels == 1


def test_state_dict_guard_rejects_missing_key() -> None:
    _, model = audit.audit_ggcnn2(GG_REPO, GG_CHECKPOINTS)
    state = OrderedDict(model.state_dict())
    state.pop(next(iter(state)))
    with pytest.raises(ValueError, match="key/shape mismatch"):
        audit.validate_state_dict_shapes(model, state, label="tampered")


def test_git_guard_rejects_dirty_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "audit@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Audit Test"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin", audit.GR_REPOSITORY], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkout is dirty"):
        audit.audit_git_checkout(
            repo, expected_remote=audit.GR_REPOSITORY, expected_commit=commit
        )


def test_cpu_smoke_and_artifact_contract(tmp_path: Path) -> None:
    result = audit.run_audit(
        run_dir=tmp_path,
        gr_repo=GR_REPO,
        gg_repo=GG_REPO,
        gg_checkpoint_root=GG_CHECKPOINTS,
        include_mps=False,
    )
    assert result["status"] == "PASS"
    expected = [
        tmp_path / "third_party/grconvnet_source_manifest.json",
        tmp_path / "third_party/ggcnn2_source_manifest.json",
        tmp_path / "environment.json",
        tmp_path / "package_lock.txt",
        tmp_path / "audit/basic_device_smoke.json",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)

    smoke = json.loads(expected[-1].read_text(encoding="utf-8"))
    assert smoke["status"] == "PASS"
    assert len(smoke["entries"]) == 2
    assert {tuple(row["output_shapes"][0]) for row in smoke["entries"]} == {
        (1, 1, 224, 224),
        (1, 1, 300, 300),
    }
    environment = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    anomaly = environment["environment_setup_anomaly"]
    assert anomaly["status"] == "corrected_before_model_smoke"
    assert anomaly["persistent_runtime_fault"] is False

    gr_manifest = json.loads(expected[0].read_text(encoding="utf-8"))
    gg_manifest = json.loads(expected[1].read_text(encoding="utf-8"))
    assert gr_manifest["provenance_disclosures"][0]["id"] == (
        "gr_input_size_and_width_scale_ambiguity"
    )
    assert gg_manifest["provenance_disclosures"][0]["id"] == (
        "gg_release_tag_source_mismatch"
    )
