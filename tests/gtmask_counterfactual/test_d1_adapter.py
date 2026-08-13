from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
import pandas as pd

from gtmask_counterfactual.d1_adapter import (
    BYTE_IDENTICAL_MEMBERS,
    CASE_B_MANIFEST_NAME,
    CandidateInventory,
    D1AdapterError,
    ORACLE_MASK_SOURCE,
    assemble_d1_scored_outputs,
    build_frozen_scorer_command,
    build_gt_candidate_command,
    build_isolated_gt_bundle_root,
    build_predicted_replay_command,
    candidate_inventory_from_summary,
    machine_blocker_payload,
    make_oracle_bundle_index_class,
    verify_isolated_bundle_root,
)
from gtmask_counterfactual.execution import FROZEN_D1_CANDIDATE_SCRIPT
from gtmask_counterfactual.io import canonical_sha256, sha256_file

_TOOL_PATH = (
    Path(__file__).resolve().parents[2] / "tools/gtmask_counterfactual/run_d1_case_b.py"
)
_TOOL_SPEC = importlib.util.spec_from_file_location("_test_run_d1_case_b", _TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
run_d1_case_b = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = run_d1_case_b
_TOOL_SPEC.loader.exec_module(run_d1_case_b)


SAMPLE_ID = "q0000000_deadbeef"


def _write_checksums(bundle: Path) -> None:
    names = (
        "color.png",
        "depth.png",
        "target_mask.png",
        "target_probability.npy",
        "language.txt",
        "intrinsics.json",
        "metadata.json",
    )
    (bundle / "checksums.sha256").write_text(
        "".join(f"{sha256_file(bundle / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    predicted = tmp_path / "predicted"
    bundle = predicted / SAMPLE_ID
    bundle.mkdir(parents=True)
    fixed = {
        "color.png": b"synthetic-rgb",
        "depth.png": b"synthetic-depth",
        "target_mask.png": b"predicted-mask",
        "target_probability.npy": b"predicted-probability",
        "language.txt": b"Grasp the synthetic object\n",
        "intrinsics.json": b'{"depth_scale":1000}\n',
    }
    for name, data in fixed.items():
        (bundle / name).write_bytes(data)
    metadata = {
        "sample_id": SAMPLE_ID,
        "sample_index": 0,
        "question_index": 0,
        "scene_id": "synthetic/scene.png",
        "query": "Grasp the synthetic object",
        "ready": True,
        "ready_for_anygrasp": True,
        "blockers": [],
        "mask_source": "predicted_mask_original_resolution",
        "oracle_artifacts_exported": False,
        "output_bundle": str(bundle),
        "prediction_mask": str(bundle / "target_mask.png"),
        "prediction_mask_sha256": sha256_file(bundle / "target_mask.png"),
        "materialization": {"target_mask.png": "copy"},
        "source_rgb": str(tmp_path / "dataset/rgb.png"),
        "source_depth": str(tmp_path / "dataset/depth.png"),
        "source_pcd": str(tmp_path / "dataset/pcd.pcd"),
    }
    (bundle / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_checksums(bundle)
    manifest_row = {
        "sample_id": SAMPLE_ID,
        "sample_index": 0,
        "question_index": 0,
        "scene_id": "synthetic/scene.png",
        "query": "Grasp the synthetic object",
        "bundle_dir": str(bundle),
        "ready": True,
        "ready_for_anygrasp": True,
        "blockers": [],
    }
    (predicted / "manifest.jsonl").write_text(
        json.dumps(manifest_row, sort_keys=True) + "\n", encoding="utf-8"
    )

    gt_mask = tmp_path / "gt-mask.png"
    gt_mask.write_bytes(b"locked-original-resolution-gt-mask")
    registry = tmp_path / "registry.parquet"
    registry.write_bytes(b"synthetic-locked-registry")
    registry_row = {
        "sample_id": SAMPLE_ID,
        "mapping_status": "PASS",
        "pixel_qa_status": "P2_MAPPING_QA_PASS",
        "bulk_gt_pixels_read": True,
        "original_gt_mask_path": str(gt_mask),
        "original_gt_mask_sha256": sha256_file(gt_mask),
    }
    authority = {
        "gt_candidate_generation_authorized": True,
        "gt_mask_registry": {
            "path": str(registry.resolve()),
            "sha256": sha256_file(registry),
        },
        "routes": {
            "d1": {
                "case": "B",
                "mask_affects_raw_sampling": True,
                "raw_candidate_regeneration_required": True,
                "filter_only_primary_allowed": False,
            }
        },
    }
    loader = tmp_path / "frozen_loader.py"
    loader.write_text("# synthetic frozen loader identity\n", encoding="utf-8")
    return {
        "predicted": predicted,
        "predicted_bundle": bundle,
        "metadata": metadata,
        "gt_mask": gt_mask,
        "registry": registry,
        "registry_row": registry_row,
        "authority": authority,
        "loader": loader,
    }


def _build(tmp_path: Path, values: dict[str, object]) -> Path:
    output = tmp_path / "oracle"
    build_isolated_gt_bundle_root(
        predicted_root=values["predicted"],
        output_root=output,
        registry_path=values["registry"],
        registry_rows=[values["registry_row"]],
        authority=values["authority"],
        expected_count=1,
        frozen_loader_source=values["loader"],
        expected_loader_sha256=sha256_file(values["loader"]),
    )
    return output


def test_isolated_bundle_replaces_only_mask_and_provenance(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    output = _build(tmp_path, values)
    source = values["predicted_bundle"]
    destination = output / SAMPLE_ID

    for name in BYTE_IDENTICAL_MEMBERS:
        assert (destination / name).read_bytes() == (source / name).read_bytes()
        assert os.stat(destination / name).st_ino == os.stat(source / name).st_ino
    assert (destination / "target_mask.png").read_bytes() == values[
        "gt_mask"
    ].read_bytes()
    assert (destination / "target_mask.png").read_bytes() != (
        source / "target_mask.png"
    ).read_bytes()
    assert (
        os.stat(destination / "target_mask.png").st_ino
        == os.stat(values["gt_mask"]).st_ino
    )

    original = values["metadata"]
    oracle = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    changed = {
        "mask_source",
        "oracle_artifacts_exported",
        "output_bundle",
        "prediction_mask",
        "prediction_mask_sha256",
        "materialization",
        "d1_case_b_oracle_provenance",
    }
    assert {key: value for key, value in oracle.items() if key not in changed} == {
        key: value for key, value in original.items() if key not in changed
    }
    assert oracle["mask_source"] == ORACLE_MASK_SOURCE
    assert oracle["oracle_artifacts_exported"] is True
    assert oracle["d1_case_b_oracle_provenance"]["filter_only_primary_allowed"] is False
    assert oracle["d1_case_b_oracle_provenance"]["probability_input_allowed"] is False

    manifest = verify_isolated_bundle_root(
        output, expected_loader_sha256=sha256_file(values["loader"])
    )
    assert manifest["sample_count"] == 1
    assert manifest["hardlink_copy_fallback_allowed"] is False
    assert (output / CASE_B_MANIFEST_NAME).is_file()


def test_isolated_bundle_executes_only_p2_pass_partition(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    unresolved_id = "q0000001_feedbeef"
    predicted = values["predicted"]
    source_bundle = values["predicted_bundle"]
    unresolved_bundle = predicted / unresolved_id
    unresolved_bundle.mkdir()
    for member in source_bundle.iterdir():
        unresolved_bundle.joinpath(member.name).write_bytes(member.read_bytes())
    metadata = json.loads(
        unresolved_bundle.joinpath("metadata.json").read_text(encoding="utf-8")
    )
    metadata["sample_id"] = unresolved_id
    metadata["output_bundle"] = str(unresolved_bundle)
    metadata["prediction_mask"] = str(unresolved_bundle / "target_mask.png")
    unresolved_bundle.joinpath("metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_checksums(unresolved_bundle)
    manifest_path = predicted / "manifest.jsonl"
    second_row = {
        "sample_id": unresolved_id,
        "sample_index": 1,
        "question_index": 1,
        "scene_id": "synthetic/scene.png",
        "query": "Grasp the unresolved object",
        "bundle_dir": str(unresolved_bundle),
        "ready": True,
        "ready_for_anygrasp": True,
        "blockers": [],
    }
    with manifest_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(second_row, sort_keys=True) + "\n")
    unresolved_row = {
        "sample_id": unresolved_id,
        "mapping_status": "UNRESOLVED",
        "pixel_qa_status": "NOT_EVALUABLE",
        "bulk_gt_pixels_read": False,
        "original_gt_mask_path": None,
        "original_gt_mask_sha256": None,
    }
    output = tmp_path / "oracle"
    build_isolated_gt_bundle_root(
        predicted_root=predicted,
        output_root=output,
        registry_path=values["registry"],
        registry_rows=[values["registry_row"], unresolved_row],
        authority=values["authority"],
        expected_count=2,
        frozen_loader_source=values["loader"],
        expected_loader_sha256=sha256_file(values["loader"]),
    )
    manifest = verify_isolated_bundle_root(
        output, expected_loader_sha256=sha256_file(values["loader"])
    )
    assert manifest["denominator_sample_count"] == 2
    assert manifest["evaluable_sample_count"] == 1
    assert manifest["unresolved_sample_count"] == 1
    assert manifest["sample_count"] == 1
    assert [row["sample_id"] for row in manifest["unresolved_samples"]] == [
        unresolved_id
    ]
    assert not (output / unresolved_id).exists()


@dataclass(frozen=True)
class _FakeSample:
    bundle_dir: Path
    metadata: dict[str, object]


class _FakeFrozenIndex:
    def __init__(self, dataset_root: Path, mask_root: Path, *, split: str = "test"):
        del dataset_root
        assert split == "test"
        self.mask_root = Path(mask_root)
        self.row = json.loads(
            (self.mask_root / "manifest.jsonl").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (self.mask_root / self.row["sample_id"] / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["mask_source"] == "predicted_mask_original_resolution"
        assert metadata["oracle_artifacts_exported"] is False

    def load_sample(self, sample_id: str, **kwargs: object) -> _FakeSample:
        assert kwargs["mask_source"] == "binary_prediction"
        bundle = self.mask_root / sample_id
        return _FakeSample(
            bundle_dir=bundle,
            metadata=json.loads((bundle / "metadata.json").read_text(encoding="utf-8")),
        )


def test_loader_reuses_frozen_class_with_only_flag_normalisation(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    loader_source = Path(__file__).resolve()
    output = tmp_path / "oracle"
    build_isolated_gt_bundle_root(
        predicted_root=values["predicted"],
        output_root=output,
        registry_path=values["registry"],
        registry_rows=[values["registry_row"]],
        authority=values["authority"],
        expected_count=1,
        frozen_loader_source=loader_source,
        expected_loader_sha256=sha256_file(loader_source),
    )
    fake_module = ModuleType("synthetic_frozen_adapter")
    fake_module.OcidVlgBundleIndex = _FakeFrozenIndex
    oracle_index = make_oracle_bundle_index_class(
        fake_module, expected_loader_sha256=sha256_file(loader_source)
    )
    index = oracle_index(tmp_path / "dataset", output)
    sample = index.load_sample(
        SAMPLE_ID, camera_frame="camera", mask_source="binary_prediction"
    )
    assert sample.bundle_dir == output / SAMPLE_ID
    assert sample.metadata["mask_source"] == ORACLE_MASK_SOURCE
    assert sample.metadata["oracle_artifacts_exported"] is True
    with pytest.raises(D1AdapterError, match="forbids predicted-probability"):
        index.load_sample(SAMPLE_ID, camera_frame="camera", mask_source="probability")


def test_unsafe_gt_symlink_is_rejected(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    linked = tmp_path / "linked-gt.png"
    linked.symlink_to(values["gt_mask"])
    row = dict(values["registry_row"])
    row["original_gt_mask_path"] = str(linked)
    with pytest.raises(D1AdapterError, match="non-symlink"):
        build_isolated_gt_bundle_root(
            predicted_root=values["predicted"],
            output_root=tmp_path / "oracle",
            registry_path=values["registry"],
            registry_rows=[row],
            authority=values["authority"],
            expected_count=1,
            frozen_loader_source=values["loader"],
            expected_loader_sha256=sha256_file(values["loader"]),
        )


def test_frozen_commands_keep_replay_and_gt_bulk_boundaries(tmp_path: Path) -> None:
    launcher = (
        Path(__file__).resolve().parents[2]
        / "tools/gtmask_counterfactual/run_d1_case_b.py"
    )
    predicted = build_predicted_replay_command(
        python=Path(os.sys.executable),
        dataset_root=tmp_path / "dataset",
        predicted_root=tmp_path / "predicted",
        output_root=tmp_path / "predicted-replay",
    )
    assert predicted[1] == str(FROZEN_D1_CANDIDATE_SCRIPT)
    assert "--registry" not in predicted

    gt = build_gt_candidate_command(
        python=Path(os.sys.executable),
        launcher=launcher,
        run_dir=tmp_path / "run",
        protocol_lock=tmp_path
        / "run/01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json",
        registry_path=tmp_path / "registry.parquet",
        resource_gate=tmp_path / "gate.json",
        dataset_root=tmp_path / "dataset",
        bundle_root=tmp_path / "oracle",
        output_root=tmp_path / "candidates",
    )
    assert gt[2] == "execute-candidates"
    assert "--registry" in gt
    assert "--bundle-root" in gt
    assert "filter" not in " ".join(gt).lower()

    scorer = build_frozen_scorer_command(
        docker=Path(os.sys.executable),
        candidate_root=tmp_path / "candidates",
        output_root=tmp_path / "scores",
        inventory=CandidateInventory(samples=2, nonempty=1, empty=1, candidates=3),
    )
    joined = " ".join(scorer)
    assert "--network none" in joined
    assert ":/candidates:ro" in joined
    assert "scripts/run_full_gqcnn_scoring.py" in joined
    assert "--expected-candidates 3" in joined


def test_candidate_inventory_and_machine_blocker(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    (candidate_root / "summary.csv").write_text(
        "sample_id,status,post_nms_count\n"
        "sample-a,success_nonempty,3\n"
        "sample-b,success_empty,0\n",
        encoding="utf-8",
    )
    inventory = candidate_inventory_from_summary(candidate_root)
    assert inventory == CandidateInventory(samples=2, nonempty=1, empty=1, candidates=3)
    blocker = machine_blocker_payload(
        blocker_code="D1_DOCKER_CLI_MISSING",
        detail="docker absent",
        resume_command=["python", "run_d1_case_b.py", "execute-candidates"],
        docker_probe={"status": "BLOCKED", "execution_attempted": False},
    )
    assert blocker["status"] == "MACHINE_BLOCKED"
    assert blocker["filter_only_primary_allowed"] is False
    assert blocker["execution_attempted"] is False
    unsigned = {key: value for key, value in blocker.items() if key != "content_sha256"}
    assert blocker["content_sha256"] == canonical_sha256(unsigned)


def test_scored_assembler_preserves_denominator_and_q_order(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    unresolved_id = "q0000001_feedbeef"
    predicted = values["predicted"]
    source_bundle = values["predicted_bundle"]
    unresolved_bundle = predicted / unresolved_id
    unresolved_bundle.mkdir()
    for member in source_bundle.iterdir():
        unresolved_bundle.joinpath(member.name).write_bytes(member.read_bytes())
    metadata = json.loads(
        unresolved_bundle.joinpath("metadata.json").read_text(encoding="utf-8")
    )
    metadata.update(
        {
            "sample_id": unresolved_id,
            "output_bundle": str(unresolved_bundle),
            "prediction_mask": str(unresolved_bundle / "target_mask.png"),
        }
    )
    unresolved_bundle.joinpath("metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_checksums(unresolved_bundle)
    with predicted.joinpath("manifest.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "sample_id": unresolved_id,
                    "sample_index": 1,
                    "question_index": 1,
                    "scene_id": "synthetic/scene.png",
                    "query": "unresolved",
                    "bundle_dir": str(unresolved_bundle),
                    "ready": True,
                    "ready_for_anygrasp": True,
                    "blockers": [],
                },
                sort_keys=True,
            )
            + "\n"
        )
    run = tmp_path / "run"
    bundle_root = run / "logs/d1_case_b/gt_oracle/bundles"
    build_isolated_gt_bundle_root(
        predicted_root=predicted,
        output_root=bundle_root,
        registry_path=values["registry"],
        registry_rows=[
            values["registry_row"],
            {
                "sample_id": unresolved_id,
                "mapping_status": "UNRESOLVED",
                "pixel_qa_status": "NOT_EVALUABLE",
                "bulk_gt_pixels_read": False,
                "original_gt_mask_path": None,
                "original_gt_mask_sha256": None,
            },
        ],
        authority=values["authority"],
        expected_count=2,
        frozen_loader_source=values["loader"],
        expected_loader_sha256=sha256_file(values["loader"]),
    )
    protocol = run / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json"
    claim = run / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}\n", encoding="utf-8")
    claim.write_text("{}\n", encoding="utf-8")
    candidate_root = run / "logs/d1_case_b/gt_oracle/candidates"
    candidate_root.mkdir(parents=True)
    candidate_root.joinpath("summary.csv").write_text(
        "sample_id,status,post_nms_count\n"
        f"{SAMPLE_ID},success_nonempty,2\n",
        encoding="utf-8",
    )
    candidate_root.joinpath("run_config.json").write_text("{}\n", encoding="utf-8")

    scored_root = run / "logs/d1_case_b/gt_oracle/scores"
    sample_root = scored_root / SAMPLE_ID
    sample_root.mkdir(parents=True)
    candidates = [
        {
            "sample_id": SAMPLE_ID,
            "candidate_id": "source-b",
            "source_candidate_index": 1,
            "gqcnn_rank": 1,
            "gqcnn_q_value": 0.9,
            "center_u_px": 20.0,
            "center_v_px": 30.0,
            "angle_deg": 10.0,
            "width_px": 40.0,
        },
        {
            "sample_id": SAMPLE_ID,
            "candidate_id": "source-a",
            "source_candidate_index": 0,
            "gqcnn_rank": 2,
            "gqcnn_q_value": 0.8,
            "center_u_px": 21.0,
            "center_v_px": 31.0,
            "angle_rad": 0.2,
            "width_px": 42.0,
        },
    ]
    payload = {
        "metadata": {
            "sample_id": SAMPLE_ID,
            "scoring_status": "scored_nonempty",
        },
        "candidates": candidates,
    }
    sample_root.joinpath("gqcnn_scored_candidates.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    required = {
        "gqcnn_scored_candidates.npz": b"npz",
        "gqcnn_scored_candidates.csv": b"csv",
        "gqcnn_top1.json": b"{}\n",
        "gqcnn_top5.json": b"{}\n",
        "scoring_metadata.json": b"{}\n",
    }
    for name, content in required.items():
        sample_root.joinpath(name).write_bytes(content)
    required_names = sorted(
        ["gqcnn_scored_candidates.json", *required]
    )
    marker = {
        "sample_id": SAMPLE_ID,
        "scoring_status": "scored_nonempty",
        "source_candidate_count": 2,
        "gqcnn_scored_count": 2,
        "top1_candidate_id": "source-b",
        "source_candidate_sha256": "candidate-source-sha",
        "model_config_hash": "model-config-sha",
        "required_files": required_names,
        "required_file_hashes": {
            name: sha256_file(sample_root / name) for name in required_names
        },
    }
    sample_root.joinpath("_SCORING_COMPLETE.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    scored_root.joinpath("run_config.json").write_text("{}\n", encoding="utf-8")
    scored_root.joinpath("progress.json").write_text(
        json.dumps(
            {
                "total_samples": 1,
                "terminal_samples": 1,
                "completed_nonempty_samples": 1,
                "skipped_empty_samples": 0,
                "failed_samples": 0,
                "scored_candidates": 2,
                "remaining_candidates": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scored_root.joinpath("run_statistics.json").write_text(
        json.dumps(
            {
                "total_samples": 1,
                "terminal_samples": 1,
                "scored_nonempty_samples": 1,
                "skipped_valid_empty_samples": 0,
                "failed_samples": 0,
                "corrupt_committed_samples": 0,
                "expected_candidates": 2,
                "scored_candidates": 2,
                "finite_q_values": 2,
                "invalid_q_values": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scored_root.joinpath("summary.csv").write_text(
        "sample_id,scoring_status,scored_candidate_count\n"
        f"{SAMPLE_ID},scored_nonempty,2\n",
        encoding="utf-8",
    )
    scored_root.joinpath("scoring_manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": SAMPLE_ID,
                "scoring_status": "scored_nonempty",
                "source_candidate_count": 2,
                "gqcnn_scored_count": 2,
                "top1_candidate_id": "source-b",
                "source_candidate_sha256": "candidate-source-sha",
                "model_config_hash": "model-config-sha",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = assemble_d1_scored_outputs(
        run_dir=run,
        bundle_root=bundle_root,
        candidate_root=candidate_root,
        scored_root=scored_root,
        protocol_lock_path=protocol,
        execution_claim_path=claim,
        expected_denominator_count=2,
        expected_loader_sha256=sha256_file(values["loader"]),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sample_count"] == 2
    assert manifest["evaluable_sample_count"] == 1
    assert manifest["technical_complement_count"] == 1
    output_candidates = pd.read_parquet(manifest["candidates"]["path"])
    output_samples = pd.read_parquet(manifest["per_sample"]["path"])
    assert output_candidates["native_score"].tolist() == [0.9, 0.8]
    assert output_candidates["source_candidate_id"].tolist() == [
        "source-b",
        "source-a",
    ]
    assert output_candidates["candidate_id"].str.fullmatch(r"[0-9a-f]{64}").all()
    unresolved = output_samples.loc[output_samples["sample_id"].eq(unresolved_id)].iloc[0]
    assert unresolved["technical_failure"]
    assert unresolved["candidate_count"] == 0
    assert unresolved["status"] == "TECHNICAL_FAILURE"

    # Resume is exact; rebinding a changed scorer payload is rejected before
    # any canonical output can be reused.
    payload["candidates"][0]["gqcnn_q_value"] = 0.7
    sample_root.joinpath("gqcnn_scored_candidates.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(D1AdapterError, match="member hash differs"):
        assemble_d1_scored_outputs(
            run_dir=run,
            bundle_root=bundle_root,
            candidate_root=candidate_root,
            scored_root=scored_root,
            protocol_lock_path=protocol,
            execution_claim_path=claim,
            expected_denominator_count=2,
            expected_loader_sha256=sha256_file(values["loader"]),
        )


@pytest.mark.parametrize(
    "entrypoint",
    [run_d1_case_b.execute_candidates, run_d1_case_b.scorer],
)
def test_heavy_entrypoints_lock_before_revalidating_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entrypoint: object
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_lock(run_dir: Path, *, purpose: str):
        del run_dir, purpose
        events.append("lock")
        try:
            yield tmp_path / ".d1_heavy_resource.lock"
        finally:
            events.append("unlock")

    class GateStop(RuntimeError):
        pass

    def stop_at_gate(path: Path) -> None:
        del path
        assert events == ["lock"]
        events.append("gate")
        raise GateStop

    monkeypatch.setattr(run_d1_case_b, "_authorise", lambda args: ({}, {}))
    monkeypatch.setattr(run_d1_case_b, "exclusive_d1_flock", fake_lock)
    monkeypatch.setattr(run_d1_case_b, "_fresh_gate", stop_at_gate)
    args = Namespace(
        run_dir=tmp_path / "run",
        resource_gate=tmp_path / "gate.json",
        execute=True,
    )
    with pytest.raises(GateStop):
        entrypoint(args)
    assert events == ["lock", "gate", "unlock"]
