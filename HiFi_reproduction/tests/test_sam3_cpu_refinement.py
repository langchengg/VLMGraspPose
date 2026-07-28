from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.run_coarse_to_fine_cpu_pipeline import _stage_complete
from src.segmentation.sam3_cpu_model import (
    Sam3CpuInferenceResult,
    TransformersSam3Cpu,
    build_pcs_processor_inputs,
    build_tracker_processor_inputs,
    configure_cpu_runtime,
    cpu_preflight,
    import_sam3_classes,
    torch_device,
)
from src.segmentation.sam3_cpu_refiner import load_hifics_bundle, refine_cpu_sample
from src.segmentation.sam3_cpu_serialization import atomic_output_directory, sha256_file
from src.segmentation.sam3_mask_selector import select_refined_mask
from src.segmentation.sam3_prompt_builder import build_visual_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (REPO_ROOT / "configs" / "sam3_cpu_refinement.yaml").read_text(encoding="utf-8")
)
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


def probability_fixture() -> np.ndarray:
    value = np.zeros((80, 100), dtype=np.float32)
    value[20:60, 25:75] = 0.9
    return value


def prompt_fixture():
    return build_visual_prompt(probability_fixture(), CONFIG["prompt"])


def test_official_four_class_import_and_forced_cpu_contract():
    torch, transformers, _, model, processor, tracker_model, tracker_processor = (
        import_sam3_classes()
    )
    assert transformers.__version__ == "5.14.1"
    assert {item.__name__ for item in (model, processor, tracker_model, tracker_processor)} == {
        "Sam3Model",
        "Sam3Processor",
        "Sam3TrackerModel",
        "Sam3TrackerProcessor",
    }
    assert str(torch_device(torch)) == "cpu"
    preflight = cpu_preflight()
    assert preflight["requested_device"] == "cpu"
    assert preflight["requested_dtype"] == "float32"
    assert preflight["mps_available_but_unused"] is True


def test_cpu_thread_controls_are_explicit(monkeypatch: pytest.MonkeyPatch):
    for key in (
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)
    result = configure_cpu_runtime(num_threads=2, interop_threads=1)
    assert result["device"] == "cpu"
    assert result["torch_num_threads"] == 2
    assert all(value == "2" for value in result["environment_threads"].values())


def test_model_source_has_no_auto_device_or_mps_transfer():
    source = inspect.getsource(TransformersSam3Cpu)
    assert 'device_map="auto"' not in source
    assert '.to("mps")' not in source
    assert '.to("cuda")' not in source
    assert "torch.float32" in source


def test_cpu_downloader_requires_current_official_processor_config():
    source = (
        REPO_ROOT / "scripts" / "sam3_cpu" / "download_sam3.py"
    ).read_text(encoding="utf-8")
    assert '"processor_config.json"' in source
    assert '"preprocessor_config.json"' not in source


def test_revision_must_be_pinned_before_model_lookup(tmp_path: Path):
    with pytest.raises(ValueError, match="revision"):
        TransformersSam3Cpu(tmp_path, revision="main")
    with pytest.raises(FileNotFoundError, match="pinned local"):
        TransformersSam3Cpu(tmp_path / "missing", revision=REVISION)
    model_path = tmp_path / REVISION
    model_path.mkdir()
    (tmp_path / "model_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "facebook/sam3",
                "resolved_revision_sha": "0" * 40,
                "local_model_path": str(model_path),
            }
        )
    )
    with pytest.raises(RuntimeError, match="do not match"):
        TransformersSam3Cpu(model_path, revision=REVISION)


def test_tracker_prompt_tensor_nesting_for_each_mode():
    image, prompt = Image.new("RGB", (100, 80)), prompt_fixture()
    point = build_tracker_processor_inputs(image, prompt, "point")
    assert "input_boxes" not in point
    assert np.asarray(point["input_points"]).shape == (
        1,
        1,
        len(prompt.positive_points_xy),
        2,
    )
    box = build_tracker_processor_inputs(image, prompt, "box")
    assert np.asarray(box["input_boxes"]).shape == (1, 1, 4)
    assert "input_points" not in box
    combined = build_tracker_processor_inputs(image, prompt, "box_point")
    assert np.asarray(combined["input_boxes"]).shape == (1, 1, 4)
    assert np.asarray(combined["input_points"]).shape[0:2] == (1, 1)
    assert np.asarray(combined["input_labels"]).shape == np.asarray(
        combined["input_points"]
    ).shape[:-1]


def test_pcs_positive_box_schema():
    inputs = build_pcs_processor_inputs(
        Image.new("RGB", (100, 80)), prompt_fixture(), "pcs_positive_box"
    )
    assert np.asarray(inputs["input_boxes"]).shape == (1, 1, 4)
    assert inputs["input_boxes_labels"] == [[1]]
    assert "text" not in inputs


def test_missing_quality_is_not_invented_and_probability_mass_is_scored():
    prompt = prompt_fixture()
    mask = prompt.cleaned_mask
    probability = probability_fixture()
    result = select_refined_mask(
        [mask],
        [mask.astype(np.float32)],
        [None],
        coarse_mask=mask,
        coarse_probability=probability,
        prompt=prompt,
        depth_m=None,
        config=CONFIG["selector"],
        accepted_source="sam3_tracker_cpu",
    )
    row = result.candidate_metrics[0]
    assert row["sam_quality"] is None
    assert row["sam_quality_available"] is False
    assert row["probability_mass_recall"] == pytest.approx(1.0)
    assert np.isfinite(row["refinement_score"])
    assert result.selected_mask_source == "sam3_tracker_cpu"


def _write_bundle(path: Path) -> None:
    path.mkdir()
    probability = probability_fixture()
    rgb = np.full((80, 100, 3), 128, dtype=np.uint8)
    Image.fromarray(rgb).save(path / "color.png")
    Image.fromarray(np.ones((80, 100), dtype=np.uint16) * 1000).save(path / "depth.png")
    Image.fromarray((probability >= 0.15).astype(np.uint8) * 255).save(
        path / "target_mask.png"
    )
    np.save(path / "target_probability.npy", probability, allow_pickle=False)
    (path / "language.txt").write_text("Grasp the object\n", encoding="utf-8")
    (path / "intrinsics.json").write_text('{"fx": 1, "fy": 1}\n', encoding="utf-8")
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "sample_id": "synthetic",
                "query": "Grasp the object",
                "scene_id": "synthetic_scene",
                "prediction_threshold": 0.15,
            }
        )
        + "\n",
        encoding="utf-8",
    )


class FakeCpuModel:
    backend = "tracker"
    load_time_seconds = 0.1
    runtime_metadata = {
        "model_id": "facebook/sam3",
        "model_revision": REVISION,
        "model_class": "Sam3TrackerModel",
        "processor_class": "Sam3TrackerProcessor",
        "processor_size": 1008,
        "torch_num_threads": 2,
        "torch_num_interop_threads": 1,
        "transformers_version": "5.14.1",
        "torch_version": "2.13.0",
    }

    def infer(self, image, prompt, *, prompt_mode, short_text=None):
        mask = prompt.cleaned_mask
        return Sam3CpuInferenceResult(
            masks=(mask,),
            probabilities=(mask.astype(np.float32),),
            qualities=(0.9,),
            backend="tracker",
            model_class="Sam3TrackerModel",
            processor_class="Sam3TrackerProcessor",
            timings={
                "preprocess_seconds": 0.01,
                "inference_seconds": 0.02,
                "postprocess_seconds": 0.01,
                "total_sample_seconds": 0.04,
                "model_load_seconds": 0.1,
            },
            memory={
                "rss_before_inference_bytes": 1,
                "peak_rss_bytes": 2,
                "rss_after_inference_bytes": 1,
            },
            output_schema={
                "keys": ["pred_masks", "iou_scores"],
                "tensors": {
                    "pred_masks": {
                        "shape": [1, 1, 1, 20, 25],
                        "dtype": "torch.float32",
                        "device": "cpu",
                        "finite": True,
                    }
                },
            },
            runtime_metadata=dict(self.runtime_metadata),
        )


def test_atomic_cpu_refinement_outputs_and_checksums(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    _write_bundle(source)
    loaded = load_hifics_bundle(source)
    assert loaded["coarse_probability"].dtype == np.float32
    metadata = refine_cpu_sample(
        source,
        output,
        model=FakeCpuModel(),
        config=CONFIG,
        prompt_mode="box_point",
    )
    assert metadata["device"] == "cpu"
    assert metadata["dtype"] == "float32"
    assert metadata["selected_mask_source"] == "sam3_tracker_cpu"
    assert metadata["ground_truth_used_for_prompt_or_selection"] is False
    assert (output / "sam3_candidate_masks.npz").is_file()
    assert (output / "timing.json").is_file()
    assert (output / "memory.json").is_file()
    for name, record in metadata["outputs"].items():
        assert sha256_file(output / record["filename"]) == record["sha256"]


def test_atomic_directory_removes_partial_output_on_error(tmp_path: Path):
    destination = tmp_path / "result"
    with pytest.raises(RuntimeError):
        with atomic_output_directory(destination) as temporary:
            (temporary / "partial.txt").write_text("partial")
            raise RuntimeError("interrupted")
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".incomplete").exists()


def test_real_input_manifest_is_prediction_only_and_checksum_complete():
    root = REPO_ROOT / "outputs" / "sam3_cpu_inputs"
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 10
    for row in rows:
        serialized = json.dumps(row).lower()
        assert "ground_truth" not in serialized.replace(
            '"ground_truth_or_annotations_included": false', ""
        )
        assert row["ground_truth_or_annotations_included"] is False
        for key, expected in row["checksums"].items():
            assert sha256_file(Path(row[key])) == expected


def test_cli_dry_run_never_loads_or_downloads_model():
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_sam3_cpu_refinement.py"),
            "--dry-run",
            "--sample-limit",
            "1",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert result["local_files_only"] is True
    assert result["samples"] == ["q0000000_b32eb3299dcd3ae9"]


def test_pipeline_stage_markers_require_the_requested_sample_count(tmp_path: Path):
    refinement = tmp_path / "run_metadata.json"
    refinement.write_text('{"sample_count": 1}\n', encoding="utf-8")
    assert not _stage_complete("refinement", refinement, 10)
    refinement.write_text('{"sample_count": 10}\n', encoding="utf-8")
    assert _stage_complete("refinement", refinement, 10)

    candidates = tmp_path / "summary.csv"
    candidates.write_text("sample_id\none\n", encoding="utf-8")
    assert not _stage_complete("dexnet", candidates, 10)


RUN_REAL = os.environ.get("RUN_SAM3_CPU_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN_REAL, reason="requires authorized pinned official SAM 3 weights")
def test_real_tracker_cpu_forward():
    model_path = Path(os.environ["SAM3_MODEL_PATH"])
    model = TransformersSam3Cpu(model_path, revision=REVISION, backend="tracker")
    result = model.infer(
        Image.new("RGB", (100, 80), "gray"),
        prompt_fixture(),
        prompt_mode="box_point",
    )
    assert result.masks
    assert all(mask.shape == (80, 100) for mask in result.masks)
    assert result.runtime_metadata["requested_device"] == "cpu"
