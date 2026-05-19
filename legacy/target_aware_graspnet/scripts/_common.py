from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.camera_loader import load_label, load_rgb
from dataset.graspnet_loader import GraspNetLoader
from dataset.sample_index import PathTemplates, SampleIndexBuilder
from main import TargetAwareGraspPipeline, frame_result_row, load_config
from target.object_language_mapping import ObjectLanguageEntry, ObjectLanguageMapper
from utils.data_types import FrameResult, GraspNetSample
from utils.io_utils import append_csv


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--target-mode", choices=["annotation", "manual", "pseudo"], default="pseudo")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")


def resolved_config(args: argparse.Namespace) -> tuple[dict, Path, Path, str, int]:
    config = load_config(args.config_dir)
    default_dataset = config.get("default", {}).get("dataset", {})
    dataset_cfg = config.get("dataset", {})
    sampler_cfg = config.get("sampler", {})
    dataset_root = args.dataset_root or Path(dataset_cfg.get("root", default_dataset.get("root", "../data/raw/graspnet")))
    output_root = args.output_root or Path(dataset_cfg.get("output_root", default_dataset.get("output_root", "outputs")))
    camera = args.camera or dataset_cfg.get("camera", default_dataset.get("camera", "realsense"))
    top_k = args.top_k or sampler_cfg.get("top_k", config.get("default", {}).get("sampler", {}).get("top_k", 5))
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    return config, dataset_root, output_root, camera, int(top_k)


def make_index_builder(config: dict) -> SampleIndexBuilder:
    dataset_cfg = config.get("dataset", {})
    return SampleIndexBuilder(PathTemplates.from_config(dataset_cfg.get("path_templates")))


def make_loader(config: dict) -> GraspNetLoader:
    default = config.get("default", {})
    dataset_cfg = config.get("dataset", {})
    return GraspNetLoader(
        depth_scale=dataset_cfg.get("depth_scale", default.get("dataset", {}).get("depth_scale", 1000.0)),
        fallback_intrinsics=dataset_cfg.get("fallback_intrinsics", default.get("dataset", {}).get("fallback_intrinsics")),
    )


def load_category_labels(config: dict) -> dict[int, str]:
    mapping_cfg = config.get("default", {}).get("target_mapping", {}) | config.get("dataset", {}).get("target_mapping", {})
    if not mapping_cfg.get("category_labels_trusted", False):
        return {}
    path_value = mapping_cfg.get("category_labels_path")
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return {int(k): str(v) for k, v in data.items()}
    if path.suffix.lower() == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        return {int(k): str(v) for k, v in data.items()}
    table = pd.read_csv(path)
    label_col = "target_label" if "target_label" in table.columns else "label"
    if label_col not in table.columns:
        label_col = "category"
    if "target_id" not in table.columns or label_col not in table.columns:
        return {}
    return {int(row["target_id"]): str(row[label_col]) for _, row in table.iterrows() if "target_id" in row}


def mapping_entries_for_samples(
    samples: list[GraspNetSample],
    config: dict,
    output_root: Path,
    all_targets: bool,
) -> list[tuple[GraspNetSample, ObjectLanguageEntry]]:
    mapping_cfg = config.get("default", {}).get("target_mapping", {}) | config.get("dataset", {}).get("target_mapping", {})
    mapper = ObjectLanguageMapper(
        output_root=output_root,
        category_labels=load_category_labels(config),
        command_mode=mapping_cfg.get("command_mode", "auto"),
    )
    pairs: list[tuple[GraspNetSample, ObjectLanguageEntry]] = []
    all_entries: list[ObjectLanguageEntry] = []
    for sample in samples:
        try:
            label = load_label(sample.label_path) if sample.label_path else None
            rgb = load_rgb(sample.rgb_path)
            entries = mapper.entries_for_sample(sample, label, rgb=rgb, all_targets=all_targets)
        except Exception as exc:
            append_csv(output_root / "failure_cases.csv", {
                "split": sample.split,
                "scene_id": sample.scene_id,
                "camera": sample.camera,
                "frame_id": sample.frame_id,
                "target_id": None,
                "command": None,
                "status": "failed",
                "error": f"mapping_failed: {type(exc).__name__}: {exc}",
                "runtime_total": 0.0,
            })
            continue
        for entry in entries:
            pairs.append((sample, entry))
            all_entries.append(entry)
    mapper.save_mapping(all_entries)
    return pairs


def sample_for_entry(sample: GraspNetSample, entry: ObjectLanguageEntry, output_root: Path) -> GraspNetSample:
    output_dir = (
        output_root
        / entry.split
        / entry.scene_id
        / entry.camera
        / entry.frame_id
        / f"target_{entry.target_id:03d}"
    )
    metadata = dict(sample.metadata)
    metadata.update({
        "target_id": entry.target_id,
        "target_label": entry.target_label,
        "command": entry.command,
        "mapping_mask_path": entry.mask_path,
    })
    return replace(sample, output_dir=output_dir, metadata=metadata)


def run_mapping_entry(
    pipeline: TargetAwareGraspPipeline,
    sample: GraspNetSample,
    entry: ObjectLanguageEntry,
    output_root: Path,
    top_k: int,
    overwrite: bool,
) -> FrameResult:
    target_sample = sample_for_entry(sample, entry, output_root)
    return pipeline.run_sample(
        target_sample,
        target_mode="annotation",
        target_id=entry.target_id,
        target_label=entry.target_label,
        command=entry.command,
        top_k=top_k,
        overwrite=overwrite,
    )


def log_failure(output_root: Path, result: FrameResult) -> None:
    if result.status != "failed":
        return
    row = frame_result_row(result)
    append_csv(output_root / "failure_cases.csv", row)
