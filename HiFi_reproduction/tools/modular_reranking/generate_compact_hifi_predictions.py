#!/usr/bin/env python3
"""Generate GT-free compact HiFi-CS predictions for one official split.

The model sees only processed RGB and language. Native RGB/depth/PCD paths are
resolved for downstream geometry, but GT masks and grasp annotations are never
written to this inference artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIFICS_ROOT = PROJECT_ROOT / "hifics"
sys.path.insert(0, str(HIFICS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from general_utils import resolve_device  # noqa: E402
from models.hifics import HierarchicalCLIPDensePredT  # noqa: E402
from score import synchronize  # noqa: E402
from src.grasping.reranking_v1.identity import (  # noqa: E402
    sha256_file,
    stable_sample_id,
)


class ImageLanguageDataset(Dataset):
    """Exact HiFi RGB preprocessing without loading a GT mask."""

    def __init__(self, records: list[dict[str, Any]], image_size: int):
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size))),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.records[index]
        path = Path(str(row["rgb_path"]).replace("\\", "/")).expanduser()
        if not path.is_absolute():
            path = HIFICS_ROOT / path
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(row["text"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status-every", type=int, default=100)
    parser.add_argument(
        "--reference-predictions-root",
        type=Path,
        help="Optional frozen test predictions for a numerical smoke comparison",
    )
    return parser.parse_args()


def _temporary_path(path: Path, tmp_root: Path) -> Path:
    tmp_root.mkdir(parents=True, exist_ok=True)
    return tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"


def atomic_text(path: Path, text: str, *, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, tmp_root)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any, *, tmp_root: Path) -> None:
    atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        tmp_root=tmp_root,
    )


def save_probability(
    path: Path, probability: np.ndarray, *, tmp_root: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, tmp_root)
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, probability=np.asarray(probability, dtype=np.float32)
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_mask(path: Path, mask: np.ndarray, *, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, tmp_root)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
        temporary, format="PNG"
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_paths(ocid_root: Path, scene_id: str) -> dict[str, Path]:
    sequence, image_name = str(scene_id).split(",", 1)
    sequence_root = (ocid_root / sequence).resolve()
    stem = Path(image_name).stem
    pcd = sequence_root / "pcd" / f"{stem}.pcd"
    # One upstream OCID-VLG sequence is stored under ``pd`` instead of the
    # otherwise canonical ``pcd`` directory. Preserve the actual source path
    # and hash rather than fabricating or skipping that scene.
    if not pcd.is_file():
        pcd = sequence_root / "pd" / f"{stem}.pcd"
    paths = {
        "rgb": sequence_root / "rgb" / image_name,
        "depth": sequence_root / "depth" / image_name,
        "pcd": pcd,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"original OCID assets missing: {missing}")
    return paths


def row_is_valid(
    row_path: Path,
    *,
    split: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
    inference_contract_sha256: str,
) -> dict[str, Any] | None:
    if not row_path.is_file():
        return None
    try:
        row = json.loads(row_path.read_text(encoding="utf-8"))
        probability_path = Path(row["probability_path"])
        mask_path = Path(row["native_mask_path"])
        if (
            row["split"] != split
            or row["manifest_sha256"] != manifest_sha256
            or row["checkpoint_sha256"] != checkpoint_sha256
            or row.get("inference_contract_sha256")
            != inference_contract_sha256
            or sha256_file(probability_path) != row["probability_sha256"]
            or sha256_file(mask_path) != row["native_mask_sha256"]
        ):
            return None
        return row
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.status_every <= 0:
        raise ValueError("batch-size and status-every must be positive")
    if not 0.0 < args.foreground_threshold < 1.0:
        raise ValueError("foreground-threshold must be in (0, 1)")
    output_root = args.output_root.expanduser().resolve()
    tmp_root = args.tmp_root.expanduser().resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.expanduser().resolve()
    annotations_path = args.annotations.expanduser().resolve()
    ocid_root = args.ocid_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    reference_root = (
        None
        if args.reference_predictions_root is None
        else args.reference_predictions_root.expanduser().resolve()
    )
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("split manifest must be a list")
    selected = records if args.limit is None else records[: int(args.limit)]
    if not selected:
        raise ValueError("no samples selected")
    annotations_payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if annotations_payload.get("info", {}).get("split") != args.split:
        raise ValueError("annotation split disagrees")
    annotations = {
        int(row["question_index"]): row for row in annotations_payload["data"]
    }
    if len(annotations) != len(annotations_payload["data"]):
        raise ValueError("annotation question_index values are not unique")
    for row in selected:
        annotation = annotations.get(int(row["question_index"]))
        if (
            annotation is None
            or annotation["image_filename"] != row["scene_id"]
            or annotation["question"] != row["text"]
        ):
            raise ValueError(
                f"annotation identity mismatch: {row['question_index']}"
            )

    config_path = run_dir / "config.yaml"
    config_sha = sha256_file(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if bool(config["invert_mask"]) is not True:
        raise ValueError("frozen HiFi run must use invert_mask=true")
    if float(args.foreground_threshold) != float(
        config["validation_threshold"]
    ):
        raise ValueError(
            "foreground threshold must equal the frozen validation threshold"
        )
    expected_contract = {
        "architecture": "models.hifics.HierarchicalCLIPDensePredT",
        "clip_backbone": "ViT-B/16",
        "projection_layers": [1, 3, 5, 7, 9],
        "decoder_dimension": 64,
        "decoder_heads": 4,
        "hierarchical_levels": 5,
        "image_resolution": 352,
        "image_resize": "bilinear",
        "input_normalization": "none",
        "foreground_probability": "sigmoid(-background_logit)",
        "foreground_threshold": 0.5,
        "model_mask": "probability>=threshold_at_352",
        "native_mask": "nearest_resize_of_binary_352_mask_to_640x480",
        "execution_batch_size": 32,
        "source_config_sha256": (
            "23cf158cc653e3af2e16d635f782d6d495b120f739a7855516c314011bf234e1"
        ),
    }
    observed_contract = {
        "architecture": config["model"],
        "clip_backbone": config["clip_backbone"],
        "projection_layers": list(config["projection_layers"]),
        "decoder_dimension": int(config["decoder_dimension"]),
        "decoder_heads": int(config.get("decoder_heads", 4)),
        "hierarchical_levels": int(config["hierarchical_levels"]),
        "image_resolution": int(config["image_resolution"]),
        "image_resize": config["image_resize"],
        "input_normalization": config["input_normalization"],
        "foreground_probability": "sigmoid(-background_logit)",
        "foreground_threshold": float(config["validation_threshold"]),
        "model_mask": "probability>=threshold_at_352",
        "native_mask": "nearest_resize_of_binary_352_mask_to_640x480",
        # The retained repeated-FiLM test bundle was generated at batch 32.
        # MPS can differ by a few ulps for a smaller final batch, so freeze this
        # execution detail to keep train/val probabilities numerically aligned.
        "execution_batch_size": int(args.batch_size),
        "source_config_sha256": config_sha,
    }
    if observed_contract != expected_contract:
        raise ValueError(
            f"retained repeated-FiLM inference contract drift: "
            f"{observed_contract}"
        )
    inference_contract_sha = hashlib.sha256(
        json.dumps(
            observed_contract,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_sha = sha256_file(manifest_path)
    annotation_sha = sha256_file(annotations_path)
    checkpoint_sha = sha256_file(checkpoint)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = resolve_device(args.device)
    model = HierarchicalCLIPDensePredT(
        version=config["clip_backbone"],
        extract_layers=tuple(config["projection_layers"]),
        reduce_dim=int(config["decoder_dimension"]),
        n_heads=int(config.get("decoder_heads", 4)),
        cond_layer=None,
        extended_film=True,
        hierarchical_film=True,
    )
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    if (
        checkpoint_payload.get("format")
        != "hifics_hierfilm_trainable_only_v1"
    ):
        raise ValueError("unsupported repeated-FiLM checkpoint format")
    expected_keys = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    trainable_state = checkpoint_payload.get("trainable_state", {})
    if set(trainable_state) != expected_keys:
        raise ValueError(
            "repeated-FiLM checkpoint trainable-state keys mismatch"
        )
    complete_state = model.state_dict()
    complete_state.update(trainable_state)
    incompatible = model.load_state_dict(complete_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"strict repeated-FiLM checkpoint load failed: {incompatible}"
        )
    checkpoint_metadata = checkpoint_payload.get("metadata", {})
    if int(checkpoint_metadata.get("global_step", -1)) != 19728:
        raise ValueError("unexpected repeated-FiLM best checkpoint step")
    model.to(device)
    model.eval()
    model.clip_model.eval()
    dataset = ImageLanguageDataset(
        selected, int(config["image_resolution"])
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    rows_dir = output_root / "rows"
    probabilities_dir = output_root / "probabilities"
    masks_dir = output_root / "native_masks"
    rows_dir.mkdir(exist_ok=True)
    probabilities_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    source_hash_cache: dict[Path, str] = {}

    def cached_hash(path: Path) -> str:
        value = source_hash_cache.get(path)
        if value is None:
            value = sha256_file(path)
            source_hash_cache[path] = value
        return value

    completed = 0
    fresh = 0
    reference_probability_max_abs = 0.0
    reference_mask_mismatches = 0
    started = time.perf_counter()
    offset = 0
    with torch.inference_mode():
        for images, queries in loader:
            batch_records = selected[offset : offset + images.shape[0]]
            pending: list[int] = []
            existing: dict[int, dict[str, Any]] = {}
            for local, record in enumerate(batch_records):
                sample_id = stable_sample_id(
                    record["scene_id"], int(record["question_index"])
                )
                cached = row_is_valid(
                    rows_dir / f"{sample_id}.json",
                    split=args.split,
                    manifest_sha256=manifest_sha,
                    checkpoint_sha256=checkpoint_sha,
                    inference_contract_sha256=inference_contract_sha,
                )
                if cached is None:
                    pending.append(local)
                else:
                    existing[local] = cached
            probabilities: np.ndarray | None = None
            inference_seconds = 0.0
            if pending:
                batch_images = images[pending].to(device)
                batch_queries = [queries[index] for index in pending]
                synchronize(device)
                inference_start = time.perf_counter()
                output = model(batch_images, batch_queries, return_features=True)
                logits = output[0] if isinstance(output, (tuple, list)) else output
                synchronize(device)
                inference_seconds = time.perf_counter() - inference_start
                probabilities = (
                    torch.sigmoid(-logits)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            pending_position = 0
            for local, record in enumerate(batch_records):
                sample_index = offset + local
                sample_id = stable_sample_id(
                    record["scene_id"], int(record["question_index"])
                )
                if local in existing:
                    completed += 1
                    continue
                assert probabilities is not None
                probability = probabilities[pending_position, 0]
                pending_position += 1
                model_mask = (
                    probability >= args.foreground_threshold
                )
                native_mask = (
                    np.asarray(
                        Image.fromarray(
                            model_mask.astype(np.uint8) * 255,
                            mode="L",
                        ).resize(
                            (640, 480),
                            resample=Image.Resampling.NEAREST,
                        ),
                        dtype=np.uint8,
                    )
                    >= 128
                )
                prefix = sample_id.rsplit("_", 1)[-1][:2]
                probability_path = (
                    probabilities_dir / prefix / f"{sample_id}.npz"
                ).resolve()
                mask_path = (masks_dir / prefix / f"{sample_id}.png").resolve()
                save_probability(
                    probability_path,
                    probability,
                    tmp_root=tmp_root,
                )
                save_mask(
                    mask_path, native_mask, tmp_root=tmp_root
                )
                paths = source_paths(ocid_root, str(record["scene_id"]))
                row = {
                    "schema_version": 1,
                    "split": args.split,
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "question_index": int(record["question_index"]),
                    "scene_id": str(record["scene_id"]),
                    "query": str(record["text"]),
                    "processed_rgb_path": str(
                        (
                            HIFICS_ROOT
                            / Path(str(record["rgb_path"]).replace("\\", "/"))
                        ).resolve()
                    ),
                    "source_rgb_path": str(paths["rgb"]),
                    "source_depth_path": str(paths["depth"]),
                    "source_pcd_path": str(paths["pcd"]),
                    "source_rgb_sha256": cached_hash(paths["rgb"]),
                    "source_depth_sha256": cached_hash(paths["depth"]),
                    "source_pcd_sha256": cached_hash(paths["pcd"]),
                    "probability_path": str(probability_path),
                    "probability_shape": list(probability.shape),
                    "probability_dtype": "float32",
                    "probability_sha256": sha256_file(probability_path),
                    "native_mask_path": str(mask_path),
                    "native_mask_shape": list(native_mask.shape),
                    "native_mask_sha256": sha256_file(mask_path),
                    "native_mask_area_px": int(np.count_nonzero(native_mask)),
                    "foreground_threshold": float(args.foreground_threshold),
                    "threshold_comparison": ">=",
                    "model_mask_transform": (
                        "probability>=threshold_at_352"
                    ),
                    "native_mask_transform": (
                        "nearest_resize_of_binary_352_mask_to_640x480"
                    ),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha,
                    "annotations_path": str(annotations_path),
                    "annotations_sha256": annotation_sha,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "config_path": str(config_path),
                    "config_sha256": config_sha,
                    "inference_contract": observed_contract,
                    "inference_contract_sha256": inference_contract_sha,
                    "checkpoint_iteration": int(
                        checkpoint_metadata.get("global_step", -1)
                    ),
                    "inference_seconds": float(
                        inference_seconds / max(len(pending), 1)
                    ),
                    "gt_artifacts_exported": False,
                    "ready": True,
                }
                if reference_root is not None:
                    reference = reference_root / sample_id
                    reference_probability_path = (
                        reference
                        / "predicted_probability_model_resolution.npy"
                    )
                    reference_mask_path = (
                        reference / "predicted_mask_original_resolution.png"
                    )
                    if (
                        not reference_probability_path.is_file()
                        or not reference_mask_path.is_file()
                    ):
                        raise FileNotFoundError(
                            f"reference prediction missing: {sample_id}"
                        )
                    expected_probability = np.load(
                        reference_probability_path, allow_pickle=False
                    )
                    difference = float(
                        np.max(np.abs(probability - expected_probability))
                    )
                    reference_probability_max_abs = max(
                        reference_probability_max_abs, difference
                    )
                    expected_mask = (
                        np.asarray(Image.open(reference_mask_path).convert("L"))
                        >= 128
                    )
                    mismatch = int(np.count_nonzero(native_mask != expected_mask))
                    reference_mask_mismatches += mismatch
                    row["reference_probability_max_abs_difference"] = difference
                    row["reference_native_mask_mismatch_px"] = mismatch
                atomic_json(
                    rows_dir / f"{sample_id}.json",
                    row,
                    tmp_root=tmp_root,
                )
                completed += 1
                fresh += 1
                if completed % args.status_every == 0:
                    print(
                        f"HiFi {args.split}: {completed}/{len(selected)} "
                        f"fresh={fresh} elapsed={time.perf_counter()-started:.1f}s",
                        flush=True,
                    )
            offset += len(batch_records)

    rows = []
    for index, record in enumerate(selected):
        sample_id = stable_sample_id(
            record["scene_id"], int(record["question_index"])
        )
        row = row_is_valid(
            rows_dir / f"{sample_id}.json",
            split=args.split,
            manifest_sha256=manifest_sha,
            checkpoint_sha256=checkpoint_sha,
            inference_contract_sha256=inference_contract_sha,
        )
        if row is None or int(row["sample_index"]) != index:
            raise RuntimeError(f"incomplete compact prediction: {sample_id}")
        rows.append(row)
    if reference_root is not None and (
        reference_probability_max_abs > 1e-6 or reference_mask_mismatches
    ):
        raise RuntimeError(
            "frozen prediction smoke mismatch: "
            f"max_abs={reference_probability_max_abs}, "
            f"mask_px={reference_mask_mismatches}"
        )
    manifest_jsonl = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    )
    atomic_text(
        output_root / "manifest.jsonl",
        manifest_jsonl,
        tmp_root=tmp_root,
    )
    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "split": args.split,
        "samples": len(rows),
        "fresh_samples": fresh,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "annotations_path": str(annotations_path),
        "annotations_sha256": annotation_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "foreground_threshold": float(args.foreground_threshold),
        "inference_contract": observed_contract,
        "inference_contract_sha256": inference_contract_sha,
        "batch_size": args.batch_size,
        "device": str(device),
        "gt_artifacts_exported": False,
        "reference_probability_max_abs_difference": (
            reference_probability_max_abs if reference_root is not None else None
        ),
        "reference_native_mask_mismatch_px": (
            reference_mask_mismatches if reference_root is not None else None
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_manifest": str((output_root / "manifest.jsonl").resolve()),
        "output_manifest_sha256": sha256_file(output_root / "manifest.jsonl"),
    }
    atomic_json(
        output_root / "summary.json", summary, tmp_root=tmp_root
    )
    atomic_text(
        output_root / "run_command.txt",
        " ".join([sys.executable, *sys.argv]) + "\n",
        tmp_root=tmp_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
