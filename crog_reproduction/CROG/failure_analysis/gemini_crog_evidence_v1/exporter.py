from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import utils.config as config
from model import build_crog
from utils.checkpoint import load_checkpoint
from utils.dataset import OCIDVLGDataset

from failure_analysis.reranking.exporter import _restore_original
from failure_analysis.reranking_v2.extract import _verify_forward_candidate_identity

from . import RENDERER_VERSION, SCHEMA_VERSION
from .evidence import extract_candidate_evidence
from .renderer import (
    deterministic_candidate_mapping,
    metadata_for_prompt,
    render_evidence_board,
)
from .security import assert_no_gt_leak, wrap_untrusted_referring_expression


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth"
)


def _read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _stable_id(split: str, local_id: int) -> str:
    return f"multiple:{split}:{int(local_id):08d}"


def _select_frozen_rows(features_path: str | Path, selected_local_ids: set[int]) -> dict[int, dict[str, Any]]:
    rows = {}
    for record in _read_jsonl(features_path):
        local_id = int(record["sample_id"])
        if local_id in selected_local_ids:
            rows[local_id] = record
    missing = selected_local_ids - set(rows)
    if missing:
        raise ValueError(f"frozen feature rows not found: {sorted(missing)[:10]}")
    return rows


def select_development_smoke_cohort(
    *,
    features_path: str | Path,
    legacy_labels_path: str | Path,
    split_manifest_path: str | Path,
    count: int = 10,
) -> tuple[list[int], list[dict[str, Any]]]:
    manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    development = {
        int(row["source_sample_id"])
        for row in manifest["rows"]
        if row["development_partition"] == "train" and row["official_split"] == "train"
    }
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for feature, label in zip(_read_jsonl(features_path), _read_jsonl(legacy_labels_path), strict=True):
        local_id = int(feature["sample_id"])
        if local_id not in development:
            continue
        correctness = [bool(item["candidate_correct"]) for item in label["candidate_labels"]]
        query = str(feature["language_instruction"]).lower()
        if correctness[0]:
            buckets["q_only_correct"].append((local_id, feature))
        elif any(correctness):
            buckets["q_only_recoverable"].append((local_id, feature))
        else:
            buckets["top5_all_wrong"].append((local_id, feature))
        if any(word in query for word in ("left of", "right of", "behind", "in front", "next to", "between")):
            buckets["relation"].append((local_id, feature))
        if any(word in query for word in ("top", "bottom", "upper", "lower", "middle")):
            buckets["location"].append((local_id, feature))
        if any(word in query for word in ("red", "green", "blue", "yellow", "white", "black", "small", "large")):
            buckets["attribute"].append((local_id, feature))
        if int(feature.get("predicted_mask_area", 10**9)) < 2500:
            buckets["small_predicted_target"].append((local_id, feature))
        candidates = feature["candidates"]
        overlaps = []
        for i in range(5):
            for j in range(i + 1, 5):
                a, b = np.asarray(candidates[i]["polygon"], np.float32), np.asarray(candidates[j]["polygon"], np.float32)
                intersection, _ = cv2.intersectConvexConvex(a, b)
                union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - intersection
                overlaps.append(0.0 if union <= 0 else float(intersection / union))
        if max(overlaps, default=0.0) >= 0.25:
            buckets["overlapping_candidates"].append((local_id, feature))
    desired = (
        "q_only_correct", "q_only_recoverable", "top5_all_wrong", "relation", "location",
        "attribute", "small_predicted_target", "overlapping_candidates",
    )
    selected: list[int] = []
    rationale: dict[int, set[str]] = defaultdict(set)
    for category in desired:
        for local_id, _ in buckets[category]:
            if local_id not in selected:
                selected.append(local_id)
                rationale[local_id].add(category)
                break
    for category in desired:
        for local_id, _ in buckets[category]:
            if local_id in selected:
                rationale[local_id].add(category)
            elif len(selected) < count:
                selected.append(local_id)
                rationale[local_id].add(category)
            if len(selected) >= count:
                break
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise AssertionError(f"could only select {len(selected)}/{count} smoke samples")
    audit_rows = [
        {"sample_id": _stable_id("train", local_id), "source_sample_id": local_id, "selection_strata": sorted(rationale[local_id])}
        for local_id in selected
    ]
    return selected, audit_rows


def _restored_maps(pred: tuple[torch.Tensor, ...], image: torch.Tensor, data: dict[str, Any]):
    m_logit, q_logit, sin_raw, cos_raw, w_logit = pred
    probabilities = (torch.sigmoid(m_logit), torch.sigmoid(q_logit), sin_raw, cos_raw, torch.sigmoid(w_logit))
    raw_values = (m_logit, q_logit, w_logit)
    kwargs = {"size": image.shape[-2:], "mode": "bicubic", "align_corners": True}
    probabilities = tuple(F.interpolate(value, **kwargs).squeeze(1) for value in probabilities)
    raw_values = tuple(F.interpolate(value, **kwargs).squeeze(1) for value in raw_values)
    for batch_index in range(image.shape[0]):
        inverse = data["inverse"][batch_index]
        height, width = map(int, data["ori_size"][batch_index])
        restored_probability = tuple(
            _restore_original(value[batch_index].cpu().numpy(), inverse, width, height)
            for value in probabilities
        )
        restored_raw = tuple(
            _restore_original(value[batch_index].cpu().numpy(), inverse, width, height)
            for value in raw_values
        )
        yield restored_probability, restored_raw


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd")


@torch.no_grad()
def export_frozen_crog_evidence(
    *,
    split: str,
    selected_local_ids: Iterable[int],
    frozen_features_path: str | Path,
    output_dir: str | Path,
    system_prompt_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "auto",
    frozen_batch_size: int = 16,
    mapping_seed: int = 47,
    keep_dense_maps: bool = True,
) -> dict[str, Any]:
    selected = sorted({int(value) for value in selected_local_ids})
    if not selected:
        raise ValueError("selected_local_ids is empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    maps_dir, boards_dir = output / "maps", output / "boards"
    maps_dir.mkdir()
    boards_dir.mkdir()
    config_path, checkpoint_path = Path(config_path).resolve(), Path(checkpoint_path).resolve()
    cfg = config.load_cfg_from_cfg_file(str(config_path))
    root = (REPO_ROOT / cfg.root_path).resolve()
    torch_device = torch.device(
        "mps" if device == "auto" and torch.backends.mps.is_available() else ("cpu" if device == "auto" else device)
    )
    frozen = _select_frozen_rows(frozen_features_path, set(selected))
    dataset = OCIDVLGDataset(
        root_dir=str(root), input_size=cfg.input_size, word_length=cfg.word_len, split=split, version=cfg.version
    )
    grouped: dict[int, list[int]] = defaultdict(list)
    for local_id in selected:
        grouped[(local_id // frozen_batch_size) * frozen_batch_size].append(local_id)
    replay_indices = []
    for start in sorted(grouped):
        replay_indices.extend(range(start, min(start + frozen_batch_size, len(dataset))))
    loader = DataLoader(
        Subset(dataset, replay_indices),
        batch_size=frozen_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=OCIDVLGDataset.collate_fn,
    )
    model, _ = build_crog(cfg)
    model = model.to(torch_device).eval()
    load_checkpoint(checkpoint_path, model, torch_device, strict=True)
    system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    candidate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    identity_max = 0.0
    written_ids: set[int] = set()
    for data in tqdm(loader, desc=f"CROG evidence replay {split}", ncols=100):
        image = data["img"].to(torch_device)
        text = data["word_vec"].to(torch_device)
        pred, _ = model(image, text)  # strict inference: no GT tensors enter CROG forward
        restored = list(_restored_maps(pred, image, data))
        for batch_index, (maps, raw_maps) in enumerate(restored):
            local_id = int(data["sent_id"][batch_index])
            if local_id not in frozen:
                continue
            feature = frozen[local_id]
            m_prob, q_prob, sin_map, cos_map, w_prob = maps
            m_raw, q_raw, w_raw = raw_maps
            identity = _verify_forward_candidate_identity(
                feature["candidates"],
                q_prob,
                sin_map,
                cos_map,
                w_prob,
                sample_context=f"{split}:{local_id}",
            )
            identity_max = max(identity_max, float(identity["maximum_value_difference"]))
            sample_id = _stable_id(split, local_id)
            evidence, sample = extract_candidate_evidence(
                sample_id=sample_id,
                frame_id=feature["scene_id"],
                candidates=feature["candidates"],
                mask_probability=m_prob,
                quality_probability=q_prob,
                sin_2theta=sin_map,
                cos_2theta=cos_map,
                width_probability=w_prob,
                mask_logit=m_raw,
                quality_raw=q_raw,
                width_raw=w_raw,
            )
            mapping = deterministic_candidate_mapping(
                [str(item["candidate_id"]) for item in feature["candidates"]], sample_id, seed=mapping_seed
            )
            for item in evidence:
                item["display_candidate_id"] = mapping["candidate_to_display"][item["candidate_id"]]
            image_bgr = cv2.imread(str(feature["image_path"]), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(feature["image_path"])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            board_path = boards_dir / f"sample_{local_id:08d}.png"
            layer_manifest = render_evidence_board(
                rgb=image_rgb,
                candidates=feature["candidates"],
                candidate_evidence=evidence,
                mask_probability=m_prob,
                quality_probability=q_prob,
                sin_2theta=sin_map,
                cos_2theta=cos_map,
                width_probability=w_prob,
                sample_id=sample_id,
                output_path=board_path,
                mapping=mapping,
            )
            (board_path.with_suffix(".layers.json")).write_text(
                json.dumps(layer_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            metadata = metadata_for_prompt(
                referring_expression_block=wrap_untrusted_referring_expression(feature["language_instruction"]),
                candidate_evidence=evidence,
                mapping=mapping,
                original_q_top1_candidate_id=sample["original_q_top1_candidate_id"],
            )
            assert_no_gt_leak({"metadata": metadata, "mapping": mapping, "layer_manifest": layer_manifest})
            map_path = maps_dir / f"sample_{local_id:08d}.npz"
            # Full-run shards normally need only the deterministic request board
            # and extracted candidate evidence.  Avoid serialising a temporary
            # 480x640 dense-map archive merely to unlink it a few lines later:
            # that adds roughly one second and ~2 MiB per sample without adding
            # any audit value.  Development ablations opt in to persisted maps.
            if keep_dense_maps:
                np.savez_compressed(
                    map_path,
                    m_logit=np.asarray(m_raw, dtype=np.float16),
                    m_probability=np.asarray(m_prob, dtype=np.float16),
                    q_raw=np.asarray(q_raw, dtype=np.float16),
                    q_probability=np.asarray(q_prob, dtype=np.float16),
                    sin_2theta=np.asarray(sin_map, dtype=np.float16),
                    cos_2theta=np.asarray(cos_map, dtype=np.float16),
                    w_raw=np.asarray(w_raw, dtype=np.float16),
                    w_probability=np.asarray(w_prob, dtype=np.float16),
                )
            sample.update(
                {
                    "language_instruction": feature["language_instruction"],
                    "image_path": str(Path(feature["image_path"]).resolve()),
                    "board_path": str(board_path.resolve()),
                    "board_sha256": layer_manifest["image_sha256"],
                    "maps_path": str(map_path.resolve()) if keep_dense_maps else None,
                    "candidate_set_sha256": hashlib.sha256(
                        json.dumps(
                            [item["candidate_checksum"] for item in feature["candidates"]],
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
            candidate_rows.extend(evidence)
            sample_rows.append(sample)
            mapping_rows.append(
                {
                    "sample_id": sample_id,
                    "mapping_json": json.dumps(mapping, sort_keys=True),
                    "mapping_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True).encode("utf-8")).hexdigest(),
                }
            )
            request_rows.append(
                {
                    "sample_id": sample_id,
                    "frame_id": feature["scene_id"],
                    "board_path": str(board_path.resolve()),
                    "board_sha256": layer_manifest["image_sha256"],
                    "metadata": metadata,
                    "mapping": mapping,
                    "system_prompt_sha256": prompt_hash,
                    "renderer_version": RENDERER_VERSION,
                    "store": False,
                    "background": False,
                    "stream": False,
                }
            )
            written_ids.add(local_id)
    if written_ids != set(selected):
        raise AssertionError(f"evidence export missing selected IDs: {sorted(set(selected) - written_ids)}")
    _write_parquet(output / "candidate_evidence.parquet", candidate_rows)
    _write_parquet(output / "sample_evidence.parquet", sample_rows)
    _write_parquet(output / "candidate_mapping.parquet", mapping_rows)
    _write_parquet(
        output / "request_manifest.parquet",
        [{**row, "mapping": json.dumps(row["mapping"], sort_keys=True)} for row in request_rows],
    )
    with (output / "request_manifest.jsonl").open("x", encoding="utf-8") as handle:
        for row in request_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    schema = {
        "schema_version": SCHEMA_VERSION,
        "candidate_fields": list(candidate_rows[0]),
        "sample_fields": list(sample_rows[0]),
        "mask_threshold": 0.35,
        "angle_period_degrees": 180,
        "decoded_width_scale_px": 100.0,
        "full_dense_maps_are_predictions_not_ground_truth": True,
    }
    (output / "evidence_schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    numeric_stats = {}
    for name in candidate_rows[0]:
        values = [row[name] for row in candidate_rows if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)]
        if values:
            array = np.asarray(values, dtype=np.float64)
            numeric_stats[name] = {"min": float(array.min()), "max": float(array.max()), "mean": float(array.mean()), "std": float(array.std())}
    statistics = {
        "sample_count": len(sample_rows),
        "candidate_count": len(candidate_rows),
        "candidates_per_sample": 5,
        "forward_identity_max_difference": identity_max,
        "numeric_fields": numeric_stats,
    }
    (output / "evidence_statistics.json").write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")
    result = {
        "status": "complete",
        "split": split,
        "sample_count": len(sample_rows),
        "candidate_count": len(candidate_rows),
        "forward_identity_max_difference": identity_max,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "system_prompt_sha256": prompt_hash,
        "no_ground_truth_forward": True,
        "device": str(torch_device),
        "frozen_batch_size": frozen_batch_size,
    }
    (output / "export_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
