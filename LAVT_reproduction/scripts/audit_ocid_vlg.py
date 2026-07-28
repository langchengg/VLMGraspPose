#!/usr/bin/env python3
"""Strict, non-skipping audit for OCID-VLG LAVT manifests."""

from __future__ import annotations

import argparse
import functools
import json
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_ocid_vlg_bert import (  # noqa: E402
    MANIFEST_REQUIRED_FIELDS,
    encode_sentence,
    load_jsonl,
    stable_sent_id,
)

ISSUE_CODES = (
    "manifest_missing_fields",
    "unstable_sent_id",
    "split_mismatch",
    "missing_rgb",
    "missing_mask",
    "unreadable_rgb",
    "unreadable_mask",
    "invalid_mask_shape",
    "shape_mismatch",
    "abnormal_instance_values",
    "non_binary_target",
    "empty_target_mask",
    "non_binary_resized_target",
    "duplicate_sent_id",
)


def _default_tokenizer(name: str):
    from bert.tokenization_bert import BertTokenizer

    return BertTokenizer.from_pretrained(name)


def _resize_for_audit(rgb: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    target = torch.from_numpy(mask.copy()).float()
    resized_rgb = torch_functional.interpolate(
        image[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )[0].permute(1, 2, 0).numpy()
    resized_mask = torch_functional.interpolate(
        target[None, None], size=(size, size), mode="nearest"
    )[0, 0].numpy().astype(np.uint8)
    return resized_rgb, resized_mask


def _resize_mask_nearest(mask: np.ndarray, size: int) -> np.ndarray:
    target = torch.from_numpy(mask.copy()).float()
    return (
        torch_functional.interpolate(
            target[None, None], size=(size, size), mode="nearest"
        )[0, 0]
        .numpy()
        .astype(np.uint8)
    )


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def audit_manifests(
    manifest_paths: dict[str, str | Path],
    tokenizer: Any,
    max_tokens: int = 20,
    image_size: int = 480,
    seed: int = 42,
    visualization_dir: str | Path | None = None,
) -> dict[str, Any]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    all_errors: list[dict[str, Any]] = []
    # Fixed-seed reservoir sampling keeps the full audit bounded in memory:
    # 37k original-resolution samples must never all be retained at once.
    valid_visuals: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    visual_rng = random.Random(seed)
    valid_visual_count = 0
    token_records: list[dict[str, Any]] = []
    split_stats: dict[str, Any] = {}
    global_issue_counts = Counter({code: 0 for code in ISSUE_CODES})

    @functools.lru_cache(maxsize=16)
    def read_rgb(path: str) -> np.ndarray:
        with Image.open(path) as handle:
            handle.load()
            return np.asarray(handle.convert("RGB"))

    @functools.lru_cache(maxsize=16)
    def read_mask(path: str) -> np.ndarray:
        with Image.open(path) as handle:
            handle.load()
            return np.asarray(handle)

    @functools.lru_cache(maxsize=64)
    def binary_and_resized_mask(
        path: str, obj_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = read_mask(path)
        if raw.ndim == 3 and raw.shape[-1] == 1:
            raw = raw[..., 0]
        target = (raw == obj_id).astype(np.uint8)
        return target, _resize_mask_nearest(target, image_size)

    for split, manifest_path in manifest_paths.items():
        rows = load_jsonl(manifest_path)
        rows_by_split[split] = rows
        sent_ids: list[str] = []
        scenes: list[str] = []
        images: list[str] = []
        sentences: list[str] = []
        counters = Counter()
        for position, row in enumerate(rows):
            missing = sorted(MANIFEST_REQUIRED_FIELDS - set(row))
            if missing:
                all_errors.append(
                    {"split": split, "row": position, "code": "manifest_missing_fields", "detail": missing}
                )
                counters["manifest_missing_fields"] += 1
                continue
            sent_id = str(row["sent_id"])
            sent_ids.append(sent_id)
            scenes.append(str(row["scene_id"]))
            images.append(str(Path(row["image_path"]).expanduser().resolve()))
            sentences.append(str(row["sentence"]))
            expected_id = stable_sent_id(row["scene_id"], row["raw_question_index"])
            if sent_id != expected_id:
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "unstable_sent_id", "detail": expected_id}
                )
                counters["unstable_sent_id"] += 1
            if row["split"] != split:
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "split_mismatch", "detail": row["split"]}
                )
                counters["split_mismatch"] += 1
            encoded = encode_sentence(tokenizer, row["sentence"], max_tokens)
            token_records.append(
                {
                    "split": split,
                    "sent_id": sent_id,
                    "whitespace_words": len(str(row["sentence"]).split()),
                    "bert_tokens": int(encoded["bert_token_count"]),
                    "truncated": bool(encoded["truncated"]),
                }
            )
            image_path = Path(row["image_path"]).expanduser()
            mask_path = Path(row["mask_path"]).expanduser()
            missing_paths = [
                str(path) for path in (image_path, mask_path) if not path.is_file()
            ]
            if missing_paths:
                for path in missing_paths:
                    code = "missing_rgb" if path == str(image_path) else "missing_mask"
                    counters[code] += 1
                    all_errors.append(
                        {"split": split, "row": position, "sent_id": sent_id,
                         "code": code, "detail": path}
                    )
                continue
            try:
                rgb = read_rgb(str(image_path.resolve()))
            except Exception as error:
                counters["unreadable_rgb"] += 1
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "unreadable_rgb", "detail": repr(error)}
                )
                continue
            try:
                raw_mask = read_mask(str(mask_path.resolve()))
            except Exception as error:
                counters["unreadable_mask"] += 1
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "unreadable_mask", "detail": repr(error)}
                )
                continue
            if raw_mask.ndim == 3 and raw_mask.shape[-1] == 1:
                raw_mask = raw_mask[..., 0]
            if raw_mask.ndim != 2:
                counters["invalid_mask_shape"] += 1
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "invalid_mask_shape", "detail": list(raw_mask.shape)}
                )
                continue
            if rgb.shape[:2] != raw_mask.shape:
                counters["shape_mismatch"] += 1
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "shape_mismatch",
                     "detail": {"rgb": list(rgb.shape[:2]), "mask": list(raw_mask.shape)}}
                )
                continue
            if not np.issubdtype(raw_mask.dtype, np.integer):
                counters["abnormal_instance_values"] += 1
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "abnormal_instance_values", "detail": str(raw_mask.dtype)}
                )
            target, resized_mask = binary_and_resized_mask(
                str(mask_path.resolve()), int(row["objID"])
            )
            target_values = set(np.unique(target).tolist())
            if not target_values <= {0, 1}:
                counters["non_binary_target"] += 1
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "non_binary_target", "detail": sorted(target_values)}
                )
            if not target.any():
                counters["empty_target_mask"] += 1
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "empty_target_mask", "detail": int(row["objID"])}
                )
            resized_values = set(np.unique(resized_mask).tolist())
            if not resized_values <= {0, 1}:
                counters["non_binary_resized_target"] += 1
                all_errors.append(
                    {"split": split, "row": position, "sent_id": sent_id,
                     "code": "non_binary_resized_target", "detail": sorted(resized_values)}
                )
            valid_visual_count += 1
            visual = (row, rgb, target)
            if len(valid_visuals) < 20:
                valid_visuals.append(visual)
            else:
                replacement = visual_rng.randrange(valid_visual_count)
                if replacement < 20:
                    valid_visuals[replacement] = visual
        duplicates = _duplicates(sent_ids)
        for duplicate in duplicates:
            all_errors.append(
                {"split": split, "code": "duplicate_sent_id", "detail": duplicate}
            )
            counters["duplicate_sent_id"] += 1
        global_issue_counts.update(counters)
        split_stats[split] = {
            "samples": len(rows),
            "unique_scenes": len(set(scenes)),
            "unique_rgb_images": len(set(images)),
            "unique_referring_expressions": len(set(sentences)),
            "duplicate_sent_ids": duplicates,
            "issue_counts": {
                code: int(counters.get(code, 0)) for code in ISSUE_CODES
            },
        }

    leakage: dict[str, Any] = {}
    split_names = list(rows_by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            left_rows, right_rows = rows_by_split[left], rows_by_split[right]
            leakage[f"{left}_{right}"] = {
                "scene_ids": sorted(
                    {str(row.get("scene_id")) for row in left_rows}
                    & {str(row.get("scene_id")) for row in right_rows}
                ),
                "sent_ids": sorted(
                    {str(row.get("sent_id")) for row in left_rows}
                    & {str(row.get("sent_id")) for row in right_rows}
                ),
            }

    bert_counts = np.asarray([row["bert_tokens"] for row in token_records], dtype=np.float64)
    word_counts = np.asarray([row["whitespace_words"] for row in token_records], dtype=np.float64)
    truncated = [row["sent_id"] for row in token_records if row["truncated"]]
    token_audit = {
        "samples": len(token_records),
        "max_tokens": max_tokens,
        "whitespace_word_count": {
            "maximum": int(word_counts.max()) if len(word_counts) else 0,
            "mean": float(word_counts.mean()) if len(word_counts) else 0.0,
            "median": float(np.median(word_counts)) if len(word_counts) else 0.0,
        },
        "bert_token_count": {
            "maximum": int(bert_counts.max()) if len(bert_counts) else 0,
            "mean": float(bert_counts.mean()) if len(bert_counts) else 0.0,
            "median": float(np.median(bert_counts)) if len(bert_counts) else 0.0,
        },
        "over_max_tokens_count": len(truncated),
        "over_max_tokens_ratio": len(truncated) / len(token_records) if token_records else 0.0,
        "truncated_sent_ids": truncated,
    }

    visualized: list[str] = []
    if visualization_dir is not None:
        destination = Path(visualization_dir)
        destination.mkdir(parents=True, exist_ok=True)
        chosen = valid_visuals
        for number, (row, rgb, target) in enumerate(chosen):
            resized_rgb, resized_mask = _resize_for_audit(rgb, target, image_size)
            figure, axes = plt.subplots(1, 4, figsize=(14, 4))
            for axis, panel, title in zip(
                axes,
                (rgb, target, resized_rgb, resized_mask),
                ("Original RGB", "Original binary GT", "Resized RGB", "Resized nearest GT"),
            ):
                axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0)
                axis.set_title(title)
                axis.axis("off")
            figure.suptitle(
                textwrap.fill(
                    f"{row['sentence']} | {row['sent_id']} | {row['scene_id']}", 120
                ),
                fontsize=9,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.86))
            output = destination / f"{number:02d}_{row['sent_id']}.png"
            figure.savefig(output, dpi=120)
            plt.close(figure)
            visualized.append(str(output.resolve()))

    leakage_errors = sum(
        len(pair[kind])
        for pair in leakage.values()
        for kind in ("scene_ids", "sent_ids")
    )
    return {
        "status": "PASS" if not all_errors and leakage_errors == 0 else "FAIL",
        "splits": split_stats,
        "leakage": leakage,
        "token_length_audit": token_audit,
        "errors": all_errors,
        "error_count": len(all_errors),
        "integrity_issue_counts": {
            code: int(global_issue_counts.get(code, 0)) for code in ISSUE_CODES
        },
        "leakage_item_count": leakage_errors,
        "visualizations": {
            "requested": 20,
            "created": len(visualized),
            "seed": seed,
            "files": visualized,
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# OCID-VLG Dataset Audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Integrity errors: {report['error_count']}",
        f"- Leakage items: {report['leakage_item_count']}",
        f"- Fixed-seed visualizations: {report['visualizations']['created']}/20",
        "",
        "## Split summary",
        "",
        "| split | samples | scenes | RGB images | expressions | duplicate sent IDs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, values in report["splits"].items():
        lines.append(
            f"| {split} | {values['samples']} | {values['unique_scenes']} | "
            f"{values['unique_rgb_images']} | {values['unique_referring_expressions']} | "
            f"{len(values['duplicate_sent_ids'])} |"
        )
    token = report["token_length_audit"]
    lines.extend(
        [
            "",
            "## Token lengths",
            "",
            f"- Maximum configured tokens: {token['max_tokens']}",
            f"- Maximum/mean/median BERT tokens: "
            f"{token['bert_token_count']['maximum']}/"
            f"{token['bert_token_count']['mean']:.3f}/"
            f"{token['bert_token_count']['median']:.3f}",
            f"- Truncated: {token['over_max_tokens_count']} "
            f"({token['over_max_tokens_ratio']:.6%})",
            "",
            "## Integrity findings",
            "",
        ]
    )
    if report["errors"]:
        lines.extend(
            f"- `{item.get('code')}` {item.get('split', '')} "
            f"{item.get('sent_id', '')}: {item.get('detail', '')}"
            for item in report["errors"][:200]
        )
        if len(report["errors"]) > 200:
            lines.append(f"- … {len(report['errors']) - 200} more; see JSON.")
    else:
        lines.append("- None.")
    lines.extend(["", "## Leakage", "", "```json", json.dumps(report["leakage"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path, default=Path("outputs/dataset_audit.json"))
    parser.add_argument("--token-output", type=Path, default=Path("outputs/token_length_audit.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/OCID_VLG_DATASET_AUDIT.md"))
    parser.add_argument("--visualization-dir", type=Path, default=Path("outputs/audit_visualizations"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = _default_tokenizer(args.tokenizer)
    report = audit_manifests(
        {
            "train": args.train_manifest,
            "val": args.val_manifest,
            "test": args.test_manifest,
        },
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        image_size=args.image_size,
        seed=args.seed,
        visualization_dir=args.visualization_dir,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.token_output.parent.mkdir(parents=True, exist_ok=True)
    args.token_output.write_text(
        json.dumps(report["token_length_audit"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.markdown_output, report)
    print(json.dumps({"status": report["status"], "errors": report["error_count"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
