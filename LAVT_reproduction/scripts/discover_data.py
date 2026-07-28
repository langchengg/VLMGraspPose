#!/usr/bin/env python3
"""Discover canonical OCID-VLG/HiFi data and emit source-resolution manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


SPLITS = ("train", "val", "test")
PRUNE_NAMES = {".git", ".venv", "__pycache__", "node_modules", "predictions", "checkpoints"}


def stable_sent_id(scene_id: str, question_index: int) -> str:
    question_index = int(question_index)
    digest = hashlib.sha256(f"{scene_id}\t{question_index}".encode("utf-8")).hexdigest()[:16]
    return f"q{question_index:07d}_{digest}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_ocid_root(path: str | Path, version: str = "unique") -> list[str]:
    root = Path(path).expanduser().resolve()
    errors: list[str] = []
    if not (root / "OCID_sub_class_dict.py").is_file():
        errors.append("missing OCID_sub_class_dict.py")
    for split in SPLITS:
        expression_path = root / "refer" / version / f"{split}_expressions.json"
        if not expression_path.is_file():
            errors.append(f"missing {expression_path.relative_to(root)}")
            continue
        try:
            payload = _read_json(expression_path)
            rows = payload["data"]
            if not rows:
                errors.append(f"empty {expression_path.relative_to(root)}")
                continue
            sequence, filename = rows[0]["image_filename"].split(",", 1)
            if not (root / sequence / "rgb" / filename).is_file():
                errors.append(f"first RGB missing for {split}")
            if not (root / sequence / "seg_mask_instances_combi" / filename).is_file():
                errors.append(f"first combined instance mask missing for {split}")
        except Exception as error:
            errors.append(f"invalid {expression_path.relative_to(root)}: {error}")
    return errors


def discover_ocid_roots(search_roots: Iterable[str | Path], version: str) -> list[Path]:
    candidates: dict[str, Path] = {}
    for search_root in search_roots:
        start = Path(search_root).expanduser()
        if not start.exists():
            continue
        for directory, names, files in os.walk(start, followlinks=False):
            names[:] = [name for name in names if name not in PRUNE_NAMES]
            if "OCID_sub_class_dict.py" not in files:
                continue
            candidate = Path(directory).resolve()
            if not validate_ocid_root(candidate, version):
                candidates[str(candidate)] = candidate
            # An OCID root contains tens of thousands of assets; no nested
            # directory can be a second root, so stop descending here.
            names[:] = []
    return sorted(candidates.values(), key=str)


def _hifi_manifest_pair(hifi_root: Path) -> tuple[Path, Path] | None:
    preferred = sorted(
        hifi_root.glob("runs/*/dataset_and_batch_metadata.json"),
        key=lambda path: (
            (path.parent / "EVALUATION_COMPLETE.json").is_file(),
            path.stat().st_mtime,
        ),
        reverse=True,
    )
    for metadata_path in preferred:
        try:
            metadata = _read_json(metadata_path)
            train = Path(metadata["train_manifest"]).expanduser().resolve()
            test = Path(metadata["test_manifest"]).expanduser().resolve()
            frozen_train = metadata_path.parent / "ocid_vlg_train.json"
            frozen_test = metadata_path.parent / "ocid_vlg_test.json"
            if frozen_train.is_file() and frozen_test.is_file():
                return frozen_train.resolve(), frozen_test.resolve()
            if train.is_file() and test.is_file():
                return train, test
        except (KeyError, OSError, ValueError, TypeError):
            continue
    direct = (
        hifi_root / "hifics" / "datasets" / "ocidvlg_final_dataset" / "train" / "ocid_vlg_train.json",
        hifi_root / "hifics" / "datasets" / "ocidvlg_final_dataset" / "test" / "ocid_vlg_test.json",
    )
    return (direct[0].resolve(), direct[1].resolve()) if all(path.is_file() for path in direct) else None


def _source_rows(root: Path, version: str, split: str) -> list[dict[str, Any]]:
    path = root / "refer" / version / f"{split}_expressions.json"
    payload = _read_json(path)
    expected_info = {"split": split, "version": version}
    if payload.get("info") != expected_info:
        raise ValueError(f"{path}: info {payload.get('info')!r} != {expected_info!r}")
    return payload["data"]


def _index_source(rows: list[dict[str, Any]]) -> dict[tuple[str, int], tuple[int, dict[str, Any]]]:
    result: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        key = (str(row["image_filename"]), int(row["question_index"]))
        if key in result:
            raise ValueError(f"duplicate source identity: {key}")
        result[key] = (index, row)
    return result


def _validate_hifi_rows(
    hifi_path: Path, source_rows: list[dict[str, Any]], split: str
) -> list[tuple[int, dict[str, Any]]]:
    frozen = _read_json(hifi_path)
    if not isinstance(frozen, list):
        raise ValueError(f"{hifi_path}: HiFi manifest must be a JSON list")
    source_by_id = _index_source(source_rows)
    aligned: list[tuple[int, dict[str, Any]]] = []
    seen: set[tuple[str, int]] = set()
    for position, row in enumerate(frozen):
        key = (str(row["scene_id"]), int(row["question_index"]))
        if key in seen:
            raise ValueError(f"{hifi_path}: duplicate identity {key}")
        seen.add(key)
        source = source_by_id.get(key)
        if source is None:
            raise ValueError(f"{hifi_path}: row {position} is absent from official {split}")
        index, source_row = source
        if str(row["text"]) != str(source_row["question"]):
            raise ValueError(f"{hifi_path}: text mismatch at row {position}")
        aligned.append((index, source_row))
    if len(aligned) != len(source_rows):
        raise ValueError(
            f"{hifi_path}: frozen {split} count {len(aligned)} != official count {len(source_rows)}"
        )
    return aligned


def build_manifest_rows(
    root: Path,
    version: str,
    split: str,
    hifi_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    source_rows = _source_rows(root, version, split)
    aligned = (
        _validate_hifi_rows(hifi_manifest, source_rows, split)
        if hifi_manifest is not None
        else list(enumerate(source_rows))
    )
    result: list[dict[str, Any]] = []
    for dataset_index, source in aligned:
        scene_id = str(source["image_filename"])
        sequence, filename = scene_id.split(",", 1)
        raw_question_index = int(source["question_index"])
        result.append(
            {
                "dataset_index": dataset_index,
                "sent_id": stable_sent_id(scene_id, raw_question_index),
                "raw_question_index": raw_question_index,
                "scene_id": scene_id,
                "image_path": str((root / sequence / "rgb" / filename).resolve()),
                "mask_path": str(
                    (root / sequence / "seg_mask_instances_combi" / filename).resolve()
                ),
                "sentence": str(source["question"]),
                "objID": int(source["answer"]),
                "split": split,
                "dataset_version": version,
            }
        )
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-root",
        action="append",
        dest="search_roots",
        default=[],
        help="Repeatable; defaults to VLMGraspPose, Downloads, and cwd.",
    )
    parser.add_argument("--ocid-root", type=Path)
    parser.add_argument("--hifi-root", type=Path)
    parser.add_argument("--ocid-api-root", type=Path)
    parser.add_argument("--version", default="unique")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--paths-yaml", type=Path, default=Path("configs/paths.local.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd().resolve()
    defaults = (cwd.parent, Path.home() / "Downloads", cwd)
    search_roots = args.search_roots or defaults
    candidates = discover_ocid_roots(search_roots, args.version)
    if args.ocid_root:
        root = args.ocid_root.expanduser().resolve()
        errors = validate_ocid_root(root, args.version)
        if errors:
            raise ValueError(f"invalid --ocid-root {root}: {errors}")
    else:
        if not candidates:
            raise FileNotFoundError(f"no valid OCID-VLG {args.version!r} root discovered")
        hifi_symlink = cwd.parent / "HiFi_reproduction" / "hifics" / "datasets" / "OCID-VLG"
        preferred = hifi_symlink.resolve() if hifi_symlink.exists() else None
        root = preferred if preferred in candidates else candidates[0]

    hifi_root = (
        args.hifi_root.expanduser().resolve()
        if args.hifi_root
        else cwd.parent / "HiFi_reproduction"
    )
    pair = _hifi_manifest_pair(hifi_root) if hifi_root.is_dir() else None
    frozen_by_split = {"train": pair[0], "test": pair[1]} if pair else {}
    hifi_prediction_manifest = (
        pair[1].parent / "predictions" / "export_manifest.json" if pair else None
    )
    manifests: dict[str, str] = {}
    for split in SPLITS:
        rows = build_manifest_rows(root, args.version, split, frozen_by_split.get(split))
        output = args.output_dir / f"dataset_manifest_{split}.jsonl"
        write_jsonl(output, rows)
        manifests[split] = str(output.resolve())

    api_root = (
        args.ocid_api_root.expanduser().resolve()
        if args.ocid_api_root
        else hifi_root / "hifics" / "datasets"
    )
    configuration = {
        "ocid_root": str(root),
        "ocid_api_root": str(api_root),
        "ocid_version": args.version,
        "hifi_root": str(hifi_root) if hifi_root.is_dir() else None,
        "hifi_frozen_train_manifest": str(pair[0]) if pair else None,
        "hifi_frozen_test_manifest": str(pair[1]) if pair else None,
        "hifi_prediction_manifest": (
            str(hifi_prediction_manifest.resolve())
            if hifi_prediction_manifest is not None
            and hifi_prediction_manifest.is_file()
            else None
        ),
        "manifests": manifests,
        "discovered_ocid_roots": [str(path) for path in candidates],
        "split_fairness": {
            "train_matches_hifi_frozen": pair is not None,
            "test_matches_hifi_frozen": pair is not None,
            "val_source": f"official_{args.version}_val",
        },
    }
    args.paths_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.paths_yaml.write_text(
        yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(json.dumps(configuration, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
