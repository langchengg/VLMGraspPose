"""OCID-VLG adapter for LAVT referring-image segmentation.

The dataset's own loader remains the source of truth for split indexing.  This
adapter deliberately reads only RGB, the selected binary instance mask, text,
and identifiers from it; grasp and depth fields never enter the returned
sample.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
MANIFEST_REQUIRED_FIELDS = {
    "dataset_index",
    "sent_id",
    "scene_id",
    "image_path",
    "mask_path",
    "sentence",
    "objID",
    "split",
    "dataset_version",
    "raw_question_index",
}


def stable_sent_id(scene_id: str, question_index: int) -> str:
    """Match the frozen HiFi-CS prediction/export identity contract."""
    question_index = int(question_index)
    identity = f"{scene_id}\t{question_index}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"q{question_index:07d}_{digest}"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: manifest row is not an object")
            rows.append(row)
    return rows


def paired_resize(
    image: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    size: int | tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize aligned RGB/mask tensors with type-appropriate interpolation."""
    if isinstance(size, int):
        output_size = (size, size)
    else:
        output_size = tuple(int(value) for value in size)
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError(f"invalid output size: {size!r}")

    image_tensor = (
        image
        if torch.is_tensor(image)
        else torch.from_numpy(np.asarray(image).copy())
    )
    if image_tensor.ndim != 3:
        raise ValueError(f"RGB image must be HxWx3 or 3xHxW, got {tuple(image_tensor.shape)}")
    if image_tensor.shape[-1] == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    if image_tensor.shape[0] != 3:
        raise ValueError(f"RGB image must have three channels, got {tuple(image_tensor.shape)}")
    image_tensor = image_tensor.contiguous().to(torch.float32)
    if image_tensor.numel() and float(image_tensor.max()) > 1.0:
        image_tensor = image_tensor / 255.0

    mask_tensor = (
        mask
        if torch.is_tensor(mask)
        else torch.from_numpy(np.asarray(mask).copy())
    )
    if mask_tensor.ndim != 2:
        raise ValueError(f"target mask must be HxW, got {tuple(mask_tensor.shape)}")
    if tuple(image_tensor.shape[-2:]) != tuple(mask_tensor.shape):
        raise ValueError(
            f"unaligned source RGB/mask shapes: {tuple(image_tensor.shape[-2:])} "
            f"!= {tuple(mask_tensor.shape)}"
        )
    original_binary = (mask_tensor != 0).to(torch.uint8)
    resized_image = torch_functional.interpolate(
        image_tensor[None],
        size=output_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    resized_mask = torch_functional.interpolate(
        original_binary[None, None].to(torch.float32),
        size=output_size,
        mode="nearest",
    )[0, 0].to(torch.int64)
    values = set(torch.unique(resized_mask).tolist())
    if not values <= {0, 1}:
        raise RuntimeError(f"nearest mask resize produced non-binary values: {values}")
    return resized_image, resized_mask


def _module_file(api_root: str | Path) -> Path:
    api_root = Path(api_root).expanduser().resolve()
    candidates = (
        api_root,
    ) if api_root.is_file() else (
        api_root / "load_ocidvlg.py",
        api_root / "dataset.py",
        api_root / "utils" / "dataset.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"no supported local OCID-VLG API found; checked: {rendered}")


def import_ocid_dataset_class(
    api_root: str | Path, ocid_root: str | Path
) -> type[Dataset]:
    """Dynamically import, without copying, a local official dataset class."""
    module_path = _module_file(api_root)
    module_name = f"_lavt_ocid_api_{hashlib.sha256(str(module_path).encode()).hexdigest()[:12]}"
    root_string = str(Path(ocid_root).expanduser().resolve())
    api_string = str(module_path.parent)
    inserted: list[str] = []
    for entry in (root_string, api_string):
        if entry not in sys.path:
            sys.path.insert(0, entry)
            inserted.append(entry)
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for entry in inserted:
            if entry in sys.path:
                sys.path.remove(entry)
    dataset_class = getattr(module, "OCIDVLGDataset", None)
    if dataset_class is None:
        raise ImportError(
            f"{module_path} does not expose OCIDVLGDataset; the CROG "
            "RefOCIDGraspDataset API is not a safe drop-in replacement"
        )
    return dataset_class


def _default_tokenizer(tokenizer_name: str):
    from bert.tokenization_bert import BertTokenizer

    return BertTokenizer.from_pretrained(tokenizer_name)


def encode_sentence(tokenizer: Any, sentence: str, max_tokens: int) -> dict[str, Any]:
    if max_tokens < 2:
        raise ValueError("max_tokens must leave room for BERT special tokens")
    full_ids = list(tokenizer.encode(text=sentence, add_special_tokens=True))
    input_ids = full_ids[:max_tokens]
    attention = [1] * len(input_ids)
    padding = max_tokens - len(input_ids)
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    pad_id = 0 if pad_id is None else int(pad_id)
    input_ids.extend([pad_id] * padding)
    attention.extend([0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0),
        "attention_mask": torch.tensor(attention, dtype=torch.int64).unsqueeze(0),
        "bert_token_count": len(full_ids),
        "truncated": len(full_ids) > max_tokens,
    }


class OCIDVLGLAVTDataset(Dataset):
    """LAVT-ready RGB/text/segmentation view over official OCID-VLG."""

    def __init__(
        self,
        ocid_root: str | Path,
        ocid_api_root: str | Path,
        split: str,
        version: str = "unique",
        image_size: int | tuple[int, int] = 480,
        max_tokens: int = 20,
        tokenizer_name: str = "bert-base-uncased",
        tokenizer: Any | None = None,
        manifest_path: str | Path | None = None,
        manifest_rows: Iterable[dict[str, Any]] | None = None,
        normalize: bool = True,
    ):
        self.root = Path(ocid_root).expanduser().resolve()
        self.split = split
        self.version = version
        self.image_size = image_size
        self.max_tokens = int(max_tokens)
        self.normalize = bool(normalize)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split!r}")
        if not self.root.is_dir():
            raise FileNotFoundError(f"OCID-VLG root does not exist: {self.root}")

        dataset_class = import_ocid_dataset_class(ocid_api_root, self.root)
        self.official_dataset = dataset_class(
            str(self.root),
            split=split,
            version=version,
            transform_img=None,
            transform_grasp=None,
            with_depth=False,
            with_segm_mask=True,
            with_grasp_masks=False,
        )
        self.tokenizer = tokenizer or _default_tokenizer(tokenizer_name)
        if manifest_path is not None and manifest_rows is not None:
            raise ValueError("pass only one of manifest_path and manifest_rows")
        selected_rows = (
            load_jsonl(manifest_path)
            if manifest_path is not None
            else list(manifest_rows) if manifest_rows is not None else None
        )
        self.records = self._validate_or_build_records(selected_rows)

    def _official_record(self, index: int) -> dict[str, Any]:
        dataset = self.official_dataset
        required_attributes = (
            "scene_ids",
            "sent_indices",
            "sentences",
            "objIDs",
            "rgb_paths",
            "mask_paths",
        )
        missing = [name for name in required_attributes if not hasattr(dataset, name)]
        if missing:
            raise AttributeError(
                "official OCIDVLGDataset lacks raw metadata required to bypass "
                f"grasp preprocessing: {missing}"
            )
        scene_id = str(dataset.scene_ids[index])
        raw_question_index = int(dataset.sent_indices[index])
        return {
            "dataset_index": index,
            "sent_id": stable_sent_id(scene_id, raw_question_index),
            "scene_id": scene_id,
            "image_path": str((self.root / dataset.rgb_paths[index]).resolve()),
            "mask_path": str((self.root / dataset.mask_paths[index]).resolve()),
            "sentence": str(dataset.sentences[index]),
            "objID": int(dataset.objIDs[index]),
            "split": self.split,
            "dataset_version": self.version,
            "raw_question_index": raw_question_index,
        }

    def _validate_or_build_records(
        self, rows: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        official = [self._official_record(index) for index in range(len(self.official_dataset))]
        if rows is None:
            return official
        by_id = {row["sent_id"]: row for row in official}
        if len(by_id) != len(official):
            raise ValueError("official split contains duplicate stable sentence IDs")
        validated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for position, row in enumerate(rows):
            missing = MANIFEST_REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"manifest row {position} is missing {sorted(missing)}")
            sent_id = str(row["sent_id"])
            if sent_id in seen:
                raise ValueError(f"duplicate manifest sent_id: {sent_id}")
            seen.add(sent_id)
            source = by_id.get(sent_id)
            if source is None:
                raise ValueError(f"manifest sent_id is absent from official {self.split}: {sent_id}")
            for field in MANIFEST_REQUIRED_FIELDS:
                expected = source[field]
                observed = row[field]
                if field in {"image_path", "mask_path"}:
                    observed = str(Path(observed).expanduser().resolve())
                if str(observed) != str(expected):
                    raise ValueError(
                        f"manifest/source mismatch for {sent_id} field {field}: "
                        f"{observed!r} != {expected!r}"
                    )
            validated.append(source)
        return validated

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        official_index = int(record["dataset_index"])
        dataset = self.official_dataset
        image = dataset.get_image_from_path(record["image_path"])
        instance_mask = dataset.get_mask_from_path(record["mask_path"])
        if image is None:
            raise OSError(f"official API could not read RGB: {record['image_path']}")
        if instance_mask is None:
            raise OSError(f"official API could not read mask: {record['mask_path']}")
        image = np.asarray(image)
        instance_mask = np.asarray(instance_mask)
        if instance_mask.ndim == 3 and instance_mask.shape[-1] == 1:
            instance_mask = instance_mask[..., 0]
        if instance_mask.ndim != 2:
            raise ValueError(f"instance mask must be 2-D: {record['mask_path']}")
        binary_original = (instance_mask == int(record["objID"])).astype(np.uint8)
        resized_image, resized_mask = paired_resize(image, binary_original, self.image_size)
        if self.normalize:
            resized_image = (resized_image - IMAGENET_MEAN) / IMAGENET_STD
        language = encode_sentence(self.tokenizer, record["sentence"], self.max_tokens)
        height, width = image.shape[:2]
        if binary_original.shape != (height, width):
            raise ValueError(
                f"RGB/mask shape mismatch after official API read: "
                f"{(height, width)} != {binary_original.shape}"
            )
        return {
            "image": resized_image,
            "target_model_resolution": resized_mask,
            "target_original_resolution": torch.from_numpy(binary_original.copy()).to(torch.int64),
            "input_ids": language["input_ids"],
            "attention_mask": language["attention_mask"],
            "sentence": record["sentence"],
            "sent_id": record["sent_id"],
            "raw_question_index": int(record["raw_question_index"]),
            "scene_id": record["scene_id"],
            "objID": int(record["objID"]),
            "original_size": torch.tensor([height, width], dtype=torch.int64),
            "image_path": record["image_path"],
            "mask_path": record["mask_path"],
            "bert_token_count": language["bert_token_count"],
            "token_truncated": language["truncated"],
            "dataset_index": official_index,
        }

    def token_length_audit(self) -> dict[str, Any]:
        counts: list[int] = []
        words: list[int] = []
        truncated: list[str] = []
        for record in self.records:
            encoded = encode_sentence(self.tokenizer, record["sentence"], self.max_tokens)
            counts.append(int(encoded["bert_token_count"]))
            words.append(len(record["sentence"].split()))
            if encoded["truncated"]:
                truncated.append(record["sent_id"])
        count_array = np.asarray(counts, dtype=np.float64)
        word_array = np.asarray(words, dtype=np.float64)
        return {
            "samples": len(counts),
            "max_tokens": self.max_tokens,
            "whitespace_word_count": {
                "maximum": int(word_array.max()) if len(words) else 0,
                "mean": float(word_array.mean()) if len(words) else 0.0,
                "median": float(np.median(word_array)) if len(words) else 0.0,
            },
            "bert_token_count": {
                "maximum": int(count_array.max()) if len(counts) else 0,
                "mean": float(count_array.mean()) if len(counts) else 0.0,
                "median": float(np.median(count_array)) if len(counts) else 0.0,
            },
            "over_max_tokens_count": len(truncated),
            "over_max_tokens_ratio": len(truncated) / len(counts) if counts else 0.0,
            "truncated_sent_ids": truncated,
        }
