"""Frozen HiFi-CS grounding and guarded decoder-only domain adaptation.

This module deliberately reuses the retained HiFi implementation instead of
reimplementing the network.  Loading is fail-closed: both immutable weight
files, the checkpoint schema, training step, exact architecture constructor,
and trainable-state key set must match before a model is returned.

The public mask convention is target foreground.  The retained network was
trained with a background-oriented logit, hence foreground probability is
``sigmoid(-background_logit)``.  RGB is resized bilinearly to 352 x 352 with
no normalization beyond conversion to ``[0, 1]``.  Foreground probability is
then resized bilinearly to the native image size before thresholding.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .io import atomic_npz, canonical_sha256, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HIFI_REPRODUCTION_ROOT = REPOSITORY_ROOT / "HiFi_reproduction"
HIFI_SOURCE_PATH = HIFI_REPRODUCTION_ROOT / "hifics/models/hifics.py"
DEFAULT_HIFI_CHECKPOINT = (
    HIFI_REPRODUCTION_ROOT
    / "runs/hifics_ocidvlg_hierfilm_20260727_214615/checkpoints/best.pth"
)
DEFAULT_CLIP_WEIGHT = Path.home() / ".cache/clip/ViT-B-16.pt"

EXPECTED_HIFI_CHECKPOINT_SHA256 = (
    "b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601"
)
EXPECTED_CLIP_WEIGHT_SHA256 = (
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
)
EXPECTED_CHECKPOINT_FORMAT = "hifics_hierfilm_trainable_only_v1"
EXPECTED_CHECKPOINT_STEP = 19_728
EXPECTED_TRAINABLE_PARAMETER_COUNT = 92
IMAGE_RESOLUTION = 352
FOREGROUND_THRESHOLD = 0.5

_ADAPTATION_PREFIXES = (
    "reduces.",
    "film_stages.",
    "blocks.",
    "trans_conv.",
)
_RGB_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_RESOLUTION, IMAGE_RESOLUTION),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
    ]
)


@dataclass(frozen=True)
class HiFiModelBundle:
    """Strictly loaded model and immutable provenance needed downstream."""

    model: torch.nn.Module
    device: torch.device
    mode: Literal["zero_shot", "adaptation"]
    checkpoint_path: Path
    checkpoint_sha256: str
    clip_weight_path: Path
    clip_weight_sha256: str
    checkpoint_metadata: Mapping[str, Any]
    adaptation_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class GroundingPrediction:
    """One real model prediction at model and native resolutions."""

    probability_352: np.ndarray
    native_probability: np.ndarray
    native_mask: np.ndarray
    native_height: int
    native_width: int
    foreground_threshold: float
    query: str
    device: str


@dataclass(frozen=True)
class AdaptationResult:
    """Validation-selected decoder adaptation result (never test-selected)."""

    model: torch.nn.Module
    best_epoch: int
    best_validation_miou: float
    epochs_completed: int
    stopped_early: bool
    history: tuple[Mapping[str, Any], ...]
    output_checkpoint: Path | None
    split_contract: Mapping[str, Any]


def _verify_file(path: Path, expected_sha256: str, label: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed} at {resolved}"
        )
    return observed


@lru_cache(maxsize=1)
def _hierarchical_model_class() -> type[torch.nn.Module]:
    """Load the retained source file without risking a ``models`` collision."""

    if not HIFI_SOURCE_PATH.is_file():
        raise FileNotFoundError(f"retained HiFi source missing: {HIFI_SOURCE_PATH}")
    module_name = "_graspnet6d_retained_hifics"
    spec = importlib.util.spec_from_file_location(module_name, HIFI_SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import retained HiFi source: {HIFI_SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    architecture = getattr(module, "HierarchicalCLIPDensePredT", None)
    if architecture is None or not issubclass(architecture, torch.nn.Module):
        raise ImportError("retained source omits HierarchicalCLIPDensePredT")
    return architecture


def resolve_grounding_device(requested: str = "cpu") -> torch.device:
    """Resolve only an explicitly requested device; CPU is the safe default."""

    normalized = str(requested).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not built and available")
        return torch.device("mps")
    raise ValueError("grounding device must be 'cpu' or 'mps'")


def _checkpoint_payload(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("HiFi checkpoint payload must be a mapping")
    if payload.get("format") != EXPECTED_CHECKPOINT_FORMAT:
        raise ValueError(
            "unsupported HiFi checkpoint format: "
            f"{payload.get('format')!r}"
        )
    metadata = payload.get("metadata")
    state = payload.get("trainable_state")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        raise ValueError("HiFi checkpoint omits metadata or trainable_state")
    if int(metadata.get("global_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError(
            "unexpected HiFi checkpoint step: "
            f"{metadata.get('global_step')!r}"
        )
    recorded_clip_sha = (
        metadata.get("weight_sources", {})
        .get("clip", {})
        .get("cache_sha256")
    )
    if recorded_clip_sha != EXPECTED_CLIP_WEIGHT_SHA256:
        raise ValueError(
            "checkpoint CLIP provenance disagrees with the retained weight"
        )
    if len(state) != EXPECTED_TRAINABLE_PARAMETER_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TRAINABLE_PARAMETER_COUNT} trainable tensors, "
            f"found {len(state)}"
        )
    for name, tensor in state.items():
        if not isinstance(name, str) or not name.startswith(_ADAPTATION_PREFIXES):
            raise ValueError(f"unexpected trainable checkpoint key: {name!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint value is not a tensor: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"non-finite checkpoint tensor: {name}")
    return payload, tuple(sorted(str(name) for name in state))


def load_hifi_model(
    *,
    device: str = "cpu",
    mode: Literal["zero_shot", "adaptation"] = "zero_shot",
    checkpoint_path: str | os.PathLike[str] = DEFAULT_HIFI_CHECKPOINT,
    clip_weight_path: str | os.PathLike[str] = DEFAULT_CLIP_WEIGHT,
) -> HiFiModelBundle:
    """Load the exact retained five-stage HiFi-CS model.

    ``zero_shot`` freezes every parameter.  ``adaptation`` exposes only the
    existing non-CLIP visual projections, five FiLM stages, five decoder
    blocks, and segmentation head.  The CLIP image and text encoders remain
    frozen in both modes.
    """

    if mode not in {"zero_shot", "adaptation"}:
        raise ValueError("mode must be 'zero_shot' or 'adaptation'")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    clip_weight = Path(clip_weight_path).expanduser().resolve()
    default_clip = DEFAULT_CLIP_WEIGHT.expanduser().resolve()
    if clip_weight != default_clip:
        raise ValueError(
            "the retained architecture calls OpenAI CLIP's default cache; "
            f"clip_weight_path must resolve to {default_clip}"
        )
    checkpoint_sha = _verify_file(
        checkpoint, EXPECTED_HIFI_CHECKPOINT_SHA256, "HiFi checkpoint"
    )
    clip_sha = _verify_file(
        clip_weight, EXPECTED_CLIP_WEIGHT_SHA256, "OpenAI CLIP ViT-B/16 weight"
    )
    payload, checkpoint_names = _checkpoint_payload(checkpoint)

    architecture = _hierarchical_model_class()
    model = architecture(
        version="ViT-B/16",
        extract_layers=(1, 3, 5, 7, 9),
        reduce_dim=64,
        n_heads=4,
        cond_layer=None,
        extended_film=True,
        hierarchical_film=True,
    )
    constructor_trainable_names = tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    if constructor_trainable_names != checkpoint_names:
        missing = sorted(set(constructor_trainable_names) - set(checkpoint_names))
        unexpected = sorted(set(checkpoint_names) - set(constructor_trainable_names))
        raise ValueError(
            "checkpoint does not match the exact retained architecture: "
            f"missing={missing}, unexpected={unexpected}"
        )

    complete_state = model.state_dict()
    complete_state.update(payload["trainable_state"])
    incompatible = model.load_state_dict(complete_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "strict retained checkpoint load failed: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if mode == "adaptation":
        by_name = dict(model.named_parameters())
        for name in checkpoint_names:
            by_name[name].requires_grad_(True)
    trainable_after_mode = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    expected_after_mode = set() if mode == "zero_shot" else set(checkpoint_names)
    if trainable_after_mode != expected_after_mode:
        raise RuntimeError("HiFi freeze policy did not produce the declared state")
    clip_trainable = [
        name
        for name, parameter in model.clip_model.named_parameters()
        if parameter.requires_grad
    ]
    if clip_trainable:
        raise RuntimeError(f"CLIP parameters unexpectedly trainable: {clip_trainable}")

    selected_device = resolve_grounding_device(device)
    model.to(selected_device)
    model.eval()
    model.clip_model.eval()
    metadata = json.loads(json.dumps(payload["metadata"], allow_nan=False))
    return HiFiModelBundle(
        model=model,
        device=selected_device,
        mode=mode,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        clip_weight_path=clip_weight,
        clip_weight_sha256=clip_sha,
        checkpoint_metadata=MappingProxyType(metadata),
        adaptation_parameter_names=checkpoint_names,
    )


def _rgb_image(source: str | os.PathLike[str] | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RGB image missing: {path}")
        with Image.open(path) as opened:
            return opened.convert("RGB").copy()
    if isinstance(source, Image.Image):
        return source.convert("RGB").copy()
    array = np.asarray(source)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"RGB array must be H x W x 3/4, received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("RGB array contains non-finite values")
    if np.issubdtype(array.dtype, np.floating):
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError("floating RGB arrays must lie in [0, 1]")
        array = np.rint(array * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        if int(array.min()) < 0 or int(array.max()) > 255:
            raise ValueError("integer RGB arrays must lie in [0, 255]")
        array = array.astype(np.uint8)
    return Image.fromarray(array[..., :3], mode="RGB")


def predict_grounding(
    bundle: HiFiModelBundle,
    rgb: str | os.PathLike[str] | Image.Image | np.ndarray,
    query: str,
    *,
    foreground_threshold: float = FOREGROUND_THRESHOLD,
) -> GroundingPrediction:
    """Run one HiFi query and return lossless native-resolution probability."""

    if bundle.mode != "zero_shot":
        # Adapted models can still infer, but the caller must first finish and
        # validation-select adaptation via ``adapt_hifi_decoder``.
        raise ValueError("direct inference requires a zero_shot model bundle")
    normalized_query = str(query).strip()
    if not normalized_query:
        raise ValueError("grounding query must be non-empty")
    if not 0.0 < float(foreground_threshold) < 1.0:
        raise ValueError("foreground_threshold must lie strictly inside (0, 1)")
    image = _rgb_image(rgb)
    native_width, native_height = image.size
    tensor = _RGB_TRANSFORM(image).unsqueeze(0).to(bundle.device)
    bundle.model.eval()
    bundle.model.clip_model.eval()
    with torch.inference_mode():
        output = bundle.model(tensor, [normalized_query], return_features=False)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if tuple(logits.shape) != (1, 1, IMAGE_RESOLUTION, IMAGE_RESOLUTION):
            raise ValueError(f"unexpected HiFi output shape: {tuple(logits.shape)}")
        probability_352_tensor = torch.sigmoid(-logits)
        native_probability_tensor = torch_functional.interpolate(
            probability_352_tensor,
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )
    probability_352 = (
        probability_352_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
    )
    native_probability = (
        native_probability_tensor[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if (
        not np.isfinite(probability_352).all()
        or not np.isfinite(native_probability).all()
        or float(probability_352.min()) < 0.0
        or float(probability_352.max()) > 1.0
        or float(native_probability.min()) < 0.0
        or float(native_probability.max()) > 1.0
    ):
        raise FloatingPointError("HiFi produced an invalid foreground probability")
    native_mask = native_probability >= float(foreground_threshold)
    return GroundingPrediction(
        probability_352=probability_352,
        native_probability=native_probability,
        native_mask=native_mask,
        native_height=native_height,
        native_width=native_width,
        foreground_threshold=float(foreground_threshold),
        query=normalized_query,
        device=str(bundle.device),
    )


def _atomic_png(path: str | os.PathLike[str], mask: np.ndarray) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            suffix=".png",
            prefix=f".{destination.stem}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                stream, format="PNG"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def save_grounding_prediction(
    prediction: GroundingPrediction,
    *,
    probability_path: str | os.PathLike[str],
    mask_path: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Atomically publish the real float32 probabilities and binary PNG."""

    probability_352 = np.asarray(prediction.probability_352, dtype=np.float32)
    native_probability = np.asarray(
        prediction.native_probability, dtype=np.float32
    )
    native_mask = np.asarray(prediction.native_mask, dtype=bool)
    expected_native_shape = (prediction.native_height, prediction.native_width)
    if probability_352.shape != (IMAGE_RESOLUTION, IMAGE_RESOLUTION):
        raise ValueError(
            f"probability_352 has shape {probability_352.shape}, expected "
            f"{(IMAGE_RESOLUTION, IMAGE_RESOLUTION)}"
        )
    if (
        native_probability.shape != expected_native_shape
        or native_mask.shape != expected_native_shape
    ):
        raise ValueError(
            "native probability/mask shape disagrees with declared native size"
        )
    if (
        not np.isfinite(probability_352).all()
        or not np.isfinite(native_probability).all()
        or float(probability_352.min()) < 0.0
        or float(probability_352.max()) > 1.0
        or float(native_probability.min()) < 0.0
        or float(native_probability.max()) > 1.0
    ):
        raise ValueError("grounding probabilities must be finite and inside [0, 1]")
    if not np.array_equal(
        native_mask,
        native_probability >= float(prediction.foreground_threshold),
    ):
        raise ValueError("native mask is inconsistent with probability and threshold")
    probability_destination = atomic_npz(
        probability_path,
        probability_352=probability_352,
        native_probability=native_probability,
        foreground_threshold=np.asarray(
            prediction.foreground_threshold, dtype=np.float32
        ),
    )
    mask_destination = _atomic_png(mask_path, native_mask)
    return MappingProxyType(
        {
            "probability_path": str(probability_destination),
            "probability_sha256": sha256_file(probability_destination),
            "mask_path": str(mask_destination),
            "mask_sha256": sha256_file(mask_destination),
            "native_height": prediction.native_height,
            "native_width": prediction.native_width,
            "foreground_threshold": prediction.foreground_threshold,
            "foreground_probability": "sigmoid(-background_logit)",
            "native_probability_resize": "bilinear_align_corners_false",
        }
    )


def _binary_mask(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask, got {array.shape}")
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"{name} must have a numeric or bool dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array != 0


def _valid_depth(value: Any, *, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(
            f"valid_depth shape {array.shape} != mask shape {expected_shape}"
        )
    if np.issubdtype(array.dtype, np.bool_):
        return array.copy()
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError("valid_depth must be a bool mask or numeric depth map")
    return np.isfinite(array) & (array > 0)


def compute_mask_metrics(
    prediction: Any,
    target: Any,
    *,
    valid_depth: Any | None = None,
    non_target_mask: Any | None = None,
    wrong_object: bool | None = None,
) -> Mapping[str, Any]:
    """Compute one sample's mask diagnostics without using grasp labels.

    When a non-target instance union is provided, ``wrong_object`` means that
    a non-empty prediction overlaps more non-target than target pixels.  A
    caller may instead supply a pre-audited boolean.  Valid-depth coverage is
    the fraction of predicted foreground pixels with finite positive depth and
    is undefined (``None``) for an empty prediction.
    """

    predicted = _binary_mask(prediction, name="prediction")
    truth = _binary_mask(target, name="target")
    if predicted.shape != truth.shape:
        raise ValueError(
            f"prediction shape {predicted.shape} != target shape {truth.shape}"
        )
    intersection = int(np.count_nonzero(predicted & truth))
    union = int(np.count_nonzero(predicted | truth))
    predicted_pixels = int(np.count_nonzero(predicted))
    target_pixels = int(np.count_nonzero(truth))
    iou = 1.0 if union == 0 else float(intersection / union)

    other_overlap: int | None = None
    derived_wrong: bool | None = None
    if non_target_mask is not None:
        other = _binary_mask(non_target_mask, name="non_target_mask")
        if other.shape != predicted.shape:
            raise ValueError("non_target_mask shape does not match prediction")
        other_overlap = int(np.count_nonzero(predicted & other))
        derived_wrong = bool(
            predicted_pixels > 0 and other_overlap > intersection
        )
    if wrong_object is not None:
        supplied_wrong = bool(wrong_object)
        if derived_wrong is not None and supplied_wrong != derived_wrong:
            raise ValueError(
                "supplied wrong_object disagrees with non_target_mask derivation"
            )
        derived_wrong = supplied_wrong

    depth_coverage: float | None = None
    valid_depth_pixels: int | None = None
    if valid_depth is not None:
        valid = _valid_depth(valid_depth, expected_shape=predicted.shape)
        valid_depth_pixels = int(np.count_nonzero(predicted & valid))
        if predicted_pixels:
            depth_coverage = float(valid_depth_pixels / predicted_pixels)

    return MappingProxyType(
        {
            "iou": iou,
            "intersection_px": intersection,
            "union_px": union,
            "predicted_foreground_px": predicted_pixels,
            "target_foreground_px": target_pixels,
            "empty_prediction": predicted_pixels == 0,
            "wrong_object": derived_wrong,
            "target_overlap_px": intersection,
            "non_target_overlap_px": other_overlap,
            "valid_depth_foreground_px": valid_depth_pixels,
            "valid_depth_coverage": depth_coverage,
        }
    )


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty mask-metric group")
    ious = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64)
    wrong_values = [
        bool(row["wrong_object"])
        for row in rows
        if row["wrong_object"] is not None
    ]
    depth_values = [
        float(row["valid_depth_coverage"])
        for row in rows
        if row["valid_depth_coverage"] is not None
    ]
    summary: dict[str, Any] = {
        "samples": len(rows),
        "mean_iou": float(ious.mean()),
        "empty_predictions": int(sum(bool(row["empty_prediction"]) for row in rows)),
        "empty_prediction_rate": float(
            np.mean([bool(row["empty_prediction"]) for row in rows])
        ),
        "wrong_object_evaluable": len(wrong_values),
        "wrong_object_count": int(sum(wrong_values)),
        "wrong_object_rate": (
            None if not wrong_values else float(np.mean(wrong_values))
        ),
        "valid_depth_evaluable": len(depth_values),
        "mean_valid_depth_coverage": (
            None if not depth_values else float(np.mean(depth_values))
        ),
        "precision_comparison": "iou > threshold",
    }
    for percent in (50, 70, 80, 90):
        passes = int(np.count_nonzero(ious > percent / 100.0))
        summary[f"p_at_{percent}"] = float(passes / len(rows))
        summary[f"p_at_{percent}_numerator"] = passes
        summary[f"p_at_{percent}_denominator"] = len(rows)
    return summary


def summarize_mask_metrics(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Derive overall and group/template mask metrics from raw masks."""

    if not records:
        raise ValueError("mask metric records must be non-empty")
    metric_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if "prediction" not in record or "target" not in record:
            raise KeyError(f"mask metric record {index} needs prediction and target")
        group = str(record.get("group", "")).strip()
        template = str(record.get("template", "")).strip()
        if not group or not template:
            raise ValueError(f"mask metric record {index} needs group and template")
        metrics = dict(
            compute_mask_metrics(
                record["prediction"],
                record["target"],
                valid_depth=record.get("valid_depth"),
                non_target_mask=record.get("non_target_mask"),
                wrong_object=record.get("wrong_object"),
            )
        )
        metrics.update(
            {
                "sample_id": str(record.get("sample_id", index)),
                "group": group,
                "template": template,
            }
        )
        metric_rows.append(metrics)

    def grouped(key_function: Any) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in metric_rows:
            buckets.setdefault(str(key_function(row)), []).append(row)
        return {key: _metric_summary(buckets[key]) for key in sorted(buckets)}

    return MappingProxyType(
        {
            "overall": _metric_summary(metric_rows),
            "by_group": grouped(lambda row: row["group"]),
            "by_template": grouped(lambda row: row["template"]),
            "by_group_template": grouped(
                lambda row: f"{row['group']}::{row['template']}"
            ),
            "rows": tuple(MappingProxyType(row) for row in metric_rows),
        }
    )


def _row_identity(row: Mapping[str, Any], index: int, role: str) -> str:
    value = row.get("sample_id", row.get("frame_id", f"{role}:{index}"))
    identity = str(value).strip()
    if not identity:
        raise ValueError(f"{role} row {index} has an empty identity")
    return identity


def validate_adaptation_splits(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Prove domain-adaptation rows are train/validation scene-disjoint.

    The guard requires explicit split labels and rejects any test-labelled row.
    Test data is therefore structurally unavailable to the optimizer and early
    stopping selector.
    """

    if not train_rows or not validation_rows:
        raise ValueError("adaptation needs non-empty train and validation rows")

    def inspect(
        rows: Sequence[Mapping[str, Any]], expected_split: str
    ) -> tuple[set[str], set[str]]:
        scenes: set[str] = set()
        identities: set[str] = set()
        for index, row in enumerate(rows):
            observed_split = str(row.get("split", "")).strip().lower()
            if observed_split != expected_split:
                raise ValueError(
                    f"{expected_split} row {index} has split "
                    f"{observed_split!r}; test and implicit splits are forbidden"
                )
            scene = str(row.get("scene_id", "")).strip()
            if not scene:
                raise ValueError(f"{expected_split} row {index} has no scene_id")
            identity = _row_identity(row, index, expected_split)
            if identity in identities:
                raise ValueError(
                    f"duplicate {expected_split} row identity: {identity}"
                )
            scenes.add(scene)
            identities.add(identity)
        return scenes, identities

    train_scenes, train_ids = inspect(train_rows, "train")
    validation_scenes, validation_ids = inspect(validation_rows, "val")
    scene_overlap = sorted(train_scenes & validation_scenes)
    identity_overlap = sorted(train_ids & validation_ids)
    if scene_overlap or identity_overlap:
        raise ValueError(
            "adaptation train/validation leakage: "
            f"scene_overlap={scene_overlap}, identity_overlap={identity_overlap}"
        )
    contract = {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_scenes": len(train_scenes),
        "validation_scenes": len(validation_scenes),
        "scene_overlap": 0,
        "identity_overlap": 0,
        "test_rows_consumed": 0,
        "early_stopping_split": "val",
        "train_identity_sha256": canonical_sha256(sorted(train_ids)),
        "validation_identity_sha256": canonical_sha256(sorted(validation_ids)),
    }
    return MappingProxyType(contract)


def _query_from_row(row: Mapping[str, Any]) -> str:
    query = str(row.get("query", row.get("text", ""))).strip()
    if not query:
        raise ValueError("adaptation row has no query/text")
    return query


def _source_from_row(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"adaptation row requires one of {keys}")


def _foreground_target(source: Any) -> torch.Tensor:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"adaptation target mask missing: {path}")
        with Image.open(path) as opened:
            array = np.asarray(opened.convert("L"), dtype=np.uint8)
    elif isinstance(source, Image.Image):
        array = np.asarray(source.convert("L"), dtype=np.uint8)
    else:
        array = np.asarray(source)
    binary = _binary_mask(array, name="adaptation target")
    image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    resized = transforms.functional.resize(
        image,
        [IMAGE_RESOLUTION, IMAGE_RESOLUTION],
        interpolation=InterpolationMode.NEAREST,
    )
    target = torch.from_numpy(
        (np.asarray(resized, dtype=np.uint8) != 0).astype(np.float32)
    )
    return target.unsqueeze(0)


class _AdaptationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self._rows[index]
        rgb = _source_from_row(row, "rgb", "rgb_path")
        if any(
            key in row for key in ("target", "target_mask", "mask_path", "gt_mask_path")
        ):
            target = _source_from_row(
                row, "target", "target_mask", "mask_path", "gt_mask_path"
            )
        else:
            label_path = Path(
                str(_source_from_row(row, "instance_label_path"))
            ).expanduser().resolve()
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"adaptation instance label missing: {label_path}"
                )
            raw_label = row.get("target_instance_label")
            if isinstance(raw_label, bool):
                raise ValueError("target_instance_label must be a positive integer")
            try:
                instance_label = int(raw_label)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "adaptation row requires target_instance_label with an instance label image"
                ) from error
            if instance_label <= 0:
                raise ValueError("target_instance_label must be a positive integer")
            with Image.open(label_path) as opened:
                labels = np.asarray(opened)
            if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("adaptation instance label must be an integer HxW image")
            target = labels == instance_label
            if not bool(target.any()):
                raise ValueError(
                    f"target instance {instance_label} is absent from {label_path}"
                )
        image_tensor = _RGB_TRANSFORM(_rgb_image(rgb))
        target_tensor = _foreground_target(target)
        return image_tensor, target_tensor, _query_from_row(row)


def _validation_miou(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    model.clip_model.eval()
    weighted_loss = 0.0
    sample_count = 0
    ious: list[float] = []
    with torch.inference_mode():
        for images, foreground_targets, queries in loader:
            images = images.to(device)
            foreground_targets = foreground_targets.to(device)
            logits = model(images, list(queries), return_features=False)[0]
            loss = torch_functional.binary_cross_entropy_with_logits(
                -logits, foreground_targets
            )
            probabilities = torch.sigmoid(-logits)
            predictions = probabilities >= FOREGROUND_THRESHOLD
            labels = foreground_targets >= 0.5
            intersection = (predictions & labels).flatten(1).sum(1).float()
            union = (predictions | labels).flatten(1).sum(1).float()
            batch_ious = torch.where(
                union == 0, torch.ones_like(union), intersection / union
            )
            count = int(images.shape[0])
            weighted_loss += float(loss.detach().cpu()) * count
            sample_count += count
            ious.extend(float(value) for value in batch_ious.cpu())
    if sample_count == 0 or not ious:
        raise RuntimeError("validation loader produced no samples")
    mean_iou = float(np.mean(np.asarray(ious, dtype=np.float64)))
    mean_loss = float(weighted_loss / sample_count)
    if not np.isfinite([mean_iou, mean_loss]).all():
        raise FloatingPointError("non-finite validation metric")
    return mean_iou, mean_loss


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            suffix=".pth",
            prefix=f".{destination.stem}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def adapt_hifi_decoder(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    device: str = "cpu",
    checkpoint_path: str | os.PathLike[str] = DEFAULT_HIFI_CHECKPOINT,
    clip_weight_path: str | os.PathLike[str] = DEFAULT_CLIP_WEIGHT,
    output_checkpoint: str | os.PathLike[str] | None = None,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    max_epochs: int = 30,
    patience: int = 5,
    minimum_improvement: float = 1e-5,
    seed: int = 20260815,
) -> AdaptationResult:
    """Adapt only the retained decoder and select solely on validation mIoU.

    Rows require explicit ``split`` and ``scene_id`` plus RGB, target mask and
    query fields.  Only ``train`` and ``val`` are accepted.  The CLIP image and
    text encoders are frozen throughout.  No test rows or test metric enter
    optimization, early stopping, or checkpoint selection.
    """

    split_contract = validate_adaptation_splits(train_rows, validation_rows)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    if not np.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    bundle = load_hifi_model(
        device=device,
        mode="adaptation",
        checkpoint_path=checkpoint_path,
        clip_weight_path=clip_weight_path,
    )
    model = bundle.model
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("decoder adaptation exposed no trainable parameters")
    optimizer = torch.optim.Adam(
        trainable, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        _AdaptationDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        _AdaptationDataset(validation_rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    best_iou = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    stopped_early = False
    history: list[Mapping[str, Any]] = []
    for epoch in range(max_epochs):
        model.train()
        model.clip_model.eval()
        weighted_loss = 0.0
        seen = 0
        for images, foreground_targets, queries in train_loader:
            images = images.to(bundle.device)
            foreground_targets = foreground_targets.to(bundle.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, list(queries), return_features=False)[0]
            loss = torch_functional.binary_cross_entropy_with_logits(
                -logits, foreground_targets
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite adaptation loss at epoch {epoch}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad and parameter.grad is not None:
                    raise RuntimeError(f"frozen parameter received a gradient: {name}")
            optimizer.step()
            count = int(images.shape[0])
            weighted_loss += float(loss.detach().cpu()) * count
            seen += count
        if seen != len(train_rows):
            raise RuntimeError(
                f"training loader covered {seen}/{len(train_rows)} rows"
            )
        validation_iou, validation_loss = _validation_miou(
            model, validation_loader, bundle.device
        )
        epoch_record = MappingProxyType(
            {
                "epoch": epoch,
                "training_loss": float(weighted_loss / seen),
                "validation_loss": validation_loss,
                "validation_mean_iou": validation_iou,
                "selection_split": "val",
                "test_rows_consumed": 0,
            }
        )
        history.append(epoch_record)
        if validation_iou > best_iou + float(minimum_improvement):
            best_iou = validation_iou
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if name in bundle.adaptation_parameter_names
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stopped_early = True
                break

    if best_state is None or best_epoch < 0 or not np.isfinite(best_iou):
        raise RuntimeError("adaptation did not produce a finite validation checkpoint")
    full_state = model.state_dict()
    full_state.update(best_state)
    incompatible = model.load_state_dict(full_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("failed to restore validation-selected adaptation state")
    model.eval()
    model.clip_model.eval()

    saved_path: Path | None = None
    if output_checkpoint is not None:
        saved_path = _atomic_torch_save(
            Path(output_checkpoint),
            {
                "format": "graspnet6d_hifi_decoder_adaptation_v1",
                "base_checkpoint_sha256": bundle.checkpoint_sha256,
                "clip_weight_sha256": bundle.clip_weight_sha256,
                "trainable_state": best_state,
                "metadata": {
                    "best_epoch": best_epoch,
                    "best_validation_mean_iou": best_iou,
                    "epochs_completed": len(history),
                    "stopped_early": stopped_early,
                    "selection_metric": "validation_mean_iou",
                    "selection_split": "val",
                    "test_rows_consumed": 0,
                    "split_contract": dict(split_contract),
                    "training_config": {
                        "batch_size": batch_size,
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "max_epochs": max_epochs,
                        "patience": patience,
                        "minimum_improvement": float(minimum_improvement),
                        "seed": seed,
                        "device": str(bundle.device),
                    },
                    "history_sha256": canonical_sha256(
                        [dict(record) for record in history]
                    ),
                },
            },
        )

    return AdaptationResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_miou=best_iou,
        epochs_completed=len(history),
        stopped_early=stopped_early,
        history=tuple(history),
        output_checkpoint=saved_path,
        split_contract=split_contract,
    )


__all__ = [
    "AdaptationResult",
    "DEFAULT_CLIP_WEIGHT",
    "DEFAULT_HIFI_CHECKPOINT",
    "EXPECTED_CHECKPOINT_FORMAT",
    "EXPECTED_CHECKPOINT_STEP",
    "EXPECTED_CLIP_WEIGHT_SHA256",
    "EXPECTED_HIFI_CHECKPOINT_SHA256",
    "FOREGROUND_THRESHOLD",
    "GroundingPrediction",
    "HiFiModelBundle",
    "IMAGE_RESOLUTION",
    "adapt_hifi_decoder",
    "compute_mask_metrics",
    "load_hifi_model",
    "predict_grounding",
    "resolve_grounding_device",
    "save_grounding_prediction",
    "summarize_mask_metrics",
    "validate_adaptation_splits",
]
