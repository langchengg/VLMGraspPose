from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from failure_analysis.failure_utils import rle_to_mask

from ..schema import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
    stable_sample_id,
)


REQUIRED_CANDIDATE_FIELDS = (
    "id",
    "target_alignment",
    "contact_quality",
    "width_fit",
    "collision_risk",
    "overall",
)


def reviewer_prompt(language_prompt: str) -> str:
    return (
        "You are reviewing five frozen robot grasp candidates for the target "
        f"described as: {language_prompt!r}. Use only the displayed RGB image, "
        "predicted target mask, optional depth view, and numbered candidates. "
        "Do not infer or request ground truth. Score every candidate from 0 to "
        "1 for target_alignment, contact_quality, width_fit, collision_risk, "
        "and overall. Return JSON only with keys best_candidate and candidates "
        "using displayed IDs 1 through 5."
    )


class VLMProvider(Protocol):
    provider_name: str
    model_snapshot: str

    def review(self, image_path: Path, prompt: str, parameters: dict[str, Any]) -> str:
        ...


def validate_vlm_response(payload: dict[str, Any], candidate_count: int = 5):
    if set(payload) != {"best_candidate", "candidates"}:
        raise ValueError("VLM response must contain only best_candidate and candidates")
    best = int(payload["best_candidate"])
    if not 1 <= best <= candidate_count:
        raise ValueError("best_candidate is outside the displayed ID range")
    candidates = payload["candidates"]
    if len(candidates) != candidate_count:
        raise ValueError("VLM response must score every displayed candidate")
    seen = set()
    for candidate in candidates:
        if set(candidate) != set(REQUIRED_CANDIDATE_FIELDS):
            raise ValueError("VLM candidate score schema mismatch")
        identifier = int(candidate["id"])
        seen.add(identifier)
        for name in REQUIRED_CANDIDATE_FIELDS[1:]:
            value = float(candidate[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
    if seen != set(range(1, candidate_count + 1)):
        raise ValueError("VLM candidate IDs are incomplete or duplicated")
    return payload


def parse_vlm_response(raw_response: str, candidate_count: int = 5):
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    return validate_vlm_response(json.loads(text), candidate_count)


def shuffled_candidate_mapping(
    candidate_ids: list[str], *, sample_id: str, seed: int
) -> dict[str, Any]:
    indices = list(range(len(candidate_ids)))
    derived_seed = int.from_bytes(
        f"{seed}:{sample_id}".encode("utf-8"), "little", signed=False
    ) % (2**32)
    random.Random(derived_seed).shuffle(indices)
    display_to_candidate = {
        str(display + 1): candidate_ids[source]
        for display, source in enumerate(indices)
    }
    return {
        "seed": int(seed),
        "derived_seed": int(derived_seed),
        "display_to_candidate": display_to_candidate,
        "candidate_to_display": {
            candidate: int(display)
            for display, candidate in display_to_candidate.items()
        },
    }


def render_marked_candidates(
    *,
    image_path: str | Path,
    candidates: list[dict[str, Any]],
    predicted_mask: np.ndarray,
    output_path: str | Path,
    sample_id: str,
    seed: int = 47,
    depth_m: np.ndarray | None = None,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mapping = shuffled_candidate_mapping(
        [str(item["candidate_id"]) for item in candidates],
        sample_id=sample_id,
        seed=seed,
    )
    by_id = {str(item["candidate_id"]): item for item in candidates}
    mask = np.asarray(predicted_mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("predicted mask shape does not match RGB")
    full = image.astype(np.float32)
    overlay = np.zeros_like(full)
    overlay[..., 1] = 255.0
    full[mask] = 0.72 * full[mask] + 0.28 * overlay[mask]
    full = np.uint8(np.clip(full, 0, 255))
    neutral_color = (255, 190, 30)
    for display_id, candidate_id in mapping["display_to_candidate"].items():
        candidate = by_id[candidate_id]
        polygon = np.asarray(candidate["polygon"], dtype=np.int32)
        cv2.polylines(full, [polygon], True, neutral_color, 3, cv2.LINE_AA)
        centre = (int(round(candidate["cx"])), int(round(candidate["cy"])))
        cv2.circle(full, centre, 5, neutral_color, -1, cv2.LINE_AA)
        cv2.putText(
            full,
            str(display_id),
            (centre[0] + 7, centre[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            neutral_color,
            2,
            cv2.LINE_AA,
        )
    mask_panel = np.zeros_like(image)
    mask_panel[mask] = (80, 205, 120)
    cv2.putText(
        mask_panel,
        "Predicted mask (not GT)",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    side_width = max(240, image.shape[1] // 2)
    top_panels = [full]
    top_panels.append(
        cv2.resize(
            mask_panel,
            (side_width, image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    )
    if depth_m is not None:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape != image.shape[:2]:
            raise ValueError("depth shape does not match RGB")
        valid = np.isfinite(depth) & (depth > 0)
        normalized = np.zeros_like(depth, dtype=np.uint8)
        if valid.any():
            low, high = np.percentile(depth[valid], (2, 98))
            scale = max(float(high - low), 1e-6)
            normalized[valid] = np.uint8(
                np.clip((depth[valid] - low) / scale * 255.0, 0, 255)
            )
        depth_panel = cv2.applyColorMap(
            normalized, cv2.COLORMAP_VIRIDIS
        )
        depth_panel = cv2.cvtColor(depth_panel, cv2.COLOR_BGR2RGB)
        depth_panel[~valid] = 0
        cv2.putText(
            depth_panel,
            "Depth (relative display)",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        top_panels.append(
            cv2.resize(
                depth_panel,
                (side_width, image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        )
    top = np.concatenate(top_panels, axis=1)
    panels = []
    panel_size = 192
    for display_id, candidate_id in mapping["display_to_candidate"].items():
        candidate = by_id[candidate_id]
        half = max(48, int(round(candidate["width_px"])))
        x, y = int(round(candidate["cx"])), int(round(candidate["cy"]))
        local = image.copy()
        polygon = np.asarray(candidate["polygon"], dtype=np.int32)
        cv2.polylines(
            local, [polygon], True, neutral_color, 3, cv2.LINE_AA
        )
        cv2.circle(local, (x, y), 5, neutral_color, -1, cv2.LINE_AA)
        crop = local[
            max(0, y - half) : min(image.shape[0], y + half),
            max(0, x - half) : min(image.shape[1], x + half),
        ]
        if crop.size == 0:
            crop = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR)
        cv2.putText(
            crop,
            f"Candidate {display_id}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            neutral_color,
            2,
            cv2.LINE_AA,
        )
        panels.append(crop)
    lower = np.concatenate(panels, axis=1)
    lower = cv2.resize(
        lower,
        (
            top.shape[1],
            max(1, top.shape[1] * panel_size // lower.shape[1]),
        ),
    )
    canvas = np.concatenate((top, lower), axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
    )
    mapping["image_path"] = str(output_path.resolve())
    mapping["image_sha256"] = sha256_file(output_path)
    mapping["depth_panel_included"] = bool(depth_m is not None)
    atomic_write_json(output_path.with_suffix(".mapping.json"), mapping)
    return mapping


def prepare_vlm_dry_run(
    *,
    features_path: str | Path,
    output_dir: str | Path,
    max_samples: int = 5,
    seed: int = 47,
    include_depth: bool = True,
) -> dict[str, Any]:
    """Render a GT-free request cohort without making any API call."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    panel_dir = output_dir / "panels"
    panel_dir.mkdir()
    request_path = output_dir / "requests.jsonl"
    count = 0
    requests = []
    for feature in read_jsonl(features_path):
        if count >= int(max_samples):
            break
        forbidden = (
            "gt",
            "ground_truth",
            "candidate_correct",
            "oracle",
            "iou",
            "angle_error",
        )
        if any(
            token in str(key).lower()
            for key in feature
            for token in forbidden
        ):
            raise ValueError("VLM source feature contains evaluation field")
        sample_id = stable_sample_id(
            feature["split"], feature["sample_id"]
        )
        depth = None
        if include_depth and feature.get("depth_path"):
            raw_depth = cv2.imread(
                str(feature["depth_path"]), cv2.IMREAD_UNCHANGED
            )
            if raw_depth is not None:
                depth = raw_depth.astype(np.float32) / 1000.0
        panel_path = panel_dir / f"sample_{count:03d}.png"
        mapping = render_marked_candidates(
            image_path=feature["image_path"],
            candidates=feature["candidates"],
            predicted_mask=rle_to_mask(feature["predicted_mask_rle"]),
            output_path=panel_path,
            sample_id=sample_id,
            seed=seed,
            depth_m=depth,
        )
        requests.append(
            {
                "schema_version": "2.0.0",
                "kind": "vlm_review_request",
                "sample_id": sample_id,
                "image_path": mapping["image_path"],
                "image_sha256": mapping["image_sha256"],
                "mapping_path": str(
                    panel_path.with_suffix(".mapping.json").resolve()
                ),
                "language_prompt": feature["language_instruction"],
                "prompt": reviewer_prompt(
                    feature["language_instruction"]
                ),
                "parameters": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": int(seed),
                },
                "live_call_authorized": False,
            }
        )
        count += 1
    atomic_write_jsonl(request_path, requests)
    result = {
        "status": "prepared_offline",
        "request_count": count,
        "request_manifest": str(request_path.resolve()),
        "live_calls": False,
        "gt_fields_included": False,
        "seed": int(seed),
    }
    atomic_write_json(output_dir / "summary.json", result)
    return result


def map_response_to_candidate_ids(
    response: dict[str, Any], mapping: dict[str, Any]
) -> dict[str, Any]:
    display_to_candidate = mapping["display_to_candidate"]
    result = {
        "best_candidate_id": display_to_candidate[str(response["best_candidate"])],
        "candidates": [],
    }
    for candidate in response["candidates"]:
        result["candidates"].append(
            {
                **candidate,
                "display_id": int(candidate["id"]),
                "candidate_id": display_to_candidate[str(candidate["id"])],
            }
        )
    return result


def cached_review(
    *,
    provider: VLMProvider | None,
    image_path: str | Path,
    prompt: str,
    mapping: dict[str, Any],
    cache_path: str | Path,
    parameters: dict[str, Any],
    retries: int = 2,
    replay: bool = False,
) -> dict[str, Any]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["cache_hit"] = True
        return cached
    if replay:
        raise FileNotFoundError(f"replay cache is missing: {cache_path}")
    if provider is None:
        return {
            "status": "blocked",
            "fallback": "q_only",
            "reason": "no configured VLM provider/credentials",
            "cache_hit": False,
        }
    error = None
    start = time.perf_counter()
    for retry in range(int(retries) + 1):
        try:
            raw = provider.review(Path(image_path), prompt, parameters)
            parsed = parse_vlm_response(raw)
            result = {
                "status": "success",
                "provider": provider.provider_name,
                "model_snapshot": provider.model_snapshot,
                "api_date": time.strftime("%Y-%m-%d"),
                "prompt": prompt,
                "parameters": parameters,
                "image_sha256": sha256_file(image_path),
                "raw_response": raw,
                "parsed_response": parsed,
                "mapped_response": map_response_to_candidate_ids(parsed, mapping),
                "retry_count": retry,
                "latency_seconds": time.perf_counter() - start,
                "cache_hit": False,
            }
            atomic_write_json(cache_path, result)
            return result
        except Exception as exc:  # provider/network/parser errors share fallback
            error = f"{type(exc).__name__}: {exc}"
    result = {
        "status": "failed",
        "fallback": "q_only",
        "error": error,
        "retry_count": int(retries),
        "latency_seconds": time.perf_counter() - start,
        "cache_hit": False,
    }
    atomic_write_json(cache_path, result)
    return result
