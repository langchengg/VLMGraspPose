from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from target.command_parser import ParsedCommand, parse_command
from target.sam_refiner import SAMMaskRefiner, SAMRefinerConfig
from utils.data_types import TargetRegion


@dataclass
class Florence2Config:
    model_id: str = "models/vlm/florence2-large-ft"
    task_prompt: str = "<CAPTION_TO_PHRASE_GROUNDING>"
    device: str = "cpu"
    max_new_tokens: int = 1024
    num_beams: int = 3
    trust_remote_code: bool = True
    local_files_only: bool = True
    multi_query: bool = True
    nms_iou_threshold: float = 0.65
    min_selection_score: float = 0.35
    min_box_area_ratio: float = 0.0002
    max_box_area_ratio: float = 0.75
    fail_on_low_quality: bool = True
    require_query_consistency: bool = True
    min_query_agreement: int = 2
    sam_enabled: bool = False
    sam_required: bool = False
    sam_checkpoint: str = "models/vlm/sam/sam_vit_b_01ec64.pth"
    sam_model_type: str = "vit_b"
    sam_device: str = "cpu"
    sam_multimask_output: bool = True
    sam_min_area_ratio: float = 0.01
    sam_max_area_ratio: float = 1.25
    sam_bbox_expansion_pixels: int = 0
    sam_bbox_expansion_ratio: float = 0.02


class HuggingFaceFlorenceBackend:
    def __init__(self, config: Florence2Config):
        self.config = config
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        cache_root = Path(__file__).resolve().parents[2] / ".hf_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / "modules").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_root))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
        os.environ.setdefault("HF_MODULES_CACHE", str(cache_root / "modules"))
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Florence-2 grounding requires optional dependencies: torch, transformers, pillow."
            ) from exc
        self._torch = torch
        self._image_cls = Image
        model_id = _resolve_model_id(self.config.model_id)
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        ).to(self.config.device)
        self._model.eval()

    def ground(self, rgb: np.ndarray, command: str) -> dict:
        results = self.ground_all(rgb, command)
        if not results:
            raise ValueError(f"Florence-2 returned no bbox for command: {command}")
        return results[0]

    def ground_all(self, rgb: np.ndarray, command: str) -> list[dict]:
        self._load()
        image = self._image_cls.fromarray(rgb.astype(np.uint8))
        prompt = f"{self.config.task_prompt}{command}"
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
                do_sample=False,
            )
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=self.config.task_prompt,
            image_size=(image.width, image.height),
        )
        return _all_grounding_results(parsed, self.config.task_prompt, command)


class Florence2Grounder:
    """Florence-2 phrase grounding wrapper returning the project's TargetRegion."""

    def __init__(
        self,
        config: Florence2Config | dict | None = None,
        backend: Any | None = None,
    ):
        if isinstance(config, dict):
            config = Florence2Config(**{key: value for key, value in config.items() if key in Florence2Config.__annotations__})
        self.config = config or Florence2Config()
        self.backend = backend or HuggingFaceFlorenceBackend(self.config)
        self._sam_refiner = None

    def ground(
        self,
        rgb: np.ndarray,
        command: str,
        target_id: int | None = None,
        target_label: str | None = None,
    ) -> TargetRegion:
        parsed = parse_command(command, target_label)
        result = self._select_grounding(rgb, parsed)
        bbox = _clip_bbox(result["bbox"], rgb.shape[:2])
        mask, mask_meta = self._mask_from_bbox(rgb, bbox)
        label = str(result.get("label") or command).strip()
        score = float(result.get("score", 1.0))
        return TargetRegion(
            target_id=target_id,
            label=label,
            bbox=bbox,
            mask=mask,
            grounding_score=score,
            center_2d=_mask_center(mask),
            command=command,
            target_source="vlm",
            metadata={
                "grounding_model": "Florence-2",
                "model_id": self.config.model_id,
                "task_prompt": self.config.task_prompt,
                "parsed_command": {
                    "target_phrase": parsed.target_phrase,
                    "target_queries": parsed.target_queries,
                    "relation": parsed.relation,
                    "reference_phrase": parsed.reference_phrase,
                    "reference_queries": parsed.reference_queries,
                    "ordinal": parsed.ordinal,
                },
                "selection_score": result.get("selection_score"),
                "selection_reason": result.get("selection_reason"),
                "candidate_count": result.get("candidate_count"),
                "selected_query": result.get("query"),
                "selected_query_role": result.get("query_role"),
                "selected_agreement_count": result.get("agreement_count"),
                "selected_agreement_queries": result.get("agreement_queries"),
                "reference_bbox": result.get("reference_bbox"),
                **mask_meta,
                "raw_grounding": result.get("raw", result),
                "all_grounding_candidates": result.get("all_candidates", []),
            },
        )

    def _mask_from_bbox(self, rgb: np.ndarray, bbox: list[int]) -> tuple[np.ndarray, dict]:
        bbox_mask = _bbox_to_mask(bbox, rgb.shape[:2])
        if not self.config.sam_enabled:
            return bbox_mask, {"target_mask_source": "vlm_bbox"}
        try:
            if self._sam_refiner is None:
                self._sam_refiner = SAMMaskRefiner(SAMRefinerConfig(
                    checkpoint=self.config.sam_checkpoint,
                    model_type=self.config.sam_model_type,
                    device=self.config.sam_device,
                    multimask_output=self.config.sam_multimask_output,
                    min_area_ratio=self.config.sam_min_area_ratio,
                    max_area_ratio=self.config.sam_max_area_ratio,
                    bbox_expansion_pixels=self.config.sam_bbox_expansion_pixels,
                    bbox_expansion_ratio=self.config.sam_bbox_expansion_ratio,
                ))
            sam_mask, sam_meta = self._sam_refiner.refine(rgb, bbox)
            if sam_mask.sum() == 0:
                raise ValueError("SAM mask is empty.")
            return sam_mask, sam_meta
        except Exception as exc:
            if self.config.sam_required:
                raise RuntimeError(f"SAM mask refinement failed: {type(exc).__name__}: {exc}") from exc
            return bbox_mask, {
                "target_mask_source": "vlm_bbox",
                "sam_error": f"{type(exc).__name__}: {exc}",
            }

    def _select_grounding(self, rgb: np.ndarray, parsed: ParsedCommand) -> dict:
        candidates = self._collect_candidates(rgb, parsed)
        if not candidates:
            raise ValueError(f"Florence-2 returned no candidate bbox for command: {parsed.command}")
        image_shape = rgb.shape[:2]
        candidates = [
            c for c in candidates
            if _box_area_ratio(_clip_bbox(c["bbox"], image_shape), image_shape) >= self.config.min_box_area_ratio
            and _box_area_ratio(_clip_bbox(c["bbox"], image_shape), image_shape) <= self.config.max_box_area_ratio
        ]
        if not candidates:
            raise ValueError(f"All Florence-2 candidates failed bbox size gate for command: {parsed.command}")

        target_candidates = [c for c in candidates if c.get("query_role") in {"full", "target"}]
        reference_candidates = [c for c in candidates if c.get("query_role") == "reference"]
        clusters = _nms_clusters(target_candidates, self.config.nms_iou_threshold, image_shape)
        reference = _best_reference(reference_candidates, image_shape)

        scored = []
        for cand in clusters:
            relation_score, relation_reason = _relation_score(cand, reference, parsed.relation, parsed.ordinal, clusters)
            score = _candidate_score(cand, relation_score, parsed)
            cand = dict(cand)
            cand["selection_score"] = score
            cand["selection_reason"] = relation_reason
            cand["reference_bbox"] = reference.get("bbox") if reference else None
            scored.append(cand)

        scored.sort(key=lambda c: c["selection_score"], reverse=True)
        selected = scored[0]
        selected["candidate_count"] = len(scored)
        selected["all_candidates"] = [
            _candidate_metadata(c, image_shape) for c in scored[:30]
        ]
        successful_target_queries = {
            c.get("query")
            for c in target_candidates
            if c.get("query") and c.get("bbox")
        }
        selected_agreement = int(selected.get("agreement_count", 1) or 1)
        if (
            self.config.fail_on_low_quality
            and self.config.require_query_consistency
            and len(successful_target_queries) >= self.config.min_query_agreement
            and selected_agreement < self.config.min_query_agreement
        ):
            raise ValueError(
                f"Inconsistent Florence-2 grounding for command '{parsed.command}': "
                f"selected bbox is supported by {selected_agreement} query, "
                f"but {len(successful_target_queries)} target queries returned bboxes."
            )
        if self.config.fail_on_low_quality and selected["selection_score"] < self.config.min_selection_score:
            raise ValueError(
                f"Low-quality Florence-2 grounding for command '{parsed.command}': "
                f"selection_score={selected['selection_score']:.3f}, reason={selected.get('selection_reason')}"
            )
        return selected

    def _collect_candidates(self, rgb: np.ndarray, parsed: ParsedCommand) -> list[dict]:
        if not self.config.multi_query:
            return self._run_query(rgb, parsed.command, "full")
        candidates = []
        for query in parsed.target_queries:
            role = "full" if query == parsed.command else "target"
            candidates.extend(self._run_query(rgb, query, role))
        for query in parsed.reference_queries or []:
            candidates.extend(self._run_query(rgb, query, "reference"))
        return candidates

    def _run_query(self, rgb: np.ndarray, query: str, role: str) -> list[dict]:
        try:
            if hasattr(self.backend, "ground_all"):
                results = self.backend.ground_all(rgb, query)
            else:
                results = [self.backend.ground(rgb, query)]
        except Exception as exc:
            return [{
                "bbox": [],
                "label": query,
                "score": 0.0,
                "query": query,
                "query_role": role,
                "error": f"{type(exc).__name__}: {exc}",
                "raw": {},
            }]
        valid = []
        for result in results:
            if not result.get("bbox"):
                continue
            item = dict(result)
            item["query"] = query
            item["query_role"] = role
            valid.append(item)
        return valid


def _first_grounding_result(parsed: dict, task_prompt: str, command: str) -> dict:
    results = _all_grounding_results(parsed, task_prompt, command)
    if not results:
        raise ValueError(f"Florence-2 returned no bbox for command: {command}")
    return results[0]


def _all_grounding_results(parsed: dict, task_prompt: str, command: str) -> list[dict]:
    payload = parsed.get(task_prompt, parsed)
    bboxes = payload.get("bboxes") or payload.get("bbox") or []
    labels = payload.get("labels") or []
    scores = payload.get("scores") or payload.get("score") or []
    if bboxes and not isinstance(bboxes[0], (list, tuple)):
        bboxes = [bboxes]
    if not isinstance(labels, list):
        labels = [labels]
    if not isinstance(scores, list):
        scores = [scores]
    results = []
    for idx, bbox in enumerate(bboxes):
        results.append({
            "bbox": bbox,
            "label": labels[idx] if idx < len(labels) and labels[idx] else command,
            "score": scores[idx] if idx < len(scores) and scores[idx] not in (None, "") else 1.0,
            "raw": parsed,
        })
    return results


def _clip_bbox(bbox: list[float], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, x2 = sorted(np.clip([x1, x2], 0, w - 1).astype(int).tolist())
    y1, y2 = sorted(np.clip([y1, y2], 0, h - 1).astype(int).tolist())
    return [x1, y1, x2, y2]


def _box_area_ratio(bbox: list[int], shape: tuple[int, int]) -> float:
    h, w = shape
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    return float(area) / float(max(1, h * w))


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _nms_clusters(candidates: list[dict], threshold: float, shape: tuple[int, int]) -> list[dict]:
    prepared = []
    for candidate in candidates:
        if not candidate.get("bbox"):
            continue
        item = dict(candidate)
        item["bbox"] = _clip_bbox(item["bbox"], shape)
        item["_base_score"] = float(item.get("score", 1.0) or 1.0) + (0.15 if item.get("query_role") == "target" else 0.05)
        prepared.append(item)
    prepared.sort(key=lambda c: c["_base_score"], reverse=True)

    clusters: list[dict] = []
    for item in prepared:
        matched = None
        for cluster in clusters:
            if _bbox_iou(item["bbox"], cluster["bbox"]) >= threshold:
                matched = cluster
                break
        if matched is None:
            item["agreement_count"] = 1
            item["agreement_queries"] = [item.get("query")]
            item["agreement_roles"] = [item.get("query_role")]
            clusters.append(item)
        else:
            matched["agreement_count"] += 1
            matched["agreement_queries"].append(item.get("query"))
            matched["agreement_roles"].append(item.get("query_role"))
            if item["_base_score"] > matched.get("_base_score", 0.0):
                keep = {
                    "agreement_count": matched["agreement_count"],
                    "agreement_queries": matched["agreement_queries"],
                    "agreement_roles": matched["agreement_roles"],
                }
                matched.update(item)
                matched.update(keep)
    return clusters


def _best_reference(candidates: list[dict], shape: tuple[int, int]) -> dict | None:
    refs = [dict(c) for c in candidates if c.get("bbox")]
    if not refs:
        return None
    for ref in refs:
        ref["bbox"] = _clip_bbox(ref["bbox"], shape)
        ref["_score"] = float(ref.get("score", 1.0) or 1.0)
    refs.sort(key=lambda c: c["_score"], reverse=True)
    return refs[0]


def _relation_score(
    candidate: dict,
    reference: dict | None,
    relation: str | None,
    ordinal: str | None,
    candidates: list[dict],
) -> tuple[float, str]:
    if ordinal in {"leftmost", "rightmost"}:
        centers = [(_bbox_center(c["bbox"])[0], c) for c in candidates]
        if not centers:
            return 0.0, "ordinal_no_candidates"
        selected_x, selected = min(centers, key=lambda item: item[0]) if ordinal == "leftmost" else max(centers, key=lambda item: item[0])
        score = 1.0 if selected is candidate else 0.15
        return score, f"{ordinal}_x={selected_x:.1f}"

    if relation is None:
        return 0.5, "no_relation"
    if reference is None:
        return 0.0, f"{relation}_missing_reference"

    cx, cy = _bbox_center(candidate["bbox"])
    rx, ry = _bbox_center(reference["bbox"])
    dx, dy = cx - rx, cy - ry
    checks = {
        "left_of": dx < 0,
        "right_of": dx > 0,
        "front_of": dy > 0,
        "behind": dy < 0,
        "rear_right_of": dx > 0 and dy < 0,
        "rear_left_of": dx < 0 and dy < 0,
        "front_right_of": dx > 0 and dy > 0,
        "front_left_of": dx < 0 and dy > 0,
    }
    ok = checks.get(relation, False)
    magnitude = min(1.0, (abs(dx) + abs(dy)) / 250.0)
    return (0.8 + 0.2 * magnitude if ok else 0.05), f"{relation}_dx={dx:.1f}_dy={dy:.1f}_ok={ok}"


def _candidate_score(candidate: dict, relation_score: float, parsed: ParsedCommand) -> float:
    roles = set(candidate.get("agreement_roles", []))
    agreement = float(candidate.get("agreement_count", 1))
    model_score = float(candidate.get("score", 1.0) or 1.0)
    role_score = 0.0
    if "target" in roles or candidate.get("query_role") == "target":
        role_score += 0.28
    if "full" in roles or candidate.get("query_role") == "full":
        role_score += 0.18
    agreement_score = min(0.22, 0.08 * agreement)
    model_score = min(0.20, 0.20 * max(0.0, min(1.0, model_score)))
    if parsed.relation or parsed.ordinal:
        relation_weight = 0.35
        no_relation_bias = 0.0
    else:
        relation_weight = 0.15
        no_relation_bias = 0.10
    return float(max(0.0, min(1.0, role_score + agreement_score + model_score + no_relation_bias + relation_weight * relation_score)))


def _candidate_metadata(candidate: dict, shape: tuple[int, int]) -> dict:
    bbox = _clip_bbox(candidate["bbox"], shape)
    return {
        "bbox": bbox,
        "label": candidate.get("label"),
        "score": candidate.get("score"),
        "query": candidate.get("query"),
        "query_role": candidate.get("query_role"),
        "selection_score": candidate.get("selection_score"),
        "selection_reason": candidate.get("selection_reason"),
        "agreement_count": candidate.get("agreement_count"),
        "agreement_queries": candidate.get("agreement_queries"),
        "area_ratio": _box_area_ratio(bbox, shape),
    }


def _resolve_model_id(model_id: str) -> str:
    path = Path(model_id)
    if path.exists():
        return str(path)
    project_relative = Path(__file__).resolve().parents[2] / model_id
    if project_relative.exists():
        return str(project_relative)
    return model_id


def _bbox_to_mask(bbox: list[int], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mask = np.zeros((h, w), dtype=bool)
    mask[max(0, y1): min(h, y2 + 1), max(0, x1): min(w, x2 + 1)] = True
    return mask


def _mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())
