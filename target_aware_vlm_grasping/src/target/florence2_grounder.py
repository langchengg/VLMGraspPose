from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from target.mask_utils import bbox_to_mask, compute_mask_center
from utils.data_types import TargetRegion


@dataclass
class Florence2Config:
    model_id: str = "microsoft/Florence-2-base-ft"
    task_prompt: str = "<CAPTION_TO_PHRASE_GROUNDING>"
    device: str = "cpu"
    max_new_tokens: int = 1024
    num_beams: int = 3
    trust_remote_code: bool = True


class HuggingFaceFlorenceBackend:
    def __init__(self, config: Florence2Config):
        self.config = config
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
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
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        ).to(self.config.device)
        self._model.eval()

    def ground(self, rgb: np.ndarray, command: str) -> dict:
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
        return _first_grounding_result(parsed, self.config.task_prompt, command)


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

    def ground(self, rgb: np.ndarray, command: str, target_id: int | None = None) -> TargetRegion:
        result = self.backend.ground(rgb, command)
        bbox = _clip_bbox(result["bbox"], rgb.shape[:2])
        mask = bbox_to_mask(bbox, rgb.shape[:2])
        label = str(result.get("label") or command).strip()
        score = float(result.get("score", 1.0))
        return TargetRegion(
            target_id=target_id,
            label=label,
            bbox=bbox,
            mask=mask,
            grounding_score=score,
            center_2d=compute_mask_center(mask),
            command=command,
            target_source="vlm",
            metadata={
                "grounding_model": "Florence-2",
                "model_id": self.config.model_id,
                "task_prompt": self.config.task_prompt,
                "raw_grounding": result.get("raw", result),
            },
        )


def _first_grounding_result(parsed: dict, task_prompt: str, command: str) -> dict:
    payload = parsed.get(task_prompt, parsed)
    bboxes = payload.get("bboxes") or payload.get("bbox") or []
    labels = payload.get("labels") or []
    scores = payload.get("scores") or payload.get("score") or []
    if not bboxes:
        raise ValueError(f"Florence-2 returned no bbox for command: {command}")
    return {
        "bbox": bboxes[0],
        "label": labels[0] if labels else command,
        "score": scores[0] if isinstance(scores, list) and scores else 1.0,
        "raw": parsed,
    }


def _clip_bbox(bbox: list[float], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, x2 = sorted(np.clip([x1, x2], 0, w - 1).astype(int).tolist())
    y1, y2 = sorted(np.clip([y1, y2], 0, h - 1).astype(int).tolist())
    return [x1, y1, x2, y2]
