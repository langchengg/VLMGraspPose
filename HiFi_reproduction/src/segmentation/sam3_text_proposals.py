"""Official Sam3Model text/box proposal generation with reusable vision features."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .proposal_types import ProposalCandidate
from .sam3_embedding_cache import EmbeddingCacheKey, Sam3EmbeddingCache


@dataclass(frozen=True)
class TextPromptSpec:
    prompt_id: str
    source_family: str
    source_variant: str
    text: str
    eligible_final: bool = True
    boxes_xyxy: tuple[tuple[float, float, float, float], ...] = ()
    box_labels: tuple[int, ...] = ()


class OfficialSam3TextProposalGenerator:
    def __init__(
        self,
        model_path: str | Path,
        *,
        revision: str,
        cache: Sam3EmbeddingCache,
        processor_sha256: str,
        num_threads: int = 8,
        micro_batch_size: int = 4,
    ):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        torch.set_num_threads(int(num_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        self.model_path = Path(model_path).expanduser().resolve()
        self.revision = revision
        self.cache = cache
        self.processor_sha256 = processor_sha256
        self.micro_batch_size = int(micro_batch_size)
        if not 1 <= self.micro_batch_size <= 16:
            raise ValueError("PCS text micro-batch must be within [1,16]")
        self.processor = Sam3Processor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = Sam3Model.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.float32,
        ).to("cpu").eval()

    def cache_key(self, rgb_sha256: str) -> EmbeddingCacheKey:
        return EmbeddingCacheKey(
            rgb_sha256=rgb_sha256,
            model_revision=self.revision,
            processor_sha256=self.processor_sha256,
            input_resolution=1008,
            dtype="float32",
            backend_class="transformers.Sam3Model",
        )

    def vision_features(self, image: Image.Image, rgb_sha256: str) -> tuple[Any, bool, float]:
        key = self.cache_key(rgb_sha256)
        started = time.perf_counter()
        if self.cache.exists(key):
            return self.cache.load_pcs(key), True, time.perf_counter() - started
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with self.torch.inference_mode():
            vision = self.model.get_vision_features(pixel_values=inputs.pixel_values)
        self.cache.save_pcs(key, vision)
        return vision, False, time.perf_counter() - started

    @staticmethod
    def _repeat_vision(vision: Any, count: int) -> Any:
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

        return Sam3VisionEncoderOutput(
            fpn_hidden_states=tuple(value.repeat(count, 1, 1, 1) for value in vision.fpn_hidden_states),
            fpn_position_encoding=tuple(
                value.repeat(count, 1, 1, 1) for value in vision.fpn_position_encoding
            ),
        )

    @staticmethod
    def _select_and_resize_outputs(
        outputs: Any,
        scores: Any,
        *,
        maximum_instances_per_prompt: int,
        output_size: tuple[int, int],
    ) -> tuple[Any, Any, Any]:
        """Select score-ranked masks before full-resolution interpolation.

        SAM 3 emits substantially more raw masks than this proposal bank is
        configured to retain.  Interpolation is independent for every mask, so
        stable score selection can happen first without changing any retained
        probability, box, rank, or raw-query index.
        """

        import torch

        order = scores.argsort(dim=1, descending=True, stable=True)
        order = order[:, : int(maximum_instances_per_prompt)]
        batch_index = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
        selected_mask_logits = outputs.pred_masks[batch_index, order]
        probabilities = torch.nn.functional.interpolate(
            selected_mask_logits.sigmoid(),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        selected_boxes = outputs.pred_boxes[batch_index, order]
        return order, probabilities, selected_boxes

    def generate(
        self,
        sample_id: str,
        image: Image.Image,
        rgb_sha256: str,
        prompts: Iterable[TextPromptSpec],
        *,
        instance_thresholds: tuple[float, ...],
        mask_thresholds: tuple[float, ...],
        maximum_instances_per_prompt: int = 20,
    ) -> tuple[list[ProposalCandidate], dict[str, Any]]:
        prompts = list(prompts)
        if not prompts:
            return [], {"prompt_count": 0, "candidate_count": 0}
        image = image.convert("RGB")
        width, height = image.size
        vision, cache_hit, vision_seconds = self.vision_features(image, rgb_sha256)
        all_candidates: list[ProposalCandidate] = []
        decoder_seconds = 0.0
        postprocess_seconds = 0.0
        for start in range(0, len(prompts), self.micro_batch_size):
            chunk = prompts[start : start + self.micro_batch_size]
            text_inputs = self.processor(
                text=[item.text for item in chunk],
                return_tensors="pt",
                padding=True,
            )
            repeated = self._repeat_vision(vision, len(chunk))
            kwargs: dict[str, Any] = {
                "vision_embeds": repeated,
                "input_ids": text_inputs.input_ids,
                "attention_mask": text_inputs.attention_mask,
            }
            if any(item.boxes_xyxy for item in chunk):
                if not all(item.boxes_xyxy for item in chunk):
                    raise ValueError("boxed and unboxed PCS prompts cannot share a micro-batch")
                boxes = []
                labels = []
                for item in chunk:
                    normalized = []
                    for x1, y1, x2, y2 in item.boxes_xyxy:
                        normalized.append(
                            [
                                (x1 + x2) / (2.0 * width),
                                (y1 + y2) / (2.0 * height),
                                (x2 - x1) / width,
                                (y2 - y1) / height,
                            ]
                        )
                    boxes.append(normalized)
                    labels.append(list(item.box_labels or (1,) * len(normalized)))
                kwargs["input_boxes"] = self.torch.tensor(boxes, dtype=self.torch.float32)
                kwargs["input_boxes_labels"] = self.torch.tensor(labels, dtype=self.torch.long)
            started = time.perf_counter()
            with self.torch.inference_mode():
                outputs = self.model(**kwargs)
            decoder_seconds += time.perf_counter() - started
            scores = outputs.pred_logits.sigmoid()
            presence = outputs.presence_logits.sigmoid() if outputs.presence_logits is not None else None
            if presence is not None:
                scores = scores * presence
            postprocess_started = time.perf_counter()
            order, probabilities, selected_boxes = self._select_and_resize_outputs(
                outputs,
                scores,
                maximum_instances_per_prompt=int(maximum_instances_per_prompt),
                output_size=(height, width),
            )
            scale = self.torch.tensor([width, height, width, height], dtype=self.torch.float32)
            boxes = selected_boxes * scale
            postprocess_seconds += time.perf_counter() - postprocess_started
            for batch_index, spec in enumerate(chunk):
                for rank, raw_index_tensor in enumerate(order[batch_index].tolist()):
                    raw_index = int(raw_index_tensor)
                    score = float(scores[batch_index, raw_index])
                    probability = probabilities[batch_index, rank].detach().cpu().numpy().astype(np.float32)
                    box = tuple(float(value) for value in boxes[batch_index, rank].tolist())
                    presence_score = (
                        None if presence is None else float(presence[batch_index].reshape(-1)[0])
                    )
                    for instance_threshold in instance_thresholds:
                        if score <= float(instance_threshold):
                            continue
                        for mask_threshold in mask_thresholds:
                            mask = probability > float(mask_threshold)
                            if not np.any(mask):
                                continue
                            variant = (
                                f"{spec.source_variant}|rank={rank}|raw={raw_index}|"
                                f"instance>{instance_threshold:.2f}|mask>{mask_threshold:.2f}"
                            )
                            provenance = {
                                "prompt_id": spec.prompt_id,
                                "text": spec.text,
                                "raw_query_index": raw_index,
                                "source_rank": rank,
                                "instance_threshold": float(instance_threshold),
                                "mask_threshold": float(mask_threshold),
                                "boxes_xyxy": spec.boxes_xyxy,
                                "box_labels": spec.box_labels,
                            }
                            all_candidates.append(
                                ProposalCandidate(
                                    sample_id=sample_id,
                                    source_family=spec.source_family,
                                    source_variant=variant,
                                    mask=mask,
                                    probability=probability,
                                    sam_score=score,
                                    presence_score=presence_score,
                                    box_xyxy=box,
                                    canonical_text_prompt=spec.text,
                                    prompt_boxes=spec.boxes_xyxy,
                                    mask_threshold=float(mask_threshold),
                                    instance_threshold=float(instance_threshold),
                                    model_revision=self.revision,
                                    rgb_checksum=rgb_sha256,
                                    eligible_final=spec.eligible_final,
                                    source_rank=rank,
                                    provenance=[provenance],
                                )
                            )
            del outputs, probabilities, repeated
            gc.collect()
        return all_candidates, {
            "prompt_count": len(prompts),
            "candidate_count": len(all_candidates),
            "vision_cache_hit": cache_hit,
            "vision_seconds": vision_seconds,
            "decoder_seconds": decoder_seconds,
            "postprocess_seconds": postprocess_seconds,
            "micro_batch_size": self.micro_batch_size,
            "maximum_instances_per_prompt": int(maximum_instances_per_prompt),
        }


__all__ = ["OfficialSam3TextProposalGenerator", "TextPromptSpec"]
