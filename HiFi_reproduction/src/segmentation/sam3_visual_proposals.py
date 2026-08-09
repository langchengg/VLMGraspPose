"""Official Sam3TrackerModel proposal generation with reusable image embeddings."""

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
class VisualPromptSpec:
    prompt_id: str
    source_family: str
    source_variant: str
    box_xyxy: tuple[float, float, float, float] | None = None
    positive_points_xy: tuple[tuple[float, float], ...] = ()
    negative_points_xy: tuple[tuple[float, float], ...] = ()
    input_mask: np.ndarray | None = None
    eligible_final: bool = True

    @property
    def point_count(self) -> int:
        return len(self.positive_points_xy) + len(self.negative_points_xy)


class OfficialSam3VisualProposalGenerator:
    def __init__(
        self,
        model_path: str | Path,
        *,
        revision: str,
        cache: Sam3EmbeddingCache,
        processor_sha256: str,
        num_threads: int = 8,
    ):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

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
        self.processor = Sam3TrackerProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = Sam3TrackerModel.from_pretrained(
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
            backend_class="transformers.Sam3TrackerModel",
        )

    def image_embeddings(
        self, image: Image.Image, rgb_sha256: str
    ) -> tuple[list[Any], bool, float]:
        key = self.cache_key(rgb_sha256)
        started = time.perf_counter()
        if self.cache.exists(key):
            return self.cache.load_tracker(key), True, time.perf_counter() - started
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with self.torch.inference_mode():
            embeddings = self.model.get_image_embeddings(inputs.pixel_values)
        self.cache.save_tracker(key, embeddings)
        return embeddings, False, time.perf_counter() - started

    @staticmethod
    def _group_prompts(prompts: list[VisualPromptSpec]) -> list[list[VisualPromptSpec]]:
        groups: dict[tuple[bool, bool, int], list[VisualPromptSpec]] = {}
        for prompt in prompts:
            key = (prompt.input_mask is not None, prompt.box_xyxy is not None, prompt.point_count)
            # The official dense-mask prompt has one mask per image rather than
            # one mask per object, so dense-mask prompts must remain separate.
            if prompt.input_mask is not None:
                groups[(True, prompt.box_xyxy is not None, prompt.point_count, prompt.prompt_id)] = [prompt]
            else:
                groups.setdefault(key, []).append(prompt)
        return list(groups.values())

    def _processor_inputs(
        self,
        specs: list[VisualPromptSpec],
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        height, width = image_shape
        kwargs: dict[str, Any] = {
            "original_sizes": [[height, width]],
            "return_tensors": "pt",
        }
        if specs[0].box_xyxy is not None:
            if not all(item.box_xyxy is not None for item in specs):
                raise ValueError("boxed and unboxed Tracker prompts cannot share a call")
            kwargs["input_boxes"] = [[list(item.box_xyxy) for item in specs]]
        if specs[0].point_count:
            if not all(item.point_count == specs[0].point_count for item in specs):
                raise ValueError("Tracker prompts in one call must have equal point counts")
            points: list[list[list[float]]] = []
            labels: list[list[int]] = []
            for item in specs:
                point_values = list(item.positive_points_xy) + list(item.negative_points_xy)
                points.append([list(value) for value in point_values])
                labels.append(
                    [1] * len(item.positive_points_xy)
                    + [0] * len(item.negative_points_xy)
                )
            kwargs["input_points"] = [points]
            kwargs["input_labels"] = [labels]
        return dict(self.processor(**kwargs))

    def generate(
        self,
        sample_id: str,
        image: Image.Image,
        rgb_sha256: str,
        prompts: Iterable[VisualPromptSpec],
        *,
        mask_thresholds: tuple[float, ...],
    ) -> tuple[list[ProposalCandidate], dict[str, Any]]:
        prompts = list(prompts)
        if not prompts:
            return [], {"prompt_count": 0, "candidate_count": 0}
        image = image.convert("RGB")
        height, width = image.height, image.width
        embeddings, cache_hit, embedding_seconds = self.image_embeddings(image, rgb_sha256)
        candidates: list[ProposalCandidate] = []
        decoder_seconds = 0.0
        calls = 0
        for specs in self._group_prompts(prompts):
            calls += 1
            inputs = self._processor_inputs(specs, (height, width))
            if specs[0].input_mask is not None:
                mask = np.asarray(specs[0].input_mask, dtype=np.float32)
                if mask.shape != (height, width) or not np.isfinite(mask).all():
                    raise ValueError("Tracker mask prompt must be finite and image-aligned")
                # The installed public forward path passes this tensor to
                # ``torch.interpolate`` and therefore requires an explicit
                # channel axis, despite the docstring abbreviating it as
                # ``[batch, height, width]``.
                inputs["input_masks"] = self.torch.from_numpy(mask)[None, None]
            started = time.perf_counter()
            with self.torch.inference_mode():
                outputs = self.model(
                    **inputs,
                    image_embeddings=embeddings,
                    multimask_output=True,
                )
            decoder_seconds += time.perf_counter() - started
            restored = self.processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                self.torch.tensor([[height, width]], dtype=self.torch.long),
                binarize=False,
            )[0]
            logits = restored.detach().float().cpu().numpy()
            if logits.ndim == 3:
                logits = logits[None]
            if logits.ndim != 4 or logits.shape[0] != len(specs):
                raise RuntimeError(
                    f"unexpected Tracker restored mask shape {logits.shape} for {len(specs)} prompts"
                )
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
            qualities = outputs.iou_scores.detach().float().cpu().numpy()
            qualities = qualities.reshape(len(specs), -1)
            if qualities.shape[1] != probabilities.shape[1]:
                raise RuntimeError("Tracker quality and multimask axes do not align")
            for object_index, spec in enumerate(specs):
                order = np.argsort(-qualities[object_index], kind="stable")
                for rank, hypothesis_index in enumerate(order.tolist()):
                    probability = np.asarray(
                        probabilities[object_index, hypothesis_index], dtype=np.float32
                    )
                    quality = float(qualities[object_index, hypothesis_index])
                    prompt_points = tuple(
                        (float(x), float(y), 1)
                        for x, y in spec.positive_points_xy
                    ) + tuple(
                        (float(x), float(y), 0)
                        for x, y in spec.negative_points_xy
                    )
                    for threshold in mask_thresholds:
                        mask = probability > float(threshold)
                        if not np.any(mask):
                            continue
                        variant = (
                            f"{spec.source_variant}|hypothesis={hypothesis_index}|"
                            f"rank={rank}|mask>{threshold:.2f}"
                        )
                        provenance = {
                            "prompt_id": spec.prompt_id,
                            "box_xyxy": spec.box_xyxy,
                            "positive_points_xy": spec.positive_points_xy,
                            "negative_points_xy": spec.negative_points_xy,
                            "uses_mask_prompt": spec.input_mask is not None,
                            "hypothesis_index": hypothesis_index,
                            "source_rank": rank,
                            "mask_threshold": float(threshold),
                        }
                        candidates.append(
                            ProposalCandidate(
                                sample_id=sample_id,
                                source_family=spec.source_family,
                                source_variant=variant,
                                mask=mask,
                                probability=probability,
                                sam_score=quality,
                                mask_quality_score=quality,
                                prompt_boxes=(() if spec.box_xyxy is None else (spec.box_xyxy,)),
                                prompt_points=prompt_points,
                                mask_threshold=float(threshold),
                                model_revision=self.revision,
                                rgb_checksum=rgb_sha256,
                                eligible_final=spec.eligible_final,
                                source_rank=rank,
                                provenance=[provenance],
                            )
                        )
            del outputs, restored, logits, probabilities
            gc.collect()
        return candidates, {
            "prompt_count": len(prompts),
            "decoder_calls": calls,
            "candidate_count": len(candidates),
            "image_embedding_cache_hit": cache_hit,
            "image_embedding_seconds": embedding_seconds,
            "decoder_seconds": decoder_seconds,
        }


__all__ = ["OfficialSam3VisualProposalGenerator", "VisualPromptSpec"]
