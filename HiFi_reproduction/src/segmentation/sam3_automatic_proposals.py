"""Official Transformers mask-generation pipeline for generic SAM 3 proposals."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .proposal_types import ProposalCandidate


class OfficialSam3AutomaticProposalGenerator:
    def __init__(
        self,
        model_path: str | Path,
        *,
        revision: str,
        num_threads: int = 8,
        shared_tracker_model: Any | None = None,
        shared_tracker_processor: Any | None = None,
    ):
        import torch
        from transformers import AutoModelForMaskGeneration, Sam3TrackerProcessor, pipeline

        self.torch = torch
        torch.set_num_threads(int(num_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        self.model_path = Path(model_path).expanduser().resolve()
        self.revision = revision
        if (shared_tracker_model is None) != (shared_tracker_processor is None):
            raise ValueError("shared Tracker model and processor must be supplied together")
        if shared_tracker_model is None:
            model = AutoModelForMaskGeneration.from_pretrained(
                self.model_path, local_files_only=True, dtype=torch.float32
            ).to("cpu").eval()
            processor = Sam3TrackerProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.shared_tracker_model = False
        else:
            model = shared_tracker_model.to("cpu").eval()
            processor = shared_tracker_processor
            self.shared_tracker_model = True
        self.model = model
        self.processor = processor
        self.pipeline = pipeline(
            "mask-generation",
            model=model,
            image_processor=processor.image_processor,
            device="cpu",
            dtype=torch.float32,
        )

    def generate(
        self,
        sample_id: str,
        image: Image.Image,
        rgb_sha256: str,
        *,
        points_per_side: int,
        points_per_batch: int,
        score_threshold: float,
        stability_threshold: float,
        nms_threshold: float,
        minimum_mask_area: int,
        maximum_proposals: int,
    ) -> tuple[list[ProposalCandidate], dict[str, Any]]:
        started = time.perf_counter()
        output = self.pipeline(
            image.convert("RGB"),
            points_per_crop=int(points_per_side),
            points_per_batch=int(points_per_batch),
            pred_iou_thresh=float(score_threshold),
            stability_score_thresh=float(stability_threshold),
            crops_n_layers=0,
            crops_nms_thresh=float(nms_threshold),
            output_bboxes_mask=True,
            batch_size=1,
        )
        masks = output["masks"]
        scores = output["scores"].detach().float().cpu().numpy()
        boxes = output.get("bounding_boxes")
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().float().cpu().numpy()
        order = np.argsort(-scores, kind="stable")
        candidates: list[ProposalCandidate] = []
        for rank, index in enumerate(order.tolist()):
            if len(candidates) >= int(maximum_proposals):
                break
            mask = np.asarray(masks[index], dtype=bool)
            if int(np.count_nonzero(mask)) < int(minimum_mask_area):
                continue
            box = None if boxes is None else tuple(float(value) for value in boxes[index])
            candidates.append(
                ProposalCandidate(
                    sample_id=sample_id,
                    source_family="AUTOMATIC",
                    source_variant=f"grid_{points_per_side}x{points_per_side}|rank={rank}",
                    mask=mask,
                    sam_score=float(scores[index]),
                    mask_quality_score=float(scores[index]),
                    box_xyxy=box,
                    model_revision=self.revision,
                    rgb_checksum=rgb_sha256,
                    source_rank=rank,
                    provenance=[
                        {
                            "official_interface": "transformers.MaskGenerationPipeline",
                            "points_per_side": int(points_per_side),
                            "points_per_batch": int(points_per_batch),
                            "score_threshold": float(score_threshold),
                            "stability_threshold": float(stability_threshold),
                            "nms_threshold": float(nms_threshold),
                            "source_rank": rank,
                        }
                    ],
                )
            )
        return candidates, {
            "candidate_count": len(candidates),
            "raw_candidate_count": len(masks),
            "runtime_seconds": time.perf_counter() - started,
            "points_per_side": int(points_per_side),
            "points_per_batch": int(points_per_batch),
            "shared_tracker_model": self.shared_tracker_model,
        }


__all__ = ["OfficialSam3AutomaticProposalGenerator"]
