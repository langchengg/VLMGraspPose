#!/usr/bin/env python3
"""Run official PCS and Tracker CPU forwards and record schema/timing/memory."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.sam3_cpu_model import (  # noqa: E402
    TransformersSam3Cpu,
    cpu_preflight,
)
from src.segmentation.sam3_cpu_serialization import (  # noqa: E402
    atomic_output_directory,
    save_strict_json,
)
from src.segmentation.sam3_cpu_visualization import save_candidate_grid  # noqa: E402
from src.segmentation.sam3_prompt_builder import build_visual_prompt  # noqa: E402


def _result_summary(result, name: str) -> dict[str, object]:
    areas = [int(np.count_nonzero(mask)) for mask in result.masks]
    if not areas or max(areas) == 0:
        raise RuntimeError(f"{name} returned no non-empty mask hypothesis")
    return {
        "hypotheses": len(result.masks),
        "nonempty_hypotheses": int(sum(area > 0 for area in areas)),
        "mask_areas_px": areas,
        "qualities": [
            None if quality is None else float(quality) for quality in result.qualities
        ],
        "probability_ranges": [
            [float(np.min(probability)), float(np.max(probability))]
            for probability in result.probabilities
        ],
        "timing": result.timings,
        "memory": result.memory,
        "output_schema": result.output_schema,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "hifics_ocidvlg_20260711_112921"
        / "anygrasp_input_predicted_mask"
        / "q0000000_b32eb3299dcd3ae9",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sam3_cpu_smoke_test",
    )
    parser.add_argument("--processor-size", type=int, choices=(1008, 560), default=1008)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--interop-threads", type=int, default=1)
    args = parser.parse_args()
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "sam3_cpu_refinement.yaml").read_text(encoding="utf-8")
    )
    image = Image.open(args.sample_dir / "color.png").convert("RGB")
    probability = np.load(
        args.sample_dir / "target_probability.npy", allow_pickle=False
    )
    prompt = build_visual_prompt(probability, config["prompt"])
    results: dict[str, object] = {}
    tracker = TransformersSam3Cpu(
        args.model_path,
        revision=args.revision,
        backend="tracker",
        processor_size=1008,
        num_threads=args.num_threads,
        interop_threads=args.interop_threads,
    )
    for name, mode in (
        ("tracker_point", "point"),
        ("tracker_box", "box"),
        ("tracker_box_point", "box_point"),
    ):
        result = tracker.infer(image, prompt, prompt_mode=mode)
        if not result.masks or not all(np.isfinite(mask).all() for mask in result.masks):
            raise RuntimeError(f"{name} did not return finite masks")
        results[name] = _result_summary(result, name)
        if name == "tracker_box_point":
            combined = result
    del tracker
    gc.collect()
    pcs = TransformersSam3Cpu(
        args.model_path,
        revision=args.revision,
        backend="pcs",
        processor_size=args.processor_size,
        num_threads=args.num_threads,
        interop_threads=args.interop_threads,
    )
    pcs_result = pcs.infer(image, prompt, prompt_mode="pcs_positive_box")
    if not pcs_result.masks or not all(np.isfinite(mask).all() for mask in pcs_result.masks):
        raise RuntimeError("PCS positive-box test did not return finite masks")
    results["pcs_positive_box"] = _result_summary(
        pcs_result, "pcs_positive_box"
    )
    output_root = args.output_root.expanduser().resolve()
    with atomic_output_directory(output_root) as temporary:
        environment = cpu_preflight()
        environment.update(
            {
                "model_revision": args.revision,
                "processor_size": args.processor_size,
                "num_threads": args.num_threads,
                "interop_threads": args.interop_threads,
            }
        )
        save_strict_json(temporary / "environment.json", environment)
        save_strict_json(
            temporary / "output_schema.json",
            {name: value["output_schema"] for name, value in results.items()},
        )
        save_strict_json(
            temporary / "timing.json",
            {name: value["timing"] for name, value in results.items()},
        )
        save_strict_json(
            temporary / "memory.json",
            {name: value["memory"] for name, value in results.items()},
        )
        save_strict_json(
            temporary / "validity.json",
            {
                name: {
                    key: value
                    for key, value in record.items()
                    if key not in {"timing", "memory", "output_schema"}
                }
                for name, record in results.items()
            },
        )
        save_candidate_grid(
            np.asarray(image),
            combined.masks,
            tuple(
                {
                    "candidate_id": f"tracker_box_point_{index}",
                    "sam_quality": quality,
                }
                for index, quality in enumerate(combined.qualities)
            ),
            temporary / "masks.png",
        )
        (temporary / "smoke_test.log").write_text(
            "\n".join(
                [
                    "status=PASSED",
                    "device=cpu",
                    "dtype=float32",
                    f"revision={args.revision}",
                    *[
                        f"{name}: hypotheses={value['hypotheses']} "
                        f"nonempty={value['nonempty_hypotheses']} "
                        f"areas_px={value['mask_areas_px']}"
                        for name, value in results.items()
                    ],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "PASSED", "tests": list(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
