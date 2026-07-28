#!/usr/bin/env python3
"""Build a strict Dex-Net bundle view whose sole changed signal is selected SAM 3 mask."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.sam3_serialization import sha256_file, write_jsonl  # noqa: E402


REQUIRED_SOURCE_FILES = ("color.png", "depth.png", "language.txt", "intrinsics.json")


def _checksums(bundle: Path) -> None:
    names = (
        "color.png",
        "depth.png",
        "target_mask.png",
        "target_probability.npy",
        "language.txt",
        "intrinsics.json",
        "metadata.json",
    )
    lines = [f"{sha256_file(bundle / name)}  {name}" for name in names]
    (bundle / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-mask-root",
        type=Path,
        default=REPO_ROOT / "runs" / "hifics_ocidvlg_20260711_112921" / "anygrasp_input_predicted_mask",
    )
    parser.add_argument(
        "--refined-root", type=Path, default=REPO_ROOT / "outputs" / "sam3_refined_masks"
    )
    parser.add_argument(
        "--bundle-root", type=Path, default=REPO_ROOT / "outputs" / "sam3_dexnet_input_bundles"
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "dexnet_candidates_sam3_ten_samples",
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT.parent / "crog_reproduction" / "OCID-VLG")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    bundle_root = args.bundle_root.expanduser().resolve()
    if bundle_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite SAM 3 Dex-Net adapter: {bundle_root}")
    if args.candidate_output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite SAM 3 candidates: {args.candidate_output}")
    run = json.loads((args.refined_root / "run_metadata.json").read_text(encoding="utf-8"))
    if run.get("experiment_status") not in {
        "real_sam3_inference_completed",
        "real_sam3_cpu_inference_completed",
    }:
        raise RuntimeError("Dex-Net regeneration is gated until at least one real SAM 3 inference completed")
    refined_rows = [
        json.loads(line)
        for line in (args.refined_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    source_rows = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (args.source_mask_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    output_rows: list[dict] = []
    if bundle_root.exists():
        output_rows = [
            json.loads(line)
            for line in (bundle_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
    else:
        temporary = bundle_root.with_name(bundle_root.name + ".incomplete")
        temporary.mkdir(parents=True)
        try:
            for refined_row in refined_rows:
                sample_id = str(refined_row["sample_id"])
                source = args.source_mask_root / sample_id
                refined = args.refined_root / sample_id
                destination = temporary / sample_id
                destination.mkdir()
                source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
                refinement = json.loads((refined / "refinement_metadata.json").read_text(encoding="utf-8"))
                for name in REQUIRED_SOURCE_FILES:
                    shutil.copy2(source / name, destination / name)
                shutil.copy2(refined / "refined_mask.png", destination / "target_mask.png")
                shutil.copy2(refined / "refined_probability.npy", destination / "target_probability.npy")
                mask_hash = sha256_file(destination / "target_mask.png")
                probability_hash = sha256_file(destination / "target_probability.npy")
                selected_mask = np.asarray(Image.open(destination / "target_mask.png")) > 0
                depth = np.asarray(Image.open(destination / "depth.png"))
                metadata = dict(source_metadata)
                metadata.update(
                    {
                        "output_bundle": str(bundle_root / sample_id),
                        "prediction_mask": str(bundle_root / sample_id / "target_mask.png"),
                        "prediction_probability": str(bundle_root / sample_id / "target_probability.npy"),
                        "prediction_mask_sha256": mask_hash,
                        "prediction_probability_sha256": probability_hash,
                        "mask_source": "predicted_mask_original_resolution",
                        "oracle_artifacts_exported": False,
                        "target_valid_point_count": int(np.count_nonzero(selected_mask & (depth > 0))),
                        "sam3_selected_mask_source": refinement["selected_mask_source"],
                        "sam3_refinement_metadata": str(refined / "refinement_metadata.json"),
                        "sam3_refinement_metadata_sha256": sha256_file(refined / "refinement_metadata.json"),
                        "ready": True,
                        "ready_for_anygrasp": True,
                        "blockers": [],
                    }
                )
                (destination / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                _checksums(destination)
                original = source_rows[sample_id]
                output_rows.append(
                    {
                        **original,
                        "output_dir": str(bundle_root / sample_id),
                        "selected_mask_source": refinement["selected_mask_source"],
                        "target_valid_point_count": metadata["target_valid_point_count"],
                        "ready": True,
                        "ready_for_anygrasp": True,
                        "blockers": [],
                    }
                )
            write_jsonl(temporary / "manifest.jsonl", output_rows)
            temporary.rename(bundle_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    command = [
        str(REPO_ROOT / ".venv-gqcnn" / "bin" / "python"),
        str(REPO_ROOT / "scripts" / "run_hifics_dexnet_candidates.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--mask-root",
        str(bundle_root),
        "--output-dir",
        str(args.candidate_output),
        "--config",
        str(REPO_ROOT / "configs" / "dexnet_candidates.yaml"),
        "--mode",
        "candidate-only",
        "--num-candidates",
        "256",
        "--top-k",
        "30",
        "--seed",
        "42",
        "--visualize",
    ]
    if args.resume:
        command.extend(["--resume", "--verify-existing"])
    if args.execute:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        return completed.returncode
    print(json.dumps({"status": "ADAPTER_READY", "samples": len(output_rows), "command": command}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
