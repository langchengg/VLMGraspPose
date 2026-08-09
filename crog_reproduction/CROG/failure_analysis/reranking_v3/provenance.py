from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Iterable

from .schema import FEATURE_GROUPS, atomic_write_json, atomic_write_text, schema_hash


PROVENANCE_COLUMNS = (
    "feature_group", "feature_name", "source_file", "source_line", "tensor_name",
    "tensor_shape", "source_resolution", "coordinate_frame", "extraction_method",
    "candidate_specific", "query_specific", "uses_text", "uses_rgb", "uses_depth",
    "raw_or_postprocessed", "train_fitted_transform", "deployment_available",
    "leakage_status", "coverage", "used_by_v2_primary", "used_by_v3_candidate",
    "final_inclusion_decision", "validation_evidence",
)


def fullchain_inventory() -> list[dict[str, Any]]:
    specs = [
        ("G0", "frozen_candidate_q_geometry", "failure_analysis/reranking_outputs/full_test_17749_v1/features.jsonl", "candidate", "[B,5]", "original 480x640", True, False, False, False, False, "postprocessed", True, True),
        ("G1", "Q_raw_and_activated_maps", "model/layers.py", "quality head", "[B,1,104,104]", "native 104x104", True, False, False, True, False, "raw+sigmoid", True, False),
        ("G2", "predicted_mask_raw_and_sigmoid", "model/layers.py", "segmentation head", "[B,1,104,104]", "native 104x104", True, False, False, True, False, "raw+sigmoid", True, False),
        ("G3", "sin2theta_cos2theta", "model/layers.py", "sin/cos heads", "2x[B,1,104,104]", "native 104x104", True, False, False, True, False, "raw identity", True, False),
        ("G4", "width_raw_and_activated", "model/layers.py", "width head", "[B,1,104,104]", "native 104x104", True, False, False, True, False, "raw+sigmoid", True, False),
        ("G5", "five_head_cross_consistency", "failure_analysis/reranking_v3/cross_head_features.py", "derived", "candidate vector", "candidate ROI", True, False, False, True, False, "derived", True, False),
        ("G6", "C3_Fv2", "model/clip.py", "C3/Fv2", "[B,512,52,52]", "52x52", True, False, False, True, False, "raw latent ROI", True, False),
        ("G6", "C4_Fv3", "model/clip.py", "C4/Fv3", "[B,1024,26,26]", "26x26", True, False, False, True, False, "raw latent ROI", True, False),
        ("G6", "C5_Fv4", "model/clip.py", "C5/Fv4", "[B,1024,13,13]", "13x13", True, False, False, True, False, "raw latent ROI", True, False),
        ("G6", "multimodal_FPN", "model/layers.py", "Fm", "[B,512,26,26]", "26x26", True, False, True, True, False, "text-modulated latent ROI", True, False),
        ("G6", "pre_decoder", "model/layers.py", "decoder input", "[B,512,26,26]", "26x26", True, False, True, True, False, "multimodal latent ROI", True, False),
        ("G6", "decoder_layer_1", "model/layers.py", "decoder.layers.0", "[676,B,512]", "26x26", True, False, True, True, False, "multimodal latent ROI", True, False),
        ("G6", "decoder_layer_2", "model/layers.py", "decoder.layers.1", "[676,B,512]", "26x26", True, False, True, True, False, "multimodal latent ROI", True, False),
        ("G6", "decoder_layer_3_post", "model/layers.py", "decoder.layers.2/Fc", "[676,B,512]", "26x26", True, False, True, True, False, "multimodal latent ROI", True, True),
        ("G6", "projector_visual_branches", "model/layers.py", "proj.vis", "5x[B,256,104,104]", "104x104", True, False, True, True, False, "pre-head ROI", True, False),
        ("G7", "CLIP_token_features", "model/clip.py", "Ft", "[B,17,512]", "token sequence", False, True, True, False, False, "frozen token inputs", True, False),
        ("G7", "CLIP_sentence_EOT", "model/clip.py", "Fs", "[B,1024]", "query global", False, True, True, False, False, "frozen sentence", True, False),
        ("G7", "cross_attention_weights", "model/layers.py", "multihead_attn weights", "3x[B,676,17]", "spatial-token", True, True, True, True, False, "averaged attention", True, False),
        ("G7", "dynamic_kernel_bias_interaction", "model/layers.py", "proj.txt", "kernel [B,256,3,3]+bias", "query global interacted with ROI", True, True, True, True, False, "candidate interaction required", True, False),
        ("G8", "aligned_RGB_predicted_maps", "failure_analysis/reranking_v3/aligned_crops.py", "aligned crop", "[B,5,C,H,W]", "gripper aligned", True, False, False, True, False, "resampled", True, True),
        ("G9", "aligned_depth_geometry", "failure_analysis/reranking_v3/depth_geometry.py", "depth+validity", "candidate vector+crop", "original image/candidate", True, False, False, False, True, "meter verified", True, True),
        ("G10", "candidate_set_relations", "failure_analysis/reranking_v3/datasets.py", "pairwise/set", "[B,5,D]", "candidate set", True, False, False, False, False, "derived", True, True),
        ("G11", "V2_OOF_prior", "failure_analysis/reranking_v2/oof.py", "OOF base/setrank/gate", "[B,5,D]", "candidate set", True, False, True, True, True, "OOF only for training", True, True),
    ]
    rows = []
    for group, name, source, tensor, shape, resolution, candidate, query, text, rgb, depth, raw, available, v2 in specs:
        rows.append({
            "feature_group": group, "feature_name": name, "source_file": source,
            "source_line": "runtime-verified; exact line recorded by audit", "tensor_name": tensor,
            "tensor_shape": shape, "source_resolution": resolution,
            "coordinate_frame": resolution, "extraction_method": "non-mutating forward hook + exact-affine candidate ROI" if group in {"G6", "G7"} else "deterministic candidate extractor",
            "candidate_specific": candidate, "query_specific": query, "uses_text": text,
            "uses_rgb": rgb, "uses_depth": depth, "raw_or_postprocessed": raw,
            "train_fitted_transform": "development-only normalization/adapter",
            "deployment_available": available, "leakage_status": "allowed_inference_signal",
            "coverage": "pending full extraction", "used_by_v2_primary": v2,
            "used_by_v3_candidate": True, "final_inclusion_decision": "pending_v3_select",
            "validation_evidence": "pending",
        })
    return rows


def write_feature_provenance(output_dir: str | Path, rows: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    values = list(rows if rows is not None else fullchain_inventory())
    json_path = output / "V3_FEATURE_PROVENANCE.json"
    csv_path = output / "V3_FEATURE_PROVENANCE.csv"
    md_path = output / "V3_FEATURE_PROVENANCE.md"
    atomic_write_json(json_path, {"schema": list(PROVENANCE_COLUMNS), "schema_hash": schema_hash(values), "features": values})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=PROVENANCE_COLUMNS)
    writer.writeheader(); writer.writerows(values)
    atomic_write_text(csv_path, buffer.getvalue())
    lines = ["# CROG V3 Feature Provenance", "", "| Feature group | Source tensor | Candidate-specific | V2 primary | V3 extracted | V3 primary | Decision reason |", "|---|---|:---:|:---:|:---:|:---:|---|"]
    for row in values:
        lines.append(f"| {row['feature_group']} {row['feature_name']} | {row['tensor_name']} | {'yes' if row['candidate_specific'] else 'no'} | {'yes' if row['used_by_v2_primary'] else 'no'} | pending | pending | {row['final_inclusion_decision']} |")
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    return {"feature_count": len(values), "schema_hash": schema_hash(values), "paths": [str(json_path), str(csv_path), str(md_path)]}

