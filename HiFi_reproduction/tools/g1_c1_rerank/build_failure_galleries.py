#!/usr/bin/env python3
"""Render deterministic post-formal qualitative galleries with GT marked analysis-only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.backends.conditioning import resize_probability_to_native  # noqa: E402


QUOTAS = {
    "recovered": 20,
    "harmful": 20,
    "unchanged_wrong_solvable": 20,
    "no_positive": 10,
    "gate_prevented_harmful": 10,
    "router_success": 10,
    "router_failure": 10,
}


def _candidate_colors(count: int) -> np.ndarray:
    """Return deterministic, non-repeating colors beyond tab20's limit."""
    if count <= 0:
        return np.empty((0, 4), dtype=float)
    index = np.arange(count, dtype=float)
    hue = (0.11 + index * 0.618033988749895) % 1.0
    saturation = 0.72 + 0.20 * ((index.astype(int) % 3) / 2.0)
    value = 0.76 + 0.20 * (((index.astype(int) // 3) % 2))
    rgb = hsv_to_rgb(np.column_stack([hue, saturation, value]))
    return np.column_stack([rgb, np.ones(count, dtype=float)])


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _corners(row: Mapping[str, Any]) -> np.ndarray:
    center = np.array([float(row["center_x"]), float(row["center_y"])])
    angle = math.radians(float(row["angle_deg"]))
    axis = np.array([math.cos(angle), math.sin(angle)])
    cross = np.array([-axis[1], axis[0]])
    half_w, half_h = float(row["width_px"]) / 2, float(row["height_px"]) / 2
    return np.stack([center-axis*half_w-cross*half_h, center+axis*half_w-cross*half_h, center+axis*half_w+cross*half_h, center-axis*half_w+cross*half_h])


def _probability(path: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            array = np.asarray(value["probability"], dtype=float)
        finally:
            value.close()
        return array
    return np.asarray(value, dtype=float)


def _overlay(ax: plt.Axes, rgb: np.ndarray, mask: np.ndarray, candidates: pd.DataFrame, *, baseline_id: str, challenger_id: str, selected_id: str, gt: Sequence[Any] | None) -> None:
    ax.imshow(rgb)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[..., 1] = 1.0
    overlay[..., 3] = mask.astype(float) * .22
    ax.imshow(overlay)
    records=candidates.sort_values(["original_rank","stable_candidate_id"],kind="mergesort").to_dict(orient="records")
    colors=_candidate_colors(len(records))
    handles=[]
    for index, row in enumerate(records):
        candidate_id = str(row["stable_candidate_id"])
        color = colors[index]
        linewidth = 1.2
        linestyle="-"
        roles=[]
        if candidate_id == baseline_id:
            linewidth,linestyle = 2.5,":"; roles.append("B")
        if candidate_id == challenger_id:
            linewidth,linestyle = 3.0,"--"; roles.append("U")
        if candidate_id == selected_id:
            linewidth,linestyle = 4.0,"-"; roles.append("F")
        polygon = _corners(row)
        ax.add_patch(plt.Polygon(polygon, fill=False, edgecolor=color, linewidth=linewidth,linestyle=linestyle))
        ax.text(float(row["center_x"]), float(row["center_y"]), str(int(row["original_rank"])), color=color, fontsize=7, weight="bold")
        role=f" [{'/'.join(roles)}]" if roles else ""
        handles.append(Line2D([0],[0],color=color,linewidth=linewidth,linestyle=linestyle,label=f"r{int(row['original_rank'])} {candidate_id[:8]}{role}"))
    if gt is not None:
        for rectangle in gt:
            polygon = np.asarray(rectangle, dtype=float)
            if polygon.shape == (4, 2):
                ax.add_patch(plt.Polygon(polygon, fill=False, edgecolor="#00FFFF", linewidth=1.5, linestyle="--"))
    if handles:
        ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(0,-.01),fontsize=4,ncol=min(4,len(handles)),frameon=False)
    ax.axis("off")


def _render(path: Path, *, sample: Mapping[str, Any], label: Mapping[str, Any], candidates: pd.DataFrame, scores: pd.DataFrame, outcome: Mapping[str, Any], category: str, backend: str) -> None:
    rgb = np.asarray(Image.open(sample["source_rgb_path"]).convert("RGB"))
    depth = np.asarray(Image.open(sample["source_depth_path"]), dtype=float) / 1000.0
    mask = np.asarray(Image.open(sample["predicted_mask_path"])) > 0
    probability = resize_probability_to_native(_probability(sample["predicted_probability_path"]), depth.shape)
    merged = candidates.merge(scores[["sample_id", "stable_candidate_id", "reranker_score"]], on=["sample_id", "stable_candidate_id"], how="left", validate="one_to_one")
    baseline_id = str(outcome.get("baseline_candidate_id", ""))
    challenger_id = str(outcome.get("ungated_candidate_id", outcome.get("selected_candidate_id", "")))
    selected_id = str(
        outcome.get(
            "final_selected_candidate_id",
            outcome.get("selected_candidate_id", challenger_id),
        )
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    _overlay(axes[0,0], rgb, mask, merged, baseline_id=baseline_id, challenger_id=challenger_id, selected_id=selected_id, gt=None)
    axes[0,0].set_title("INFERENCE VIEW: frozen candidates + predicted mask")
    _overlay(axes[0,1], rgb, mask, merged, baseline_id=baseline_id, challenger_id=challenger_id, selected_id=selected_id, gt=label["gt_grasp_rectangles"])
    axes[0,1].set_title("ANALYSIS ONLY: GT rectangles in cyan")
    axes[0,2].imshow(probability, cmap="magma", vmin=0, vmax=1); axes[0,2].set_title("HiFi probability"); axes[0,2].axis("off")
    finite = depth[np.isfinite(depth) & (depth > 0)]
    vmin, vmax = (np.quantile(finite,[.02,.98]) if len(finite) else (0,1))
    axes[1,0].imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax); axes[1,0].set_title("Depth (m)"); axes[1,0].axis("off")
    axes[1,1].imshow(rgb); axes[1,1].imshow(mask, cmap="Greens", alpha=.3); axes[1,1].set_title("Predicted mask overlay"); axes[1,1].axis("off")
    axes[1,2].axis("off")
    columns = [column for column in ("original_rank","original_score","reranker_score","grasp_axis_mask_support","width_compatibility","sweep_minimum_clearance_proxy") if column in merged]
    table = merged.sort_values("original_rank").head(10)[columns].round(3)
    axes[1,2].table(cellText=table.values, colLabels=[name.replace("_","\n") for name in columns], loc="center", cellLoc="center", fontsize=6)
    delta_columns=[column for column in ("original_score","reranker_score","grasp_axis_mask_support","width_compatibility","sweep_minimum_clearance_proxy") if column in merged]
    delta_text=[]
    baseline_rows=merged.loc[merged["stable_candidate_id"].astype(str).eq(baseline_id)]
    challenger_rows=merged.loc[merged["stable_candidate_id"].astype(str).eq(challenger_id)]
    selected_rows=merged.loc[merged["stable_candidate_id"].astype(str).eq(selected_id)]
    if not baseline_rows.empty:
        baseline_row=baseline_rows.iloc[0]
        for column in delta_columns:
            if not challenger_rows.empty:
                delta=float(challenger_rows.iloc[0][column])-float(baseline_row[column])
                delta_text.append(f"ΔU−B {column.replace('_',' ')}={delta:+.3f}")
            if not selected_rows.empty:
                delta=float(selected_rows.iloc[0][column])-float(baseline_row[column])
                delta_text.append(f"ΔF−B {column.replace('_',' ')}={delta:+.3f}")
    axes[1,2].set_title("Candidate score / evidence table\n"+"; ".join(delta_text),fontsize=7)
    fig.suptitle(f"{backend.upper()} · {category} · {sample['sample_id']}\nQuery: {sample['language']}\nDistinct color = candidate; dotted B=baseline, dashed U=ungated, solid F=final; rank printed at centre", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _contact_sheet(paths: Sequence[Path], destination: Path) -> None:
    if not paths:
        return
    thumbs=[]
    for path in paths:
        image=Image.open(path).convert("RGB"); image.thumbnail((420,240)); thumbs.append(image.copy())
    columns=4; rows=(len(thumbs)+columns-1)//columns
    sheet=Image.new("RGB",(columns*420,rows*240),(245,245,245))
    for index,image in enumerate(thumbs): sheet.paste(image,((index%columns)*420,(index//columns)*240))
    destination.parent.mkdir(parents=True,exist_ok=True); sheet.save(destination)


def main(argv: Sequence[str] | None = None) -> int:
    args=_parse(argv); base=args.base_run.expanduser().resolve(); run=args.run_dir.expanduser().resolve()
    if (run / "COMPLETE").exists(): raise PermissionError("refusing to mutate sealed failure galleries")
    samples=pd.read_parquet(base/"manifests"/"test_samples.parquet"); raw_labels=pd.read_parquet(run/"09_formal_test"/"test_analysis_labels.parquet")
    sample_by={str(row.sample_id):row._asdict() for row in samples.itertuples(index=False)}
    label_by={str(row.sample_id):row._asdict() for row in raw_labels.itertuples(index=False)}
    lock=json.loads((run/"08_lock"/"PRIMARY_METHOD_LOCK.json").read_text())
    router_outcome=pd.read_parquet(run/"09_formal_test"/"outcomes"/"cross_backend"/"router.parquet").set_index("sample_id")
    manifest_rows=[]
    for backend in ("g1","c1"):
        selected=lock["selected_methods"]["backend"][backend.upper()]; pool=str(selected["primary_pool"]); method=str(selected["primary_ungated_method"])
        candidates=pd.read_parquet(run/"02_features"/"test"/backend/f"{pool}_features.parquet")
        labels=pd.read_parquet(run/"09_formal_test"/f"{backend}_candidate_labels.parquet")
        joined=candidates.merge(labels[["sample_id","stable_candidate_id","candidate_correct"]],on=["sample_id","stable_candidate_id"],validate="one_to_one")
        scores=pd.read_parquet(run/"09_formal_test"/"candidate_scores"/backend/pool/f"{method}_ensemble.parquet")
        ungated=pd.read_parquet(run/"09_formal_test"/"outcomes"/backend/pool/f"{method}_ensemble.parquet").set_index("sample_id")
        gated=pd.read_parquet(run/"09_formal_test"/"outcomes"/backend/pool/"primary_gated.parquet").set_index("sample_id")
        positive=joined.groupby("sample_id")["candidate_correct"].any()
        categories={
            "recovered":ungated.index[ungated["recovered"]].tolist(),
            "harmful":ungated.index[ungated["harmful"]].tolist(),
            "unchanged_wrong_solvable":ungated.index[(~ungated["final_correct"]) & ungated.index.to_series().map(positive).fillna(False).to_numpy()].tolist(),
            "no_positive":[sample_id for sample_id, value in positive.items() if not bool(value)],
            "gate_prevented_harmful":ungated.index[ungated["harmful"] & ~gated["harmful"]].tolist(),
        }
        for category,ids in categories.items():
            chosen=sorted(map(str,ids))[:QUOTAS[category]]; rendered=[]
            for index,sample_id in enumerate(chosen):
                local_candidates=joined.loc[joined["sample_id"].astype(str).eq(sample_id)]
                if local_candidates.empty:
                    continue
                outcome=(ungated.loc[sample_id].to_dict() if sample_id in ungated.index else gated.loc[sample_id].to_dict())
                if sample_id in ungated.index:
                    outcome["ungated_candidate_id"]=ungated.loc[sample_id,"selected_candidate_id"]
                if sample_id in gated.index:
                    outcome["final_selected_candidate_id"]=gated.loc[sample_id,"selected_candidate_id"]
                path=run/"12_failure_galleries"/backend/category/f"{index:03d}_{sample_id.replace('/','_')}.png"
                _render(path,sample=sample_by[sample_id],label=label_by[sample_id],candidates=local_candidates,scores=scores,outcome=outcome,category=category,backend=backend)
                rendered.append(path)
                manifest_rows.append({"backend":backend.upper(),"category":category,"sample_id":sample_id,"path":str(path)})
            _contact_sheet(rendered,run/"12_failure_galleries"/backend/f"{category}_contact_sheet.png")
            manifest_rows.append({"backend":backend.upper(),"category":category,"sample_id":"__SUMMARY__","eligible":len(ids),"requested":QUOTAS[category],"rendered":len(rendered),"shortfall":max(QUOTAS[category]-len(rendered),0)})
    # Router panels use the actual cross-backend union so candidate IDs and
    # colors are never rendered against the wrong backend's pool.
    cross_candidates=[]; cross_scores=[]
    for backend in ("g1","c1"):
        selected=lock["selected_methods"]["backend"][backend.upper()]
        method=str(selected["primary_ungated_method"])
        frame=pd.read_parquet(run/"02_features"/"test"/backend/"allnms_features.parquet")
        score=pd.read_parquet(run/"09_formal_test"/"candidate_scores"/backend/"allnms"/f"{method}_ensemble.parquet")
        cross_candidates.append(frame)
        cross_scores.append(score)
    union_candidates=pd.concat(cross_candidates,ignore_index=True)
    union_scores=pd.concat(cross_scores,ignore_index=True)
    for category,ids in {
        "router_success":router_outcome.index[router_outcome["recovered"]].tolist(),
        "router_failure":router_outcome.index[router_outcome["harmful"]].tolist(),
    }.items():
        chosen=sorted(map(str,ids))[:QUOTAS[category]]; rendered=[]
        for index,sample_id in enumerate(chosen):
            local_candidates=union_candidates.loc[union_candidates["sample_id"].astype(str).eq(sample_id)]
            if local_candidates.empty: continue
            outcome=router_outcome.loc[sample_id].to_dict()
            outcome["ungated_candidate_id"]=outcome.get("selected_candidate_id","")
            outcome["final_selected_candidate_id"]=outcome.get("selected_candidate_id","")
            path=run/"12_failure_galleries"/"cross_backend"/category/f"{index:03d}_{sample_id.replace('/','_')}.png"
            _render(path,sample=sample_by[sample_id],label=label_by[sample_id],candidates=local_candidates,scores=union_scores,outcome=outcome,category=category,backend="cross_backend")
            rendered.append(path)
            manifest_rows.append({"backend":"G1+C1","category":category,"sample_id":sample_id,"path":str(path)})
        _contact_sheet(rendered,run/"12_failure_galleries"/"cross_backend"/f"{category}_contact_sheet.png")
        manifest_rows.append({"backend":"G1+C1","category":category,"sample_id":"__SUMMARY__","eligible":len(ids),"requested":QUOTAS[category],"rendered":len(rendered),"shortfall":max(QUOTAS[category]-len(rendered),0)})
    manifest=pd.DataFrame(manifest_rows); manifest.to_csv(run/"12_failure_galleries"/"gallery_manifest.csv",index=False)
    summary=manifest.loc[manifest["sample_id"].eq("__SUMMARY__")]
    (run/"12_failure_galleries"/"README.md").write_text("# Failure Galleries\n\nGT rectangles appear only in panels explicitly marked **ANALYSIS ONLY**. Shortfalls are reported rather than filled with duplicate or synthetic cases.\n\n"+summary.to_markdown(index=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","rendered":int(manifest["path"].notna().sum()) if "path" in manifest else 0,"summary":str(run/"12_failure_galleries"/"README.md")},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
