"""Deterministic reporting figures derived only from machine-readable tables."""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PIL import Image  # noqa: E402

from reranking.data_contracts import CROG_CANONICAL_CANDIDATES, MODULAR_CANONICAL_RUN, REPO_ROOT


FIGURE_NAMES = (
    "reliability",
    "method_comparison",
    "confidence_intervals",
    "switch_analysis",
    "q_drop_pareto",
    "ablation",
)

GALLERY_CATEGORIES = (
    "recovered",
    "harmful",
    "unchanged",
    "bothwrong",
    "empty",
    "no-positive",
    "rank>5",
)

FAILURE_STAGES = tuple(f"F{index}" for index in range(11))
FAILURE_GROUPS = {
    "grounding": ("F1", "F2"),
    "candidate-generation": ("F3", "F4"),
    "ranking": ("F5", "F6", "F7"),
}

CROG_RAW_PREDICTIONS = CROG_CANONICAL_CANDIDATES.with_name("predictions.jsonl")
MODULAR_ANNOTATIONS = REPO_ROOT / "crog_reproduction/OCID-VLG/refer/unique/test_expressions.json"


def _inference_pool_edgecolor(candidate: Mapping[str, Any]) -> str:
    """Use one GT-independent style for every frozen inference candidate."""

    del candidate
    return "#6B7280"


def _save_pair(fig: plt.Figure, output: Path, name: str) -> dict[str, str]:
    png = output / f"{name}.png"
    pdf = output / f"{name}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", metadata={"Software": "reranking.visualize"})
    fig.savefig(pdf, bbox_inches="tight", metadata={"Creator": "reranking.visualize"})
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _empty(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _reliability(data: Mapping[str, Any]) -> plt.Figure:
    frame = data["reliability"]
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    valid = frame.dropna(subset=["confidence", "accuracy"]) if not frame.empty else frame
    ax.plot([0, 1], [0, 1], "--", color="0.45", label="perfect calibration")
    if valid.empty:
        _empty(ax, "No calibrated probabilities in machine-readable predictions")
    else:
        ax.plot(valid["confidence"], valid["accuracy"], "o-", label="locked primary")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Empirical 2D-consistency rate")
        ax.legend()
    ax.set_title("Reliability diagram")
    return fig


def _method_comparison(data: Mapping[str, Any]) -> plt.Figure:
    frame = data["validation"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    id_col = next((c for c in ("method", "configuration", "experiment_id") if c in frame), None)
    metric_col = next((c for c in ("j_at_1", "selected_j_at_1", "validation_j_at_1") if c in frame), None)
    if id_col is None or metric_col is None:
        _empty(ax, "No validation J@1 column in registry")
    else:
        values = pd.to_numeric(frame[metric_col], errors="coerce")
        plot = frame.loc[values.notna(), [id_col]].copy()
        plot[metric_col] = values.loc[values.notna()]
        if plot.empty:
            _empty(ax, "No finite validation J@1 values")
        else:
            plot = plot.groupby(id_col, sort=True)[metric_col].mean().sort_values()
            ax.barh(plot.index.astype(str), plot.values, color="#3b82f6")
            ax.set_xlabel("Validation J@1 (2D consistency)")
    ax.set_title("Validation method comparison")
    return fig


def _confidence_intervals(data: Mapping[str, Any]) -> plt.Figure:
    frame = pd.DataFrame(data["bootstrap_rows"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if frame.empty:
        _empty(ax, "No scene-cluster bootstrap intervals")
    else:
        frame = frame.sort_values("point_estimate").reset_index(drop=True)
        y = np.arange(len(frame))
        x = frame["point_estimate"].to_numpy(float)
        lower = x - frame["ci_lower"].to_numpy(float)
        upper = frame["ci_upper"].to_numpy(float) - x
        ax.errorbar(x, y, xerr=[lower, upper], fmt="o", capsize=4)
        ax.axvline(0.0, color="0.4", linestyle="--")
        ax.set_yticks(y, frame["experiment_id"].astype(str))
        ax.set_xlabel("Paired J@1 difference (95% scene-cluster CI)")
    ax.set_title("Clustered confidence intervals")
    return fig


def _switch_analysis(data: Mapping[str, Any]) -> plt.Figure:
    summary = data["primary_summary"]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    names = ["Recovered", "Harmful", "Net", "All switches"]
    values = [
        summary["recovered"],
        summary["harmful"],
        summary["net_recovered"],
        summary["switch_count"],
    ]
    ax.bar(names, values, color=["#16a34a", "#dc2626", "#2563eb", "#64748b"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Query count")
    ax.set_title("Locked-primary switch outcomes")
    return fig


def _q_drop(data: Mapping[str, Any]) -> plt.Figure:
    frame = data["primary_outcomes"]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    if "q_drop" not in frame:
        _empty(ax, "No q_drop field in locked-primary predictions")
    else:
        q_drop = pd.to_numeric(frame["q_drop"], errors="coerce")
        gain = frame["selected_correct"].astype(int) - frame["baseline_correct"].astype(int)
        valid = q_drop.notna() & np.isfinite(q_drop)
        if not valid.any():
            _empty(ax, "No finite q_drop values")
        else:
            colors = np.where(gain.loc[valid] > 0, "#16a34a", np.where(gain.loc[valid] < 0, "#dc2626", "#94a3b8"))
            ax.scatter(q_drop.loc[valid], gain.loc[valid], c=colors, alpha=0.55, s=18)
            ax.axhline(0, color="0.4", linestyle="--")
            ax.set_xlabel("Baseline q drop caused by selected switch")
            ax.set_ylabel("Outcome change (-1, 0, +1)")
    ax.set_title("J@1 gain versus q-drop")
    return fig


def _ablation(data: Mapping[str, Any]) -> plt.Figure:
    frame = data["validation"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    label_col = next((c for c in ("ablation_group", "feature_group", "ablation", "configuration") if c in frame), None)
    metric_col = next((c for c in ("delta_j_at_1", "delta_vs_baseline", "j_at_1") if c in frame), None)
    if label_col is None or metric_col is None:
        _empty(ax, "No ablation rows in validation registry")
    else:
        values = pd.to_numeric(frame[metric_col], errors="coerce")
        plot = frame.loc[values.notna(), [label_col]].copy()
        plot[metric_col] = values.loc[values.notna()]
        if "category" in frame:
            category = frame.loc[plot.index, "category"].astype(str).str.lower()
            selected = category.str.contains("ablation|leave|remove|without|reference")
            if selected.any():
                plot = plot.loc[selected]
        if plot.empty:
            _empty(ax, "No finite ablation metrics")
        else:
            grouped = plot.groupby(label_col, sort=True)[metric_col].mean().sort_values()
            ax.barh(grouped.index.astype(str), grouped.values, color="#8b5cf6")
            ax.axvline(0, color="0.4", linewidth=0.8)
            ax.set_xlabel(metric_col)
    ax.set_title("Validation ablation evidence")
    return fig


_BUILDERS: dict[str, Callable[[Mapping[str, Any]], plt.Figure]] = {
    "reliability": _reliability,
    "method_comparison": _method_comparison,
    "confidence_intervals": _confidence_intervals,
    "switch_analysis": _switch_analysis,
    "q_drop_pareto": _q_drop,
    "ablation": _ablation,
}


def build_visualizations(
    output_dir: str | Path,
    *,
    validation_registry: pd.DataFrame,
    primary_outcomes: pd.DataFrame,
    reliability: pd.DataFrame,
    bootstrap_rows: list[dict[str, Any]],
    primary_summary: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Build the six required PNG/PDF figure pairs from supplied values."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = {
        "validation": validation_registry,
        "primary_outcomes": primary_outcomes,
        "reliability": reliability,
        "bootstrap_rows": bootstrap_rows,
        "primary_summary": primary_summary,
    }
    return {
        name: _save_pair(builder(data), output, name)
        for name, builder in _BUILDERS.items()
    }


def _gallery_categories(row: pd.Series) -> list[str]:
    candidate_count = int(row.get("candidate_count", 1))
    positive_count = int(row.get("positive_count", int(bool(row["oracle"]))))
    before = bool(row["baseline_correct"])
    after = bool(row["selected_correct"])
    if candidate_count == 0:
        categories = ["empty"]
    elif positive_count == 0:
        categories = ["no-positive"]
    elif not before and after:
        categories = ["recovered"]
    elif before and not after:
        categories = ["harmful"]
    elif before and after:
        categories = ["unchanged"]
    else:
        categories = ["bothwrong"]
    if float(row.get("first_positive_rank", 0)) > 5:
        categories.append("rank>5")
    return categories


def _base_sample_id(value: Any) -> str:
    text = str(value)
    return text.split("/", 1)[1] if "/" in text else text


def _crog_source_id(value: Any) -> str | None:
    base = _base_sample_id(value)
    tail = base.rsplit(":", 1)[-1]
    try:
        return str(int(tail))
    except ValueError:
        return None


def _enrich_posthoc_evidence(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical visualization-only evidence after all test scoring.

    The returned fields are never fed to a model.  They are read only by the
    post-lock gallery stage and make the inference/evaluation boundary
    explicit in every rendered case.
    """

    result = outcomes.copy()
    if "dataset" not in result:
        return result
    dataset = result["dataset"].astype(str)
    crog_rows = result.index[dataset.str.startswith("crog")]
    crog_index: dict[str, list[int]] = {}
    for index in crog_rows:
        source_id = _crog_source_id(result.at[index, "sample_id"])
        if source_id is not None:
            crog_index.setdefault(source_id, []).append(int(index))
    if crog_index and CROG_RAW_PREDICTIONS.is_file():
        with CROG_RAW_PREDICTIONS.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                raw = json.loads(line)
                indices = crog_index.get(str(raw.get("sample_id")))
                if not indices:
                    continue
                fields = {
                    "image_path": raw.get("image_path"),
                    "depth_path": raw.get("depth_path"),
                    "language_instruction": raw.get("language_instruction"),
                    "predicted_mask_rle_json": json.dumps(
                        raw.get("predicted_mask_rle"), separators=(",", ":")
                    ),
                    "gt_grasps_json": json.dumps(
                        raw.get("gt_grasps", []), separators=(",", ":")
                    ),
                    "mask_iou": raw.get("mask_iou"),
                    "gt_mask_path": raw.get("mask_path"),
                }
                for index in indices:
                    for name, value in fields.items():
                        result.at[index, name] = value

    modular_rows = result.index[dataset.str.startswith("modular")]
    manifest_path = MODULAR_CANONICAL_RUN / "input_manifest.csv"
    if len(modular_rows) and manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        manifest["sample_id"] = manifest["sample_id"].astype(str)
        by_sample = manifest.set_index("sample_id", drop=False)
        mask_metadata_path = (
            MODULAR_CANONICAL_RUN / "masks/hierfilm/per_sample_mask_metadata.csv"
        )
        mask_iou = {}
        if mask_metadata_path.is_file():
            metadata = pd.read_csv(mask_metadata_path)
            mask_iou = dict(
                zip(
                    metadata["sample_id"].astype(str),
                    pd.to_numeric(metadata["standard_iou_352"], errors="coerce"),
                    strict=False,
                )
            )
        annotations: list[dict[str, Any]] = []
        if MODULAR_ANNOTATIONS.is_file():
            annotations = list(json.loads(MODULAR_ANNOTATIONS.read_text(encoding="utf-8")).get("data", []))
        for index in modular_rows:
            sample_id = _base_sample_id(result.at[index, "sample_id"])
            if sample_id not in by_sample.index:
                continue
            source = by_sample.loc[sample_id]
            question_index = int(source["question_index"])
            bundle = MODULAR_CANONICAL_RUN / "masks/hierfilm/bundles" / sample_id
            fields = {
                "image_path": source.get("native_rgb_path", source.get("rgb_path")),
                "depth_path": source.get("native_depth_path", source.get("depth_path")),
                "language_instruction": source.get("query", ""),
                "predicted_mask_path": str(bundle / "target_mask.png"),
                "gt_mask_path": source.get("gt_mask_path", ""),
                "mask_iou": mask_iou.get(sample_id),
                "gt_grasps_json": json.dumps(
                    annotations[question_index].get("grasps", [])
                    if 0 <= question_index < len(annotations)
                    else [],
                    separators=(",", ":"),
                ),
            }
            for name, value in fields.items():
                result.at[index, name] = value

    if "mask_iou" in result:
        values = pd.to_numeric(result["mask_iou"], errors="coerce")
        grounding = values.lt(0.5) & result["candidate_count"].gt(0)
        result.loc[grounding, "failure_stage"] = "F1"
    return result


def _decode_rle(value: Any) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, Mapping):
        return None
    size = raw.get("size")
    counts = np.asarray(raw.get("counts", []), dtype=np.int64)
    if not isinstance(size, list) or len(size) != 2:
        return None
    if counts.size == 0:
        return np.zeros((int(size[0]), int(size[1])), dtype=bool)
    alternating = (
        np.arange(counts.size, dtype=np.uint8) + int(raw.get("start_value", 0))
    ) % 2
    flat = np.repeat(alternating, counts)
    if flat.size != int(size[0]) * int(size[1]):
        raise ValueError("predicted-mask RLE size mismatch")
    return flat.reshape((int(size[0]), int(size[1]))).astype(bool)


def _rectangle_polygon(candidate: Mapping[str, Any]) -> np.ndarray:
    x = float(candidate.get("x_px", 0.0))
    y = float(candidate.get("y_px", 0.0))
    width = max(float(candidate.get("width_px", 0.0)), 2.0)
    height = max(float(candidate.get("height_px", 20.0)), 2.0)
    angle = float(candidate.get("angle_rad", 0.0))
    local = np.asarray(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    return local @ rotation.T + np.asarray([x, y])


def _gt_polygon(grasp: Any) -> np.ndarray | None:
    raw = np.asarray(grasp, dtype=np.float64)
    if raw.shape == (4, 2):
        return raw
    if raw.ndim == 1 and raw.size >= 5:
        candidate = {
            "x_px": raw[0],
            "y_px": raw[1],
            "height_px": raw[2],
            "width_px": raw[3],
            "angle_rad": math.radians(float(raw[4])),
        }
        return _rectangle_polygon(candidate)
    return None


def _load_mask(row: pd.Series, shape: tuple[int, int]) -> np.ndarray | None:
    raw_path = row.get("predicted_mask_path")
    if raw_path is not None and str(raw_path).strip():
        path = Path(str(raw_path))
        if path.is_file() and not path.is_symlink():
            mask = np.asarray(Image.open(path).convert("L")) > 0
            if mask.shape == shape:
                return mask
    mask = _decode_rle(row.get("predicted_mask_rle_json"))
    return mask if mask is None or mask.shape == shape else None


def _render_gallery_case(row: pd.Series, output_path: Path) -> None:
    image_path = Path(str(row.get("image_path", "")))
    if image_path.is_symlink() or not image_path.is_file():
        raise FileNotFoundError(f"gallery RGB is missing: {image_path}")
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    pool = json.loads(str(row.get("candidate_pool_json", "[]")))
    if not isinstance(pool, list):
        raise ValueError("candidate_pool_json must contain a list")
    baseline_id = str(row.get("baseline_candidate_id", ""))
    selected_id = str(row.get("selected_candidate_id", ""))
    mask = _load_mask(row, rgb.shape[:2])
    depth = None
    depth_path = Path(str(row.get("depth_path", "")))
    if depth_path.is_file() and not depth_path.is_symlink():
        depth = np.asarray(Image.open(depth_path), dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
    grasps = json.loads(str(row.get("gt_grasps_json", "[]")))
    deltas = json.loads(str(row.get("feature_delta_json", "[]")))

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = axes.ravel()
    panels[0].imshow(rgb)
    panels[0].set_title("RGB (inference input)")

    overlay = rgb.copy().astype(np.float64)
    if mask is not None:
        tint = np.asarray([204.0, 121.0, 167.0])
        overlay[mask] = 0.68 * overlay[mask] + 0.32 * tint
    panels[1].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    panels[1].set_title("Predicted mask + frozen candidate pool")
    for candidate in pool:
        panels[1].add_patch(
            Polygon(
                _rectangle_polygon(candidate),
                closed=True,
                fill=False,
                edgecolor=_inference_pool_edgecolor(candidate),
                linewidth=1.2,
            )
        )

    if depth is None:
        panels[2].text(0.5, 0.5, "Depth unavailable", ha="center", va="center")
    else:
        valid = np.isfinite(depth) & (depth > 0)
        shown = depth.copy()
        if valid.any():
            low, high = np.percentile(shown[valid], [2, 98])
            shown[~valid] = np.nan
            panels[2].imshow(shown, cmap="magma_r", vmin=low, vmax=max(high, low + 1e-12))
        else:
            panels[2].text(0.5, 0.5, "No valid depth", ha="center", va="center")
    panels[2].set_title("Depth visualization (inference input)")

    panels[3].imshow(rgb)
    for candidate in pool:
        candidate_id = str(candidate.get("candidate_id", ""))
        if candidate_id not in {baseline_id, selected_id}:
            continue
        if candidate_id == baseline_id == selected_id:
            color, width, label = "#7C3AED", 5.0, "BASELINE = RERANK"
        elif candidate_id == baseline_id:
            color, width, label = "#EF4444", 5.0, "BASELINE"
        else:
            color, width, label = "#22C55E", 4.0, "RERANK"
        polygon = _rectangle_polygon(candidate)
        panels[3].add_patch(Polygon(polygon, closed=True, fill=False, edgecolor=color, linewidth=width, label=label))
    if panels[3].patches:
        handles, labels = panels[3].get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        panels[3].legend(unique.values(), unique.keys(), loc="lower left", fontsize=8)
    panels[3].set_title("Baseline Top-1 vs reranked Top-1")

    panels[4].imshow(rgb)
    for index, grasp in enumerate(grasps, start=1):
        polygon = _gt_polygon(grasp)
        if polygon is not None:
            panels[4].add_patch(
                Polygon(polygon, closed=True, fill=False, edgecolor="#00E5FF", linewidth=1.8)
            )
    panels[4].set_title("EVALUATION ONLY — GT grasps")

    panels[5].axis("off")
    by_id = {str(item.get("candidate_id")): item for item in pool}
    baseline = by_id.get(baseline_id, {})
    selected = by_id.get(selected_id, {})
    outcome = (
        "recovered" if not bool(row["baseline_correct"]) and bool(row["selected_correct"])
        else "harmful" if bool(row["baseline_correct"]) and not bool(row["selected_correct"])
        else "unchanged" if bool(row["baseline_correct"]) and bool(row["selected_correct"])
        else "unsolved"
    )
    lines = [
        f"query: {row.get('language_instruction', '')}",
        f"dataset: {row.get('dataset', '')}",
        f"outcome: {outcome}    failure stage: {row.get('failure_stage', 'NONE')}",
        f"gate: {row.get('gate_id', 'G0')}    switch: {bool(row.get('switch_applied', False))}",
        (
            f"baseline: {baseline_id}  q={baseline.get('q', float('nan')):.5g}  "
            f"rank={baseline.get('original_rank', 'n/a')}"
        ),
        (
            f"reranked: {selected_id}  q={selected.get('q', float('nan')):.5g}  "
            f"score={selected.get('rerank_score', float('nan')):.5g}  "
            f"rank={selected.get('rerank_rank', 'n/a')}"
        ),
        "",
        "largest selected − baseline feature deltas:",
    ]
    lines.extend(
        f"  {item['feature']}: {item['delta']:+.4g}"
        for item in deltas[:8]
    )
    if not deltas:
        lines.append("  no shared audited scalar delta")
    panels[5].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    panels[5].set_title("Decision audit")
    for panel in panels[:5]:
        panel.axis("off")
    figure.suptitle(
        f"{row['sample_id']} — post-lock offline analysis (GT isolated)", fontsize=13
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight", metadata={"Software": "reranking.visualize"})
    plt.close(figure)


def _selection_blocks(
    outcomes: pd.DataFrame,
    matcher: Callable[[pd.Series], bool],
    *,
    quota: int,
) -> tuple[list[pd.Series], dict[str, dict[str, int]]]:
    selected: list[pd.Series] = []
    counts: dict[str, dict[str, int]] = {}
    datasets = sorted(outcomes.get("dataset", pd.Series("all", index=outcomes.index)).astype(str).unique())
    for dataset in datasets:
        view = outcomes.loc[
            outcomes.get("dataset", pd.Series("all", index=outcomes.index)).astype(str).eq(dataset)
        ]
        eligible = [row for _, row in view.iterrows() if matcher(row)]
        chosen = eligible[:quota]
        selected.extend(chosen)
        counts[dataset] = {
            "requested": int(quota),
            "eligible": len(eligible),
            "selected": len(chosen),
            "materialized": 0,
        }
    return selected, counts


def build_galleries(
    output_dir: str | Path,
    outcomes: pd.DataFrame,
    *,
    quota_per_category: int = 25,
) -> dict[str, Any]:
    """Build a browsable index, copying only real declared image assets.

    Cases remain listed even when no image material exists.  The index records
    eligible, selected and materialized counts separately, so a shortfall can
    never be mistaken for a generated qualitative example.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    quota = int(quota_per_category)
    if quota < 1:
        raise ValueError("quota_per_category must be positive")
    ordered = _enrich_posthoc_evidence(outcomes).sort_values(
        [column for column in ("dataset", "scene_id", "sample_id") if column in outcomes],
        kind="mergesort",
    )
    index_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    asset_columns = ("gallery_image_path", "image_path", "visualization_path")
    render_cache: dict[tuple[str, str], Path] = {}

    def materialize(
        row: pd.Series,
        *,
        category: str,
        directory: Path,
        per_dataset: dict[str, dict[str, int]],
    ) -> None:
        dataset = str(row.get("dataset", "all"))
        source = None
        for column in asset_columns:
            raw = row.get(column)
            if raw is not None and str(raw).strip():
                candidate = Path(str(raw))
                if candidate.is_file() and not candidate.is_symlink():
                    source = candidate
                    break
        copied = None
        error = ""
        if source is not None:
            identity = f"{dataset}\0{row['sample_id']}"
            filename = hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".png"
            copied = directory / filename
            copied.parent.mkdir(parents=True, exist_ok=True)
            cache_key = (dataset, str(row["sample_id"]))
            try:
                rendered = render_cache.get(cache_key)
                if rendered is None:
                    rendered = output / "_rendered" / filename
                    rendered.parent.mkdir(parents=True, exist_ok=True)
                    if "candidate_pool_json" in row.index and pd.notna(
                        row.get("candidate_pool_json")
                    ):
                        _render_gallery_case(row, rendered)
                    else:
                        # Compatibility path for callers that only provide a
                        # declared qualitative asset. Formal matrix outcomes
                        # always use the audited six-panel renderer above.
                        shutil.copy2(source, rendered)
                    render_cache[cache_key] = rendered
                shutil.copy2(rendered, copied)
                per_dataset[dataset]["materialized"] += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as caught:
                copied = None
                error = f"{type(caught).__name__}: {caught}"
        index_rows.append(
            {
                "category": category,
                "dataset": dataset,
                "sample_id": str(row["sample_id"]),
                "scene_id": str(row["scene_id"]),
                "baseline_correct": bool(row["baseline_correct"]),
                "selected_correct": bool(row["selected_correct"]),
                "oracle": bool(row["oracle"]),
                "candidate_count": int(row.get("candidate_count", 1)),
                "positive_count": int(
                    row.get("positive_count", int(bool(row["oracle"])))
                ),
                "first_positive_rank": int(row.get("first_positive_rank", 0)),
                "failure_stage": str(row.get("failure_stage", "NONE")),
                "gate_id": str(row.get("gate_id", "")),
                "switch_applied": bool(row.get("switch_applied", False)),
                "asset_available": copied is not None,
                "asset_path": ""
                if copied is None
                else copied.relative_to(output).as_posix(),
                "render_error": error,
            }
        )

    for category in GALLERY_CATEGORIES:
        safe_category = category.replace(">", "_gt_").replace("-", "_")
        directory = output / safe_category
        directory.mkdir()
        selected, counts = _selection_blocks(
            ordered,
            lambda row, name=category: name in _gallery_categories(row),
            quota=quota,
        )
        for row in selected:
            dataset = str(row.get("dataset", "all"))
            target = directory if dataset == "all" else directory / dataset.replace("/", "_")
            materialize(row, category=category, directory=target, per_dataset=counts)
        summary[category] = {
            "requested": sum(value["requested"] for value in counts.values()),
            "eligible": sum(value["eligible"] for value in counts.values()),
            "selected": sum(value["selected"] for value in counts.values()),
            "materialized": sum(value["materialized"] for value in counts.values()),
            "shortfall": sum(
                max(0, value["requested"] - value["selected"])
                for value in counts.values()
            ),
            "per_dataset": counts,
        }

    failure_root = output / "failure_stages"
    failure_root.mkdir()
    stage_summary: dict[str, Any] = {}
    for stage in FAILURE_STAGES:
        directory = failure_root / stage
        directory.mkdir()
        selected, counts = _selection_blocks(
            ordered,
            lambda row, expected=stage: str(row.get("failure_stage", "NONE"))
            == expected,
            quota=quota,
        )
        for row in selected:
            dataset = str(row.get("dataset", "all"))
            target = directory if dataset == "all" else directory / dataset.replace("/", "_")
            materialize(
                row,
                category=f"failure_stage:{stage}",
                directory=target,
                per_dataset=counts,
            )
        stage_summary[stage] = {
            "requested": sum(value["requested"] for value in counts.values()),
            "eligible": sum(value["eligible"] for value in counts.values()),
            "selected": sum(value["selected"] for value in counts.values()),
            "materialized": sum(value["materialized"] for value in counts.values()),
            "per_dataset": counts,
            "evidence_status": "UNAVAILABLE"
            if stage in {"F0", "F2", "F8", "F9", "F10"}
            and not any(value["eligible"] for value in counts.values())
            else "AVAILABLE",
        }
    summary["failure_stages"] = stage_summary

    group_summary: dict[str, Any] = {}
    for group_name, stages in FAILURE_GROUPS.items():
        directory = failure_root / group_name.replace("-", "_")
        directory.mkdir()
        selected, counts = _selection_blocks(
            ordered,
            lambda row, expected=stages: str(row.get("failure_stage", "NONE"))
            in expected,
            quota=quota,
        )
        for row in selected:
            dataset = str(row.get("dataset", "all"))
            target = directory if dataset == "all" else directory / dataset.replace("/", "_")
            materialize(
                row,
                category=f"failure_group:{group_name}",
                directory=target,
                per_dataset=counts,
            )
        group_summary[group_name] = {
            "requested": sum(value["requested"] for value in counts.values()),
            "eligible": sum(value["eligible"] for value in counts.values()),
            "selected": sum(value["selected"] for value in counts.values()),
            "materialized": sum(value["materialized"] for value in counts.values()),
            "per_dataset": counts,
        }
    summary["failure_groups"] = group_summary

    shutil.rmtree(output / "_rendered", ignore_errors=True)
    index = pd.DataFrame(index_rows)
    index_path = output / "index.csv"
    index.to_csv(index_path, index=False)
    rows_html = []
    for row in index_rows:
        asset = (
            f'<a href="{html.escape(row["asset_path"])}"><img loading="lazy" '
            f'alt="{html.escape(row["category"])}" src="{html.escape(row["asset_path"])}"></a>'
            if row["asset_available"]
            else "material unavailable: " + html.escape(row["render_error"])
        )
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(row['category'])}</td>"
            f"<td>{html.escape(row['dataset'])}</td>"
            f"<td>{html.escape(row['sample_id'])}</td>"
            f"<td>{html.escape(row['scene_id'])}</td>"
            f"<td>{html.escape(row['failure_stage'])}</td>"
            f"<td>{asset}</td>"
            "</tr>"
        )
    counts = "".join(
        f"<li>{html.escape(name)}: eligible={value['eligible']}, "
        f"selected={value['selected']}, materialized={value['materialized']}</li>"
        for name, value in summary.items()
        if name in GALLERY_CATEGORIES
    )
    (output / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Failure gallery</title>"
        "<style>body{font-family:sans-serif;max-width:1500px;margin:auto}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:.4rem;vertical-align:top}img{width:420px;height:auto}</style></head><body>"
        "<h1>Locked-primary failure gallery</h1>"
        "<p>Six-panel images separate inference-visible RGB/mask/depth/candidates from the explicitly labelled offline GT panel.</p>"
        f"<ul>{counts}</ul><table><thead><tr><th>Category</th><th>Dataset</th><th>Sample</th>"
        f"<th>Scene</th><th>Failure stage</th><th>Asset</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
        "</body></html>\n",
        encoding="utf-8",
    )
    (output / "gallery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "index_html": str(output / "index.html"),
        "index_csv": str(index_path),
        "summary_json": str(output / "gallery_summary.json"),
        "categories": summary,
    }


__all__ = [
    "FIGURE_NAMES",
    "GALLERY_CATEGORIES",
    "build_galleries",
    "build_visualizations",
]
