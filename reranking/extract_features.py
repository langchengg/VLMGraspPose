"""Inference-feature schema, leakage audit, and data-quality analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from reranking.models.tabular import assert_no_forbidden_columns


IDENTITY_COLUMNS = frozenset(
    {
        "sample_id",
        "scene_id",
        "frame_id",
        "expression_id",
        "candidate_id",
        "candidate_identity_sha256",
        "route",
        "pool_type",
    }
)

LABEL_COLUMNS = frozenset(
    {
        "candidate_correct",
        "label",
        "best_iou_same_gt",
        "best_angle_error_same_gt",
        "baseline_top1_correct",
        "pool_has_positive",
        "failure_mode",
        "evaluator_version",
        "evaluator_track",
        "source_candidate_position",
    }
)

NON_MODEL_COLUMNS = frozenset(
    {
        *IDENTITY_COLUMNS,
        "language_instruction",
        "image_path",
        "depth_path",
        "pcd_path",
        "split",
        "question_index",
        "sample_index",
        "pipeline",
        "stage",
        "candidate_json",
        "endpoints_uv_json",
        "contact_points_uv_json",
        "contact_normals_json",
        "center_camera_xyz_m_json",
        "pose_matrix_json",
        "candidate_seed",
        "rejection_reason",
        "valid",
    }
)

CROG_NATIVE_FORBIDDEN_EVIDENCE_TOKENS = (
    "depth",
    "clearance",
    "collision",
    "obstacle",
    "contact_depth",
    "normal",
    "surface",
    "crop_",
    "scanline",
    "axis_valid",
    "z_m",
    "z_reference",
    "safety",
)


class FeatureAuditError(ValueError):
    """Raised when an inference table contains leakage or invalid features."""


def feature_group(name: str) -> str:
    lower = name.lower()
    if lower.startswith("q_") or lower in {
        "q_raw",
        "original_rank",
        "rank_percentile",
        "candidate_count",
        "top1_margin",
        "top2_margin",
        "score_entropy",
        "score_concentration",
        "score_prominence",
    }:
        return "baseline"
    if any(token in lower for token in ("mask", "coverage", "component", "boundary")):
        return "mask"
    if any(token in lower for token in ("width", "thickness", "overreach", "underreach")):
        return "width"
    if any(token in lower for token in ("depth", "crop", "surface", "roughness", "gradient", "plane")):
        return "depth"
    if any(token in lower for token in ("contact", "normal", "friction")):
        return "contact"
    if any(token in lower for token in ("collision", "clearance", "sweep", "obstacle", "clutter", "intrusion")):
        return "clearance"
    if any(token in lower for token in ("cluster", "neighbor", "overlap", "consensus", "density", "uniqueness")):
        return "relations"
    if any(token in lower for token in ("reliability", "missing", "available", "valid_ratio")):
        return "reliability"
    if any(token in lower for token in ("embedding", "latent", "decoder", "roi")):
        return "embedding"
    if lower in {"x_px", "y_px", "z_m", "angle_rad", "width_px", "width_m", "height_px"}:
        return "candidate_geometry"
    return "other_scalar"


def select_model_feature_columns(
    frame: pd.DataFrame,
    *,
    requested: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Select numeric inference columns and reject label-derived names."""

    if requested is None:
        candidates = [
            str(column)
            for column in frame.columns
            if column not in NON_MODEL_COLUMNS
            and column not in LABEL_COLUMNS
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
    else:
        candidates = [str(value) for value in requested]
    missing = sorted(set(candidates) - set(frame.columns))
    if missing:
        raise FeatureAuditError(f"requested features are missing: {missing}")
    leakage = sorted(set(candidates) & LABEL_COLUMNS)
    if leakage:
        raise FeatureAuditError(f"label columns requested as model features: {leakage}")
    try:
        assert_no_forbidden_columns(candidates)
    except ValueError as error:
        raise FeatureAuditError(str(error)) from error
    return tuple(sorted(dict.fromkeys(candidates)))


def select_crog_native_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Select only CROG-native RGB/language evidence, never depth proxies."""

    candidates = select_model_feature_columns(frame)
    selected = tuple(
        column
        for column in candidates
        if not any(
            token in column.lower()
            for token in CROG_NATIVE_FORBIDDEN_EVIDENCE_TOKENS
        )
    )
    if not selected:
        raise FeatureAuditError("CROG-native feature schema is empty")
    return selected


def schema_sha256(columns: Sequence[str], groups: Mapping[str, str]) -> str:
    payload = json.dumps(
        {"columns": list(columns), "groups": dict(sorted(groups.items()))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_feature_schema(
    frame: pd.DataFrame,
    *,
    requested: Sequence[str] | None = None,
) -> dict[str, Any]:
    columns = select_model_feature_columns(frame, requested=requested)
    groups = {name: feature_group(name) for name in columns}
    return {
        "schema_version": 1,
        "feature_count": len(columns),
        "feature_columns": list(columns),
        "feature_groups": groups,
        "feature_schema_sha256": schema_sha256(columns, groups),
        "labels_stored_separately": True,
        "fit_time_transformations": [
            "missing-value imputation",
            "missing indicators",
            "scaling",
            "score calibration",
        ],
        "feature_source_constraint": "inference-time-only",
    }


def _safe_univariate_metrics(labels: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    if not finite.any():
        return {"roc_auc": None, "pr_auc": None, "spearman": None}
    y = labels[finite]
    x = values[finite]
    if np.unique(y).size < 2 or np.unique(x).size < 2:
        return {"roc_auc": None, "pr_auc": None, "spearman": None}
    correlation = spearmanr(x, y).statistic
    return {
        "roc_auc": float(roc_auc_score(y, x)),
        "pr_auc": float(average_precision_score(y, x)),
        "spearman": None if not np.isfinite(correlation) else float(correlation),
    }


def _univariate_j1(
    identity: pd.DataFrame,
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    work = identity[["sample_id", "candidate_id"]].copy()
    work["value"] = np.where(np.isfinite(values), values, -np.inf)
    work["label"] = labels
    top = (
        work.sort_values(
            ["sample_id", "value", "candidate_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("sample_id", sort=False)
        .first()
    )
    return 0.0 if top.empty else float(top["label"].mean())


def audit_feature_table(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    requested: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Quantify feature validity without fitting any model."""

    keys = ["sample_id", "candidate_id"]
    embedded_labels = sorted(set(features.columns) & LABEL_COLUMNS)
    if embedded_labels:
        raise FeatureAuditError(
            "feature table contains forbidden label columns despite the required "
            f"storage separation: {embedded_labels}"
        )
    for name, frame in (("features", features), ("labels", labels)):
        missing = sorted(set(keys) - set(frame.columns))
        if missing:
            raise FeatureAuditError(f"{name} table missing keys: {missing}")
        if frame.duplicated(keys).any():
            raise FeatureAuditError(f"{name} table has duplicate candidate keys")
    if "candidate_correct" not in labels.columns:
        raise FeatureAuditError("labels table lacks candidate_correct")
    joined = features.merge(
        labels[keys + ["candidate_correct"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(features) or len(joined) != len(labels):
        raise FeatureAuditError("feature/label candidate key sets differ")
    schema = build_feature_schema(features, requested=requested)
    y = joined["candidate_correct"].astype(bool).to_numpy()
    rows: list[dict[str, Any]] = []
    for name in schema["feature_columns"]:
        values = pd.to_numeric(joined[name], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(values)
        finite_values = values[finite]
        quantiles = (
            [None] * 7
            if finite_values.size == 0
            else [
                float(value)
                for value in np.quantile(
                    finite_values, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
                )
            ]
        )
        positive_values = values[finite & y]
        negative_values = values[finite & (~y)]
        metrics = _safe_univariate_metrics(y, values)
        unique_count = int(np.unique(finite_values).size)
        rows.append(
            {
                "feature": name,
                "group": schema["feature_groups"][name],
                "dtype": str(features[name].dtype),
                "missing_count": int((~finite).sum()),
                "missing_rate": float((~finite).mean()),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
                "unique_finite_count": unique_count,
                "constant": unique_count <= 1,
                "near_constant": bool(
                    finite_values.size > 0
                    and pd.Series(finite_values).value_counts(normalize=True).iloc[0]
                    >= 0.995
                ),
                "quantiles_0_1_10_50_90_99_100": quantiles,
                "positive_mean": (
                    None if positive_values.size == 0 else float(positive_values.mean())
                ),
                "negative_mean": (
                    None if negative_values.size == 0 else float(negative_values.mean())
                ),
                "univariate_j_at_1": _univariate_j1(joined, values, y),
                **metrics,
            }
        )
    numeric = joined[schema["feature_columns"]].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corr(method="spearman", min_periods=20)
    redundant: list[dict[str, Any]] = []
    for left_index, left in enumerate(correlations.columns):
        for right in correlations.columns[left_index + 1 :]:
            value = correlations.loc[left, right]
            if np.isfinite(value) and abs(float(value)) >= 0.98:
                redundant.append(
                    {"feature_a": left, "feature_b": right, "spearman": float(value)}
                )
    return {
        "schema_version": 1,
        "candidate_count": int(len(joined)),
        "query_count": int(joined["sample_id"].nunique()),
        "positive_candidate_count": int(y.sum()),
        "positive_candidate_rate": float(y.mean()),
        "feature_schema": schema,
        "features": rows,
        "high_redundancy_pairs_abs_spearman_ge_0_98": redundant,
        "forbidden_column_scanner_passed": True,
        "label_join_keys": keys,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_feature_audit(
    audit: Mapping[str, Any], json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    md_tmp = markdown_path.with_name(f".{markdown_path.name}.{os.getpid()}.tmp")
    safe = _json_safe(dict(audit))
    json_tmp.write_text(
        json.dumps(safe, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    feature_rows = safe["features"]
    lines = [
        "# Feature audit",
        "",
        f"- Candidates: {safe['candidate_count']}",
        f"- Queries: {safe['query_count']}",
        f"- Features: {safe['feature_schema']['feature_count']}",
        f"- Positive rate: {safe['positive_candidate_rate']:.6f}",
        f"- Leakage scanner: {'PASS' if safe['forbidden_column_scanner_passed'] else 'FAIL'}",
        "",
        "| feature | group | missing | ROC-AUC | PR-AUC | univariate J@1 | constant |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in feature_rows:
        def render(value: Any) -> str:
            return "N/A" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)

        lines.append(
            "| {feature} | {group} | {missing_rate} | {roc_auc} | {pr_auc} | "
            "{univariate_j_at_1} | {constant} |".format(
                **{key: render(value) for key, value in row.items()}
            )
        )
    md_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(json_tmp, json_path)
    os.replace(md_tmp, markdown_path)


def _feature_unit(name: str) -> str:
    lower = name.lower()
    if lower.endswith("_rad") or "angle_rad" in lower:
        return "radian"
    if lower.endswith("_deg") or "angle_error" in lower:
        return "degree"
    if lower.endswith("_px") or "distance_px" in lower or "area_px" in lower:
        return "pixel"
    if lower.endswith("_m") or "depth_m" in lower:
        return "metre"
    if "rank" in lower or "count" in lower or lower.endswith("_id"):
        return "count/index"
    if any(
        token in lower
        for token in (
            "probability",
            "coverage",
            "ratio",
            "fraction",
            "support",
            "reliability",
            "entropy",
            "score",
            "q_",
        )
    ):
        return "dimensionless"
    return "source-native dimensionless"


def _feature_source(route: str, group: str) -> str:
    if route == "crog":
        sources = {
            "baseline": "frozen CROG quality/rank output",
            "mask": "frozen CROG predicted mask/probability and diagnostics",
            "width": "frozen CROG candidate geometry and predicted-mask diagnostics",
            "depth": "not admitted to the CROG-native model schema",
            "contact": "frozen CROG candidate diagnostics when present",
            "clearance": "frozen CROG RGB/mask diagnostics; no native depth evidence",
            "relations": "relations derived only from frozen CROG candidates",
            "reliability": "availability/reliability flags from frozen CROG artifacts",
            "embedding": "frozen CROG latent/ROI feature when present",
            "candidate_geometry": "immutable frozen CROG candidate pose",
            "other_scalar": "frozen CROG inference artifact",
        }
    else:
        sources = {
            "baseline": "official frozen GQ-CNN quality/rank output",
            "mask": "frozen predicted HiFi probability/binary mask",
            "width": "immutable Dex-Net geometry plus predicted-mask thickness",
            "depth": "input depth and official candidate-aligned GQ-CNN preprocessing",
            "contact": "frozen Dex-Net contact geometry and local depth",
            "clearance": "single-view 2.5D depth/mask proxy",
            "relations": "relations derived only from frozen Dex-Net candidates",
            "reliability": "inference-time availability/reliability flags",
            "embedding": "frozen backbone/ROI embedding when present",
            "candidate_geometry": "immutable frozen Dex-Net candidate pose",
            "other_scalar": "frozen Modular inference artifact",
        }
    return sources.get(group, sources["other_scalar"])


def write_feature_audit_bundle(
    audits: Sequence[Mapping[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    schema_path: Path,
) -> None:
    """Write one honest multi-route audit instead of exposing only the first dataset."""

    if not audits:
        raise FeatureAuditError("at least one feature audit is required")
    enriched: list[dict[str, Any]] = []
    for raw in audits:
        audit = _json_safe(dict(raw))
        route = str(audit.get("route", "unknown"))
        for row in audit.get("features", []):
            name = str(row["feature"])
            group = str(row["group"])
            row["definition"] = (
                f"Canonical inference-time scalar `{name}`; the implementation "
                "field and feature-schema hash are the executable definition."
            )
            row["unit"] = _feature_unit(name)
            row["source"] = _feature_source(route, group)
        enriched.append(audit)

    schemas = [
        {
            "route": str(audit.get("route", "unknown")),
            "pool": str(audit.get("pool", "unknown")),
            **dict(audit["feature_schema"]),
        }
        for audit in enriched
    ]
    schema_payload: dict[str, Any] = {
        "schema_version": 1,
        "datasets": schemas,
        "labels_stored_separately": True,
        "feature_source_constraint": "inference-time-only",
    }
    schema_payload["combined_feature_schema_sha256"] = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    payload = {"schema_version": 1, "audits": enriched}

    for path in (json_path, markdown_path, schema_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    md_tmp = markdown_path.with_name(f".{markdown_path.name}.{os.getpid()}.tmp")
    schema_tmp = schema_path.with_name(f".{schema_path.name}.{os.getpid()}.tmp")
    json_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    schema_tmp.write_text(
        json.dumps(schema_payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Feature audit",
        "",
        "All tables below use label-separated, inference-time-only feature files. "
        "The hashed machine schema and implementation field are the authoritative "
        "definitions; units and sources are stated explicitly here.",
        "",
    ]
    for audit in enriched:
        lines.extend(
            [
                f"## {audit.get('route', 'unknown')} / {audit.get('pool', 'unknown')}",
                "",
                f"- Candidates: {audit['candidate_count']}",
                f"- Queries: {audit['query_count']}",
                f"- Features: {audit['feature_schema']['feature_count']}",
                f"- Positive rate: {audit['positive_candidate_rate']:.6f}",
                f"- Leakage scanner: {'PASS' if audit['forbidden_column_scanner_passed'] else 'FAIL'}",
                "",
                "| feature | group | unit | source | missing | ROC-AUC | PR-AUC | univariate J@1 | constant |",
                "|---|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in audit["features"]:
            def render(value: Any) -> str:
                if value is None:
                    return "N/A"
                if isinstance(value, float):
                    return f"{value:.6f}"
                return str(value).replace("|", "\\|")

            lines.append(
                "| {feature} | {group} | {unit} | {source} | {missing_rate} | "
                "{roc_auc} | {pr_auc} | {univariate_j_at_1} | {constant} |".format(
                    **{key: render(value) for key, value in row.items()}
                )
            )
        lines.append("")
    md_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(json_tmp, json_path)
    os.replace(md_tmp, markdown_path)
    os.replace(schema_tmp, schema_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    features = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.labels)
    audit = audit_feature_table(features, labels)
    args.schema.parent.mkdir(parents=True, exist_ok=True)
    args.schema.write_text(
        json.dumps(audit["feature_schema"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_feature_audit(audit, args.audit_json, args.audit_markdown)
    print(json.dumps(audit["feature_schema"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
