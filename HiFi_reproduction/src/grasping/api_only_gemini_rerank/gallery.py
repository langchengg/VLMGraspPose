"""Materialize evaluation-only failure galleries from frozen candidate decisions."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .constants import EXACT_MODEL_IDS
from .io import atomic_json, utc_now
from .renderer import PALETTE, _draw_candidate, _font


GALLERY_CATEGORIES = (
    "recovered", "harmful", "neutral_correct", "neutral_wrong",
    "missed_recoverable", "unrecoverable", "disagreement", "unstable",
    "api_failure",
)


def outcome_categories(row: Mapping[str, Any]) -> set[str]:
    """Return every preregistered gallery category applicable to one decision."""
    baseline = bool(row["baseline_correct"])
    final = bool(row["final_correct"])
    changed = str(row["baseline_candidate_id"]) != str(row["final_candidate_id"])
    recoverable = bool(row.get("top5_any_correct", row.get("recoverable_error", False)))
    categories: set[str] = set()
    if not baseline and final:
        categories.add("recovered")
    if baseline and not final:
        categories.add("harmful")
    if baseline and final and changed:
        categories.add("neutral_correct")
    if not baseline and not final:
        categories.add("neutral_wrong")
    if not baseline and recoverable and not final:
        categories.add("missed_recoverable")
    if not recoverable:
        categories.add("unrecoverable")
    return categories


def _model_for_method(method: str) -> str | None:
    for model in EXACT_MODEL_IDS:
        if model in method:
            return model
    return None


def _development_policy_decisions(run: Path, baseline: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct development API outcomes without treating them as validation."""
    path = run / "stage_results" / "policy_selection.parquet"
    if not path.is_file():
        return pd.DataFrame()
    results = pd.read_parquet(path)
    source = baseline.loc[baseline["split"].eq("validation")]
    indexed = {
        (str(row["backend"]), str(row["sample_id"])): row
        for row in source.to_dict(orient="records")
    }
    rows = []
    for response in results.loc[results["replicate_id"].eq(1)].to_dict(orient="records"):
        item = indexed.get((str(response["backend"]), str(response["sample_id"])))
        if item is None:
            continue
        original = str(item["top1_candidate_id"])
        final = str(response["selected_candidate_id"]) if response["status"] == "SUCCEEDED" else original
        correctness = json.loads(str(item["candidate_correctness_json"]))
        rows.append({
            "backend": response["backend"],
            "method": f"{response['backend']}_{response['model_id']}_{response['protocol']}_development",
            "sample_id": response["sample_id"], "baseline_candidate_id": original,
            "final_candidate_id": final, "baseline_correct": bool(item["top1_correct"]),
            "final_correct": bool(correctness.get(final, False)),
            "top5_any_correct": bool(item["top5_any_correct"]),
            "evaluation_split": "validation", "provider_stage": "policy_selection",
            "provider_protocol": response["protocol"], "provider_evidence_variant": response["evidence_variant"],
            "evaluation_scope": "development_policy_selection",
        })
    return pd.DataFrame(rows)


def _read_raw_reason(run: Path, response: Mapping[str, Any] | None) -> tuple[list[str], str]:
    if response is None:
        return [], "No successful provider response; original Top-1 fallback."
    model = str(response.get("model_id", ""))
    model_dir = "er2" if model == EXACT_MODEL_IDS[0] else "flash"
    request_hash = str(response.get("request_hash", ""))
    path = run / "raw_api" / model_dir / f"{request_hash}.json"
    if not path.is_file():
        return [], str(response.get("fallback_reason") or "Raw structured response unavailable.")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return [], "Raw structured response unreadable."
    return list(map(str, value.get("reason_codes", []))), str(value.get("brief_rationale", ""))


def _fit_rgb(path: str, size: tuple[int, int] = (640, 480)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _fit_mask(path: str, size: tuple[int, int] = (640, 480)) -> Image.Image:
    raw = np.asarray(Image.open(path).convert("L"))
    if raw.shape[::-1] != size:
        raw = np.asarray(Image.fromarray(raw).resize(size, Image.Resampling.NEAREST))
    binary = raw > 0
    visual = np.zeros((*binary.shape, 3), dtype=np.uint8)
    visual[binary] = (70, 210, 210)
    return Image.fromarray(visual)


def _render_case(
    destination: Path,
    *,
    decision: Mapping[str, Any],
    candidates: pd.DataFrame,
    evidence: Mapping[str, Any],
    response: Mapping[str, Any] | None,
    category: str,
    run: Path,
) -> None:
    rgb = _fit_rgb(str(evidence["source_rgb_path"]))
    mask = _fit_mask(str(evidence["predicted_mask_path"]))
    original = str(decision["baseline_candidate_id"])
    selected = str(decision["final_candidate_id"])
    legend = []
    for item in candidates.sort_values("original_rank").to_dict(orient="records"):
        rank = int(item["original_rank"])
        colour = PALETTE[(rank - 1) % len(PALETTE)]
        suffix = ""
        if str(item["candidate_id"]) == original:
            suffix += " ORIG"
        if str(item["candidate_id"]) == selected:
            suffix += " API"
        label = f"R{rank}{suffix}"
        _draw_candidate(rgb, item, label, colour)
        _draw_candidate(mask, item, label, colour)
        legend.append(
            f"R{rank}: {str(item['candidate_id'])[:12]} score={float(item['original_score']):.5g}{suffix}"
        )
    reason_codes, rationale = _read_raw_reason(run, response)
    model = str(response.get("model_id")) if response else (_model_for_method(str(decision["method"])) or "provider-fallback")
    protocol = str(response.get("protocol")) if response else "fallback"
    provider_decision = (
        str(response.get("decision"))
        if response and response.get("status") == "SUCCEEDED"
        else "FALLBACK_KEEP_ORIGINAL"
    )
    confidence = response.get("switch_confidence") if response else None
    reliability = response.get("evidence_reliability") if response else None
    canvas = Image.new("RGB", (1280, 900), "white")
    canvas.paste(rgb, (0, 70)); canvas.paste(mask, (640, 70))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1280, 70), fill=(242, 242, 242))
    draw.text((12, 8), f"{category} | {decision['backend']} | {decision['sample_id']}", fill=(20, 20, 20), font=_font(20))
    language = str(evidence.get("language", ""))
    draw.text((12, 39), "Language: " + textwrap.shorten(language, width=150, placeholder="..."), fill=(20, 20, 20), font=_font(15))
    draw.text((12, 552), "RGB + all frozen candidates", fill=(20, 20, 20), font=_font(16))
    draw.text((652, 552), "Predicted mask + identical frozen candidates", fill=(20, 20, 20), font=_font(16))
    details = [
        f"Evaluation scope: {decision.get('evaluation_scope', 'locked_validation_or_formal')}",
        f"Model: {model}", f"Protocol: {protocol}",
        f"Provider decision: {provider_decision}; reliability={reliability}; confidence={confidence}",
        f"Reason codes: {reason_codes or ['UNAVAILABLE']}",
        f"Original Top-1: {original}; API/final Top-1: {selected}",
        f"Final correct: {bool(decision['final_correct'])}; original correct: {bool(decision['baseline_correct'])}",
        f"Candidate-pool recoverable: {bool(decision.get('top5_any_correct', decision.get('recoverable_error', False)))}",
        *legend,
        "Rationale: " + rationale,
    ]
    y = 585
    for detail in details:
        for line in textwrap.wrap(str(detail), width=155) or [""]:
            draw.text((12, y), line, fill=(20, 20, 20), font=_font(14))
            y += 18
            if y > 884:
                break
        if y > 884:
            break
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def materialize_failure_gallery(run_dir: str | Path, *, limit: int = 20) -> dict[str, Any]:
    """Build deterministic evaluation galleries; unavailable categories stay explicitly empty."""
    run = Path(run_dir)
    decision_parts: list[pd.DataFrame] = []
    formal_path = run / "FORMAL_PER_SAMPLE_DECISIONS.parquet"
    formal_backends: set[str] = set()
    if formal_path.is_file():
        formal = pd.read_parquet(formal_path).copy()
        formal["evaluation_split"] = "test"
        formal["provider_stage"] = "formal"
        formal_backends = set(formal["backend"].astype(str))
        decision_parts.append(formal)
    validation_path = run / "validation_per_sample_decisions.parquet"
    if validation_path.is_file():
        validation = pd.read_parquet(validation_path).copy()
        validation = validation.loc[~validation["backend"].astype(str).isin(formal_backends)]
        validation["evaluation_split"] = "validation"
        validation["provider_stage"] = "untouched_validation"
        decision_parts.append(validation)
    decisions = pd.concat(decision_parts, ignore_index=True, sort=False) if decision_parts else pd.DataFrame()
    baseline = pd.read_parquet(run / "baseline_per_sample.parquet")
    if len(decisions):
        decisions = decisions.merge(
            baseline[["backend", "split", "sample_id", "top5_any_correct", "unrecoverable_error"]],
            left_on=["backend", "evaluation_split", "sample_id"],
            right_on=["backend", "split", "sample_id"], how="left", validate="many_to_one",
        )
    stage_results: dict[str, pd.DataFrame] = {}
    for stage in ("diagnostic", "policy_selection", "untouched_validation", "formal"):
        path = run / "stage_results" / f"{stage}.parquet"
        stage_results[stage] = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
    candidates = {
        backend: pd.read_parquet(run / f"CANDIDATE_MANIFEST_{backend}.parquet")
        for backend in ("G1", "C1")
    }
    evidence = {
        backend: pd.read_parquet(run / f"EVIDENCE_FEATURES_{backend}.parquet")
        for backend in ("G1", "C1")
    }
    selected: dict[str, list[dict[str, Any]]] = {name: [] for name in GALLERY_CATEGORIES}
    development = _development_policy_decisions(run, baseline)
    gallery_decisions = pd.concat([decisions, development], ignore_index=True, sort=False)
    if len(gallery_decisions):
        gallery_decisions["evaluation_scope"] = gallery_decisions.get(
            "evaluation_scope", pd.Series(index=gallery_decisions.index, dtype=object)
        ).fillna("locked_validation_or_formal")
        for row in gallery_decisions.sort_values(["evaluation_scope", "backend", "method", "sample_id"]).to_dict(orient="records"):
            for category in outcome_categories(row):
                if len(selected[category]) < limit:
                    selected[category].append(row)
    stability_path = run / "diagnostic_stability.csv"
    if stability_path.is_file():
        stability = pd.read_csv(stability_path)
        for row in stability.loc[~stability["top1_stable"].astype(bool)].sort_values(["backend", "sample_id"]).head(limit).to_dict(orient="records"):
            base = baseline.loc[(baseline["backend"] == row["backend"]) & (baseline["split"] == "validation") & (baseline["sample_id"].astype(str) == str(row["sample_id"]))]
            if len(base):
                item = base.iloc[0]
                ids = json.loads(str(row["selected_ids"]))
                correctness = json.loads(str(item["candidate_correctness_json"]))
                selected["unstable"].append({
                    "backend": row["backend"], "method": f"{row['backend']}_{row['model_id']}_diagnostic",
                    "sample_id": row["sample_id"], "baseline_candidate_id": item["top1_candidate_id"],
                    "final_candidate_id": ids[0], "baseline_correct": bool(item["top1_correct"]),
                    "final_correct": bool(correctness.get(str(ids[0]), False)),
                    "top5_any_correct": bool(item["top5_any_correct"]), "evaluation_split": "validation",
                    "provider_stage": "diagnostic",
                })
    for stage, results in stage_results.items():
        if results.empty or "status" not in results:
            continue
        for response in results.loc[~results["status"].eq("SUCCEEDED")].sort_values(
            ["backend", "sample_id", "model_id", "replicate_id"]
        ).to_dict(orient="records"):
            if len(selected["api_failure"]) >= limit:
                break
            split = "test" if stage == "formal" else "validation"
            base = baseline.loc[
                baseline["backend"].eq(response["backend"]) & baseline["split"].eq(split)
                & baseline["sample_id"].astype(str).eq(str(response["sample_id"]))
            ]
            if base.empty:
                continue
            item = base.iloc[0]
            original = str(item["top1_candidate_id"])
            selected["api_failure"].append({
                "backend": response["backend"],
                "method": f"{response['backend']}_{response['model_id']}_{response['protocol']}_{stage}_failure",
                "sample_id": response["sample_id"], "baseline_candidate_id": original,
                "final_candidate_id": original, "baseline_correct": bool(item["top1_correct"]),
                "final_correct": bool(item["top1_correct"]),
                "top5_any_correct": bool(item["top5_any_correct"]),
                "evaluation_split": split, "provider_stage": stage,
                "provider_protocol": response["protocol"], "provider_evidence_variant": response["evidence_variant"],
                "evaluation_scope": f"{stage}_technical_fallback",
            })
    # The exact two-model disagreement categories are deliberately not imputed
    # when a required model is unavailable.
    manifest: dict[str, Any] = {"generated_at_utc": utc_now(), "limit_per_category": limit, "categories": {}}
    for category, rows in selected.items():
        directory = run / "failure_gallery" / category
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.png"):
            stale.unlink()
        artifacts = []
        for index, row in enumerate(rows[:limit]):
            backend, sample_id = str(row["backend"]), str(row["sample_id"])
            split = str(row["evaluation_split"])
            candidate_group = candidates[backend].loc[
                candidates[backend]["split"].eq(split) & candidates[backend]["sample_id"].astype(str).eq(sample_id)
            ]
            evidence_group = evidence[backend].loc[
                evidence[backend]["split"].eq(split) & evidence[backend]["sample_id"].astype(str).eq(sample_id)
            ]
            if candidate_group.empty or evidence_group.empty:
                continue
            response = None
            stage = str(row.get("provider_stage", ""))
            results = stage_results.get(stage, pd.DataFrame())
            model = _model_for_method(str(row["method"]))
            if len(results) and model is not None:
                matched = results.loc[
                    results["backend"].eq(backend) & results["sample_id"].astype(str).eq(sample_id)
                    & results["model_id"].eq(model) & results["replicate_id"].eq(1)
                ]
                expected_protocol = row.get("provider_protocol")
                if expected_protocol is not None and not pd.isna(expected_protocol):
                    matched = matched.loc[matched["protocol"].eq(str(expected_protocol))]
                expected_variant = row.get("provider_evidence_variant")
                if expected_variant is not None and not pd.isna(expected_variant):
                    matched = matched.loc[matched["evidence_variant"].eq(str(expected_variant))]
                if stage == "diagnostic" and category == "unstable":
                    matched = matched.loc[matched["evidence_variant"].eq("E3_RGBD_GEOMETRY_SCORE_AWARE")]
                if len(matched):
                    response = matched.iloc[0].to_dict()
            safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["method"]))[:80]
            path = directory / f"{index + 1:02d}_{backend}_{sample_id}_{safe_method}.png"
            _render_case(path, decision=row, candidates=candidate_group,
                         evidence=evidence_group.iloc[0].to_dict(), response=response,
                         category=category, run=run)
            artifacts.append(str(path.relative_to(run)))
        manifest["categories"][category] = {
            "eligible_examples": len(rows), "materialized": len(artifacts), "artifacts": artifacts,
            "status": "COMPLETE" if artifacts else "NO_APPLICABLE_CASES",
        }
    atomic_json(run / "failure_gallery" / "MANIFEST.json", manifest)
    return manifest
