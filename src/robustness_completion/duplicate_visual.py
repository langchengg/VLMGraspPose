"""Deterministic, outcome-blind visual audit of the locked duplicate map."""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .common import atomic_json, atomic_text, require_run_dir, sha256_file, verify_preregistration
from .duplicate_map import _aligned_views, _load_arrays, _verify_locked_outputs


def _stable(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return (
        frame.assign(
            _order=frame["pair_id"].astype(str).map(
                lambda value: hashlib.sha256(value.encode()).hexdigest()
            )
        )
        .sort_values(["_order", "pair_id"])
        .head(limit)
        .drop(columns="_order")
    )


def _boundary_rejections(frame: pd.DataFrame) -> pd.DataFrame:
    rejected = frame[
        ~(frame["exact_match"] | frame["strict_match"] | frame["moderate_match"])
    ].copy()
    if rejected.empty:
        return rejected
    # Outcome-blind closeness to the moderate conjunction. Zero means a value
    # is already within its individual bound; positive values are violations.
    violation = (
        np.maximum(rejected["phash_hamming"] / 8.0 - 1.0, 0.0)
        + np.maximum(rejected["dhash_hamming"] / 10.0 - 1.0, 0.0)
        + np.maximum(rejected["translation_magnitude_px"] / 8.0 - 1.0, 0.0)
        + np.maximum(0.980 - rejected["luminance_ssim"], 0.0) / 0.020
        + np.maximum(rejected["normalised_rgb_mae"] / 0.030 - 1.0, 0.0)
    )
    # Non-estimable SSIM is possible only for a sub-7px aligned overlap.  Put
    # those pairs last rather than letting NaN make boundary selection depend
    # on pandas' implementation details.
    return rejected.assign(_violation=violation.fillna(np.inf)).sort_values(
        ["_violation", "pair_id"]
    ).head(50).drop(columns="_violation")


def _render_pair(
    row: pd.Series, category: str, lookup: pd.DataFrame, output: Path
) -> None:
    train = lookup.loc[str(row["train_observation_id"])]
    test = lookup.loc[str(row["test_observation_id"])]
    train_arrays = _load_arrays(str(train["canonical_array_path"]))
    test_arrays = _load_arrays(str(test["canonical_array_path"]))
    views, _, _, _ = _aligned_views(train_arrays, test_arrays)
    train_rgb, test_rgb, _, _, train_depth, test_depth, train_valid, test_valid = views
    rgb_difference = np.abs(train_rgb.astype(np.int16) - test_rgb.astype(np.int16)).astype(np.uint8)
    joint = train_valid & test_valid & np.isfinite(train_depth) & np.isfinite(test_depth)
    depth_difference = np.full_like(train_depth, np.nan, dtype=np.float32)
    depth_difference[joint] = np.abs(train_depth[joint] - test_depth[joint])
    panels = [
        (train_rgb, "Train RGB", None),
        (test_rgb, "Test RGB", None),
        (rgb_difference, "Absolute RGB difference", None),
        (train_depth, "Train depth (m)", "viridis"),
        (test_depth, "Test depth (m)", "viridis"),
        (depth_difference, "Absolute depth difference (m)", "magma"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for axis, (image, title, colourmap) in zip(axes.ravel(), panels, strict=True):
        axis.imshow(image, cmap=colourmap)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    metric_text = (
        f"{category} | train={row['train_sequence_id']}/{row['train_frame_id']} | "
        f"test={row['test_sequence_id']}/{row['test_frame_id']}\n"
        f"pHash={int(row['phash_hamming'])}; dHash={int(row['dhash_hamming'])}; "
        f"shift={float(row['translation_magnitude_px']):.3f}px; "
        f"SSIM={float(row['luminance_ssim']):.6f}; RGB MAE={float(row['normalised_rgb_mae']):.6f}; "
        f"depth overlap={float(row['valid_depth_overlap']):.6f}; "
        f"median/p95 relative depth={float(row['median_relative_depth_error']):.6g}/"
        f"{float(row['p95_relative_depth_error']):.6g}; RGB-only={bool(row['rgb_only_match'])}"
    )
    fig.suptitle(metric_text, fontsize=8.2, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def audit_duplicate_map(repo: Path, run_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    repo = repo.resolve()
    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    root = run_dir / "duplicate_audit"
    completion_path = root / "visual_audit_completion.json"
    if completion_path.exists() and resume:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        for relative, expected in completion["output_sha256"].items():
            if sha256_file(root / relative) != expected:
                raise RuntimeError(f"visual audit hash mismatch: {relative}")
        return {"status": "COMPLETE", "resumed": True, **completion["counts"]}
    lock = json.loads((root / "DUPLICATE_MAP_LOCK.json").read_text(encoding="utf-8"))
    _verify_locked_outputs(root, lock)
    pairs = pd.read_parquet(root / "all_candidate_pairs.parquet")
    observations = pd.read_parquet(root / "canonical_observations.parquet")
    fingerprints = pd.read_parquet(root / "fingerprints.parquet")
    lookup = observations.merge(fingerprints, on="observation_id", validate="one_to_one").set_index("observation_id")
    selections = {
        "strict": _stable(pairs[pairs["exact_match"] | pairs["strict_match"]], 50),
        "moderate_only": _stable(
            pairs[pairs["moderate_match"] & ~pairs["strict_match"] & ~pairs["exact_match"]], 50
        ),
        "nearest_rejected_boundary": _boundary_rejections(pairs),
        "exact": _stable(pairs[pairs["exact_match"]], 20),
    }
    visual_dir = root / "visual_audit"
    visual_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for category, selected in selections.items():
        for number, (_, row) in enumerate(selected.iterrows()):
            filename = f"{category}_{number:03d}_{str(row['pair_id'])[:12]}.png"
            path = visual_dir / filename
            if not path.exists():
                _render_pair(row, category, lookup, path)
            manifest_rows.append(
                {
                    "category": category,
                    "pair_id": str(row["pair_id"]),
                    "panel_path": str(path.relative_to(root)),
                    "panel_sha256": sha256_file(path),
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    from .common import atomic_frame

    atomic_frame(visual_dir / "manifest.csv", manifest)
    html_rows = []
    for record in manifest_rows:
        pair = pairs[pairs["pair_id"] == record["pair_id"]].iloc[0]
        html_rows.append(
            "<section><h2>"
            + html.escape(f"{record['category']}: {record['pair_id']}")
            + "</h2><img loading='lazy' src='"
            + html.escape(record["panel_path"])
            + "' alt='duplicate audit panel'><pre>"
            + html.escape(json.dumps(pair.to_dict(), indent=2, default=str))
            + "</pre></section>"
        )
    html_path = root / "duplicate_audit.html"
    atomic_text(
        html_path,
        "<!doctype html><meta charset='utf-8'><title>RGB-D duplicate audit</title>"
        "<style>body{font:14px sans-serif;max-width:1200px;margin:auto}img{width:100%}"
        "section{break-inside:avoid;border-bottom:1px solid #bbb;padding:1rem}</style>"
        "<h1>AI-assisted visual audit, not independent human annotation</h1>"
        "<p>Panels are selected deterministically without model outcomes. They cannot alter thresholds or membership.</p>"
        + "".join(html_rows),
    )
    pdf_path = root / "duplicate_audit.pdf"
    descriptor, temporary = tempfile.mkstemp(prefix=".duplicate_audit.", suffix=".pdf", dir=root)
    os.close(descriptor)
    try:
        with PdfPages(temporary) as document:
            for record in manifest_rows:
                image = plt.imread(root / record["panel_path"])
                fig, axis = plt.subplots(figsize=(11.69, 8.27))
                axis.imshow(image)
                axis.axis("off")
                document.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        os.replace(temporary, pdf_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    methods_path = root / "AI_ASSISTED_AUDIT.md"
    counts = {category: int(len(frame)) for category, frame in selections.items()}
    atomic_text(
        methods_path,
        "# AI-assisted visual audit\n\n"
        "This is an **AI-assisted visual audit, not independent human annotation**. "
        "Pair selection used the preregistered, outcome-blind deterministic rules. "
        "No panel was accepted/rejected manually and the audit did not modify any threshold.\n\n"
        + "\n".join(f"- {name}: {count} panels" for name, count in counts.items())
        + "\n",
    )
    names = [
        "visual_audit/manifest.csv",
        "duplicate_audit.html",
        "duplicate_audit.pdf",
        "AI_ASSISTED_AUDIT.md",
    ]
    completion = {
        "status": "COMPLETE",
        "duplicate_map_lock_sha256": sha256_file(root / "DUPLICATE_MAP_LOCK.json"),
        "counts": counts,
        "output_sha256": {name: sha256_file(root / name) for name in names},
    }
    atomic_json(completion_path, completion)
    return {"status": "COMPLETE", "resumed": False, **counts}
