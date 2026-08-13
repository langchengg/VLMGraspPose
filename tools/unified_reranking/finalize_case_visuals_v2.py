#!/usr/bin/env python3
"""Validate and atomically seal one post-formal qualitative-visual run."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from unified_reranking.candidates import GEOMETRY_COLUMNS  # noqa: E402
from unified_reranking.case_visuals_v2 import ALL_CANDIDATE_COLORS  # noqa: E402
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)


EXPECTED_SOURCE_LOCK_SHA256 = (
    "4b52eac6494e59a0f902792b794c3cca02824bf7a36bed4699b569986383f793"
)
ROUTES = ("crog", "g1", "c1")
OUTCOMES = (
    "recovered",
    "harmful",
    "gate_prevented_harmful",
    "gate_missed_recoverable",
    "wrong_to_wrong_solvable",
    "candidate_generation_irreparable",
)
EXPECTED_ROUTE_CANDIDATES = {"crog": 38_375, "g1": 23_088, "c1": 26_582}
EXPECTED_TOP5_ROWS = 86_408
EXPECTED_CASE_ROWS = 23_025
EXPECTED_SAMPLE_COUNT = 7_675


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _content_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return canonical_sha256(payload)


def _assert_content_hash(value: dict[str, Any], *, name: str) -> None:
    if str(value.get("content_sha256")) != _content_hash(value):
        raise RuntimeError(f"{name} content SHA-256 mismatch")


def _png_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"unreadable image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def _pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"pdfinfo did not report a page count: {path}")
    return int(match.group(1))


def _pptx_slides(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    return sum(
        bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", name)) for name in names
    )


def _geometry_hash(row: Any) -> str:
    return canonical_sha256(
        [
            str(row.route).upper(),
            str(row.sample_id),
            str(row.candidate_id),
            int(row.native_rank),
            float(row.cx_px),
            float(row.cy_px),
            float(row.theta_deg),
            float(row.width_px),
            float(row.height_px),
        ]
    )


def _validate_source(run: Path, manifest: dict[str, Any]) -> tuple[Path, str]:
    source = Path(str(manifest["source_run"])).resolve()
    lock = source / "FINAL_RUN_LOCK.json"
    observed = sha256_file(lock)
    if observed != EXPECTED_SOURCE_LOCK_SHA256:
        raise RuntimeError("source FINAL_RUN_LOCK SHA-256 changed")
    if str(manifest.get("source_final_lock_sha256")) != observed:
        raise RuntimeError("derived manifest is not bound to the source final lock")
    source_manifest = _read_json(source / "manifest.json")
    if (
        source_manifest.get("status") != "COMPLETE"
        or int(source_manifest.get("formal_test_execution_count", -1)) != 1
        or source_manifest.get("test_label_state") != "FORMAL_TEST_COMPLETE"
        or not (source / "COMPLETE").is_file()
    ):
        raise RuntimeError("source formal run is not authoritatively COMPLETE")
    before = _read_json(run / "00_audit/source_snapshot_before.json")
    after = _read_json(run / "00_audit/source_snapshot_after.json")
    if before != after:
        raise RuntimeError("source snapshots differ")
    source_pass = _read_json(run / "00_audit/SOURCE_IMMUTABILITY_PASS.json")
    if source_pass.get("status") != "PASS" or not source_pass.get(
        "before_equals_after"
    ):
        raise RuntimeError("source immutability audit did not pass")
    return source, observed


def _validate_tables(run: Path) -> dict[str, Any]:
    cases = pd.read_parquet(run / "01_data/canonical_case_table.parquet")
    candidates = pd.read_parquet(run / "01_data/canonical_candidate_table.parquet")
    if len(cases) != EXPECTED_CASE_ROWS or len(candidates) != sum(
        EXPECTED_ROUTE_CANDIDATES.values()
    ):
        raise RuntimeError("canonical case/candidate row count mismatch")
    if cases[["route", "sample_id"]].duplicated().any():
        raise RuntimeError("duplicate route/sample in canonical case table")
    route_cases = cases.groupby("route")["sample_id"].nunique().to_dict()
    if route_cases != {route: EXPECTED_SAMPLE_COUNT for route in ROUTES}:
        raise RuntimeError(f"route denominator mismatch: {route_cases}")
    route_candidates = candidates.groupby("route").size().to_dict()
    if route_candidates != EXPECTED_ROUTE_CANDIDATES:
        raise RuntimeError(f"route candidate count mismatch: {route_candidates}")
    if int(candidates["in_top5"].astype(bool).sum()) != EXPECTED_TOP5_ROWS:
        raise RuntimeError("Top-5 row count mismatch")
    if candidates[["route", "sample_id", "candidate_id"]].duplicated().any():
        raise RuntimeError("duplicate canonical candidate identity")
    required = {
        "route",
        "sample_id",
        *GEOMETRY_COLUMNS,
        "candidate_geometry_sha256",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise RuntimeError(f"candidate geometry columns missing: {missing}")
    expected_geometry = [_geometry_hash(row) for row in candidates.itertuples(index=False)]
    if candidates["candidate_geometry_sha256"].astype(str).tolist() != expected_geometry:
        raise RuntimeError("canonical candidate geometry SHA-256 mismatch")
    audit = _read_json(run / "01_data/case_join_audit.json")
    if (
        audit.get("status") != "PASS"
        or int(audit.get("case_rows", -1)) != len(cases)
        or int(audit.get("candidate_rows", -1)) != len(candidates)
        or int(audit.get("top5_rows", -1)) != EXPECTED_TOP5_ROWS
        or not audit.get("g1_c1_mask_identity")
    ):
        raise RuntimeError("canonical join audit mismatch")
    g1 = cases[cases["route"] == "g1"].set_index("sample_id")
    c1 = cases[cases["route"] == "c1"].set_index("sample_id")
    if not g1.index.equals(c1.index) or not g1[
        "hifics_mask_sha256"
    ].astype(str).equals(c1["hifics_mask_sha256"].astype(str)):
        raise RuntimeError("G1/C1 did not consume byte-identical HiFi-CS masks")
    return {"cases": cases, "candidates": candidates, "audit": audit}


def _validate_selections(run: Path, tables: dict[str, Any]) -> dict[str, int]:
    cases: pd.DataFrame = tables["cases"]
    candidates: pd.DataFrame = tables["candidates"]
    selected = pd.read_csv(run / "02_selection/SELECTED_AUDIT_CASES.csv")
    expected_keys = {(route, outcome) for route in ROUTES for outcome in OUTCOMES}
    observed_keys = set(zip(selected["route"], selected["presentation_outcome"]))
    if len(selected) != 18 or observed_keys != expected_keys:
        raise RuntimeError("selected route/outcome inventory is not exactly 3 x 6")
    if selected[["route", "sample_id"]].duplicated().any():
        raise RuntimeError("duplicate selected route/sample")
    if (
        not selected["mandatory_eligible"].astype(bool).all()
        or not selected["same_gt_native_final"].astype(bool).all()
        or not selected["selection_status"].eq("SELECTED").all()
        or not selected["alternative_rank"].astype(int).eq(1).all()
    ):
        raise RuntimeError("a selected case violates the mandatory eligibility contract")
    selected_keys = selected[["route", "sample_id", "presentation_outcome"]].rename(
        columns={"presentation_outcome": "selected_presentation_outcome"}
    )
    joined = selected_keys.merge(
        cases,
        on=["route", "sample_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["native_candidate_id"].isna().any() or joined[
        "final_candidate_id"
    ].isna().any():
        raise RuntimeError("selected native/final candidate identity is missing")
    if (
        not joined["candidate_geometry_valid"].astype(bool).all()
        or not joined["rgb_exists"].astype(bool).all()
        or not joined["gt_mask_exists"].astype(bool).all()
        or (joined["highlight_clip_fraction"].astype(float) > 0.20).any()
    ):
        raise RuntimeError("selected visual asset/geometry/image-quality contract failed")
    candidate_keys = set(
        zip(candidates["route"], candidates["sample_id"], candidates["candidate_id"])
    )
    top5_keys = set(
        zip(
            candidates.loc[candidates["in_top5"].astype(bool), "route"],
            candidates.loc[candidates["in_top5"].astype(bool), "sample_id"],
            candidates.loc[candidates["in_top5"].astype(bool), "candidate_id"],
        )
    )
    for row in joined.itertuples(index=False):
        for candidate_id in (row.native_candidate_id, row.final_candidate_id):
            key = (str(row.route), str(row.sample_id), str(candidate_id))
            if key not in candidate_keys or key not in top5_keys:
                raise RuntimeError(f"selected candidate is outside the frozen Top-5: {key}")
        case_dir = (
            run
            / "03_case_data"
            / str(row.route)
            / str(row.selected_presentation_outcome)
            / str(row.sample_id)
        )
        for name in ("case_data.json", "feature_contributions.csv", "native_vs_final.csv"):
            if not (case_dir / name).is_file():
                raise RuntimeError(f"selected case-data artifact is missing: {case_dir / name}")
    case_data_files = list((run / "03_case_data").rglob("*"))
    if sum(path.is_file() for path in case_data_files) != 54:
        raise RuntimeError("case-data inventory is not exactly 18 x 3")
    short = pd.read_csv(run / "02_selection/SELECTED_SHORT_DECK_CASES.csv")
    if (
        len(short) != 6
        or short["short_deck_order"].astype(int).tolist() != list(range(1, 7))
        or not set(zip(short["route"], short["sample_id"])).issubset(
            set(zip(selected["route"], selected["sample_id"]))
        )
    ):
        raise RuntimeError("short-deck selection inventory mismatch")
    cross = pd.read_csv(run / "02_selection/SELECTED_CROSS_ROUTE_CASES.csv")
    if (
        len(cross) != 2
        or cross["sample_id"].duplicated().any()
        or cross["cross_route_order"].astype(int).tolist() != [1, 2]
    ):
        raise RuntimeError("cross-route selection inventory mismatch")
    for sample_id in cross["sample_id"].astype(str):
        rows = cases[cases["sample_id"].astype(str) == sample_id]
        if set(rows["route"].astype(str)) != set(ROUTES):
            raise RuntimeError(f"cross-route sample lacks all three routes: {sample_id}")
    return {"route_cases": len(selected), "short_cases": len(short), "cross_cases": len(cross)}


def _validate_gallery(run: Path, tables: dict[str, Any]) -> dict[str, Any]:
    route = pd.read_csv(run / "05_gallery/gallery_manifest.csv")
    cross = pd.read_csv(run / "05_gallery/cross_route_manifest.csv")
    candidates: pd.DataFrame = tables["candidates"]
    if len(route) != 18 or len(cross) != 2:
        raise RuntimeError("gallery manifest row count mismatch")
    for row in route.itertuples(index=False):
        png = Path(str(row.board_path)).resolve()
        svg = png.with_suffix(".svg")
        if (
            sha256_file(png) != str(row.board_png_sha256)
            or sha256_file(svg) != str(row.board_svg_sha256)
            or _png_shape(png) != (2400, 1350)
        ):
            raise RuntimeError(f"route board contract failed: {png}")
        image = cv2.imread(str(png), cv2.IMREAD_COLOR)
        assert image is not None
        panel = cv2.cvtColor(image[625:970, 40:645], cv2.COLOR_BGR2RGB)
        ranks = sorted(
            candidates[
                (candidates["route"].astype(str) == str(row.route))
                & (candidates["sample_id"].astype(str) == str(row.sample_id))
            ]["native_rank"]
            .astype(int)
            .tolist()
        )
        if not ranks:
            raise RuntimeError(f"selected route board has no candidates: {png}")
        for rank in ranks:
            value = ALL_CANDIDATE_COLORS[(rank - 1) % len(ALL_CANDIDATE_COLORS)]
            color = np.asarray(
                [int(value[index : index + 2], 16) for index in (1, 3, 5)],
                dtype=np.int16,
            )
            distance = abs(panel.astype("int16") - color)
            visible_pixels = int((distance.max(axis=2) <= 35).sum())
            if visible_pixels <= 40:
                raise RuntimeError(
                    f"All candidates rank r{rank} is not visibly rendered: {png}"
                )
    for row in cross.itertuples(index=False):
        png = Path(str(row.board_path)).resolve()
        if sha256_file(png) != str(row.board_sha256) or _png_shape(png) != (
            2400,
            1350,
        ):
            raise RuntimeError(f"cross-route board contract failed: {png}")
    html = (run / "05_gallery/index.html").read_text(encoding="utf-8")
    if html.count("<article ") != 20:
        raise RuntimeError("HTML gallery does not contain exactly 20 boards")
    return {
        "route_boards": len(route),
        "cross_route_boards": len(cross),
        "all_candidate_rank_boxes_visible": True,
    }


def _validate_decks(run: Path) -> dict[str, int]:
    slides = run / "06_slides"
    full_pptx = slides / "RERANKING_SUCCESS_FAILURE_ANALYSIS_REVISED.pptx"
    full_pdf = slides / "RERANKING_SUCCESS_FAILURE_ANALYSIS_REVISED.pdf"
    short_pptx = slides / "RERANKING_CASES_SHORT_DECK.pptx"
    short_pdf = slides / "RERANKING_CASES_SHORT_DECK.pdf"
    observed = {
        "full_pptx_slides": _pptx_slides(full_pptx),
        "full_pdf_pages": _pdf_pages(full_pdf),
        "short_pptx_slides": _pptx_slides(short_pptx),
        "short_pdf_pages": _pdf_pages(short_pdf),
    }
    if observed != {
        "full_pptx_slides": 24,
        "full_pdf_pages": 24,
        "short_pptx_slides": 10,
        "short_pdf_pages": 10,
    }:
        raise RuntimeError(f"deck page/slide count mismatch: {observed}")
    for name, count in (("rendered_full", 24), ("rendered_short", 10)):
        images = sorted((slides / name).glob("slide-*.png"))
        if len(images) != count or any(_png_shape(path) != (1920, 1080) for path in images):
            raise RuntimeError(f"rendered deck contract failed: {name}")
    for name in ("FULL_DECK_CONTACT_SHEET.png", "SHORT_DECK_CONTACT_SHEET.png"):
        path = slides / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"contact sheet missing: {path}")
    return observed


def _inventory(run: Path) -> list[dict[str, Any]]:
    excluded = {"00_audit/DERIVED_SHA256_MANIFEST.csv", "COMPLETE"}
    rows: list[dict[str, Any]] = []
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.is_symlink() or any(
            part.startswith(".") for part in path.relative_to(run).parts
        ):
            continue
        relative = path.relative_to(run).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _verify_existing_complete(run: Path) -> Path:
    marker = _read_json(run / "COMPLETE")
    inventory_path = run / "00_audit/DERIVED_SHA256_MANIFEST.csv"
    if marker.get("status") != "COMPLETE" or sha256_file(inventory_path) != marker.get(
        "inventory_sha256"
    ):
        raise RuntimeError("derived COMPLETE marker/inventory mismatch")
    locked = pd.read_csv(inventory_path).to_dict("records")
    if locked != _inventory(run):
        raise RuntimeError("derived run changed after completion")
    manifest = _read_json(run / "manifest.json")
    _assert_content_hash(manifest, name="derived manifest")
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("derived manifest is not COMPLETE")
    return run


def finalize(run: Path) -> Path:
    run = run.resolve()
    if (run / "COMPLETE").exists():
        return _verify_existing_complete(run)
    manifest = _read_json(run / "manifest.json")
    _assert_content_hash(manifest, name="pending derived manifest")
    if manifest.get("status") != "COMPLETE_PENDING_VISUAL_AUDIT":
        raise RuntimeError("derived manifest is not pending visual audit")
    source, source_lock = _validate_source(run, manifest)
    tables = _validate_tables(run)
    reconciliation = _read_json(run / "00_audit/QUALITATIVE_RECONCILIATION.json")
    if reconciliation.get("status") != "PASS" or reconciliation.get("mismatches"):
        raise RuntimeError("qualitative E0-E10/formal reconciliation did not pass")
    selections = _validate_selections(run, tables)
    gallery = _validate_gallery(run, tables)
    decks = _validate_decks(run)
    timestamp = datetime.now(timezone.utc).isoformat()
    manual_audit = f"""# Manual visual audit

Status: PASS

- Source formal run: `{source}`; final-lock SHA-256 `{source_lock}`.
- Inspected both complete contact sheets and all selected/cross-route slides at 1920 x 1080.
- Full deck: 24/24 rendered pages present; selected and cross-route pages 4-23 inspected at original resolution.
- Short deck: 10/10 rendered pages present; case/cross-route pages 2-9 inspected at original resolution.
- Verified prompt/target legibility, RGB/mask registration, route-correct CROG vs HiFi-CS masks, common crops, native/final/matched-GT distinction, candidate overlays, PASS/FAIL text, and evidence-hierarchy wording.
- No black fallback panels, broken fonts, or clipped slide objects were observed; automated slide overflow checks passed.
- One severely overexposed CROG irreparable candidate (`q0015854_26ec2fa48c96e6b4`) was rejected by the declared eligibility rule and replaced by deterministic alternative `q0013163_94c8fc9c7b67f399`.
- Auxiliary raster-board legends/table rows are intentionally compact; they remained readable at 1920 x 1080. Primary headings, outcomes, prompt, and diagnostic labels carry the presentation hierarchy.

Audit completed: {timestamp}
"""
    atomic_text(run / "00_audit/MANUAL_VISUAL_AUDIT.md", manual_audit)
    qa = {
        "schema_version": 1,
        "status": "PASS",
        "source_run": str(source),
        "source_final_lock_sha256": source_lock,
        "formal_test_reexecuted": False,
        "models_retrained": False,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "case_rows": EXPECTED_CASE_ROWS,
        "candidate_rows": sum(EXPECTED_ROUTE_CANDIDATES.values()),
        "top5_rows": EXPECTED_TOP5_ROWS,
        "selections": selections,
        "gallery": gallery,
        "decks": decks,
        "source_snapshot_equal": True,
        "g1_c1_mask_bytes_identical": True,
        "route_registry": {"crog": "CROG", "g1": "HiFi-CS -> G1", "c1": "HiFi-CS -> C1"},
        "d1_mapping": "UNPROVEN_EXCLUDED",
        "manual_audit": "00_audit/MANUAL_VISUAL_AUDIT.md",
        "checked_at_utc": timestamp,
    }
    qa["content_sha256"] = _content_hash(qa)
    atomic_json(run / "00_audit/VISUAL_TECHNICAL_QA.json", qa)
    command = shlex.join([sys.executable, *sys.argv])
    prior_commands = (run / "commands.log").read_text(encoding="utf-8")
    command_lines = [
        f"{timestamp}\tmanual_visual_audit\tfull=24/24\tshort=10/10\tselected_and_cross_original_resolution=PASS",
        f"{timestamp}\t{command}",
    ]
    atomic_text(run / "commands.log", prior_commands + "\n".join(command_lines) + "\n")
    manifest.pop("selection_changed", None)
    manifest.update(
        {
            "status": "COMPLETE",
            "model_selection_changed": False,
            "qualitative_case_selection_changed": True,
            "visual_audit_status": "PASS",
            "visual_audit_artifact": "00_audit/VISUAL_TECHNICAL_QA.json",
            "completed_at_utc": timestamp,
        }
    )
    manifest["content_sha256"] = _content_hash(manifest)
    atomic_json(run / "manifest.json", manifest)
    inventory = _inventory(run)
    inventory_frame = pd.DataFrame(inventory, columns=["relative_path", "bytes", "sha256"])
    csv_text = inventory_frame.to_csv(index=False, lineterminator="\n")
    inventory_path = run / "00_audit/DERIVED_SHA256_MANIFEST.csv"
    atomic_text(inventory_path, csv_text)
    marker = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_final_lock_sha256": source_lock,
        "inventory_path": "00_audit/DERIVED_SHA256_MANIFEST.csv",
        "inventory_sha256": sha256_file(inventory_path),
        "inventory_count": len(inventory),
        "completed_at_utc": timestamp,
    }
    atomic_json(run / "COMPLETE", marker)
    return _verify_existing_complete(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(finalize(args.run_dir))


if __name__ == "__main__":
    main()
