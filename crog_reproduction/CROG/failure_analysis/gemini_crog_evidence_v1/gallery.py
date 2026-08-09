from __future__ import annotations

import hashlib
import html
import json
import os
import uuid
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from textwrap import fill
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from PIL import Image


ER2_MODEL = "gemini-robotics-er-2-preview"
FLASH_MODEL = "gemini-3.6-flash"
ER2_DIRECT = "gemini_robotics_er2_crog_evidence_direct"
ER2_SAFE = "gemini_robotics_er2_crog_evidence_safe"
FLASH_DIRECT = "gemini_3_6_flash_crog_evidence_direct"
FLASH_SAFE = "gemini_3_6_flash_crog_evidence_safe"
CONSENSUS_SAFE = "gemini_dual_consensus_safe"
Q_ONLY = "crog_q_only"
LOCKED_PRIMARY = "locked_gemini_primary"

CATEGORY_REQUESTS: OrderedDict[str, int] = OrderedDict(
    (
        ("er2_recovered", 25),
        ("er2_harmful", 25),
        ("flash_recovered", 25),
        ("flash_harmful", 25),
        ("consensus_switches", 20),
        ("model_disagreements", 20),
        ("abstain", 10),
        ("technical_fallback", 10),
        ("both_wrong", 15),
        ("q_only_correctly_kept", 15),
    )
)

METHOD_LABELS = OrderedDict(
    (
        (Q_ONLY, "q-only"),
        (ER2_DIRECT, "ER2 direct"),
        (ER2_SAFE, "ER2 safe"),
        (FLASH_DIRECT, "Flash direct"),
        (FLASH_SAFE, "Flash safe"),
        (CONSENSUS_SAFE, "dual consensus safe"),
        (LOCKED_PRIMARY, "locked primary"),
    )
)

REQUIRED_BOARD_PANELS = {
    "rgb",
    "predicted_m",
    "predicted_q",
    "predicted_angle",
    "predicted_width",
    "candidate_cards",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows(value: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        return [dict(row) for row in pq.read_table(value).to_pylist()]
    return [dict(row) for row in value]


def _raw_candidate_id(sample_id: str, stable_id: Any) -> str | None:
    if stable_id is None:
        return None
    value = str(stable_id)
    prefix = f"{sample_id}/"
    return value[len(prefix) :] if value.startswith(prefix) else value


def _legacy_transition(row: Mapping[str, Any]) -> str:
    before = bool(row["legacy_q_only_correct"])
    after = bool(row["legacy_selected_correct"])
    if not before and after:
        return "Recovered"
    if before and not after:
        return "Harmful"
    return "Correct kept" if before else "Both wrong"


def _method_index(
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in outcomes:
        row = dict(source)
        sample_id, method = str(row["sample_id"]), str(row["method"])
        if method in result[sample_id]:
            raise ValueError(f"duplicate outcome for {sample_id}/{method}")
        result[sample_id][method] = row
    return dict(result)


def _decision_index(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in decisions:
        row = dict(source)
        if int(row.get("replicate_id", 0) or 0) != 0:
            continue
        protocol = str(row.get("protocol", "P1_full_crog_evidence"))
        if protocol not in {"P1", "P1_full_crog_evidence"}:
            continue
        sample_id, model = str(row["sample_id"]), str(row["model_id"])
        if model in result[sample_id]:
            raise ValueError(f"duplicate decision for {sample_id}/{model}")
        result[sample_id][model] = row
    return dict(result)


def select_gallery_cases(
    *,
    per_method_outcomes: Iterable[Mapping[str, Any]],
    per_model_decisions: Iterable[Mapping[str, Any]],
    evidence_sample_ids: Iterable[str],
    requested_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Deterministically fill each preregistered category without artificial shortage."""

    outcomes = _method_index(list(per_method_outcomes))
    decisions = _decision_index(list(per_model_decisions))
    evidence_ids = {str(value) for value in evidence_sample_ids}
    requested = OrderedDict(CATEGORY_REQUESTS)
    if requested_counts is not None:
        unknown = set(requested_counts) - set(requested)
        if unknown:
            raise ValueError(f"unknown gallery categories: {sorted(unknown)}")
        for category, count in requested_counts.items():
            if int(count) < 0:
                raise ValueError("gallery counts must be non-negative")
            requested[category] = int(count)

    def outcome(sample_id: str, method: str) -> dict[str, Any] | None:
        return outcomes.get(sample_id, {}).get(method)

    def decision(sample_id: str, model: str) -> dict[str, Any] | None:
        return decisions.get(sample_id, {}).get(model)

    samples = sorted(set(outcomes) | set(decisions))
    eligible: dict[str, list[str]] = {category: [] for category in requested}
    for sample_id in samples:
        if sample_id not in evidence_ids:
            continue
        er2 = outcome(sample_id, ER2_DIRECT)
        flash = outcome(sample_id, FLASH_DIRECT)
        consensus = outcome(sample_id, CONSENSUS_SAFE)
        primary = outcome(sample_id, LOCKED_PRIMARY) or consensus
        er2_decision = decision(sample_id, ER2_MODEL)
        flash_decision = decision(sample_id, FLASH_MODEL)
        if er2 and _legacy_transition(er2) == "Recovered":
            eligible["er2_recovered"].append(sample_id)
        if er2 and _legacy_transition(er2) == "Harmful":
            eligible["er2_harmful"].append(sample_id)
        if flash and _legacy_transition(flash) == "Recovered":
            eligible["flash_recovered"].append(sample_id)
        if flash and _legacy_transition(flash) == "Harmful":
            eligible["flash_harmful"].append(sample_id)
        if consensus and bool(consensus.get("switch", False)):
            eligible["consensus_switches"].append(sample_id)
        if (
            er2_decision
            and flash_decision
            and str(er2_decision.get("selected_stable_candidate_id"))
            != str(flash_decision.get("selected_stable_candidate_id"))
        ):
            eligible["model_disagreements"].append(sample_id)
        if any(bool(row.get("abstain", False)) for row in (er2_decision, flash_decision) if row):
            eligible["abstain"].append(sample_id)
        if any(
            bool(row.get("technical_fallback", False))
            or str(row.get("lifecycle_status")) == "TECHNICAL_FALLBACK"
            for row in (er2_decision, flash_decision)
            if row
        ):
            eligible["technical_fallback"].append(sample_id)
        if (
            er2
            and flash
            and not bool(er2["legacy_q_only_correct"])
            and not bool(er2["legacy_selected_correct"])
            and not bool(flash["legacy_selected_correct"])
        ):
            eligible["both_wrong"].append(sample_id)
        if (
            primary
            and bool(primary["legacy_q_only_correct"])
            and bool(primary["legacy_selected_correct"])
            and not bool(primary.get("switch", False))
        ):
            eligible["q_only_correctly_kept"].append(sample_id)

    cases: list[dict[str, Any]] = []
    categories: dict[str, Any] = OrderedDict()
    for category, count in requested.items():
        available = list(dict.fromkeys(eligible[category]))
        selected = available[:count]
        for index, sample_id in enumerate(selected):
            cases.append(
                {
                    "category": category,
                    "category_index": index,
                    "sample_id": sample_id,
                }
            )
        categories[category] = {
            "requested": count,
            "eligible_unique": len(eligible[category]),
            "available_unique_within_category": len(available),
            "selected": len(selected),
            "shortfall": count - len(selected),
            "selected_sample_ids": selected,
        }
    return {
        "schema_version": "1.0",
        "selection_track": "Legacy evaluator",
        "deduplication_policy": "unique_within_category; cross-category reuse allowed",
        "categories": categories,
        "cases": cases,
    }


def _load_evidence_catalog(
    evidence_index_path: str | Path,
    required_sample_ids: set[str],
) -> dict[str, dict[str, Any]]:
    payload = _read_json(Path(evidence_index_path))
    index_rows = {str(row["sample_id"]): row for row in payload["rows"]}
    missing = required_sample_ids - set(index_rows)
    if missing:
        raise ValueError(f"gallery samples absent from evidence index: {sorted(missing)[:3]}")
    by_source: dict[Path, set[str]] = defaultdict(set)
    for sample_id in required_sample_ids:
        by_source[Path(index_rows[sample_id]["source"]).resolve()].add(sample_id)

    result: dict[str, dict[str, Any]] = {}
    for source, sample_ids in by_source.items():
        requests = {
            str(row["sample_id"]): row
            for row in _read_jsonl(source / "request_manifest.jsonl")
            if str(row["sample_id"]) in sample_ids
        }
        samples = {
            str(row["sample_id"]): row
            for row in pq.read_table(source / "sample_evidence.parquet").to_pylist()
            if str(row["sample_id"]) in sample_ids
        }
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pq.read_table(source / "candidate_evidence.parquet").to_pylist():
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                candidates[sample_id].append(row)
        for sample_id in sample_ids:
            if sample_id not in requests or sample_id not in samples:
                raise ValueError(f"incomplete evidence catalog for {sample_id}")
            request = requests[sample_id]
            sample = samples[sample_id]
            if len(candidates[sample_id]) != 5:
                raise ValueError(f"gallery requires five frozen candidates for {sample_id}")
            mapping = request["mapping"]
            if isinstance(mapping, str):
                mapping = json.loads(mapping)
            displays = mapping.get("display_to_candidate", {})
            inverse = mapping.get("candidate_to_display", {})
            if set(displays) != set("ABCDE") or inverse != {
                value: key for key, value in displays.items()
            }:
                raise ValueError(f"invalid candidate mapping for {sample_id}")
            board_path = Path(request.get("board_path") or sample["board_path"]).resolve()
            if not board_path.is_file():
                raise FileNotFoundError(board_path)
            board_sha256 = _sha256_file(board_path)
            expected_hashes = {
                str(value)
                for value in (
                    request.get("board_sha256"),
                    sample.get("board_sha256"),
                    index_rows[sample_id].get("board_sha256"),
                )
                if value
            }
            if expected_hashes != {board_sha256}:
                raise ValueError(f"input-board hash mismatch for {sample_id}")
            layers_path = board_path.with_suffix(".layers.json")
            if not layers_path.is_file():
                raise ValueError(f"input board lacks layer audit: {sample_id}")
            layers = _read_json(layers_path)
            if layers.get("evaluation_overlay_included") is not False:
                raise ValueError(f"input board contains or ambiguously reports evaluation overlay: {sample_id}")
            if not REQUIRED_BOARD_PANELS.issubset(set(layers.get("panels", []))):
                raise ValueError(f"input board panel contract is incomplete: {sample_id}")
            if layers.get("image_sha256") != board_sha256:
                raise ValueError(f"input-board layer hash mismatch for {sample_id}")
            result[sample_id] = {
                "source": str(source),
                "request": request,
                "sample": sample,
                "candidate_evidence": candidates[sample_id],
                "mapping": mapping,
                "board_path": board_path,
                "board_sha256": board_sha256,
                "layers_path": layers_path,
                "layers": layers,
            }
    return result


def _display_selection(
    sample_id: str,
    row: Mapping[str, Any] | None,
    mapping: Mapping[str, Any],
) -> str:
    if row is None:
        return "not available"
    raw = _raw_candidate_id(sample_id, row.get("selected_stable_candidate_id"))
    if raw is None:
        raw = row.get("selected_candidate_id")
    display = mapping["candidate_to_display"].get(str(raw), "?")
    return f"{display} → {sample_id}/{raw}"


def _reason_text(decision: Mapping[str, Any] | None) -> str:
    if decision is None:
        return "not available"
    reasons = decision.get("global_reason_codes") or decision.get("reason_codes") or []
    if isinstance(reasons, str):
        try:
            reasons = json.loads(reasons)
        except json.JSONDecodeError:
            reasons = [reasons]
    reason = ", ".join(map(str, reasons)) if reasons else str(decision.get("fallback_reason") or "not saved")
    return (
        f"decision={decision.get('decision', decision.get('status', 'unknown'))}; "
        f"confidence={float(decision.get('confidence', 0.0) or 0.0):.3f}; "
        f"margin={float(decision.get('score_margin_top1_top2', 0.0) or 0.0):.3f}; "
        f"reasons={reason}"
    )


def _render_case(
    *,
    case: Mapping[str, Any],
    evidence: Mapping[str, Any],
    outcomes: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    sample_id = str(case["sample_id"])
    mapping = evidence["mapping"]
    board_path = Path(evidence["board_path"])
    before_hash = _sha256_file(board_path)
    board = Image.open(board_path).convert("RGB")

    selected_lines = []
    for method, label in METHOD_LABELS.items():
        row = outcomes.get(method)
        if row is not None:
            selected_lines.append(f"{label:20s} {_display_selection(sample_id, row, mapping)}")
    mapping_lines = [
        f"{display} → {sample_id}/{candidate}"
        for display, candidate in sorted(mapping["display_to_candidate"].items())
    ]
    evaluation_lines = []
    for method, label in METHOD_LABELS.items():
        row = outcomes.get(method)
        if row is None:
            continue
        evaluation_lines.append(
            f"{label:20s} Legacy={_legacy_transition(row):12s} "
            f"Corrected q={bool(row['corrected_q_only_correct'])!s:5s} "
            f"selected={bool(row['corrected_selected_correct'])!s:5s}"
        )
    decision_lines = [
        "ER2: " + _reason_text(decisions.get(ER2_MODEL)),
        "Flash: " + _reason_text(decisions.get(FLASH_MODEL)),
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
        }
    )
    fig = plt.figure(figsize=(18, 11), constrained_layout=True, facecolor="#FAFAF7")
    grid = fig.add_gridspec(1, 2, width_ratios=(2.2, 1.0))
    board_axis = fig.add_subplot(grid[0, 0])
    eval_axis = fig.add_subplot(grid[0, 1])
    board_axis.imshow(board)
    board_axis.set_title(
        "Gemini input board — byte-identical, predicted evidence only, no evaluation overlay",
        fontsize=12,
        weight="bold",
        color="#264653",
    )
    board_axis.axis("off")
    eval_axis.set_facecolor("#FFF7F5")
    for spine in eval_axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#D55E00")
        spine.set_linewidth(2.0)
    eval_axis.set_xticks([])
    eval_axis.set_yticks([])
    expression = str(evidence["sample"].get("language_instruction", ""))
    text = (
        "EVALUATION-ONLY PANEL — NEVER SENT TO GEMINI\n"
        "================================================\n"
        f"Category: {case['category']}\n"
        f"Sample: {sample_id}\n"
        f"Expression: {fill(expression, 58)}\n\n"
        "Frozen stable/display mapping\n"
        + "\n".join(mapping_lines)
        + "\n\nSaved selections\n"
        + "\n".join(selected_lines)
        + "\n\nModel confidence and reason codes\n"
        + "\n".join(fill(line, 72) for line in decision_lines)
        + "\n\nIndependent evaluator outcomes\n"
        + "\n".join(evaluation_lines)
        + "\n\nGT is used only for the correctness outcomes above; "
        "it is absent from the byte-identical input board."
    )
    eval_axis.text(
        0.035,
        0.975,
        text,
        transform=eval_axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        family="monospace",
        color="#2E3440",
        linespacing=1.25,
        wrap=True,
    )
    fig.suptitle(
        fill(expression, 115), fontsize=13, weight="bold", color="#2E3440"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    after_hash = _sha256_file(board_path)
    if before_hash != after_hash or before_hash != evidence["board_sha256"]:
        raise RuntimeError("gallery rendering modified the immutable input board")
    return {
        **dict(case),
        "language_instruction": expression,
        "image_path": str(output_path.resolve()),
        "input_board_path": str(board_path.resolve()),
        "input_board_sha256_before": before_hash,
        "input_board_sha256_after": after_hash,
        "input_board_evaluation_overlay_included": False,
        "evaluation_panel_in_composite": True,
        "selected": {
            method: _display_selection(sample_id, row, mapping)
            for method, row in outcomes.items()
            if method in METHOD_LABELS
        },
    }


def build_evaluation_gallery(
    *,
    per_method_outcomes_path: str | Path,
    per_model_decisions_path: str | Path,
    evidence_index_path: str | Path,
    output_dir: str | Path,
    requested_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build the preregistered formal gallery entirely from saved artifacts."""

    destination = Path(output_dir).resolve()
    existing_manifest = destination / "gallery.json"
    if existing_manifest.is_file():
        return _read_json(existing_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        os.replace(
            destination,
            destination.with_name(f".{destination.name}.incomplete-{uuid.uuid4().hex}"),
        )
    output = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    output.mkdir()
    outcome_rows = _rows(per_method_outcomes_path)
    decision_rows = _rows(per_model_decisions_path)
    evidence_payload = _read_json(Path(evidence_index_path))
    evidence_ids = {str(row["sample_id"]) for row in evidence_payload["rows"]}
    selection = select_gallery_cases(
        per_method_outcomes=outcome_rows,
        per_model_decisions=decision_rows,
        evidence_sample_ids=evidence_ids,
        requested_counts=requested_counts,
    )
    required = {str(case["sample_id"]) for case in selection["cases"]}
    catalog = _load_evidence_catalog(evidence_index_path, required)
    outcomes = _method_index(outcome_rows)
    decisions = _decision_index(decision_rows)

    records = []
    for case in selection["cases"]:
        sample_id = str(case["sample_id"])
        filename = f"{int(case['category_index']):02d}_{sample_id.replace(':', '_').replace('/', '_')}.png"
        record = _render_case(
            case=case,
            evidence=catalog[sample_id],
            outcomes=outcomes.get(sample_id, {}),
            decisions=decisions.get(sample_id, {}),
            output_path=output / str(case["category"]) / filename,
        )
        records.append(record)

    rows = "\n".join(
        "<article><h2>"
        + html.escape(str(record["category"]))
        + " — "
        + html.escape(str(record["sample_id"]))
        + "</h2><p>"
        + html.escape(str(record["language_instruction"]))
        + "</p><img loading=\"lazy\" src=\""
        + html.escape(str(Path(record["image_path"]).relative_to(output)))
        + "\" alt=\"Gemini CROG evaluation case\"></article>"
        for record in records
    )
    _atomic_text(
        output / "index.html",
        "<!doctype html><meta charset=\"utf-8\"><title>CROG Gemini evidence "
        "gallery</title><style>body{font-family:system-ui;max-width:1600px;"
        "margin:auto;padding:24px;background:#fafaf7;color:#2e3440}img{width:100%;"
        "height:auto}article{margin:0 0 52px}h2{color:#264653}</style>"
        "<h1>CROG Gemini Evidence V1 — formal evaluation gallery</h1>"
        "<p>The left panel is the byte-identical GT-free Gemini input board. Ground-truth "
        "correctness appears only in the separately bordered evaluation-only panel.</p>"
        + rows,
    )
    relocated_records = []
    for record in records:
        relocated = dict(record)
        image_path = Path(str(relocated["image_path"])).resolve()
        relocated["image_path"] = str(destination / image_path.relative_to(output))
        relocated_records.append(relocated)
    result = {
        **selection,
        "case_count": len(relocated_records),
        "cases": relocated_records,
        "inputs": {
            "per_method_outcomes": str(Path(per_method_outcomes_path).resolve()),
            "per_model_decisions": str(Path(per_model_decisions_path).resolve()),
            "evidence_index": str(Path(evidence_index_path).resolve()),
        },
        "safety": {
            "input_boards_byte_identical": True,
            "input_board_evaluation_overlay_included": False,
            "gt_only_in_separate_evaluation_panel": True,
            "samples_unique_within_each_category": True,
            "cross_category_reuse_allowed": True,
        },
        "index_html": str((destination / "index.html").resolve()),
    }
    _atomic_json(output / "gallery.json", result)
    summary_lines = [
        "# CROG Gemini Evidence V1 evaluation gallery",
        "",
        "Ground truth appears only in the separate evaluation panel; input boards remain byte-identical.",
        "",
        "| Category | Requested | Eligible | Available unique | Selected | Shortfall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, row in result["categories"].items():
        summary_lines.append(
            f"| {category} | {row['requested']} | {row['eligible_unique']} | "
            f"{row['available_unique_within_category']} | {row['selected']} | {row['shortfall']} |"
        )
    _atomic_text(output / "summary.md", "\n".join(summary_lines) + "\n")
    os.replace(output, destination)
    return result
