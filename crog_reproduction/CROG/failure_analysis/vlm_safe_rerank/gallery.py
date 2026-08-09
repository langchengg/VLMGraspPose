from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import load_feature_index
from .renderer import PerturbationVariant, render_pairwise_board


def _evaluation_figure(
    board: bytes,
    *,
    outcome: Mapping[str, Any],
    parsed: Mapping[str, Any] | None,
    output: Path,
) -> None:
    image = plt.imread(io.BytesIO(board), format="png")
    fig = plt.figure(figsize=(18, 14), dpi=100)
    grid = fig.add_gridspec(2, 1, height_ratios=(6, 1))
    ax = fig.add_subplot(grid[0]); ax.imshow(image); ax.axis("off")
    evaluation = fig.add_subplot(grid[1]); evaluation.axis("off")
    scores = "no valid critic output"
    if parsed:
        scores = (
            f"decision={parsed.get('decision')} reliable={parsed.get('evidence_reliable')} | "
            f"target B/C={parsed.get('baseline_target_alignment')}/{parsed.get('challenger_target_alignment')} | "
            f"contact B/C={parsed.get('baseline_contact_geometry')}/{parsed.get('challenger_contact_geometry')} | "
            f"collision B/C={parsed.get('baseline_collision_risk')}/{parsed.get('challenger_collision_risk')}"
        )
    stage = (
        "critic/gate" if outcome["baseline_correct"] != outcome["selected_correct"]
        else "no recoverable candidate" if outcome["cohort"] == "unrecoverable_error"
        else "query/challenger/critic"
    )
    evaluation.text(
        .01, .92,
        "EVALUATION-ONLY PANEL (never sent to Gemini)\n"
        f"sample={outcome['sample_id']} model={outcome['model_id']} cohort={outcome['cohort']}\n"
        f"baseline_correct={outcome['baseline_correct']} challenger_correct={outcome['challenger_correct']} "
        f"hard_rule_switch={outcome['switched']} selected_correct={outcome['selected_correct']}\n"
        f"failure_stage={stage}\n"
        "calibrated_p_benefit/p_harm=not computed in diagnostic cohort; "
        f"gate={'hard-rule pass' if outcome['switched'] else 'KEEP baseline'}\n{scores}",
        va="top", family="monospace", fontsize=10,
    )
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)


def generate_failure_gallery(
    run_dir: str | Path,
    *,
    phase: str = "diagnostic",
    limit_per_category: int = 10,
) -> dict[str, int]:
    root = Path(run_dir); phase_dir = root / phase
    outcomes = pq.read_table(phase_dir / "pairwise_outcomes.parquet").to_pylist()
    responses = pq.read_table(phase_dir / "pairwise_responses.parquet").to_pylist()
    response_index = {
        (row["sample_id"], row["challenger_candidate_id"], row["model_id"], row["variant"]): row
        for row in responses
    }
    unstable_pairs: set[tuple[str, str, str]] = set()
    perturbation_path = root / "diagnostic_perturbations/pairwise_responses.parquet"
    if phase in {"diagnostic", "diagnostic_expanded", "p3_diagnostic"} and perturbation_path.is_file():
        perturbations = pq.read_table(perturbation_path).to_pylist()
        decisions: dict[tuple[str, str, str], set[str]] = {}
        for row in [*responses, *perturbations]:
            parsed = row.get("parsed")
            if not isinstance(parsed, Mapping):
                continue
            key = (str(row["sample_id"]), str(row["challenger_candidate_id"]), str(row["model_id"]))
            decisions.setdefault(key, set()).add(str(parsed.get("decision")))
        unstable_pairs = {key for key, values in decisions.items() if len(values) > 1}
    inference = json.loads((phase_dir / "inference_manifest.json").read_text())
    sample_ids = {str(row["sample_id"]) for row in outcomes}
    features = load_feature_index(inference["feature_file"], sample_ids)
    categories = {
        "harmful": lambda row: row["baseline_correct"] and not row["selected_correct"],
        "recovered": lambda row: not row["baseline_correct"] and row["selected_correct"],
        "missed_recovery": lambda row: not row["baseline_correct"] and row["challenger_correct"] and not row["switched"],
        "unrecoverable_switch": lambda row: row["cohort"] == "unrecoverable_error" and row["switched"],
        "unstable": lambda row: (
            str(row["sample_id"]),
            str(row["challenger_candidate_id"]),
            str(row["model_id"]),
        ) in unstable_pairs,
    }
    counts = {}
    for category, predicate in categories.items():
        chosen = [row for row in outcomes if row["variant"] == "original" and predicate(row)][:limit_per_category]
        counts[category] = len(chosen)
        directory = root / "failure_gallery" / category; directory.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(chosen):
            feature = features[row["sample_id"]]
            board, _ = render_pairwise_board(feature, row["challenger_candidate_id"], variant=PerturbationVariant.ORIGINAL)
            response = response_index[(row["sample_id"], row["challenger_candidate_id"], row["model_id"], "original")]
            _evaluation_figure(
                board, outcome=row, parsed=response.get("parsed"),
                output=directory / f"{index:02d}_{row['model_id'].replace('/', '_')}_{row['sample_id'].replace(':', '_')}.png",
            )
        if not chosen:
            (directory / "EMPTY.md").write_text(f"No {category} cases were observed in {phase}.\n", encoding="utf-8")
    return counts
