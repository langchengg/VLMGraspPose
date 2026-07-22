"""Truth-preserving report assembly for the full OCID-VLG VGN experiment."""

from __future__ import annotations

import csv
import html
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .render_gallery import build_gallery


OFFICIAL_VGN_REPOSITORY = "https://github.com/ethz-asl/vgn"
OFFICIAL_VGN_PAPER = "https://proceedings.mlr.press/v155/breyer21a.html"
REAL_ROBOT_ABSENCE_REASON = "no physical robot execution logs"


def _atomic_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _first_json(paths: Sequence[Path]) -> tuple[dict[str, Any] | None, Path | None]:
    for path in paths:
        if path.is_file():
            return _json(path), path
    return None, None


def _rows(root: Path) -> list[dict[str, Any]]:
    for path in (
        root / "metrics" / "per_sample.csv",
        root / "per_sample.csv",
        root / "summary.csv",
    ):
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as stream:
                return [dict(row) for row in csv.DictReader(stream)]
    return []


def _null_metric(metric_name: str, reason: str, scope: str) -> dict[str, Any]:
    return {
        "metric_name": metric_name,
        "metric_scope": scope,
        "value": None,
        "numerator": None,
        "denominator": None,
        "confidence_interval_95": None,
        "reason": reason,
    }


def _coverage_metric(
    aggregate: Mapping[str, Any] | None,
    key: str,
    label: str,
) -> dict[str, Any]:
    proportions = aggregate.get("proportions") if aggregate else None
    raw = proportions.get(key) if isinstance(proportions, Mapping) else None
    if not isinstance(raw, Mapping):
        return _null_metric(
            label,
            "offline aggregate metric is unavailable",
            "ocid_vlg_offline_deployment",
        )
    return {
        "metric_name": label,
        "metric_scope": "ocid_vlg_offline_deployment",
        "value": raw.get("estimate"),
        "numerator": raw.get("numerator"),
        "denominator": raw.get("denominator"),
        "confidence_interval_95": [raw.get("ci_lower"), raw.get("ci_upper")],
        "interval_method": raw.get("method"),
        "reason": None,
        "is_physical_success_metric": False,
    }


def _simulation_metric(aggregate: Mapping[str, Any] | None) -> dict[str, Any]:
    if aggregate is None:
        return _null_metric(
            "simulated_grasp_success_rate",
            "no PyBullet simulation aggregate found",
            "pybullet_simulated_physical_execution",
        )
    status = str(aggregate.get("status", "unknown"))
    value = aggregate.get("simulated_grasp_success_rate")
    if status != "completed" or value is None:
        metric = _null_metric(
            "simulated_grasp_success_rate",
            str(aggregate.get("reason") or f"simulation status is {status}"),
            "pybullet_simulated_physical_execution",
        )
        metric["status"] = status
        metric["blockers"] = aggregate.get("blockers", [])
        return metric
    config = aggregate.get("config")
    formal = bool(config.get("official_paper_protocol")) if isinstance(config, Mapping) else False
    return {
        "metric_name": "simulated_grasp_success_rate",
        "metric_scope": "pybullet_simulated_physical_execution",
        "value": value,
        "numerator": aggregate.get("successful_executions"),
        "denominator": aggregate.get("attempted_executions"),
        "confidence_interval_95": aggregate.get("scene_cluster_bootstrap_95_ci"),
        "reason": None,
        "status": status,
        "protocol": aggregate.get("protocol"),
        "protocol_tier": "full_official_protocol" if formal else "smoke_or_custom_round_count",
        "is_real_world_success_metric": False,
    }


def _real_robot_metric(aggregate: Mapping[str, Any] | None) -> dict[str, Any]:
    if aggregate is None:
        return _null_metric(
            "real_robot_grasp_success_rate",
            REAL_ROBOT_ABSENCE_REASON,
            "real_robot_physical_execution",
        )
    value = aggregate.get("real_robot_grasp_success_rate")
    physical = aggregate.get("real_robot_grasp_success")
    if value is None:
        metric = _null_metric(
            "real_robot_grasp_success_rate",
            str(aggregate.get("reason") or REAL_ROBOT_ABSENCE_REASON),
            "real_robot_physical_execution",
        )
        metric["status"] = aggregate.get("status")
        return metric
    details = physical if isinstance(physical, Mapping) else {}
    return {
        "metric_name": "real_robot_grasp_success_rate",
        "metric_scope": "real_robot_physical_execution",
        "value": value,
        "numerator": details.get("numerator"),
        "denominator": details.get("denominator"),
        "confidence_interval_95": details.get("wilson_95_ci"),
        "reason": None,
        "status": aggregate.get("status"),
    }


def _percent(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "not available"


def _metric_line(metric: Mapping[str, Any]) -> str:
    value = metric.get("value")
    if value is None:
        return f"- `{metric['metric_name']}`: **null** — {metric.get('reason', 'not available')}"
    numerator, denominator = metric.get("numerator"), metric.get("denominator")
    counts = (
        f" ({numerator}/{denominator})"
        if numerator is not None and denominator is not None
        else ""
    )
    interval = metric.get("confidence_interval_95")
    ci = ""
    if isinstance(interval, (list, tuple)) and len(interval) == 2 and None not in interval:
        ci = f"; 95% CI [{_percent(interval[0])}, {_percent(interval[1])}]"
    return f"- `{metric['metric_name']}`: **{_percent(value)}**{counts}{ci}"


def _status_table(aggregate: Mapping[str, Any] | None) -> str:
    counts = aggregate.get("status_counts") if aggregate else None
    if not isinstance(counts, Mapping) or not counts:
        return "No evaluated status table was found."
    lines = ["| status | samples |", "|---|---:|"]
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(counts.items()))
    return "\n".join(lines)


def _markdown(
    executive: Mapping[str, Any],
    offline: Mapping[str, Any] | None,
    simulation: Mapping[str, Any] | None,
    robot: Mapping[str, Any] | None,
    oracle: Mapping[str, Any] | None,
    auxiliary: Mapping[str, Any] | None,
    run_config: Mapping[str, Any] | None,
) -> str:
    official = executive["offline_candidate_coverage"]["official"]
    target = executive["offline_candidate_coverage"]["target"]
    simulated = executive["simulated_physical_success"]
    real = executive["real_robot_success"]
    manifest_count = offline.get("manifest_count") if offline else None
    registered = offline.get("registered_row_count") if offline else None
    truth = offline.get("truthfulness") if offline else None
    limitations = (run_config or {}).get("limitations") or [
        "single-view TSDF adaptation",
        "OCID-VLG has no complete 6-DoF grasp ground truth",
        "offline candidate coverage is not physical grasp success",
    ]
    limitation_lines = "\n".join(f"- {item}" for item in limitations)
    sim_scope = (
        f"Status: `{simulation.get('status')}`. Protocol: `{simulation.get('protocol')}`."
        if simulation
        else "No simulation aggregate was supplied."
    )
    robot_scope = (
        f"Status: `{robot.get('status')}`."
        if robot
        else "No physical robot aggregate was supplied."
    )
    oracle_text = (
        "```json\n" + json.dumps(oracle, indent=2, ensure_ascii=False) + "\n```"
        if oracle
        else "GT-mask oracle aggregate is not available. No oracle delta is claimed."
    )
    auxiliary_text = (
        "This is an **auxiliary cross-representation metric, not 6-DoF ground truth**.\n\n"
        "```json\n" + json.dumps(auxiliary, indent=2, ensure_ascii=False) + "\n```"
        if auxiliary
        else (
            "No `projected_4dof_auxiliary_metric` result is available. It is not inferred. "
            "If later computed, it must be labelled an **auxiliary cross-representation metric, "
            "not 6-DoF ground truth**."
        )
    )
    return f"""# OCID-VLG × HiFi-CS × official VGN evaluation

## Executive separation of metric scopes

{_metric_line(official)}
{_metric_line(target)}
{_metric_line(simulated)}
{_metric_line(real)}

Offline candidate coverage measures deployment/candidate availability only. It is **not** a physical grasp success rate. Simulated physics and real-robot execution are reported separately and never substituted for one another.

## 1. Experimental scope

The offline experiment applies HiFi-CS predicted target masks to a target-centred, context-preserving, single-view TSDF and the pinned official VGN detector. Candidate ranking uses official processed VGN quality with no custom re-ranking. The official implementation and paper are linked in the reproduction references.

## 2. Dataset and manifest completeness

- Manifest rows declared: `{manifest_count if manifest_count is not None else 'not available'}`
- Per-sample rows registered: `{registered if registered is not None else 'not available'}`
- Technical failures remain in the reported status table and denominator accounting.

## 3. Geometry and calibration quality

Geometry/calibration distributions are read from the evaluation aggregate. Missing calibration or support-plane estimates are terminal technical outcomes; they are not silently removed. Detailed numeric distributions remain in `metrics/aggregate_metrics.json`.

## 4. Full OCID-VLG offline results

{_metric_line(official)}
{_metric_line(target)}

These metrics are named **coverage/availability**, not grasp success. VGN quality is a model output, not an observed execution label.

## 5. Failure taxonomy

{_status_table(offline)}

## 6. Predicted-mask vs GT-mask oracle

{oracle_text}

## 7. Target-consistency analysis

When GT masks are available, projected top-1 inclusion, nearest GT target point distance, and projected depth error are target-consistency diagnostics. They are not physical grasp success and not 6-DoF ground-truth accuracy.

## 8. Optional projected 4-DoF auxiliary results

{auxiliary_text}

## 9. Official VGN simulation success

{sim_scope}

{_metric_line(simulated)}

Only completed PyBullet physical executions under the declared protocol contribute to `simulated_grasp_success_rate`. A smoke/custom run is not presented as the full paper protocol.

## 10. Optional GraspNet official 6-DoF AP

`graspnet_official_evaluation`: **null** — GraspNet-1Billion test data and/or dex models were not supplied to this report builder. No AP is inferred from OCID-VLG.

## 11. Real robot results or explicit absence

{robot_scope}

{_metric_line(real)}

Neither offline VGN quality nor PyBullet results are used as real-robot labels.

## 12. Qualitative success/failure cases

See [gallery.html](gallery.html). Each available link points to an existing sample artifact; filters expose status, query type, category, mask IoU, VGN quality, and official candidate count.

## 13. Limitations

{limitation_lines}

## 14. Reproduction commands

```bash
python -m scripts.evaluate_full_ocid_vgn --output outputs/hifics_vgn_full --bootstrap-replicates 10000 --cluster-key scene_id --seed 42
python -m scripts.build_vgn_report --ocid-output outputs/hifics_vgn_full --sim-output outputs/vgn_sim_benchmark --output outputs/hifics_vgn_full/report
```

### References

- Official VGN repository (BSD-3-Clause): {OFFICIAL_VGN_REPOSITORY}
- VGN CoRL paper: {OFFICIAL_VGN_PAPER}

### Truthfulness audit

```json
{json.dumps(truth, indent=2, ensure_ascii=False) if truth is not None else 'null'}
```
"""


def _html_report(markdown: str) -> str:
    try:
        import markdown as markdown_renderer

        body = markdown_renderer.markdown(
            markdown, extensions=("fenced_code", "tables"), output_format="html5"
        )
    except ImportError:
        body = f"<pre>{html.escape(markdown)}</pre>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCID-VLG × HiFi-CS × official VGN evaluation</title>
<style>body{{max-width:1100px;margin:32px auto;padding:0 20px;background:#fbfcfe;color:#17202a;font:15px/1.55 system-ui,sans-serif}}h1,h2{{line-height:1.2}}h2{{margin-top:2em;border-bottom:1px solid #dbe2ea;padding-bottom:.25em}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#111820;color:#edf3fa;border-radius:7px;padding:14px}}code{{background:#edf1f5;padding:.12em .3em;border-radius:3px}}pre code{{background:none;padding:0}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccd5df;padding:6px 10px}}a{{color:#1769aa}}</style></head>
<body><p><a href="gallery.html">Open qualitative gallery</a></p><main>{body}</main></body></html>"""


def build_report(
    ocid_output: str | Path,
    sim_output: str | Path | None,
    output: str | Path,
    *,
    real_robot_output: str | Path | None = None,
) -> dict[str, Path]:
    """Build report artifacts without synthesizing missing evaluation results."""

    ocid = Path(ocid_output).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for directory in ("tables", "figures", "paper_figures"):
        (destination / directory).mkdir(exist_ok=True)

    offline, offline_path = _first_json(
        (
            ocid / "metrics" / "aggregate_metrics.json",
            ocid / "aggregate_metrics.json",
            ocid / "metrics.json",
        )
    )
    run_config = _json(ocid / "run_config.json")
    oracle, _ = _first_json(
        (ocid / "metrics" / "oracle_delta.json", ocid / "oracle_delta.json")
    )
    auxiliary, _ = _first_json(
        (
            ocid / "metrics" / "projected_4dof_auxiliary_metric.json",
            ocid / "projected_4dof_auxiliary_metric.json",
        )
    )
    simulation = (
        _json(Path(sim_output).expanduser().resolve() / "aggregate.json")
        if sim_output is not None
        else None
    )
    robot_paths: list[Path] = []
    if real_robot_output is not None:
        robot_paths.append(Path(real_robot_output).expanduser().resolve() / "aggregate.json")
    robot_paths.extend(
        (ocid / "real_robot" / "aggregate.json", ocid / "robot" / "aggregate.json")
    )
    robot, _ = _first_json(tuple(robot_paths))

    official_coverage = _coverage_metric(
        offline, "official_candidate_availability", "official_vgn_candidate_coverage"
    )
    target_coverage = _coverage_metric(
        offline, "target_candidate_availability", "target_candidate_coverage"
    )
    simulated = _simulation_metric(simulation)
    real = _real_robot_metric(robot)
    executive: dict[str, Any] = {
        "schema_version": 1,
        "offline_candidate_coverage": {
            "metric_scope": "ocid_vlg_offline_deployment",
            "official": official_coverage,
            "target": target_coverage,
            "physical_success_claimed": False,
        },
        "simulated_physical_success": simulated,
        "real_robot_success": real,
        "source_files": {
            "offline_aggregate": str(offline_path) if offline_path else None,
            "simulation_aggregate": (
                str(Path(sim_output).expanduser().resolve() / "aggregate.json")
                if sim_output is not None
                else None
            ),
        },
        "scope_separation_verified": (
            official_coverage["metric_scope"] != simulated["metric_scope"]
            and simulated["metric_scope"] != real["metric_scope"]
        ),
    }
    markdown = _markdown(
        executive, offline, simulation, robot, oracle, auxiliary, run_config
    )
    report_md = _atomic_text(destination / "report.md", markdown)
    report_html = _atomic_text(destination / "report.html", _html_report(markdown))
    summary_path = _atomic_json(destination / "executive_summary.json", executive)
    gallery_path = build_gallery(
        _rows(ocid), ocid / "samples", destination / "gallery.html"
    )
    return {
        "report_md": report_md,
        "report_html": report_html,
        "executive_summary": summary_path,
        "gallery": gallery_path,
    }


__all__ = ["REAL_ROBOT_ABSENCE_REASON", "build_report"]
