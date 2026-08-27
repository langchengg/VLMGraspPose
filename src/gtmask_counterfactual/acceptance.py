"""Independent P9/P10 acceptance gates for saved counterfactual outputs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import transition_pipeline_status
from .contracts import RunState
from .gallery_pipeline import verify_complete_gallery
from .independent import (
    NATIVE_CLASSES,
    POST_R7_CLASSES,
    independent_recompute_from_frames,
)
from .io import artifact_record, atomic_json, canonical_sha256, sha256_file


GALLERY_ACCEPTANCE_RELATIVE_PATH = Path(
    "12_case_selection/P9_GALLERY_ACCEPTANCE.json"
)
INDEPENDENT_ACCEPTANCE_RELATIVE_PATH = Path(
    "16_independent_recompute/INDEPENDENT_VALIDATION.json"
)


def _object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain one JSON object")
    return value


def _verify_self_hash(value: Mapping[str, Any], *, name: str) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash differs")


def _verify_record(record: Mapping[str, Any], *, root: Path, name: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"{name} escapes the counterfactual run") from error
    if (
        path.is_symlink()
        or not path.is_file()
        or record.get("sha256") != sha256_file(path)
        or int(record.get("bytes", -1)) != path.stat().st_size
    ):
        raise RuntimeError(f"{name} artifact differs")
    return path


def _publish_exact(path: Path, payload: Mapping[str, Any], *, resume: bool) -> Path:
    value = dict(payload)
    value["content_sha256"] = canonical_sha256(value)
    if path.exists():
        observed = _object(path, name=path.name)
        if not resume or observed != value:
            raise RuntimeError(f"existing acceptance artifact differs: {path}")
        return path
    return atomic_json(path, value)


def _pipeline(root: Path) -> dict[str, Any]:
    return _object(root / "pipeline_status.json", name="pipeline status")


def accept_gallery(
    run_dir: str | Path,
    *,
    resume: bool = False,
    expected_sample_count: int = 7_675,
) -> Path:
    """Accept only an exact, manually reviewed deterministic gallery bundle."""

    root = Path(run_dir).expanduser().resolve()
    pipeline = _pipeline(root)
    if pipeline.get("status") not in {
        RunState.P8_STATISTICS_COMPLETE.value,
        RunState.P9_GALLERIES_COMPLETE.value,
    }:
        raise PermissionError("P9 core acceptance requires P8")
    gallery_path = root / "14_galleries/GALLERY_MANIFEST.json"
    gallery = verify_complete_gallery(
        root, expected_sample_count=expected_sample_count
    )
    eligible_path = _verify_record(
        gallery["eligible"], root=root, name="gallery eligible table"
    )
    selected_path = _verify_record(
        gallery["selected"], root=root, name="gallery selected table"
    )
    eligible = pd.read_parquet(eligible_path)
    selected = pd.read_parquet(selected_path)
    boards = gallery["boards"]
    manual_acceptance = _object(
        _verify_record(
            gallery["manual_qa_acceptance"],
            root=root,
            name="manual QA acceptance",
        ),
        name="manual QA acceptance",
    )
    manual_frame = pd.read_csv(
        _verify_record(
            manual_acceptance["manual_qa"], root=root, name="manual QA CSV"
        ),
        keep_default_na=False,
    )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "gallery_manifest": artifact_record(gallery_path),
        "postprocess_manifest": artifact_record(
            root / "08_metrics/POSTPROCESS_MANIFEST.json"
        ),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "board_count": len(boards),
        "manual_qa_count": len(manual_frame),
    }
    destination = _publish_exact(
        root / GALLERY_ACCEPTANCE_RELATIVE_PATH, payload, resume=resume
    )
    if pipeline.get("status") == RunState.P8_STATISTICS_COMPLETE.value:
        transition_pipeline_status(
            root,
            RunState.P9_GALLERIES_COMPLETE,
            first_incomplete_stage=RunState.P10_INDEPENDENT_RECOMPUTE_PASS.value,
        )
    return destination


def _load_ground_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    column = (
        "gt_grasp_rectangles"
        if "gt_grasp_rectangles" in frame.columns
        else "gt_grasp_list_json"
    )
    def plain(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return plain(value.tolist())
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    values = []
    for value in frame[column]:
        if isinstance(value, str):
            value = json.loads(value)
        values.append(plain(value))
    return frame[["sample_id"]].assign(gt_grasp_rectangles=values)


def _load_postprocess_graph(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output_path = root / "08_metrics/POSTPROCESS_MANIFEST.json"
    output = _object(output_path, name="postprocess manifest")
    _verify_self_hash(output, name="postprocess manifest")
    input_record = output.get("postprocess_inputs")
    if not isinstance(input_record, Mapping):
        raise RuntimeError("postprocess manifest lacks input authority")
    input_path = _verify_record(input_record, root=root, name="postprocess inputs")
    inputs = _object(input_path, name="postprocess inputs")
    _verify_self_hash(inputs, name="postprocess inputs")
    return output, inputs


def _table_branch_metrics(root: Path, output: Mapping[str, Any]) -> pd.DataFrame:
    record = output.get("artifacts", {}).get("table_bundle")
    if not isinstance(record, Mapping):
        raise RuntimeError("postprocess output lacks table bundle")
    manifest = _object(
        _verify_record(record, root=root, name="table bundle"), name="table bundle"
    )
    _verify_self_hash(manifest, name="table bundle")
    branch_record = manifest.get("tables", {}).get("branch_metrics.csv")
    if not isinstance(branch_record, Mapping):
        raise RuntimeError("table bundle lacks branch metrics")
    return pd.read_csv(
        _verify_record(branch_record, root=root, name="branch metrics")
    )


def _assert_metric_rows(
    computed: Mapping[str, Mapping[str, Any]], observed: pd.DataFrame
) -> None:
    if observed.duplicated(["route", "branch"]).any():
        raise RuntimeError("saved branch metrics contain duplicate identities")
    indexed = observed.set_index(["route", "branch"])
    fields = {
        "N": "N",
        "no_output": "no_output",
        "native_j_at_1_numerator": "native_correct",
        "oracle_at_5_numerator": "oracle_at_5",
        "oracle_all_numerator": "oracle_all",
        "candidate_count_mean": "candidate_count_mean",
        "candidate_count_median": "candidate_count_median",
        "candidate_count_p95": "candidate_count_p95",
        "native_j_at_1": "native_j_at_1",
        "j_at_5": "j_at_5",
        "mrr": "mrr",
        "positive_candidates_per_sample_mean": "positive_candidates_per_sample_mean",
        "positive_candidates_per_sample_median": "positive_candidates_per_sample_median",
        "positive_candidates_per_sample_p95": "positive_candidates_per_sample_p95",
    }
    integer_fields = {
        "N",
        "no_output",
        "native_j_at_1_numerator",
        "oracle_at_5_numerator",
        "oracle_all_numerator",
    }
    for key, metrics in computed.items():
        route, branch = key.split("|", 1)
        if (route, branch) not in indexed.index:
            raise RuntimeError(f"saved branch metrics omit {key}")
        row = indexed.loc[(route, branch)]
        for computed_name, observed_name in fields.items():
            exact = (
                int(metrics[computed_name]) == int(row[observed_name])
                if computed_name in integer_fields
                else math.isclose(
                    float(metrics[computed_name]),
                    float(row[observed_name]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
            if not exact:
                raise RuntimeError(
                    f"independent branch metric differs: {key}.{computed_name}"
                )
        if route == "D1" and (
            int(metrics["oracle_at_10_numerator"]) != int(row["oracle_at_10"])
            or not math.isclose(
                float(metrics["j_at_10"]),
                float(row["j_at_10"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise RuntimeError(f"independent D1 Top-10 metric differs: {key}")
        distributions = {
            "first_positive_rank_distribution": "first_positive_rank_distribution_json",
            "positive_candidates_per_sample_distribution": "positive_candidates_per_sample_distribution_json",
        }
        for computed_name, observed_name in distributions.items():
            if metrics[computed_name] != json.loads(str(row[observed_name])):
                raise RuntimeError(
                    f"independent branch distribution differs: {key}.{computed_name}"
                )


def accept_independent_recompute(
    run_dir: str | Path,
    *,
    resume: bool = False,
    expected_sample_count: int = 7_675,
) -> Path:
    """Reopen saved frames and independently recompute before accepting P10."""

    root = Path(run_dir).expanduser().resolve()
    pipeline = _pipeline(root)
    if pipeline.get("status") not in {
        RunState.P9_GALLERIES_COMPLETE.value,
        RunState.P10_INDEPENDENT_RECOMPUTE_PASS.value,
    }:
        raise PermissionError("P10 core acceptance requires P9")
    gallery_acceptance = root / GALLERY_ACCEPTANCE_RELATIVE_PATH
    verify_complete_gallery(root, expected_sample_count=expected_sample_count)
    gallery = _object(gallery_acceptance, name="P9 gallery acceptance")
    _verify_self_hash(gallery, name="P9 gallery acceptance")
    if gallery.get("status") != "PASS":
        raise RuntimeError("P9 gallery acceptance did not PASS")

    output, inputs = _load_postprocess_graph(root)
    artifacts = inputs.get("artifacts")
    outputs = output.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(outputs, Mapping):
        raise RuntimeError("postprocess artifact graph is malformed")
    sample = pd.read_parquet(
        _verify_record(artifacts["sample_manifest"], root=root, name="sample manifest")
    )
    sample = sample[["sample_id", "scene_id", "frame_id"]]
    ground_truth = _load_ground_truth(
        _verify_record(artifacts["ground_truth"], root=root, name="ground truth")
    )
    candidate_frames = []
    for key, record in sorted(artifacts["candidates"].items()):
        candidate_frames.append(
            pd.read_parquet(
                _verify_record(record, root=root, name=f"candidate geometry {key}")
            )
        )
    candidates = pd.concat(candidate_frames, ignore_index=True)
    final = pd.read_parquet(
        _verify_record(artifacts["final_outcomes"], root=root, name="final outcomes")
    )
    labels = pd.read_parquet(
        _verify_record(outputs["candidate_labels"], root=root, name="candidate labels")
    )
    outcomes = pd.read_parquet(
        _verify_record(outputs["sample_outcomes"], root=root, name="sample outcomes")
    )
    native = pd.read_parquet(
        _verify_record(outputs["native_taxonomy"], root=root, name="native taxonomy")
    )
    post = pd.read_parquet(
        _verify_record(outputs["post_r7_taxonomy"], root=root, name="post-R7 taxonomy")
    )
    statistical = pd.read_parquet(
        _verify_record(outputs["statistical_inputs"], root=root, name="stat inputs")
    )
    branch_table = _table_branch_metrics(root, output)
    checked: dict[str, Mapping[str, Any]] = {}
    routes = tuple(route for route, status in output["routes"].items() if status == "COMPLETE")
    for route in routes:
        route_manifest = sample.copy()
        technical = set(
            native.loc[
                native["route"].eq(route) & native["technical_failure"].astype(bool),
                "sample_id",
            ].astype(str)
        )
        route_manifest["technical_failure"] = route_manifest["sample_id"].astype(
            str
        ).isin(technical)
        result = independent_recompute_from_frames(
            route_manifest,
            candidates.loc[candidates["route"].eq(route)],
            ground_truth,
            k_by_route={route: (5, 10) if route == "D1" else (5,)},
            final_outcomes=final.loc[final["route"].eq(route)],
            taxonomy_definitions={
                "native_classes": list(NATIVE_CLASSES),
                "post_r7_classes": list(POST_R7_CLASSES),
            },
            saved_candidate_labels=labels.loc[labels["route"].eq(route)],
            saved_sample_outcomes=outcomes.loc[outcomes["route"].eq(route)],
            saved_native_taxonomy=native.loc[native["route"].eq(route)],
            saved_post_r7_taxonomy=post.loc[post["route"].eq(route)],
            saved_statistical_inputs=statistical.loc[statistical["route"].eq(route)],
        )
        required = {
            "candidate_labels",
            "sample_outcomes",
            "native_taxonomy",
            "post_r7_taxonomy",
            "statistical_inputs",
        }
        if result.get("status") != "PASS" or not all(
            result["exact_checks"].get(name) is True for name in required
        ):
            raise RuntimeError(f"independent saved-frame replay failed: {route}")
        checked.update(result["branch_metrics"])
    _assert_metric_rows(checked, branch_table)
    inline_record = outputs.get("independent_recompute")
    if not isinstance(inline_record, Mapping):
        raise RuntimeError("postprocess lacks inline independent diagnostic")
    inline_path = _verify_record(
        inline_record, root=root, name="inline independent diagnostic"
    )
    inline = _object(inline_path, name="inline independent diagnostic")
    _verify_self_hash(inline, name="inline independent diagnostic")
    if inline.get("status") != "PASS":
        raise RuntimeError("inline independent diagnostic did not PASS")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "process_role": "standalone saved-frame independent recompute",
        "forbidden_modules_imported": False,
        "postprocess_manifest": artifact_record(
            root / "08_metrics/POSTPROCESS_MANIFEST.json"
        ),
        "postprocess_inputs": output["postprocess_inputs"],
        "gallery_acceptance": artifact_record(gallery_acceptance),
        "inline_diagnostic": artifact_record(inline_path),
        "source_candidate_geometry": artifact_record(
            Path(str(outputs["candidate_labels"]["path"]))
        ),
        "routes": list(routes),
        "per_sample_exact_match": True,
        "metrics_exact_match": True,
        "taxonomy_exact_match": True,
        "paired_inputs_exact_match": True,
    }
    destination = _publish_exact(
        root / INDEPENDENT_ACCEPTANCE_RELATIVE_PATH, payload, resume=resume
    )
    if pipeline.get("status") == RunState.P9_GALLERIES_COMPLETE.value:
        transition_pipeline_status(
            root,
            RunState.P10_INDEPENDENT_RECOMPUTE_PASS,
            first_incomplete_stage=RunState.COMPLETE.value,
        )
    return destination


__all__ = [
    "GALLERY_ACCEPTANCE_RELATIVE_PATH",
    "INDEPENDENT_ACCEPTANCE_RELATIVE_PATH",
    "accept_gallery",
    "accept_independent_recompute",
]
