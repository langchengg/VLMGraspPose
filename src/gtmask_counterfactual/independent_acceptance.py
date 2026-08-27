"""Standalone P10 acceptance from hash-bound saved scientific frames.

Unlike :mod:`gtmask_counterfactual.acceptance`, this module has no dependency
on the P9 gallery producer.  A fresh process trusts only the hash-bound P9
acceptance record, reopens the saved postprocess graph, and uses the independent
NumPy/Pandas evaluator primitives to reproduce the scientific outputs.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from unified_reranking.hashing import canonical_sha256, sha256_file

from .contracts import RUN_STATE_ORDER, RunState
from .independent import (
    NATIVE_CLASSES,
    POST_R7_CLASSES,
    independent_recompute_from_frames,
)


GALLERY_ACCEPTANCE_RELATIVE_PATH = Path("12_case_selection/P9_GALLERY_ACCEPTANCE.json")
INDEPENDENT_ACCEPTANCE_RELATIVE_PATH = Path(
    "16_independent_recompute/INDEPENDENT_VALIDATION.json"
)

# These modules can generate candidates, select/rank them, produce taxonomies or
# reports, or validate/build the gallery.  P10 must be launched in a fresh
# process and must not inherit any of them.  Exact-prefix matching avoids false
# positives from unrelated third-party module names.
FORBIDDEN_MODULE_PREFIXES = (
    "gtmask_counterfactual.acceptance",
    "gtmask_counterfactual.candidate_matching",
    "gtmask_counterfactual.d1_adapter",
    "gtmask_counterfactual.execution",
    "gtmask_counterfactual.figures",
    "gtmask_counterfactual.g1_c1_adapter",
    "gtmask_counterfactual.galleries",
    "gtmask_counterfactual.gallery_pipeline",
    "gtmask_counterfactual.metrics",
    "gtmask_counterfactual.postprocess",
    "gtmask_counterfactual.reporting",
    "gtmask_counterfactual.resource",
    "gtmask_counterfactual.statistics",
    "gtmask_counterfactual.taxonomy",
    "unified_reranking.candidates",
    "unified_reranking.feature_extractors",
    "unified_reranking.gate",
    "unified_reranking.metrics",
    "unified_reranking.models",
    "unified_reranking.postformal_reporting",
    "unified_reranking.route_router",
    "unified_reranking.rules",
    "unified_reranking.statistics",
    "unified_reranking.training",
    "d1_reranking.candidates",
    "grasping",
    "src.grasping",
    "lightgbm",
    "torch",
)


def forbidden_modules_imported(
    modules: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Return actually loaded modules forbidden in the standalone P10 process."""

    loaded = sys.modules if modules is None else modules
    return tuple(
        sorted(
            name
            for name in loaded
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_MODULE_PREFIXES
            )
        )
    )


def _assert_import_isolation(*, stage: str) -> tuple[str, ...]:
    observed = forbidden_modules_imported()
    if observed:
        raise RuntimeError(
            f"standalone P10 imported forbidden modules at {stage}: "
            + ", ".join(observed)
        )
    return observed


def _object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain one JSON object")
    return value


def _verify_self_hash(value: Mapping[str, Any], *, name: str) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash differs")


def _artifact_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            dict(value), indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _publish_exact(path: Path, payload: Mapping[str, Any], *, resume: bool) -> Path:
    value = dict(payload)
    value["content_sha256"] = canonical_sha256(value)
    if path.exists():
        observed = _object(path, name=path.name)
        if not resume or observed != value:
            raise RuntimeError(f"existing acceptance artifact differs: {path}")
        return path
    return _atomic_json(path, value)


def _pipeline(root: Path) -> dict[str, Any]:
    return _object(root / "pipeline_status.json", name="pipeline status")


def _transition_to_p10(root: Path) -> None:
    """Perform the same monotonic lifecycle transition without audit imports."""

    status_path = root / "pipeline_status.json"
    current = _pipeline(root)
    observed = str(current.get("status", ""))
    target = RunState.P10_INDEPENDENT_RECOMPUTE_PASS.value
    if observed == target:
        return
    if (
        observed not in RUN_STATE_ORDER
        or RUN_STATE_ORDER[target] != RUN_STATE_ORDER[observed] + 1
    ):
        raise PermissionError(
            f"pipeline status must advance exactly one stage: {observed} -> {target}"
        )
    updated = {
        **current,
        "previous_status": observed,
        "status": target,
        "first_incomplete_stage": RunState.COMPLETE.value,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(status_path, updated)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        manifest = _object(manifest_path, name="counterfactual manifest")
        manifest["status"] = target
        _atomic_json(manifest_path, manifest)


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


def _verify_p9_acceptance(
    root: Path, *, postprocess_record: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = root / GALLERY_ACCEPTANCE_RELATIVE_PATH
    value = _object(path, name="P9 gallery acceptance")
    _verify_self_hash(value, name="P9 gallery acceptance")
    gallery_record = value.get("gallery_manifest")
    if not isinstance(gallery_record, Mapping):
        raise RuntimeError("P9 gallery acceptance lacks its gallery binding")
    gallery_path = _verify_record(
        gallery_record, root=root, name="P9 accepted gallery manifest"
    )
    counts = {
        name: value.get(name)
        for name in (
            "eligible_count",
            "selected_count",
            "board_count",
            "manual_qa_count",
        )
    }
    if (
        value.get("status") != "PASS"
        or value.get("postprocess_manifest") != dict(postprocess_record)
        or gallery_path != root / "14_galleries/GALLERY_MANIFEST.json"
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in counts.values()
        )
        or any(int(item) < 0 for item in counts.values())
        or int(counts["selected_count"]) == 0
        or int(counts["eligible_count"]) < int(counts["selected_count"])
        or len(
            {
                int(counts["selected_count"]),
                int(counts["board_count"]),
                int(counts["manual_qa_count"]),
            }
        )
        != 1
    ):
        raise RuntimeError("P9 gallery acceptance contract differs")
    return path, value


def _load_ground_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "gt_grasp_rectangles" in frame.columns:
        column = "gt_grasp_rectangles"
    elif "gt_grasp_list_json" in frame.columns:
        column = "gt_grasp_list_json"
    else:
        raise RuntimeError("ground truth lacks grasp rectangles")

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
    result = frame[["sample_id"]].assign(gt_grasp_rectangles=values)
    result["sample_id"] = result["sample_id"].astype(str)
    if result["sample_id"].eq("").any() or result["sample_id"].duplicated().any():
        raise RuntimeError("ground-truth sample identities differ")
    return result


def _table_branch_metrics(root: Path, output: Mapping[str, Any]) -> pd.DataFrame:
    artifacts = output.get("artifacts")
    record = artifacts.get("table_bundle") if isinstance(artifacts, Mapping) else None
    if not isinstance(record, Mapping):
        raise RuntimeError("postprocess output lacks table bundle")
    manifest = _object(
        _verify_record(record, root=root, name="table bundle"), name="table bundle"
    )
    _verify_self_hash(manifest, name="table bundle")
    tables = manifest.get("tables")
    branch_record = (
        tables.get("branch_metrics.csv") if isinstance(tables, Mapping) else None
    )
    if not isinstance(branch_record, Mapping):
        raise RuntimeError("table bundle lacks branch metrics")
    return pd.read_csv(_verify_record(branch_record, root=root, name="branch metrics"))


def _assert_metric_rows(
    computed: Mapping[str, Mapping[str, Any]], observed: pd.DataFrame
) -> None:
    if observed.duplicated(["route", "branch"]).any():
        raise RuntimeError("saved branch metrics contain duplicate identities")
    expected_identities = {tuple(key.split("|", 1)) for key in computed}
    observed_identities = set(
        observed[["route", "branch"]].astype(str).itertuples(index=False, name=None)
    )
    if observed_identities != expected_identities:
        raise RuntimeError("saved branch metric identities differ")
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
            "positive_candidates_per_sample_distribution": (
                "positive_candidates_per_sample_distribution_json"
            ),
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
    """Reopen saved frames in an import-isolated process and accept P10."""

    _assert_import_isolation(stage="entry")
    root = Path(run_dir).expanduser().resolve()
    pipeline = _pipeline(root)
    if pipeline.get("status") not in {
        RunState.P9_GALLERIES_COMPLETE.value,
        RunState.P10_INDEPENDENT_RECOMPUTE_PASS.value,
    }:
        raise PermissionError("P10 core acceptance requires P9")

    output, inputs = _load_postprocess_graph(root)
    postprocess_path = root / "08_metrics/POSTPROCESS_MANIFEST.json"
    postprocess_record = _artifact_record(postprocess_path)
    gallery_acceptance, _ = _verify_p9_acceptance(
        root, postprocess_record=postprocess_record
    )
    artifacts = inputs.get("artifacts")
    outputs = output.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(outputs, Mapping):
        raise RuntimeError("postprocess artifact graph is malformed")

    sample = pd.read_parquet(
        _verify_record(artifacts["sample_manifest"], root=root, name="sample manifest")
    )
    required_sample = {"sample_id", "scene_id", "frame_id"}
    if not required_sample.issubset(sample.columns):
        raise RuntimeError("sample manifest lacks independent identity fields")
    sample = sample[["sample_id", "scene_id", "frame_id"]].copy()
    sample["sample_id"] = sample["sample_id"].astype(str)
    if (
        len(sample) != int(expected_sample_count)
        or sample["sample_id"].eq("").any()
        or sample["sample_id"].duplicated().any()
    ):
        raise RuntimeError("independent sample denominator differs")
    denominator = set(sample["sample_id"])
    ground_truth = _load_ground_truth(
        _verify_record(artifacts["ground_truth"], root=root, name="ground truth")
    )
    if set(ground_truth["sample_id"]) != denominator:
        raise RuntimeError("independent ground-truth denominator differs")

    candidate_records = artifacts.get("candidates")
    if not isinstance(candidate_records, Mapping) or not candidate_records:
        raise RuntimeError("postprocess inputs lack candidate geometry")
    candidate_frames = [
        pd.read_parquet(
            _verify_record(record, root=root, name=f"candidate geometry {key}")
        )
        for key, record in sorted(candidate_records.items())
    ]
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

    route_status = output.get("routes")
    if not isinstance(route_status, Mapping):
        raise RuntimeError("postprocess output lacks route status")
    routes = tuple(
        route for route, status in route_status.items() if status == "COMPLETE"
    )
    if not routes or not set(routes).issubset({"G1", "C1", "D1"}):
        raise RuntimeError("independent route availability differs")
    checked: dict[str, Mapping[str, Any]] = {}
    for route in routes:
        route_manifest = sample.copy()
        technical = set(
            native.loc[
                native["route"].eq(route) & native["technical_failure"].astype(bool),
                "sample_id",
            ].astype(str)
        )
        route_manifest["technical_failure"] = route_manifest["sample_id"].isin(
            technical
        )
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
        required_checks = {
            "candidate_labels",
            "sample_outcomes",
            "native_taxonomy",
            "post_r7_taxonomy",
            "statistical_inputs",
        }
        if result.get("status") != "PASS" or not all(
            result["exact_checks"].get(name) is True for name in required_checks
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

    observed_forbidden = _assert_import_isolation(stage="pre-publication")
    import_audit = {
        "status": "PASS",
        "forbidden_module_prefixes": list(FORBIDDEN_MODULE_PREFIXES),
        "observed_forbidden_modules": list(observed_forbidden),
    }
    payload = {
        "schema_version": 2,
        "status": "PASS",
        "process_role": "standalone saved-frame independent recompute",
        "forbidden_modules_imported": bool(observed_forbidden),
        "forbidden_import_audit": import_audit,
        "postprocess_manifest": postprocess_record,
        "postprocess_inputs": output["postprocess_inputs"],
        "gallery_acceptance": _artifact_record(gallery_acceptance),
        "inline_diagnostic": _artifact_record(inline_path),
        "source_candidate_geometry": _artifact_record(
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
        _transition_to_p10(root)
    return destination


__all__ = [
    "FORBIDDEN_MODULE_PREFIXES",
    "GALLERY_ACCEPTANCE_RELATIVE_PATH",
    "INDEPENDENT_ACCEPTANCE_RELATIVE_PATH",
    "accept_independent_recompute",
    "forbidden_modules_imported",
]
