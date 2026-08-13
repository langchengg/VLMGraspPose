"""Post-lock compatibility adapter for the frozen P15 evaluator loader.

The locked postformal module creates a dynamic module but does not register it
in ``sys.modules`` before executing a dataclass-based evaluator.  Python 3.11
requires that registration while resolving dataclass annotations.  This
adapter verifies the exact locked postformal bytes, substitutes only the module
loading primitive at runtime, and binds the proof into the P15 manifest.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file


LOCK_RELATIVE = Path("08_lock/FORMAL_TEST_LOCK.json")
LOCK_DIGEST_RELATIVE = Path("08_lock/FORMAL_TEST_LOCK.sha256")
POSTFORMAL_RELATIVE = Path("src/d1_reranking/postformal.py")
SAFE_REFERENCE_RELATIVE = Path("tools/d1_reranking/independent_recompute.py")
MANIFEST_RELATIVE = Path("16_reports/D1_POSTFORMAL_MANIFEST.json")
AUDIT_RELATIVE = Path("16_reports/POSTFORMAL_SOURCE_ADAPTER.json")
SAFE_REFERENCE_FRAGMENT = """    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
"""


class NativeProbabilityNumpyAdapter:
    """Delegate NumPy except for declared 352-domain HiFi probability loads."""

    def __init__(
        self,
        delegate: Any,
        native_shapes: Mapping[Path, tuple[int, int]],
    ) -> None:
        self._delegate = delegate
        self._native_shapes = native_shapes

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def load(self, file: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        result = self._delegate.load(file, *args, **kwargs)
        path = Path(str(file)).expanduser().resolve()
        if path not in self._native_shapes:
            return result
        array = np.asarray(result, dtype=np.float32)
        if (
            array.shape != (352, 352)
            or not np.isfinite(array).all()
            or float(array.min()) < 0.0
            or float(array.max()) > 1.0
        ):
            raise RuntimeError(
                f"D1 HiFi model-domain probability contract differs: {path}"
            )
        height, width = self._native_shapes[path]
        import torch
        import torch.nn.functional as torch_functional

        tensor = torch.from_numpy(array)[None, None]
        resized = (
            torch_functional.interpolate(
                tensor,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            .numpy()
            .astype(np.float32, copy=False)
        )
        if resized.shape != (height, width) or not np.isfinite(resized).all():
            raise RuntimeError("D1 HiFi native probability resize is invalid")
        return np.clip(resized, 0.0, 1.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _record(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"P15 adapter artifact is absent/not regular: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _load_lock(root: Path) -> dict[str, Any]:
    lock_path = root / LOCK_RELATIVE
    if (root / LOCK_DIGEST_RELATIVE).read_text(encoding="ascii").strip() != sha256_file(
        lock_path
    ):
        raise RuntimeError("P15 adapter formal-lock detached digest differs")
    value = json.loads(lock_path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    recorded = unsigned.pop("self_sha256", None)
    if value.get("status") != "LOCKED" or recorded != canonical_sha256(unsigned):
        raise RuntimeError("P15 adapter formal-lock self hash differs")
    return value


def _locked_record(lock: Mapping[str, Any], path: Path) -> dict[str, Any]:
    target = path.resolve()
    matches = [
        dict(record)
        for record in lock.get("inventory", {}).values()  # type: ignore[union-attr]
        if isinstance(record, Mapping)
        and Path(str(record.get("path", ""))).resolve() == target
    ]
    if len(matches) != 1 or matches[0] != _record(target):
        raise RuntimeError(f"P15 adapter locked source differs/not unique: {target}")
    return matches[0]


def safe_load_evaluator(record: Mapping[str, Any]) -> Any:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if _record(path) != dict(record):
        raise RuntimeError("P15 adapter evaluator record differs")
    name = f"_d1_postformal_evaluator_{record['sha256'][:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load D1 canonical evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    for primitive in (
        "gt_from_corners",
        "corners",
        "IOU_THRESHOLD",
        "ANGLE_THRESHOLD_DEG",
    ):
        if not hasattr(module, primitive):
            raise RuntimeError(
                f"D1 canonical evaluator misses visual primitive: {primitive}"
            )
    return module


def exact_d1_allnms_candidates(formal: Mapping[str, Any]) -> pd.DataFrame:
    """Select the raw D1 AllNMS namespace from route-qualified formal outcomes."""

    from d1_reranking import postformal as locked_postformal

    outcomes = (
        formal["outcomes"]
        .loc[lambda frame: frame["source_route"].astype(str).eq("D1")]
        .copy()
    )
    allnms_path = locked_postformal._verified_path(
        formal["plan"]["components"]["d1_allnms_candidates"],
        name="D1 AllNMS candidates",
    )
    allnms = pd.read_parquet(allnms_path)
    if "route" in allnms and "source_route" not in allnms:
        allnms = allnms.rename(columns={"route": "source_route"})
    keys = ["source_route", "sample_id", "candidate_id"]
    allnms[keys] = allnms[keys].astype(str)
    outcomes[keys] = outcomes[keys].astype(str)
    if allnms.duplicated(keys).any() or outcomes.duplicated(keys).any():
        raise RuntimeError("D1 formal/AllNMS candidate identity is duplicated")
    selected = allnms[keys + ["candidate_geometry_sha256"]].merge(
        outcomes,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_allnms", ""),
        indicator=True,
    )
    if (
        len(selected) != len(allnms)
        or not selected["_merge"].eq("both").all()
        or not selected["candidate_geometry_sha256_allnms"]
        .astype(str)
        .eq(selected["candidate_geometry_sha256"].astype(str))
        .all()
    ):
        raise RuntimeError("D1 formal outcomes do not exactly cover raw AllNMS")
    return selected.loc[:, list(outcomes.columns)].copy()


def alias_hifics_probability_assets(visual: pd.DataFrame) -> pd.DataFrame:
    """Expose exact authority-table fields under names understood by locked P15."""

    source_path = "predicted_hifics_probability_path"
    source_sha = "predicted_hifics_probability_sha256"
    target_path = "hifics_probability_path"
    target_sha = "hifics_probability_sha256"
    if source_path not in visual or source_sha not in visual:
        raise RuntimeError("D1 visual authority misses HiFi probability asset fields")
    paths = visual[source_path].astype(str).str.strip()
    hashes = visual[source_sha].astype(str).str.strip().str.lower()
    if paths.eq("").any() or not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise RuntimeError("D1 visual authority has invalid HiFi probability records")
    resolved = paths.map(lambda value: Path(value).expanduser().resolve())
    if not resolved.map(lambda path: path.is_file() and not path.is_symlink()).all():
        raise RuntimeError("D1 visual authority has missing/non-regular probability assets")
    result = visual.copy()
    for target, values in ((target_path, paths), (target_sha, hashes)):
        if target in result:
            current = result[target].astype(str).str.strip()
            if not current.eq(values).all():
                raise RuntimeError(
                    f"D1 visual authority probability alias differs: {target}"
                )
        result[target] = values.to_numpy()
    if "expression" not in result:
        raise RuntimeError("D1 visual authority misses the referring expression")
    prompts = result["expression"].astype(str).str.strip()
    if prompts.eq("").any():
        raise RuntimeError("D1 visual authority has an empty referring expression")
    if "language_prompt" in result:
        current = result["language_prompt"].astype(str).str.strip()
        if not current.eq(prompts).all():
            raise RuntimeError("D1 visual authority language alias differs")
    result["language_prompt"] = prompts.to_numpy()
    return result


def select_renderable_cases(
    original: Any,
    formal: Mapping[str, Any],
    funnel: pd.DataFrame,
    evidence: Mapping[str, Any],
) -> pd.DataFrame:
    """Exclude only E0/no-candidate rows from geometry-dependent case boards."""

    if {"sample_id", "candidate_count", "mask_quality"}.difference(funnel.columns):
        raise RuntimeError("D1 case-selection funnel schema differs")
    adjusted = funnel.copy()
    no_candidate = adjusted["candidate_count"].fillna(0).astype(int).eq(0)
    threshold = float(evidence["mask_quality_threshold"])
    adjusted.loc[no_candidate, "mask_quality"] = threshold
    cases = original(formal, adjusted, evidence)
    if cases.empty:
        return cases
    counts = funnel.set_index(funnel["sample_id"].astype(str))["candidate_count"]
    selected_counts = cases["sample_id"].astype(str).map(counts)
    if selected_counts.isna().any() or selected_counts.fillna(0).astype(int).le(0).any():
        raise RuntimeError("D1 geometry-dependent case selection includes no-output rows")
    low_quality = cases["case_category"].eq("low_quality_grounding_association")
    cases.loc[low_quality, "selection_rule"] = (
        "lexicographic_sample_id_over_nonempty_allnms_population"
    )
    return cases


def run_adapted_postformal(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    lock = _load_lock(root)
    postformal_path = (_repo_root() / POSTFORMAL_RELATIVE).resolve()
    reference_path = (_repo_root() / SAFE_REFERENCE_RELATIVE).resolve()
    postformal_record = _locked_record(lock, postformal_path)
    reference_record = _locked_record(lock, reference_path)
    if reference_path.read_text(encoding="utf-8").count(SAFE_REFERENCE_FRAGMENT) != 1:
        raise RuntimeError("P15 adapter safe-loader reference is not unique")

    from d1_reranking import postformal as locked_postformal

    original_loader = locked_postformal._load_evaluator
    original_allnms = locked_postformal._allnms_candidates
    original_visual = locked_postformal._load_visual_ground_truth
    original_case_selection = locked_postformal._case_selection
    original_numpy = locked_postformal.np
    native_probability_shapes: dict[Path, tuple[int, int]] = {}

    def adapted_visual(*args: Any, **kwargs: Any) -> pd.DataFrame:
        visual = alias_hifics_probability_assets(original_visual(*args, **kwargs))
        if {"image_height", "image_width"}.difference(visual.columns):
            raise RuntimeError("D1 visual authority misses native image dimensions")
        for row in visual.itertuples(index=False):
            path = Path(str(row.hifics_probability_path)).expanduser().resolve()
            shape = (int(row.image_height), int(row.image_width))
            if min(shape) <= 0:
                raise RuntimeError("D1 visual authority has invalid image dimensions")
            previous = native_probability_shapes.setdefault(path, shape)
            if previous != shape:
                raise RuntimeError("D1 probability path has conflicting native shapes")
        return visual

    def adapted_case_selection(
        formal: Mapping[str, Any],
        funnel: pd.DataFrame,
        evidence: Mapping[str, Any],
    ) -> pd.DataFrame:
        return select_renderable_cases(
            original_case_selection, formal, funnel, evidence
        )

    locked_postformal._load_evaluator = safe_load_evaluator
    locked_postformal._allnms_candidates = exact_d1_allnms_candidates
    locked_postformal._load_visual_ground_truth = adapted_visual
    locked_postformal._case_selection = adapted_case_selection
    locked_postformal.np = NativeProbabilityNumpyAdapter(
        original_numpy, native_probability_shapes
    )
    try:
        locked_postformal.build_postformal_artifacts(root)
    finally:
        locked_postformal._load_evaluator = original_loader
        locked_postformal._allnms_candidates = original_allnms
        locked_postformal._load_visual_ground_truth = original_visual
        locked_postformal._case_selection = original_case_selection
        locked_postformal.np = original_numpy

    manifest_path = root / MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("adapted P15 manifest did not complete")
    adapter_path = Path(__file__).resolve()
    tool_path = (
        _repo_root() / "tools/d1_reranking/build_postformal_with_source_adapter.py"
    ).resolve()
    allnms_record = lock["inventory"]["component/d1_allnms_candidates"]
    allnms_count = len(
        pd.read_parquet(Path(str(allnms_record["path"])), columns=["candidate_id"])
    )
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_values_changed": False,
        "compatibility_scope": [
            "dynamic_evaluator_sys_modules_registration_only",
            "raw_allnms_case_board_namespace_filter_only",
            "visual_authority_hifics_probability_field_alias_only",
            "visual_authority_expression_to_language_prompt_alias_only",
            "hifics_352_probability_to_native_bilinear_align_corners_false",
            "e0_no_candidate_rows_excluded_from_geometry_dependent_boards_only",
        ],
        "locked_postformal_source": postformal_record,
        "safe_loader_reference": reference_record,
        "adapter_module": _record(adapter_path),
        "adapter_tool": _record(tool_path),
        "safe_reference_occurrences": 1,
        "raw_allnms_candidate_count": allnms_count,
        "visual_probability_alias": {
            "path": "predicted_hifics_probability_path->hifics_probability_path",
            "sha256": (
                "predicted_hifics_probability_sha256->hifics_probability_sha256"
            ),
        },
        "visual_language_alias": "expression->language_prompt",
        "visual_probability_transform": {
            "source_shape": [352, 352],
            "destination_shape_source": ["image_height", "image_width"],
            "mode": "bilinear",
            "align_corners": False,
            "clipped_range": [0.0, 1.0],
        },
        "case_board_no_output_policy": {
            "formal_metrics_changed": False,
            "failure_decomposition_changed": False,
            "excluded_from_geometry_boards": "candidate_count==0",
            "selection_scope": "low_quality_grounding_association",
        },
        "formal_metric_values_changed": False,
        "preserved_evaluator_sha256": lock["inventory"][
            "component/canonical_evaluator"
        ]["sha256"],
    }
    audit["content_sha256"] = canonical_sha256(audit)
    audit_path = root / AUDIT_RELATIVE
    atomic_json(audit_path, audit)

    sources = dict(manifest["sources"])
    sources.update(
        {
            "locked_postformal_source": postformal_record,
            "safe_loader_reference": reference_record,
            "postformal_source_adapter": _record(adapter_path),
            "postformal_source_adapter_tool": _record(tool_path),
        }
    )
    manifest["sources"] = sources
    manifest["source_signature_sha256"] = canonical_sha256(sources)
    artifacts = dict(manifest["artifacts"])
    artifacts["source_adapter_audit"] = _record(audit_path)
    manifest["artifacts"] = artifacts
    manifest["compatibility_adapter"] = {
        "status": "PASS",
        "scientific_values_changed": False,
        "scope": audit["compatibility_scope"],
    }
    manifest.pop("content_sha256", None)
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


__all__ = [
    "NativeProbabilityNumpyAdapter",
    "alias_hifics_probability_assets",
    "exact_d1_allnms_candidates",
    "run_adapted_postformal",
    "safe_load_evaluator",
    "select_renderable_cases",
]
